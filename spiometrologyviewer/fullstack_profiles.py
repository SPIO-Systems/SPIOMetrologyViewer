"""Glass plates and the interfaces between them, from a cross-section image of a glued glass stack.

Ported from ThicknessImageProcessor (thickness/profiles.py).

The image is levelled (tilt), then reduced to a texture profile, one value per row over the full
width. The glass is speckled with vertical streaks; glue and polymer are smoother. Two measures:

  anisotropy   mean |d/dx| / mean |d/dy| after a light blur: high in the vertically streaked glass,
               low at the interfaces; flat inside the glass, so scratches hardly show
  high-pass    mean |image - blurred image|: the speckle energy

The texture is divided by the local glass level; rows below a threshold mark an interface, which
extends while the texture stays below a second, higher level (hysteresis). Runs closer than the
minimum glass thickness are one interface (a thick polymer band can look glass-like in its middle);
the N - 1 deepest interfaces separate the N glass plates.

Inside each interface the layers are bounded by thin dark lines. The projected intensity divided by
a running median (the local background) gives a depth profile whose peaks are those lines, the same
in dark and over-exposed images. The deepest lines in a window around the interface bound the
polymer layers. A glue layer is light between two dark lines: it lies between a rising and a falling
edge of the vertical intensity gradient (Sobel).
All positions are in px of the levelled image.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, percentile_filter
from scipy.signal import find_peaks, peak_widths

ANISOTROPY, HIGHPASS = 0, 1


# ---------------------------------------------------------------- image

def to_gray(rgb):
    """float32 gray image; the LEXT TIFFs are gray stored as RGB, so one channel is enough."""
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    if np.array_equal(rgb[::97, ::97, 0], rgb[::97, ::97, 2]):
        return rgb[..., 0].astype(np.float32)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def project(img, median=False):
    """One value per row: the mean (= the sum along x / width) or the median along x."""
    return np.median(img, axis=1) if median else img.mean(axis=1)


def refine_extremum(g, i):
    """Sub-pixel position of the extremum of g at index i (parabola through i-1, i, i+1)."""
    if 0 < i < len(g) - 1:
        a, b, c = g[i - 1], g[i], g[i + 1]
        d = a - 2 * b + c
        if d != 0:
            return i + float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
    return float(i)


# ---------------------------------------------------------------- tilt

def block_profiles(gray, blocks):
    """Mean profile of each of `blocks` vertical blocks (blocks, H) and the block centres (x, px)."""
    edges = np.linspace(0, gray.shape[1], blocks + 1).astype(int)
    profs = np.stack([gray[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])])
    return profs, (edges[:-1] + edges[1:]) / 2


def tilt_sharpness(gray, angles_deg, blocks=16, sigma=2.0):
    """Sharpness of the projection (sum of squared gradient) when the rows are sheared by each angle.

    Shearing the block profiles instead of the image keeps this cheap; a correctly levelled stack
    gives the sharpest projected lines."""
    profs, xc = block_profiles(gray, blocks)
    profs = gaussian_filter1d(profs, sigma, axis=1)
    y = np.arange(profs.shape[1], dtype=np.float64)
    xc = xc - gray.shape[1] / 2
    out = []
    for a in angles_deg:
        shift = xc * np.tan(np.radians(a))
        p = sum(np.interp(y + s, y, pr) for s, pr in zip(shift, profs))
        out.append(float(np.sum(np.diff(p) ** 2)))
    return np.array(out)


def estimate_tilt(gray, range_deg=1.0, steps=81, blocks=16):
    """(best angle in degrees, angles, sharpness). Positive = the layers fall to the right."""
    angles = np.linspace(-range_deg, range_deg, steps)
    sharp = tilt_sharpness(gray, angles, blocks)
    i = int(np.argmax(sharp))
    best = angles[i]
    if 0 < i < len(angles) - 1:   # parabola through the top three for a finer angle
        a, b, c = sharp[i - 1:i + 2]
        d = a - 2 * b + c
        if d < 0:
            best += 0.5 * (a - c) / d * (angles[1] - angles[0])
    return float(best), angles, sharp


def deskew(gray, angle_deg):
    """Shear rows so that layers tilted by `angle_deg` become horizontal (small angles)."""
    if abs(angle_deg) < 1e-6:
        return gray
    h, w = gray.shape
    t = np.tan(np.radians(angle_deg))
    m = np.float32([[1, 0, 0], [t, 1, -t * w / 2]])
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- air edges

def air_edges(intensity, threshold, sigma=2.0, search=60):
    """(top, bottom) of the stack: where the smoothed intensity first / last exceeds `threshold`,
    refined to the steepest gradient within `search` px."""
    smooth = gaussian_filter1d(intensity.astype(np.float64), sigma)
    above = np.flatnonzero(smooth > threshold)
    if len(above) < 2:
        raise ValueError(f"No row brighter than the air threshold {threshold}")
    g = np.gradient(smooth)
    a, b = above[0], above[-1]
    lo, hi = max(a - search, 0), min(a + search, len(g))
    top = refine_extremum(g, lo + int(np.argmax(g[lo:hi])))
    lo, hi = max(b - search, 0), min(b + search, len(g))
    bottom = refine_extremum(g, lo + int(np.argmin(g[lo:hi])))
    return top, bottom


# ---------------------------------------------------------------- texture

def highpass_energy(gray, sigma):
    """|image - blurred image|: large in the speckled glass, small in smooth glue and air."""
    return np.abs(gray - cv2.GaussianBlur(gray, (0, 0), sigma))


def texture_profile(gray, measure=ANISOTROPY, blur_sigma=1.0, highpass_sigma=3.0, smooth_sigma=15.0):
    """Texture per row over the full width, smoothed along y by smooth_sigma."""
    if measure == HIGHPASS:
        return gaussian_filter1d(project(highpass_energy(gray, highpass_sigma)).astype(np.float64), smooth_sigma)
    b = cv2.GaussianBlur(gray, (0, 0), blur_sigma) if blur_sigma > 0 else gray
    gx = gaussian_filter1d(np.abs(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)).mean(1).astype(np.float64), smooth_sigma)
    gy = gaussian_filter1d(np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)).mean(1).astype(np.float64), smooth_sigma)
    return gx / np.maximum(gy, 1e-6)


# ---------------------------------------------------------------- glass plates

@dataclass
class Plates:
    top: float                  # air -> first glass
    bottom: float               # last glass -> air
    plates: list                # [(name, top, bottom)] top to bottom
    interfaces: list            # [(start, end, name above, name below)] top to bottom
    texture: np.ndarray         # texture / glass level (glass ~ 1)
    level: np.ndarray           # glass level per row (same units as the raw texture)
    nonglass: list              # every non-glass run [(start, end)] before grouping
    notes: list


def runs(mask):
    """[(start, end)] of the True runs of a boolean array (end exclusive)."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def glass_level(texture, lo, hi, percentile=75, window=1501):
    """Texture of the glass per row: a running `percentile` over `window` px (follows slow drifts
    along the stack), or one value for the whole stack if window is 0. Outside lo:hi it is constant."""
    seg = texture[lo:hi]
    level = np.full(len(texture), np.percentile(seg, percentile))
    if window:
        run = percentile_filter(seg, percentile, size=int(window) | 1, mode="nearest")
        level[lo:hi] = run
        level[:lo], level[hi:] = run[0], run[-1]
    return level


def find_glass_plates(texture, top, bottom, names, threshold=0.85, extend=0.95, min_glass=600, edge_margin=300,
                      glass_percentile=75, level_window=1501):
    """The glass plates `names` (top to bottom) and the interfaces between them.

    The texture is divided by the glass level (glass ~ 1). Inside the stack, ignoring `edge_margin`
    px at the air edges (where the glass is often chipped), rows below `threshold` mark an
    interface, which extends over the neighbouring rows below `extend` (hysteresis: a polymer band
    only a little smoother than glass is kept with its glue line). Runs closer than `min_glass` are
    one interface; the len(names) - 1 deepest interfaces are kept."""
    n = len(names)
    lo, hi = int(np.ceil(top + edge_margin)), int(np.floor(bottom - edge_margin))
    if hi - lo < 10:
        raise ValueError("The stack is too short for the edge margin")
    level = glass_level(texture, lo, hi, glass_percentile, level_window)
    norm = texture / np.maximum(level, 1e-9)
    core = runs(norm[lo:hi] < threshold)
    found = [(int(lo + a), int(lo + b)) for a, b in runs(norm[lo:hi] < max(extend, threshold))
             if any(a <= c < b for c, _ in core)]
    groups = []
    for a, b in found:
        if groups and a - groups[-1][1] < min_glass:
            groups[-1] = (groups[-1][0], b)
        else:
            groups.append((a, b))
    notes = []
    if len(groups) > n - 1:
        depth = [norm[a:b].min() for a, b in groups]
        keep = sorted(np.argsort(depth)[:n - 1])
        notes.append(f"{len(groups)} interfaces found, kept the {n - 1} deepest")
        groups = [groups[i] for i in keep]
    elif len(groups) < n - 1:
        notes.append(f"Found {len(groups)} of {n - 1} interfaces: raise the threshold or lower min glass")
    edges = [top] + [v for g in groups for v in g] + [bottom]
    k = len(groups) + 1
    plates = [(names[i] if k == n else f"glass {i + 1}", float(edges[2 * i]), float(edges[2 * i + 1]))
              for i in range(k)]
    interfaces = [(float(a), float(b), plates[i][0], plates[i + 1][0]) for i, (a, b) in enumerate(groups)]
    for name, a, b in plates:
        if b - a < min_glass:
            notes.append(f"{name} is only {b - a:.0f} px: check the threshold")
    return Plates(top, bottom, plates, interfaces, norm, level, found, notes)


def interface_crops(img, plates, margin=30):
    """[(name, row0, row1, crop)] of each interface with `margin` px into the glass on both sides."""
    out = []
    for a, b, above, below in plates.interfaces:
        r0, r1 = max(int(np.floor(a - margin)), 0), min(int(np.ceil(b + margin)), img.shape[0])
        out.append((f"{above}|{below}", r0, r1, img[r0:r1]))
    return out


# ---------------------------------------------------------------- dark lines in the interfaces

def relative_depth(intensity, sigma=1.5, bg_window=301):
    """(smoothed intensity, background, depth): depth = 1 - smoothed / background, where the
    background is a running median over `bg_window` px. Dark lines are its peaks."""
    smooth = gaussian_filter1d(intensity.astype(np.float64), sigma)
    bg = median_filter(smooth, size=int(bg_window) | 1, mode="nearest")
    return smooth, bg, 1 - smooth / np.maximum(bg, 1e-6)


@dataclass
class Lines:
    pos: np.ndarray       # sub-pixel row of each dark line, top to bottom
    depth: np.ndarray     # prominence in the depth profile (0..1)
    left: np.ndarray      # upper and lower edge at half depth (sub-pixel rows)
    right: np.ndarray
    peak: np.ndarray      # integer row of each line

    def __len__(self):
        return len(self.pos)


def find_lines(depth, lo, hi, min_depth=0.15, min_sep=4, min_steepness=0.005):
    """Dark lines between rows lo and hi: peaks of `depth` with prominence >= min_depth and
    prominence / (width at half depth) >= min_steepness. The steepness test drops broad, shallow
    brightness waves in the glass, which are not boundaries."""
    lo, hi = int(max(lo, 1)), int(min(hi, len(depth) - 1))
    seg = depth[lo:hi]
    peaks, props = find_peaks(seg, prominence=min_depth, distance=max(1, int(min_sep)))
    if len(peaks) == 0:
        return Lines(*(np.zeros(0) for _ in range(4)), np.zeros(0, int))
    _, _, left, right = peak_widths(seg, peaks, rel_height=0.5, prominence_data=(
        props["prominences"], props["left_bases"], props["right_bases"]))
    keep = props["prominences"] / np.maximum(right - left, 1) >= min_steepness
    pos = np.array([refine_extremum(seg, p) for p in peaks[keep]]) + lo
    return Lines(pos, props["prominences"][keep], left[keep] + lo, right[keep] + lo, peaks[keep] + lo)


@dataclass
class Window:
    """The rows around one interface and the dark lines found in them."""
    label: str                    # e.g. "P1_Glass|P2_Glass"
    lo: int
    hi: int
    lines: Lines
    extract: list                 # layers of this interface to extract (top to bottom)
    n_lines: int                  # lines needed for them (see interface_plan)
    used: list = field(default_factory=list)   # indices of the lines used as boundaries (top to bottom)
    edges: tuple = None           # (top, bottom) of the glue found in this window (gradient edges)
    search: tuple = None          # rows (lo, hi) searched for the glue


@dataclass
class Layer:
    name: str
    top: float
    bottom: float
    method: str                   # "lines" (line centres), "edges" / "dark band" (gradient edges, glue)
    interface: str

    @property
    def thickness(self):
        return self.bottom - self.top


def interface_plan(between):
    """(layers to extract, dark lines needed) for the layers between two glasses. The extracted
    layers must be the bottom part of the interface (the layers above them are left out): their
    boundaries are one line each, plus one line for the upper glass edge if layers are left out."""
    ext = [lay for lay in between if lay.get("extract", True)]
    if not ext:
        return [], 0
    if between[len(between) - len(ext):] != ext:
        raise ValueError("Only the layers at the bottom of an interface can be extracted: "
                         + ", ".join(lay["name"] for lay in between))
    return ext, len(ext) + 1 + (len(ext) < len(between))


def interface_windows(depth, plates, layers, search=150, min_depth=0.15, min_steepness=0.005, min_sep=4):
    """A Window with its dark lines for every interface, `search` px beyond the texture interface
    on both sides (the texture can stop inside a glass-like polymer)."""
    out = []
    for (a, b, above, below), between in zip(plates.interfaces, interfaces(layers)):
        lo, hi = int(a - search), int(b + search)
        ext, n = interface_plan(between)
        out.append(Window(f"{above}|{below}", max(lo, 0), min(hi, len(depth)),
                          find_lines(depth, lo, hi, min_depth, min_sep, min_steepness), ext, n))
    return out


def interfaces(layers):
    """The layers between consecutive glasses: [[layer, ...], ...] (one list per interface)."""
    out, cur = [], None
    for lay in layers:
        if lay["type"] == "glass":
            if cur is not None:
                out.append(cur)
            cur = []
        elif cur is None:
            raise ValueError(f"The stack must start with a glass, not {lay['name']}")
        else:
            cur.append(lay)
    return out


def deepest(lines, n):
    """Indices of the n deepest lines, top to bottom; None if there are fewer."""
    if len(lines) < n:
        return None
    return sorted(int(i) for i in np.argsort(lines.depth)[::-1][:n])


def polymer_layers(windows):
    """{name: Layer} of every extracted polymer that lies directly on the glass below its interface:
    between the two lowest of the window's n_lines deepest lines. Sets Window.used."""
    out = {}
    for w in windows:
        if not w.extract or w.extract[-1]["type"] != "polymer":
            continue
        use = deepest(w.lines, w.n_lines)
        if use is None:
            continue
        w.used = use
        out[w.extract[-1]["name"]] = Layer(w.extract[-1]["name"], float(w.lines.pos[use[-2]]),
                                           float(w.lines.pos[use[-1]]), "lines", w.label)
    return out


# ---------------------------------------------------------------- glue layers: gradient edges

def gradient_profile(level, sigma=1.5, bg_window=301):
    """(relative intensity, relative gradient) per row: the Gaussian-blurred image projected along x,
    and its vertical Sobel gradient (intensity per px) averaged along x, both divided by the running
    median background so that dark and over-exposed images compare."""
    b = cv2.GaussianBlur(level, (0, 0), sigma) if sigma > 0 else level
    prof = b.mean(axis=1).astype(np.float64)
    bg = np.maximum(median_filter(prof, size=int(bg_window) | 1, mode="nearest"), 1e-6)
    grad = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3).mean(axis=1).astype(np.float64) / 8
    return prof / bg, grad / bg


def glue_edges(rel, grad, lo, hi, dark_max=0.85, flank=10, min_t=3, max_t=150, min_contrast=0.02, max_band=60):
    """(top, bottom, method) of a glue layer between rows lo and hi, or None.

    A glue layer is light between dark boundary lines: going down, a rising edge (dark -> light)
    then a falling edge (light -> dark). The rising edge must come out of a dark line (relative
    intensity <= dark_max within `flank` px above it), the falling edge must go into one (within
    `flank` px below it), and the interior must be at least min_contrast lighter than the darker of
    the two lines. Of those pairs, min_t..max_t px apart, the one with the largest
    (rising - falling gradient) x interior contrast wins: method "edges".
    A glue too thin to show a light interior is one dark band: then the strongest falling edge
    followed by a rising edge at most max_band px further, with a dark row between: "dark band"."""
    lo, hi = int(max(lo, 1)), int(min(hi, len(grad) - 1))
    i = np.arange(lo, hi)
    peak = (grad[i] >= grad[i - 1]) & (grad[i] >= grad[i + 1])
    dip = (grad[i] <= grad[i - 1]) & (grad[i] <= grad[i + 1])
    rising, falling = i[peak & (grad[i] > 0)], i[dip & (grad[i] < 0)]
    best, pair = 0.0, None
    for r in rising:
        above = rel[max(r - flank, 0):r + 1].min()
        if above > dark_max:
            continue
        for f in falling[(falling >= r + min_t) & (falling <= r + max_t)]:
            below = rel[f:f + flank + 1].min()
            if below > dark_max:
                continue
            contrast = rel[r:f + 1].mean() - max(above, below)
            score = (grad[r] - grad[f]) * contrast
            if contrast >= min_contrast and score > best:
                best, pair = score, (r, f)
    if pair:
        return refine_extremum(grad, pair[0]), refine_extremum(grad, pair[1]), "edges"
    for f in falling:
        rs = rising[(rising > f) & (rising <= f + max_band)]
        rs = [r for r in rs if rel[f:r + 1].min() <= dark_max]
        if rs:
            r = max(rs, key=lambda r: grad[r])
            score = grad[r] - grad[f]
            if score > best:
                best, pair = score, (f, r)
    if pair:
        return refine_extremum(grad, pair[0]), refine_extremum(grad, pair[1]), "dark band"
    return None


def glue_layers(windows, rel, grad, polymers=None, dark_max=0.85, flank=10, max_t=150, min_contrast=0.02,
                max_band=60):
    """{name: Layer} of the extracted glue layers, by gradient edges (see glue_edges), each searched
    in its interface window. Where an extracted polymer lies below the glue (P1P2_Glue above
    P2_Polymer) the search stops 10 px below the polymer's top, so the polymer band is not taken.
    Sets Window.edges and Window.search."""
    polymers = polymers or {}
    out = {}
    for w in windows:
        glue = [lay for lay in w.extract if lay["type"] == "glue"]
        if not glue:
            continue
        lo, hi = w.lo, w.hi
        below = w.extract[w.extract.index(glue[0]) + 1:]
        if below and below[0]["name"] in polymers:
            hi = min(hi, int(polymers[below[0]["name"]].top) + 10)
        w.search = (lo, hi)
        found = glue_edges(rel, grad, lo, hi, dark_max, flank, 3, max_t, min_contrast, max_band)
        if found is None:
            continue
        top, bottom, method = found
        w.edges = (top, bottom)
        out[glue[0]["name"]] = Layer(glue[0]["name"], top, bottom, method, w.label)
    return out


# ---------------------------------------------------------------- combined layers and the full stack

def mixed_layers(windows):
    """{name: Layer} of the extracted layers of type "mixed" (several thin or ill-defined layers
    measured together, e.g. P3_Polymer + P3P4_Glue): between the dark lines chosen by
    polymer_layers (Window.used), one boundary per extracted layer of the interface."""
    out = {}
    for w in windows:
        if not w.used or not any(lay["type"] == "mixed" for lay in w.extract):
            continue
        b = [float(w.lines.pos[i]) for i in w.used[-(len(w.extract) + 1):]]
        for j, lay in enumerate(w.extract):
            if lay["type"] == "mixed":
                out[lay["name"]] = Layer(lay["name"], b[j], b[j + 1], "lines", w.label)
    return out


def stack_layers(layers, plates, found):
    """Every layer of the stack top to bottom as a Layer: the measured ones from `found`
    ({name: Layer}), and each glass between the measured layers next to it (the air edges at the
    top and bottom). A glass next to an unmeasured layer ends at its texture plate edge."""
    edges = {name: (a, b) for name, a, b in plates.plates}
    out = []
    for k, lay in enumerate(layers):
        if lay["type"] != "glass":
            if lay["name"] in found:
                out.append(found[lay["name"]])
            continue
        above = next((found.get(l["name"]) for l in reversed(layers[:k]) if l["type"] != "glass"), None) if k else None
        below = next((found.get(l["name"]) for l in layers[k + 1:] if l["type"] != "glass"), None)
        top = plates.top if k == 0 else (above.bottom if above else edges.get(lay["name"], (np.nan,))[0])
        last = all(l["type"] != "glass" for l in layers[k + 1:])
        bottom = plates.bottom if last else (below.top if below else edges.get(lay["name"], (np.nan, np.nan))[1])
        out.append(Layer(lay["name"], float(top), float(bottom), "glass", ""))
    return out
