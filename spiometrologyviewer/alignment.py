"""P3 alignment marks: x / y offset of the dark-line pattern against the bright-line pattern, and dirt.

Both patterns are made of straight vertical and horizontal lines, mirror-symmetric about their own
centre (the bright pattern: four boxes with one bright line through the gap between them). Each
pattern's line positions are found from the column / row profiles of its long straight lines; its
centre is the position most line pairs are symmetric about. Offset = dark centre - bright centre
(+x right, +y down in the image).

Dirt is searched everywhere except on the lines (and the white square / label): particles, fibres,
rings (droplet / bubble rims: a dark closed outline with a light inside) and stains (blurred
low-contrast spots).
"""
import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import measure, morphology

from . import detect, features, gui_steps, preprocess
from .gui_steps import P, StepUI, outline, tint
from .p3mirror import detect_dark_hysteresis
from .pipeline import Analysis, step_overlay
from .wafer import Metric

CLASSES = ("particle", "fibre", "ring", "stain")
BRIGHT_RGB, DARK_RGB = (255, 200, 0), (0, 160, 255)    # line colours in the views


# ---------------------------------------------------------------- lines and centres

def line_masks(gray, cfg):
    """Bright and dark thin-line masks (top-hat / black-hat), the dark one bridged across the bright
    lines (a dark line is interrupted where a bright one crosses it)."""
    lc = cfg["lines"]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lc["width_px"], lc["width_px"]))
    g = gray.astype(np.float32)
    bright = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k) > lc["bright_contrast"]
    dark = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k) > lc["dark_contrast"]
    return bright, dark


def straight(mask, length):
    """Only the long straight vertical and horizontal parts of `mask`: (vertical, horizontal)."""
    m = mask.astype(np.uint8)
    return (cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8)) > 0,
            cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, length), np.uint8)) > 0)


def peaks(lines_mask, axis, length):
    """Sub-pixel positions of the lines in a straight-line mask: x of vertical lines (axis 0), y of
    horizontal ones (axis 1)."""
    prof = lines_mask.sum(axis=0 if axis == 0 else 1).astype(float)
    prof = np.convolve(prof, [0.25, 0.5, 0.25], "same")
    if prof.max() <= 0:
        return np.zeros(0)
    idx, _ = find_peaks(prof, height=max(0.15 * prof.max(), length), distance=5)
    return np.array([float(np.dot(np.arange(p - 2, p + 3), prof[p - 2:p + 3]) / max(prof[p - 2:p + 3].sum(), 1e-6))
                     for p in idx if 2 <= p < len(prof) - 2])


def symmetry_centre(pos, lo, hi, tol):
    """The position in [lo, hi] that most lines have a mirror partner about (within `tol` px), as the
    mean midpoint of the matched pairs, and the share of lines that were matched. (nan, 0) if none."""
    best = (0, np.nan, 0.0)
    mid = (lo + hi) / 2
    for i in range(len(pos)):
        for j in range(i, len(pos)):
            c = (pos[i] + pos[j]) / 2
            if not lo <= c <= hi:
                continue
            mids = []
            for a in pos:
                d = np.abs(pos - (2 * c - a))
                if d.min() <= tol:
                    mids.append((a + pos[np.argmin(d)]) / 2)
            n = len(mids)
            if n > best[0] or (n == best[0] and abs(c - mid) < abs(best[1] - mid)):
                best = (n, float(np.mean(mids)), n / len(pos))
    return best[1], best[2]


def find_patterns(gray, cfg):
    """Line masks, line positions and centres of the bright and the dark pattern."""
    lc = cfg["lines"]
    L = int(lc["min_length_um"] / cfg["um_per_px"])
    bright, dark = line_masks(gray, cfg)
    bright_d = cv2.dilate(bright.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    bv, bh = straight(bright, L)
    dv, dh = straight(dark | bright_d, L)
    dv &= ~bright_d
    dh &= ~bright_d
    h, w = gray.shape
    search = lc["centre_search_um"] / cfg["um_per_px"]
    out = {"bright_mask": bv | bh, "dark_mask": dv | dh}
    bx, by = peaks(bv, 0, L), peaks(bh, 1, L)
    # the bright lines show up in the bridged dark mask too: keep only real dark lines
    dx = np.array([x for x in peaks(straight(dark | bright_d, L)[0], 0, L) if not len(bx) or np.abs(bx - x).min() > 3])
    dy = np.array([y for y in peaks(straight(dark | bright_d, L)[1], 1, L) if not len(by) or np.abs(by - y).min() > 3])
    for name, xs, ys in (("bright", bx, by), ("dark", dx, dy)):
        cx, qx = symmetry_centre(xs, w / 2 - search, w / 2 + search, lc["pair_tol_px"])
        cy, qy = symmetry_centre(ys, h / 2 - search, h / 2 + search, lc["pair_tol_px"])
        out[name] = {"x": xs, "y": ys, "centre": np.array([cx, cy]), "quality": (qx, qy)}
    return out


# ---------------------------------------------------------------- pipeline steps

def step_lines(s, cfg):
    return {"patterns": find_patterns(s["gray"], cfg)}


def step_offset(s, cfg):
    p, px = s["patterns"], cfg["um_per_px"]
    b, d = p["bright"]["centre"], p["dark"]["centre"]
    off = (d - b) * px
    q = min(*p["bright"]["quality"], *p["dark"]["quality"])
    return {"offset": {
        "bright_cx_px": float(b[0]), "bright_cy_px": float(b[1]),
        "dark_cx_px": float(d[0]), "dark_cy_px": float(d[1]),
        "offset_x_um": float(off[0]), "offset_y_um": float(off[1]), "offset_um": float(np.hypot(*off)),
        "fit_quality": float(q),
        "fit_error": float(1 - q),   # for the tolerances, which are "lower is better"
        "offset_ok": bool(np.all(np.isfinite(off)) and q >= cfg["lines"]["min_quality"]),
        "bright_lines": f"{len(p['bright']['x'])}+{len(p['bright']['y'])}",
        "dark_lines": f"{len(p['dark']['x'])}+{len(p['dark']['y'])}",
    }}


def step_region(s, cfg):
    """Where dirt is searched. The line pixels (plus a small margin) are inpainted, so dirt crossing a
    line (rings!) stays in one piece, and dirt is detected on that cleaned image everywhere except
    the white square / label, the scale bar and the image border. The dirt coverage is per surface area,
    lines included. "Inner" = inside the mark (box around its lines)."""
    px = cfg["um_per_px"]
    h, w = s["gray"].shape
    p = s["patterns"]
    margin = 2 * int(round(cfg["dirt"]["line_margin_um"] / px)) + 1
    lines = cv2.dilate((p["bright_mask"] | p["dark_mask"]).astype(np.uint8), np.ones((margin, margin), np.uint8)) > 0
    # the white square / label: thick bright blobs (an opening removes the thin lines first)
    bright = (cv2.morphologyEx(s["gray"].astype(np.float32), cv2.MORPH_TOPHAT, np.ones((61, 61), np.uint8)) > 60)
    bright = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    big_bright = morphology.remove_small_objects(bright, max_size=int(cfg["dirt"]["label_min_area_um2"] / px ** 2))
    k = 2 * int(cfg["dirt"]["label_margin_um"] / px) + 1     # the square casts a shadow
    labels = cv2.dilate(big_bright.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    overlay = cv2.dilate(s["overlay"].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    border = int(cfg["dirt"]["border_ignore_um"] / px)
    image = np.zeros((h, w), bool)
    image[border:h - border, border:w - border] = True
    search = image & ~labels & ~overlay
    valid = search & ~lines     # where dirt can really be seen (the lines are filled in)
    clean = cv2.inpaint(np.clip(s["gray"], 0, 255).astype(np.uint8), lines.astype(np.uint8), 5,
                        cv2.INPAINT_TELEA).astype(np.float32)

    xs = np.concatenate([p["bright"]["x"], p["dark"]["x"]])
    ys = np.concatenate([p["bright"]["y"], p["dark"]["y"]])
    mark = np.zeros((h, w), bool)
    if len(xs) and len(ys):
        mark[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1] = True
    corners = (np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]])
               if len(xs) and len(ys) else np.zeros((4, 2)))
    region = preprocess.Mirror(corners, image, cv2.distanceTransform(image.astype(np.uint8), cv2.DIST_L2, 5),
                               np.zeros((h, w), bool), bool(len(xs) and len(ys)), [])
    # coverage is per surface area, lines included (dirt on a line is filled in with the line)
    return {"mirror": region, "lines_zone": lines, "labels": labels, "search": search, "valid": valid,
            "clean": clean, "inner": search & mark, "edge_band": search & ~mark}


def directional_background(img, length_px, blur_px):
    """Background that keeps long straight features (the shaded channels between close lines) but not
    compact dirt or rings: the lower of a median along x and a median along y, at half resolution."""
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    n = max(3, int(length_px / 2) | 1)
    bg = np.minimum(ndimage.median_filter(small, size=(n, 1)), ndimage.median_filter(small, size=(1, n)))
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.maximum(cv2.GaussianBlur(bg, (0, 0), max(blur_px, 0.5)), 1.0)


def step_lighting(s, cfg):
    px = cfg["um_per_px"]
    ic = cfg["illumination"]
    bg = directional_background(s["clean"], ic["median_length_um"] / px, ic["blur_um"] / px)
    ratio = np.where(s["search"], s["clean"] / bg, 1.0).astype(np.float32)
    return {"ratio": ratio, "bg": bg, "saturated": (s["gray"] >= cfg["saturation_level"]) & s["valid"]}


def step_dark(s, cfg):
    dark = detect_dark_hysteresis(s["ratio"], s["search"], cfg)
    # inpainting leftovers (mostly at line crossings) lie (almost) entirely on or right next to a line
    lab = measure.label(dark, connectivity=2)
    near_line = cv2.dilate(s["lines_zone"].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    on_line = ndimage.mean(near_line, lab, index=np.arange(1, lab.max() + 1)) if lab.max() else []
    drop = np.flatnonzero(np.asarray(on_line) > cfg["dirt"]["max_on_line"]) + 1
    dark &= ~np.isin(lab, drop)
    st = cfg["stain"]
    sigma = st["smooth_um"] / cfg["um_per_px"]
    m = s["search"].astype(np.float32)
    smooth = cv2.GaussianBlur(s["ratio"] * m, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-3)
    k = 2 * int(st["dark_margin_um"] / cfg["um_per_px"]) + 1    # halos around rings and particles
    near_dark = cv2.dilate(dark.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    stain = (smooth < st["t_ratio"]) & s["valid"] & ~near_dark
    stain = morphology.remove_small_objects(stain, max_size=int(st["min_area_um2"] / cfg["um_per_px"] ** 2) - 1)
    return {"dark": dark, "stain": stain, "stain_ratio": smooth}


def classify(o, filled_ratio, cfg):
    c = cfg["classify"]
    if filled_ratio >= c["ring_min_fill_ratio"] and o["feret_um"] >= c["ring_min_diam_um"]:
        return "ring"
    long_shape = o["elongation"] > c["fibre_min_elongation"] or o["thinness"] > c["fibre_min_thinness"]
    if long_shape and o["feret_um"] >= c["fibre_min_length_um"]:
        return "fibre"
    return "particle"


def step_classify(s, cfg):
    lab = measure.label(s["dark"], connectivity=2).astype(np.int32)
    n = lab.max()
    # ring: the outline (small gaps bridged) encloses much more than itself
    k = 2 * cfg["classify"]["ring_gap_px"] + 1
    filled = {}
    for r in measure.regionprops(lab):
        im = np.pad(r.image, k).astype(np.uint8)
        im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        filled[r.label] = ndimage.binary_fill_holes(im).sum() / max(r.area, 1)
    stains = measure.label(s["stain"], connectivity=2)
    take = (stains > 0) & (lab == 0)
    lab[take] = stains[take] + n
    objects = features.blob_features(lab, s["ratio"], s["rgb"], s["bg"], s["mirror"], s["saturated"], s["inner"], cfg)
    for o in objects:
        o["class"] = "stain" if o["label_id"] > n else classify(o, filled.get(o["label_id"], 1.0), cfg)
        o["chip_depth_um"] = np.nan
    a = Analysis(s["name"], s["rgb"], s["mirror"], s["inner"], s["edge_band"], s["saturated"], lab, objects,
                 um_per_px=cfg["um_per_px"], classes=CLASSES,
                 annotate=lambda img, p=s["patterns"], off=s["offset"]: draw_offset(img, p, off, bgr=True))
    return {"analysis": a}


def image_metrics(a, offset, cfg):
    px = cfg["um_per_px"]
    p = cfg["presence"]
    inner_n, edge_n = a.inner.sum(), a.edge_band.sum()
    by = {c: [o for o in a.objects if o["class"] == c] for c in CLASSES}
    dirt = a.objects_lab > 0

    def pct(mask, reg, n):
        return 100.0 * (mask & reg).sum() / n if n else np.nan

    d = [o["eq_diam_um"] for o in by["particle"]]
    return {
        "image": a.name,
        "um_per_px": px,
        **offset,
        "dirt_coverage_mark_pct": pct(dirt, a.inner, inner_n),
        "dirt_coverage_outside_pct": pct(dirt, a.edge_band, edge_n),
        "dirt_coverage_total_pct": pct(dirt, a.inner | a.edge_band, inner_n + edge_n),
        "particle_count": len(d),
        "particle_max_diam_um": max(d, default=0.0),
        "particles_present": any(x >= p["particle_min_diam_um"] for x in d),
        "fibre_count": len(by["fibre"]),
        "fibre_max_length_um": max((o["feret_um"] for o in by["fibre"]), default=0.0),
        "ring_count": len(by["ring"]),
        "ring_max_diam_um": max((o["feret_um"] for o in by["ring"]), default=0.0),
        "stain_count": len(by["stain"]),
        "stain_area_pct": pct(np.isin(a.objects_lab, [o["label_id"] for o in by["stain"]]), a.inner | a.edge_band,
                              inner_n + edge_n),
    }


def step_metrics(s, cfg):
    return {"metrics": image_metrics(s["analysis"], s["offset"], cfg)}


STEPS = [
    ("overlay", step_overlay),
    ("lines", step_lines),
    ("offset", step_offset),
    ("region", step_region),
    ("lighting", step_lighting),
    ("dark", step_dark),
    ("classify", step_classify),
    ("metrics", step_metrics),
]

METRICS = [
    Metric("offset_x_um", "|Offset x| [um]", 50, 100, 5, abs=True),
    Metric("offset_y_um", "|Offset y| [um]", 50, 100, 5, abs=True),
    Metric("offset_um", "Total offset [um]", 75, 125, 5),
    Metric("fit_error", "Offset fit error (1 - quality)", 0.3, 0.5, 0.05),
    Metric("dirt_coverage_total_pct", "Dirt coverage [%]", 0.5, 2, 0.05),
    Metric("particle_count", "Particle count", 20, 50, 1),
    Metric("ring_count", "Ring count", 5, 20, 1),
]


# ---------------------------------------------------------------- drawing

def draw_offset(img, patterns, offset, bgr=False):
    """Line positions (short ticks at the image edge), both centres and the offset arrow."""
    col = (lambda c: c[::-1]) if bgr else (lambda c: c)
    h, w = img.shape[:2]
    for name, c in (("bright", BRIGHT_RGB), ("dark", DARK_RGB)):
        p = patterns[name]
        for x in p["x"]:
            cv2.line(img, (int(round(x)), 0), (int(round(x)), 14), col(c), 2)
        for y in p["y"]:
            cv2.line(img, (w - 14, int(round(y))), (w - 1, int(round(y))), col(c), 2)
    b = (offset["bright_cx_px"], offset["bright_cy_px"])
    d = (offset["dark_cx_px"], offset["dark_cy_px"])
    if np.all(np.isfinite(b)):
        cv2.drawMarker(img, tuple(int(round(v)) for v in b), col(BRIGHT_RGB), cv2.MARKER_CROSS, 40, 2)
    if np.all(np.isfinite(d)):
        cv2.drawMarker(img, tuple(int(round(v)) for v in d), col(DARK_RGB), cv2.MARKER_TILTED_CROSS, 30, 2)
    if np.all(np.isfinite(b)) and np.all(np.isfinite(d)):
        cv2.arrowedLine(img, tuple(int(round(v)) for v in b), tuple(int(round(v)) for v in d), col((255, 0, 0)), 2,
                        tipLength=0.25)
    text = (f"offset dark - bright: x {offset['offset_x_um']:+.1f} um, y {offset['offset_y_um']:+.1f} um"
            f"  (fit {offset['fit_quality']:.2f})")
    cv2.putText(img, text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col((255, 0, 0) if not offset["offset_ok"] else (0, 90, 0)),
                2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- tuning GUI

def v_masks(s, cfg):
    bright, dark = line_masks(s["gray"], cfg)
    img = (s["rgb"] * 0.4).astype(np.uint8)
    img[dark] = DARK_RGB
    img[bright] = BRIGHT_RGB
    return img


def v_lines(s, cfg):
    p = s["patterns"]
    img = (s["rgb"] * 0.5).astype(np.uint8)
    img[p["dark_mask"]] = DARK_RGB
    img[p["bright_mask"]] = BRIGHT_RGB
    return img


def i_lines(s, cfg):
    p = s["patterns"]
    fmt = lambda a: ", ".join(f"{v:.0f}" for v in a)
    return (f"Bright lines  x: {fmt(p['bright']['x'])}\n"
            f"              y: {fmt(p['bright']['y'])}\n"
            f"Dark lines    x: {fmt(p['dark']['x'])}\n"
            f"              y: {fmt(p['dark']['y'])}")


def v_offset(s, cfg):
    img = v_lines(s, cfg)
    return draw_offset(img, s["patterns"], s["offset"])


def v_offset_image(s, cfg):
    return draw_offset(s["rgb"].copy(), s["patterns"], s["offset"])


def i_offset(s, cfg):
    o, p = s["offset"], s["patterns"]
    q = lambda name: "x {:.0%}, y {:.0%}".format(*p[name]["quality"])
    return (f"Bright centre: ({o['bright_cx_px']:.1f}, {o['bright_cy_px']:.1f}) px   lines paired: {q('bright')}\n"
            f"Dark centre:   ({o['dark_cx_px']:.1f}, {o['dark_cy_px']:.1f}) px   lines paired: {q('dark')}\n"
            f"Offset dark - bright: x {o['offset_x_um']:+.1f} um, y {o['offset_y_um']:+.1f} um"
            f"  (total {o['offset_um']:.1f} um)\n"
            f"Fit quality {o['fit_quality']:.2f}: {'OK' if o['offset_ok'] else 'TOO LOW, check the lines'}")


def v_region(s, cfg):
    img = tint(s["rgb"], s["lines_zone"] | s["labels"], (120, 120, 120), 0.6)
    outline(img, s["search"], gui_steps.GREEN)
    return outline(img, s["inner"], gui_steps.YELLOW)


def i_region(s, cfg):
    px2 = cfg["um_per_px"] ** 2 / 1e6
    return (f"Searched for dirt: {s['valid'].sum() * px2:.2f} mm2 "
            f"(inside the mark {s['inner'].sum() * px2:.2f}, outside {s['edge_band'].sum() * px2:.2f})\n"
            f"Lines (+ margin, filled in, not counted): {s['lines_zone'].sum() * px2:.2f} mm2, "
            f"square / label: {s['labels'].sum() * px2:.2f} mm2")


def v_dark(s, cfg):
    img = outline(s["rgb"].copy(), s["search"], gui_steps.GREEN)
    outline(img, s["stain"], (140, 60, 30), 2)
    return outline(img, s["dark"], gui_steps.RED, 2)


def v_stain(s, cfg):
    img = gui_steps.colormap(np.clip((1.0 - s["stain_ratio"]) / (2 * max(1 - cfg["stain"]["t_ratio"], 1e-3)), 0, 1))
    img[~s["valid"]] = (img[~s["valid"]] * 0.25).astype(np.uint8)
    return outline(img, s["stain"], (255, 255, 255), 2)


def i_dark(s, cfg):
    return (f"Dark objects: {measure.label(s['dark'], connectivity=2).max()}\n"
            f"Stains: {measure.label(s['stain'], connectivity=2).max()}")


def i_classify(s, cfg):
    objs = s["analysis"].objects
    lines = []
    for cls in CLASSES:
        o = [x for x in objs if x["class"] == cls]
        lines.append(f"{cls:9s} {len(o):3d}   max size {max((x['feret_um'] for x in o), default=0):.0f} um")
    return "\n".join(lines)


def i_metrics(s, cfg):
    return gui_steps.format_metrics(s["metrics"], [
        "offset_x_um", "offset_y_um", "offset_um", "fit_quality", "offset_ok", "dirt_coverage_mark_pct",
        "dirt_coverage_outside_pct", "particle_count", "fibre_count", "ring_count", "ring_max_diam_um",
        "stain_count", "stain_area_pct"])


def step_ui():
    return [
        gui_steps.overlay_ui(local=False),
        StepUI(
            "lines", "Find the line patterns",
            "Thin bright lines stand out in a top-hat filter, thin dark lines in a black-hat filter (orange / "
            "blue in 'Line pixels'). Only long straight vertical and horizontal pieces are kept, so dirt, rings "
            "and the white square drop out. A dark line is interrupted where a bright one crosses it, so the "
            "bright lines are bridged over for the dark ones. Line positions come from the column / row "
            "profiles (ticks along the top and right edge).",
            [P(("lines", "width_px"), "Line filter width [px]", 5, 41, 2,
               "Wider than the lines, narrower than the gaps between them."),
             P(("lines", "bright_contrast"), "Bright line contrast", 5, 150, 1,
               "Grey levels above the local background."),
             P(("lines", "dark_contrast"), "Dark line contrast", 2, 60, 1,
               "Grey levels below the local background (some dark lines are faint and grey)."),
             P(("lines", "min_length_um"), "Min. line length [um]", 100, 1000, 10,
               "Shorter straight pieces are not lines.")],
            {"Straight lines": v_lines,
             "Line pixels": v_masks,
             "Image": lambda s, c: s["rgb"]},
            i_lines),
        StepUI(
            "offset", "Offset dark vs bright",
            "Each pattern is mirror-symmetric about its centre. The centre is the position that most of its lines "
            "have a mirror partner about (within the tolerance); missing or extra lines hardly matter. Offset = "
            "dark centre (blue X) - bright centre (orange +), red arrow; +x right, +y down. Fit quality = the "
            "smallest share of lines that found a partner (over x and y of both patterns).",
            [P(("lines", "pair_tol_px"), "Pair tolerance [px]", 0.5, 8, 0.5),
             P(("lines", "centre_search_um"), "Centre search [um]", 50, 1000, 10,
               "The centre is searched this far from the image centre."),
             P(("lines", "min_quality"), "Min. fit quality", 0, 1, 0.05,
               "Below this the offset is flagged (offset_ok = no).")],
            {"Offset": v_offset,
             "Offset on image": v_offset_image,
             "Image": lambda s, c: s["rgb"]},
            i_offset),
        StepUI(
            "region", "Where dirt is searched",
            "The line pixels (plus a small margin, grey) are filled in from their surroundings ('Lines removed'), "
            "so dirt crossing a line stays in one piece; dirt is then searched everywhere except the white "
            "square / label and the image border. Yellow = "
            "inside the mark (reported separately from outside).",
            [P(("dirt", "line_margin_um"), "Margin around lines [um]", 0, 15, 0.5),
             P(("dirt", "max_on_line"), "Max. share on a line", 0.1, 1, 0.05,
               "Objects lying more than this much on or right next to a line are leftovers of the line removal."),
             P(("dirt", "border_ignore_um"), "Image border ignored [um]", 0, 100, 5),
             P(("dirt", "label_margin_um"), "Margin around square / label [um]", 0, 150, 5),
             P(("dirt", "label_min_area_um2"), "Square / label min. area [um2]", 500, 50000, 500,
               "Bright blobs at least this big are the square or label, not dirt.")],
            {"Search area": v_region,
             "Lines removed": lambda s, c: np.dstack([np.clip(s["clean"], 0, 255).astype(np.uint8)] * 3),
             "Image": lambda s, c: s["rgb"]},
            i_region),
        StepUI(
            "lighting", "Even out the lighting",
            "The background brightness is the lower of a median along x and a median along y (on the image with "
            "the lines removed): long straight shading, like the channels between close lines, stays in the "
            "background, compact dirt and rings don't. Every pixel is divided by it (1.0 = clean).",
            [P(("illumination", "median_length_um"), "Median length [um]", 50, 600, 10,
               "Longer than the dirt (rings!), shorter than the line spacing variations."),
             P(("illumination", "blur_um"), "Smoothing [um]", 1, 100, 1)],
            {"Flattened (image / background)": gui_steps.v_ratio,
             "Background estimate": gui_steps.v_background,
             "Image": lambda s, c: s["rgb"]},
            lambda s, c: ""),
        StepUI(
            "dark", "Dark objects and stains",
            "Dark cores (below the threshold) grown into touching fainter pixels (red). Stains are broad, faint "
            "darkenings: the flattened image smoothed and below the stain threshold, away from the dark objects "
            "(brown).",
            [P(("dark", "t_ratio"), "Dark threshold", 0.2, 0.98, 0.01),
             P(("dark", "t_ratio_low"), "Grow threshold", 0.5, 1.0, 0.01),
             P(("dark", "min_area_px"), "Min. core size [px]", 1, 50, 1),
             P(("stain", "t_ratio"), "Stain threshold", 0.5, 1.0, 0.01),
             P(("stain", "smooth_um"), "Stain smoothing [um]", 5, 100, 5),
             P(("stain", "min_area_um2"), "Stain min. area [um2]", 0, 50000, 500),
             P(("stain", "dark_margin_um"), "Stain distance from dark objects [um]", 0, 100, 5,
               "Halos around rings and particles are not stains.")],
            {"Detections": v_dark,
             "Stain map": v_stain,
             "Below threshold": gui_steps.v_dark_threshold,
             "Image": lambda s, c: s["rgb"]},
            i_dark),
        StepUI(
            "classify", "Classify",
            "Ring = a dark outline that encloses much more than itself (droplet / bubble rim): filled area / own "
            "area above the limit. Fibre = long and thin. The rest are particles; stains keep their class.",
            [P(("classify", "ring_min_fill_ratio"), "Ring min. filled / own area", 1, 10, 0.1),
             P(("classify", "ring_gap_px"), "Ring gaps bridged [px]", 0, 10, 1,
               "Gaps in the outline up to about twice this are closed before filling."),
             P(("classify", "ring_min_diam_um"), "Ring min. size [um]", 0, 200, 5),
             P(("classify", "fibre_min_elongation"), "Fibre min. elongation", 1, 15, 0.5),
             P(("classify", "fibre_min_thinness"), "Fibre min. thinness", 1, 30, 0.5),
             P(("classify", "fibre_min_length_um"), "Fibre min. length [um]", 0, 300, 5)],
            {"Classes": gui_steps.v_classes,
             "Image": lambda s, c: s["rgb"]},
            i_classify),
        StepUI(
            "metrics", "Results",
            "Per-image numbers: the offset and its fit quality, and dirt inside / outside the mark.",
            [P(("presence", "particle_min_diam_um"), "Particle min. diameter [um]", 0, 100, 1)],
            {"Classes": gui_steps.v_classes,
             "Image": lambda s, c: s["rgb"]},
            i_metrics),
    ]
