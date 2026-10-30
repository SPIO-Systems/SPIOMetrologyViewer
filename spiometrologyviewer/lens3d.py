"""P4 3D profile: shape of the imprinted lenses from LEXT 3D microscope scans (.lext).

Each scan is one stitched height map of one lens (a dome in a recess, the flat surface around it in
the corners). The analysis:

  load       height map with our calibration factors (X, Y, Z) on top of the file's own scale,
             binned (bin_px) to keep it fast; noisy pixels (the steep lens rim) marked unreliable
  level      plane fitted to the corners outside the lens and subtracted
  lens       rim = lowest point around the lens (circle fit): diameter, roundness, sag
  sphere     sphere fitted to the lens (geometric least squares) -> radius of curvature R, its
             deviation from nominal_roc_um, R as a function of the fitted aperture, and a conic fit
  shape      residual to the sphere; Zernike terms (astigmatism, coma, trefoil, spherical) and the
             rest (mid spatial frequencies)
  stitching  the stitch-tile pattern (residual folded with the tile period) and its strength
  defects    bumps (particles) and pits: local deviations from the smoothed residual

All lengths are um, residual heights nm. x right, y down, as in the image.
"""
import math

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import ndimage, optimize
from skimage import measure, morphology

from . import gui_steps, io, report
from .grating import region
from .gui_steps import P, StepUI, outline, tint
from .pipeline import Analysis
from .wafer import Metric

CLASSES = ("bump", "pit")
ZERNIKE_NAMES = {5: "astig 45", 6: "astig 0", 7: "coma y", 8: "coma x", 9: "trefoil y", 10: "trefoil x",
                 11: "spherical"}


# ---------------------------------------------------------------- helpers

def bin_mean(a, valid, b):
    """Mean of the valid pixels in b x b blocks (NaN where a block has none)."""
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    v = valid[:h, :w].reshape(h // b, b, w // b, b)
    s = np.where(v, a[:h, :w].reshape(h // b, b, w // b, b), 0).sum(axis=(1, 3), dtype=np.float64)
    n = v.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan).astype(np.float32)


def smooth(z, sigma):
    """Gaussian smoothing that ignores NaN."""
    ok = np.isfinite(z)
    num = ndimage.gaussian_filter(np.where(ok, z, 0).astype(np.float64), sigma)
    den = ndimage.gaussian_filter(ok.astype(np.float64), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.2, num / den, np.nan)


def grid(s):
    """x, y (um) of the binned pixel centres."""
    ny, nx = s["z_raw"].shape
    return (np.arange(nx) + 0.5) * s["sx"], (np.arange(ny) + 0.5) * s["sy"]


def radius_map(s, cx, cy):
    x, y = grid(s)
    return np.hypot(x[None, :] - cx, y[:, None] - cy)


def subsample(mask, n_max):
    """Indices (rows, cols) of at most ~n_max pixels of `mask`, evenly spread."""
    rr, cc = np.nonzero(mask)
    step = max(1, int(np.ceil(len(rr) / n_max)))
    return rr[::step], cc[::step]


def points(s, mask, z, n_max=300_000):
    rr, cc = subsample(mask, n_max)
    x, y = grid(s)
    return x[cc], y[rr], z[rr, cc].astype(np.float64)


def fit_circle(x, y):
    """Algebraic (Kasa) circle fit -> cx, cy, r."""
    A = np.c_[x, y, np.ones_like(x)]
    c = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    return cx, cy, np.sqrt(c[2] + cx * cx + cy * cy)


def fit_sphere(x, y, z, centre=None, robust_um=0.5):
    """Geometric sphere fit: minimises the distance of the points to the sphere. With `centre`
    (xc, yc) only the centre height and R are fitted. -> xc, yc, zc, R (zc = height of the centre)."""
    # start: algebraic fit x^2+y^2+z^2 + Dx + Ey + Fz + G = 0
    A = np.c_[x, y, z, np.ones_like(x)]
    D, E, F, G = np.linalg.lstsq(A, -(x * x + y * y + z * z), rcond=None)[0]
    x0 = np.array([-D / 2, -E / 2, -F / 2, np.sqrt(max(D * D / 4 + E * E / 4 + F * F / 4 - G, 1.0))])
    if centre is not None:
        r2 = (x - centre[0]) ** 2 + (y - centre[1]) ** 2
        apex = z.max()
        R0 = x0[3] if np.isfinite(x0[3]) else 1e4
        res = optimize.least_squares(lambda p: np.sqrt(r2 + (z - p[0]) ** 2) - p[1], [apex - R0, R0],
                                     loss="soft_l1", f_scale=robust_um, x_scale=[100, 100])
        return centre[0], centre[1], res.x[0], res.x[1]
    res = optimize.least_squares(lambda p: np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2) - p[3],
                                 x0, loss="soft_l1", f_scale=robust_um, x_scale=[10, 10, 100, 100])
    return tuple(res.x)


def fit_conic(x, y, z, xc, yc, R):
    """Conic (asphere) z = z_apex - c r^2 / (1 + sqrt(1 - (1 + k) c^2 r^2)), c = 1/R0, centre fixed.
    -> R0, k."""
    r2 = (x - xc) ** 2 + (y - yc) ** 2

    def sag(p):
        c = 1 / p[1]
        return p[0] - c * r2 / (1 + np.sqrt(np.clip(1 - (1 + p[2]) * c * c * r2, 1e-9, None)))

    res = optimize.least_squares(lambda p: sag(p) - z, [z.max(), R, 0.0], x_scale=[1, 100, 1],
                                 bounds=([-np.inf, 100, -50], [np.inf, 1e6, 50]))
    return res.x[1], res.x[2]


def noll_nm(j):
    """Noll index -> (n, m), m signed (+ cos, - sin): 4 -> (2, 0) focus, 5/6 -> (2, -2)/(2, 2) astigmatism."""
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    k = j - n * (n + 1) // 2          # 1-based position within row n
    m = 2 * (k // 2) if n % 2 == 0 else 2 * ((k - 1) // 2) + 1
    return n, (0 if m == 0 else m if j % 2 == 0 else -m)


def zernike(j, rho, theta):
    """Zernike polynomial (Noll index j), normalised to unit rms over the unit disk."""
    n, m = noll_nm(j)
    ma = abs(m)
    R = np.zeros_like(rho)
    for k in range((n - ma) // 2 + 1):
        c = (-1) ** k * math.factorial(n - k) / (
            math.factorial(k) * math.factorial((n + ma) // 2 - k) * math.factorial((n - ma) // 2 - k))
        R = R + c * rho ** (n - 2 * k)
    if m == 0:
        return np.sqrt(n + 1) * R
    return np.sqrt(2 * (n + 1)) * R * (np.cos(ma * theta) if m > 0 else np.sin(ma * theta))


def n_terms(n_max):
    return (n_max + 1) * (n_max + 2) // 2


def tile_periods(res, sx, sy, ranges, highpass_um=150):
    """Stitch-tile period (um) along x and y: the strongest peak of the 2D autocorrelation of the
    high-passed residual at a lag along x (y) within ranges[0] (ranges[1]). The high-pass leaves the
    tile edges and the marks every tile repeats, not the lens shape."""
    ok = np.isfinite(res)
    hp = np.where(ok, res - smooth(res, highpass_um / min(sx, sy)), 0.0)
    w = ok.astype(np.float64)
    shape = [2 * n for n in hp.shape]   # zero padding: no wrap-around
    ac = np.fft.irfft2(np.abs(np.fft.rfft2(hp, shape)) ** 2, shape)
    nw = np.fft.irfft2(np.abs(np.fft.rfft2(w, shape)) ** 2, shape)   # number of overlapping pixels per lag
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = np.where(nw > 0.2 * nw[0, 0], ac / nw, 0.0)
    out = []
    for line, step, (lo, hi) in ((ac[0], sx, ranges[0]), (ac[:, 0], sy, ranges[1])):
        i0, i1 = int(np.floor(lo / step)), int(np.ceil(hi / step))
        seg = line[i0:i1 + 1]
        if len(seg) < 3:
            out.append(np.nan)
            continue
        i = int(np.argmax(seg)) + i0
        a, b, c = line[i - 1], line[i], line[i + 1]   # parabola through the peak: sub-pixel lag
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) < 0 else 0.0
        out.append((i + float(np.clip(d, -0.5, 0.5))) * step)
    return out


def residual_rgb(res_nm, base, rng):
    """Residual (nm) in blue (below the sphere) - white - red (above), grey `base` elsewhere."""
    t = np.clip(np.nan_to_num(res_nm / rng), -1, 1)
    r = np.where(t > 0, 1.0, 1.0 + t)
    b = np.where(t < 0, 1.0, 1.0 - t)
    g = 1.0 - np.abs(t)
    img = (np.dstack([r, g, b]) * 255).astype(np.uint8)
    out = np.where(np.isfinite(res_nm)[..., None], img, base)
    return out.astype(np.uint8)


def gray01(x):
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros(x.shape + (3,), np.uint8)
    lo, hi = np.percentile(x[ok], [1, 99.5])
    return gui_steps.gray(np.nan_to_num(x, nan=lo), lo, hi if hi > lo else lo + 1)


def height_rgb(z):
    ok = np.isfinite(z)
    if not ok.any():
        return np.zeros(z.shape + (3,), np.uint8)
    lo, hi = np.percentile(z[ok], [0.5, 99.5])
    img = gui_steps.colormap((np.nan_to_num(z, nan=lo) - lo) / max(hi - lo, 1e-6))
    img[~ok] = 0
    return img


def figure_rgb(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def circle(img, s, cx, cy, r, color, thickness=1):
    c = (int(round(cx / s["sx"] - 0.5)), int(round(cy / s["sy"] - 0.5)))
    cv2.ellipse(img, c, (int(round(r / s["sx"])), int(round(r / s["sy"]))), 0, 0, 360, color, thickness)
    return img


# ---------------------------------------------------------------- pipeline steps

def calibration(path, cfg):
    """Calibration factors (%) for the scan: the wafer folder's calibration CSV if there is one,
    else the config's."""
    cal = io.find_lext_calibration(path.parent)
    if cal is not None:
        return cal
    c = cfg["calibration"]
    return {"x_pct": c["x_pct"], "y_pct": c["y_pct"], "z_pct": c["z_pct"], "source": "config",
            "x_res": np.nan, "y_res": np.nan, "z_res": np.nan, "ignored": []}


def step_load(s, cfg):
    path = s["path"]
    d = io.load_lext(path)
    cal = calibration(path, cfg)
    b = max(1, int(cfg["bin_px"]))
    file_sx, file_sy = d["um_per_px"]
    sx, sy = file_sx * cal["x_pct"] / 100 * b, file_sy * cal["y_pct"] / 100 * b
    ok = ~d["invalid"]
    z = bin_mean(d["height_um"], ok, b) * np.float32(cal["z_pct"] / 100)
    inten = bin_mean(d["intensity"].astype(np.float32), np.ones_like(ok), b)
    # unreliable: invalid, or far off the median of its neighbours (the steep rim gives noise of tens of um)
    med = ndimage.median_filter(np.nan_to_num(z, nan=float(np.nanmedian(z))), size=3)
    reliable = np.isfinite(z) & (np.abs(z - med) <= cfg["noise"]["max_step_um"])
    reliable = ~ndimage.binary_dilation(~reliable, iterations=1)
    return {"z_raw": z, "reliable": reliable, "sx": sx, "sy": sy, "file_px": (file_sx, file_sy), "cal": cal,
            "rgb": gray01(inten), "hover": hover}


def step_level(s, cfg):
    lc = cfg["level"]
    z = s["z_raw"]
    ny, nx = z.shape
    r = radius_map(s, nx * s["sx"] / 2, ny * s["sy"] / 2)
    corner = (r > lc["corner_radius_um"]) & s["reliable"]
    x, y = grid(s)
    X, Y = np.meshgrid(x, y)
    keep = corner.copy()
    coef = np.array([0.0, 0.0, float(np.nanmedian(z[corner])) if corner.any() else 0.0])
    for _ in range(5):
        if keep.sum() < 100:
            break
        A = np.c_[X[keep], Y[keep], np.ones(keep.sum())]
        coef = np.linalg.lstsq(A, z[keep].astype(np.float64), rcond=None)[0]
        res = z - (coef[0] * X + coef[1] * Y + coef[2])
        sd = np.nanstd(res[keep])
        keep = corner & (np.abs(res) <= lc["clip_sigma"] * sd)
    plane = coef[0] * X + coef[1] * Y + coef[2]
    zl = (z - plane).astype(np.float32)
    return {"z": zl, "corner": corner, "corner_used": keep, "tilt": (coef[0] * 100, coef[1] * 100),
            "corner_rms_um": float(np.nanstd(zl[keep])) if keep.any() else np.nan}


def step_lens(s, cfg):
    lc = cfg["lens"]
    z = s["z"]
    ny, nx = z.shape
    zs = smooth(np.where(s["reliable"], z, np.nan), 1)
    r_img = radius_map(s, nx * s["sx"] / 2, ny * s["sy"] / 2)
    # (NaN excluded: argmax would return the first NaN, e.g. a defect near the apex)
    central = np.where((r_img < lc["rim_search_um"][0]) & np.isfinite(zs), zs, -np.inf)
    iy, ix = np.unravel_index(np.argmax(central), z.shape)
    x, y = grid(s)
    cx0, cy0 = x[ix], y[iy]
    # rim: the lowest point along rays from the apex
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    step = min(s["sx"], s["sy"])
    radii = np.arange(lc["rim_search_um"][0], lc["rim_search_um"][1], step)
    px = (cx0 + radii[None, :] * np.cos(angles)[:, None]) / s["sx"] - 0.5
    py = (cy0 + radii[None, :] * np.sin(angles)[:, None]) / s["sy"] - 0.5
    prof = ndimage.map_coordinates(np.nan_to_num(zs, nan=np.inf), [py, px], order=0, mode="constant", cval=np.inf)
    i = np.argmin(prof, axis=1)
    found = np.isfinite(prof[np.arange(len(angles)), i])
    rim_r, rim_z = radii[i], prof[np.arange(len(angles)), i]
    rx, ry = cx0 + rim_r * np.cos(angles), cy0 + rim_r * np.sin(angles)
    use = found.copy()
    cx, cy, R = cx0, cy0, np.nan
    for _ in range(3):   # circle fit, dropping rays that are far off (a defect or noise at the rim)
        if use.sum() < 20:
            break
        cx, cy, R = fit_circle(rx[use], ry[use])
        d = np.hypot(rx - cx, ry - cy) - R
        mad = np.median(np.abs(d[use])) * 1.4826 + 1e-6
        use = found & (np.abs(d) <= max(3 * mad, step))
    d = np.hypot(rx[use] - cx, ry[use] - cy)
    roundness = float(np.percentile(d, 98) - np.percentile(d, 2)) if use.sum() >= 20 else np.nan
    apex = float(np.nanmax(np.where(radius_map(s, cx, cy) < 0.2 * R, zs, np.nan))) if np.isfinite(R) else np.nan
    aperture = R - lc["edge_margin_um"] if np.isfinite(R) else np.nan
    r = radius_map(s, cx, cy)
    fit_area = s["reliable"] & (r < aperture) & (r >= lc["apex_exclude_um"]) if np.isfinite(aperture) else \
        np.zeros(z.shape, bool)
    return {"rim_pts": (rx[use], ry[use]), "rim_centre": (cx, cy), "rim_radius": R, "rim_roundness": roundness,
            "sag": apex - float(np.median(rim_z[use])) if use.any() else np.nan,
            "aperture": aperture, "fit_area": fit_area, "lens_found": bool(np.isfinite(R) and fit_area.sum() > 1000)}


def step_sphere(s, cfg):
    sc = cfg["sphere"]
    if not s["lens_found"]:
        return {"sphere": None, "roc_curve": [], "conic": (np.nan, np.nan)}
    cx, cy = s["rim_centre"]
    r = radius_map(s, cx, cy)
    a = s["aperture"]
    x, y, z = points(s, s["fit_area"] & (r < sc["fit_frac"] * a), s["z"])
    xc, yc, zc, R = fit_sphere(x, y, z)
    curve = []
    for f in sc["fractions"]:
        m = s["fit_area"] & (r < f * a)
        if m.sum() < 200:
            curve.append((f, np.nan))
            continue
        xf, yf, zf = points(s, m, s["z"], 150_000)
        curve.append((f, fit_sphere(xf, yf, zf, centre=(xc, yc))[3]))
    try:
        conic = fit_conic(x, y, z, xc, yc, R)
    except (ValueError, np.linalg.LinAlgError):
        conic = (np.nan, np.nan)
    return {"sphere": (xc, yc, zc, R), "roc_curve": curve, "conic": conic}


def step_shape(s, cfg):
    if s["sphere"] is None:
        return {"residual": np.full(s["z"].shape, np.nan, np.float32), "zernike": {}, "midfreq": None}
    xc, yc, zc, R = s["sphere"]
    x, y = grid(s)
    X, Y = np.meshgrid(x, y)
    area = s["fit_area"]
    res = np.full(s["z"].shape, np.nan, np.float32)
    res[area] = (np.sqrt((X[area] - xc) ** 2 + (Y[area] - yc) ** 2 + (s["z"][area] - zc) ** 2) - R) * 1000
    res[area] -= np.nanmean(res[area])
    # Zernike terms over the unit disk of the aperture
    a = s["aperture"]
    J = n_terms(cfg["zernike"]["n_max"])
    rr, cc = subsample(area, 200_000)
    rho = np.hypot(X[rr, cc] - xc, Y[rr, cc] - yc) / a
    th = np.arctan2(Y[rr, cc] - yc, X[rr, cc] - xc)
    Z = np.stack([zernike(j, rho, th) for j in range(1, J + 1)], axis=1)
    coef = np.linalg.lstsq(Z, res[rr, cc].astype(np.float64), rcond=None)[0]
    rho_all = np.hypot(X[area] - xc, Y[area] - yc) / a
    th_all = np.arctan2(Y[area] - yc, X[area] - xc)
    fit = np.zeros(rho_all.shape)
    for j in range(1, J + 1):
        fit += coef[j - 1] * zernike(j, rho_all, th_all)
    mid = np.full(res.shape, np.nan, np.float32)
    mid[area] = res[area] - fit
    return {"residual": res, "zernike": {j: float(coef[j - 1]) for j in range(1, J + 1)}, "midfreq": mid}


def step_stitching(s, cfg):
    st = cfg["stitching"]
    mid = s["midfreq"]
    empty = {"stitch": None, "stitch_map": None, "stitch_layout": None}
    if mid is None or not np.isfinite(mid).any():
        return empty
    layout = io.lext_stitching(s["path"].parent)
    if st["period_um"]:
        px_um = py_um = float(st["period_um"])
    else:
        # The stitched image's tile step is not the stage step from the layout (17013_2_P4: ~575 um vs
        # 515.6 um for 644.5 um tiles with 20 % overlap), so the layout only narrows the search.
        if layout is not None:
            seeds = [t * (1 - layout["overlap_pct"] / 100) for t in layout["tile_um"]]
            ranges = [(p * 0.85, p * 1.2) for p in seeds]
        else:
            ranges = [(400, 800), (400, 800)]
        px_um, py_um = tile_periods(mid, s["sx"], s["sy"], ranges)
    if not (np.isfinite(px_um) and np.isfinite(py_um)):
        return {**empty, "stitch_layout": layout}
    ok = np.isfinite(mid)
    x, y = grid(s)
    X, Y = np.meshgrid(x, y)
    bins = int(st["bins"])
    phx, phy = (X[ok] / px_um) % 1.0, (Y[ok] / py_um) % 1.0
    idx = np.minimum((phy * bins).astype(int), bins - 1) * bins + np.minimum((phx * bins).astype(int), bins - 1)
    v = mid[ok].astype(np.float64)
    n = np.bincount(idx, minlength=bins * bins)
    sums = np.bincount(idx, weights=v, minlength=bins * bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        pattern = np.where(n > 0, sums / n, 0.0)
    pattern -= np.sum(pattern * n) / max(n.sum(), 1)
    rms = float(np.sqrt(np.sum(n * pattern ** 2) / max(n.sum(), 1)))
    smap = np.full(mid.shape, np.nan, np.float32)
    smap[ok] = pattern[idx]
    return {"stitch": {"period_um": (px_um, py_um), "rms_nm": rms, "pattern": pattern.reshape(bins, bins)},
            "stitch_map": smap, "stitch_layout": layout}


def step_defects(s, cfg):
    dc = cfg["defect"]
    area = s["fit_area"]
    base = s["midfreq"] if s["midfreq"] is not None else np.full(s["z"].shape, np.nan, np.float32)
    if cfg["stitching"]["remove"] and s["stitch_map"] is not None:
        base = base - s["stitch_map"]
    filled = np.nan_to_num(base, nan=0.0).astype(np.float32)
    k = 2 * int(round(dc["window_um"] / min(s["sx"], s["sy"]) / 2)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(filled, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(filled, cv2.MORPH_BLACKHAT, kernel)
    px2 = s["sx"] * s["sy"]
    min_px = max(1, int(np.ceil(dc["min_area_um2"] / px2)))
    bumps = morphology.remove_small_objects((tophat > dc["min_height_nm"]) & area, max_size=min_px - 1)
    pits = morphology.remove_small_objects((blackhat > dc["min_height_nm"]) & area & ~bumps, max_size=min_px - 1)
    lab = measure.label(bumps, connectivity=2).astype(np.int32)
    nb = lab.max()
    pl = measure.label(pits, connectivity=2)
    lab[pl > 0] = pl[pl > 0] + nb
    px = (s["sx"] + s["sy"]) / 2
    cx0, cy0 = s["rim_centre"]
    objects = []
    for r in measure.regionprops(lab):
        rr, cc = r.coords[:, 0], r.coords[:, 1]
        bump = r.label <= nb
        skel = float(morphology.skeletonize(r.image).sum())
        major, minor = r.axis_major_length, max(r.axis_minor_length, 1.0)
        dist = np.hypot((r.centroid[1] + 0.5) * s["sx"] - cx0, (r.centroid[0] + 0.5) * s["sy"] - cy0)
        objects.append({
            "label_id": r.label, "class": "bump" if bump else "pit",
            "cy": r.centroid[0], "cx": r.centroid[1], "bbox": r.bbox,
            "area_um2": r.area * px2, "eq_diam_um": r.equivalent_diameter_area * px,
            "length_um": max(major, skel) * px, "feret_um": float(r.feret_diameter_max) * px,
            "elongation": major / minor, "thinness": skel ** 2 / r.area, "solidity": r.solidity,
            "edge_dist_um": float(s["aperture"] - dist), "in_inner": True, "saturated_frac": 0.0,
            "height_nm": float(tophat[rr, cc].max()) if bump else -float(blackhat[rr, cc].max()),
        })
    return {"defect_lab": lab, "objects": objects, "tophat": tophat, "blackhat": blackhat}


def build_analysis(s, cfg):
    area = s["fit_area"]
    res = s["residual"]
    ok = np.isfinite(res)
    rng = cfg["output"]["residual_range_nm"] or (float(np.percentile(np.abs(res[ok]), 99)) if ok.any() else 1.0)
    base = (gray01(s["z"]) * 0.6).astype(np.uint8)
    img = residual_rgb(res, base, max(rng, 1.0))
    if np.isfinite(s["rim_radius"]):
        cx, cy = s["rim_centre"]
        circle(img, s, cx, cy, s["rim_radius"], (0, 200, 0), 1)
        circle(img, s, cx, cy, s["aperture"], (255, 200, 0), 1)
    reg = region(area)
    zero = np.zeros(area.shape, bool)
    a = Analysis(s["name"], s["rgb"], reg, area, zero, zero, s["defect_lab"], s["objects"],
                 um_per_px=(s["sx"] + s["sy"]) / 2, classes=CLASSES, result_rgb=img)
    m = s["metrics_row"]

    def annotate(bgr):
        lines = [f"R {m['roc_um'] / 1000:.4f} mm  ({m['roc_dev_pct']:+.2f}% vs nominal)",
                 f"residual rms {m['residual_rms_nm']:.0f} nm, colour scale +-{rng:.0f} nm",
                 f"astig {m['astig_nm']:.0f}  coma {m['coma_nm']:.0f}  spherical {m['spherical_nm']:.0f} nm",
                 f"stitching {m['stitch_rms_nm']:.0f} nm rms"]
        y = bgr.shape[0] - 12 - 22 * (len(lines) - 1)
        for t in lines:
            report.outlined_text(bgr, t, (10, y))
            y += 22

    a.annotate = annotate
    a.plots = plots_rgb(s, cfg)
    return a


def image_metrics(s, cfg):
    zc = s["zernike"]
    mag = lambda *js: float(np.sqrt(sum(zc.get(j, np.nan) ** 2 for j in js))) if zc else np.nan
    res = s["residual"]
    ok = np.isfinite(res)
    sph = s["sphere"]
    R = sph[3] if sph else np.nan
    nominal = cfg["nominal_roc_um"]
    curve = {f"roc_{int(round(f * 100))}pct": rv for f, rv in s["roc_curve"]}
    spread = [rv for f, rv in s["roc_curve"] if f >= 0.3 and np.isfinite(rv)]
    objs = s["objects"]
    heights = [o["height_nm"] for o in objs if o["class"] == "bump"]
    depths = [-o["height_nm"] for o in objs if o["class"] == "pit"]
    st = s["stitch"]
    cal = s["cal"]
    J = n_terms(cfg["zernike"]["n_max"])
    return {
        "image": s["name"],
        "um_per_px": (s["sx"] + s["sy"]) / 2,
        "cal_x_pct": cal["x_pct"], "cal_y_pct": cal["y_pct"], "cal_z_pct": cal["z_pct"], "cal_source": cal["source"],
        "tilt_x_pct": s["tilt"][0], "tilt_y_pct": s["tilt"][1], "corner_rms_um": s["corner_rms_um"],
        "lens_found": s["lens_found"],
        "lens_diameter_um": 2 * s["rim_radius"], "rim_roundness_um": s["rim_roundness"], "sag_um": s["sag"],
        "aperture_um": 2 * s["aperture"],
        "valid_pct": 100.0 * s["fit_area"].sum() / max(1, (radius_map(s, *s["rim_centre"]) < s["aperture"]).sum())
        if np.isfinite(s["aperture"]) else np.nan,
        "roc_mm": R / 1000, "roc_um": R, "roc_dev_pct": 100.0 * (R - nominal) / nominal,
        **curve,
        "roc_spread_pct": 100.0 * (max(spread) - min(spread)) / nominal if spread else np.nan,
        "conic_roc_um": s["conic"][0], "conic_k": s["conic"][1],
        "residual_rms_nm": float(np.sqrt(np.mean(res[ok] ** 2))) if ok.any() else np.nan,
        "residual_pv_nm": float(np.diff(np.percentile(res[ok], [0.1, 99.9]))[0]) if ok.any() else np.nan,
        "astig_nm": mag(5, 6), "astig_angle_deg": float(np.degrees(np.arctan2(zc[5], zc[6])) / 2) if zc else np.nan,
        "coma_nm": mag(7, 8), "trefoil_nm": mag(9, 10), "spherical_nm": abs(zc.get(11, np.nan)) if zc else np.nan,
        "lowfreq_rms_nm": mag(*range(5, J + 1)),
        "midfreq_rms_nm": float(np.sqrt(np.nanmean(s["midfreq"] ** 2))) if s["midfreq"] is not None else np.nan,
        "stitch_rms_nm": st["rms_nm"] if st else np.nan,
        "stitch_period_x_um": st["period_um"][0] if st else np.nan,
        "stitch_period_y_um": st["period_um"][1] if st else np.nan,
        "bump_count": len(heights), "pit_count": len(depths),
        "bump_max_height_nm": max(heights, default=0.0), "pit_max_depth_nm": max(depths, default=0.0),
        "defect_max_nm": max(heights + depths, default=0.0),
        "defect_area_um2": float(sum(o["area_um2"] for o in objs)),
    }


def step_metrics(s, cfg):
    m = image_metrics(s, cfg)
    s2 = {**s, "metrics_row": m}
    return {"metrics": m, "analysis": build_analysis(s2, cfg)}


STEPS = [
    ("load", step_load),
    ("level", step_level),
    ("lens", step_lens),
    ("sphere", step_sphere),
    ("shape", step_shape),
    ("stitching", step_stitching),
    ("defects", step_defects),
    ("metrics", step_metrics),
]

METRICS = [
    Metric("roc_mm", "Fitted R [mm]", 0, 0, 0.001, enabled=False, follows="roc_dev_pct", decimals=4),
    Metric("roc_dev_pct", "|R deviation from nominal| [%]", 0.5, 1.0, 0.05, abs=True),
    Metric("roc_spread_pct", "R spread over aperture [%]", 0.5, 1.0, 0.05),
    Metric("residual_rms_nm", "Residual to sphere rms [nm]", 300, 500, 10),
    Metric("astig_nm", "Astigmatism [nm]", 300, 600, 10),
    Metric("coma_nm", "Coma [nm]", 200, 400, 10),
    Metric("spherical_nm", "Spherical aberration [nm]", 200, 400, 10),
    Metric("stitch_rms_nm", "Stitching pattern rms [nm]", 50, 100, 5),
    # set from the spread on wafer 17013_2_P4_26 (typical 20-40 bumps and pits, dirt particles of several um)
    Metric("rim_roundness_um", "Rim roundness [um]", 15, 30, 1),
    Metric("bump_count", "Bumps", 30, 60, 1),
    Metric("pit_count", "Pits", 45, 70, 1),
    Metric("defect_max_nm", "Largest bump / pit [nm]", 10000, 25000, 500),
]


# ---------------------------------------------------------------- plots

def plots_rgb(s, cfg):
    """R vs aperture, radial residual profile, Zernike terms and the stitch-tile pattern, as one image."""
    fig = Figure(figsize=(12, 8), dpi=90)
    ax = fig.subplots(2, 2)
    nominal = cfg["nominal_roc_um"] / 1000
    curve = [(f, rv) for f, rv in s["roc_curve"] if np.isfinite(rv)]
    a = ax[0, 0]
    if curve:
        f, rv = zip(*curve)
        a.plot(np.array(f) * 100, np.array(rv) / 1000, "o-", label="fitted R")
    a.axhline(nominal, color="k", ls="--", lw=1, label=f"nominal {nominal:.3f} mm")
    if s["sphere"]:
        a.axhline(s["sphere"][3] / 1000, color="C3", lw=1, label=f"R at {cfg['sphere']['fit_frac']:.0%}: "
                                                                    f"{s['sphere'][3] / 1000:.4f} mm")
    a.set_xlabel("fitted aperture [% of aperture radius]")
    a.set_ylabel("R [mm]")
    a.set_title("Radius of curvature vs fitted aperture")
    outer = [rv / 1000 for f, rv in curve if f >= 0.3]   # small apertures scatter a lot: keep them off the scale
    lo, hi = min(outer + [nominal]), max(outer + [nominal])
    pad = max((hi - lo) * 0.3, nominal * 0.002)
    a.set_ylim(lo - pad, hi + pad)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    res = s["residual"]
    if s["sphere"] is not None and np.isfinite(res).any():
        r = radius_map(s, s["sphere"][0], s["sphere"][1])
        ok = np.isfinite(res)
        edges = np.arange(0, s["aperture"] + 25, 25)
        idx = np.digitize(r[ok], edges)
        mean = np.array([res[ok][idx == i].mean() if (idx == i).any() else np.nan for i in range(1, len(edges))])
        std = np.array([res[ok][idx == i].std() if (idx == i).any() else np.nan for i in range(1, len(edges))])
        mid = (edges[:-1] + edges[1:]) / 2
        a.fill_between(mid, mean - std, mean + std, alpha=0.3, label="+-1 std around the circle")
        a.plot(mid, mean, label="mean around the circle")
        a.axhline(0, color="k", lw=0.8)
        a.legend(fontsize=8)
    a.set_xlabel("distance from the lens centre [um]")
    a.set_ylabel("residual to sphere [nm]")
    a.set_title("Radial residual profile")
    a.grid(alpha=0.3)

    a = ax[1, 0]
    zc = s["zernike"]
    if zc:
        js = [j for j in zc if j >= 5]
        a.bar(range(len(js)), [zc[j] for j in js], color=["C1" if j in ZERNIKE_NAMES else "C0" for j in js])
        a.set_xticks(range(len(js)))
        a.set_xticklabels([f"Z{j}\n{ZERNIKE_NAMES.get(j, '')}" for j in js], fontsize=6)
        a.axhline(0, color="k", lw=0.8)
    a.set_ylabel("rms [nm]")
    a.set_title("Zernike terms of the residual (Noll; piston, tilt, focus left out)")

    a = ax[1, 1]
    st = s.get("stitch")
    if st:
        v = np.max(np.abs(st["pattern"])) or 1
        im = a.imshow(st["pattern"], cmap="RdBu_r", vmin=-v, vmax=v,
                      extent=[0, st["period_um"][0], st["period_um"][1], 0])
        fig.colorbar(im, ax=a, label="nm")
        a.set_title(f"Mean stitch tile: {st['rms_nm']:.0f} nm rms "
                    f"(period {st['period_um'][0]:.0f} x {st['period_um'][1]:.0f} um)")
        a.set_xlabel("um")
    else:
        a.set_title("Stitch tile: not found")
    fig.suptitle(s["name"])
    fig.tight_layout()
    return figure_rgb(fig)


# ---------------------------------------------------------------- tuning GUI

def hover(s, x, y):
    """Mouse-over text in the tuning tab (the views are on the binned grid)."""
    if not (0 <= y < s["z_raw"].shape[0] and 0 <= x < s["z_raw"].shape[1]):
        return ""
    t = f"x {(x + 0.5) * s['sx']:6.0f}  y {(y + 0.5) * s['sy']:6.0f} um   height {s['z_raw'][y, x]:8.3f} um"
    if "z" in s:
        t += f"   leveled {s['z'][y, x]:8.3f} um"
    if s.get("residual") is not None and np.isfinite(s["residual"][y, x]):
        t += f"   residual {s['residual'][y, x]:+6.0f} nm"
    return t


def v_load(s, cfg):
    return tint(s["rgb"], ~s["reliable"], gui_steps.RED, 0.6)


def i_load(s, cfg):
    c, (fx, fy) = s["cal"], s["file_px"]
    b = max(1, int(cfg["bin_px"]))
    text = (f"Calibration from {c['source']}: X {c['x_pct']:.6f} %  Y {c['y_pct']:.6f} %  Z {c['z_pct']:.6f} %\n"
            f"File scale: {fx:.6f} x {fy:.6f} um/px  ->  calibrated {fx * c['x_pct'] / 100:.6f} x "
            f"{fy * c['y_pct'] / 100:.6f} um/px; binned {b} x {b}: {s['sx']:.3f} x {s['sy']:.3f} um/px\n"
            f"Unreliable (red): {100 * (~s['reliable']).mean():.1f}% of the scan")
    if c.get("ignored"):
        text += f"\nOther calibration CSVs ignored: {', '.join(c['ignored'])}"
    if np.isfinite(c.get("x_res", np.nan)) and abs(c["x_res"] / fx - 1) > 0.001:
        text += (f"\nWARNING: the CSV's resolution ({c['x_res']:.6f} um) differs from this scan's "
                 f"({fx:.6f} um): it was exported from another scan.")
    return text


def v_level(s, cfg):
    return height_rgb(s["z"])


def v_corners(s, cfg):
    img = (height_rgb(s["z"]) * 0.35).astype(np.uint8)
    res = np.where(s["corner"], s["z"], np.nan)
    sd = s["corner_rms_um"] if np.isfinite(s["corner_rms_um"]) else 1.0
    img = np.where(s["corner"][..., None], residual_rgb(res * 1000, img, 3000 * sd), img).astype(np.uint8)
    return outline(img, s["corner_used"], gui_steps.GREEN)


def i_level(s, cfg):
    return (f"Plane tilt: x {s['tilt'][0]:+.4f} %, y {s['tilt'][1]:+.4f} %\n"
            f"Corner points used: {s['corner_used'].sum()} of {s['corner'].sum()}; "
            f"their rms after leveling {s['corner_rms_um']:.3f} um")


def v_lens(s, cfg):
    img = height_rgb(s["z"])
    rx, ry = s["rim_pts"]
    for x, y in zip(rx, ry):
        cv2.circle(img, (int(x / s["sx"]), int(y / s["sy"])), 2, gui_steps.YELLOW, -1)
    if np.isfinite(s["rim_radius"]):
        cx, cy = s["rim_centre"]
        circle(img, s, cx, cy, s["rim_radius"], (255, 255, 255), 2)
        circle(img, s, cx, cy, s["aperture"], gui_steps.MAGENTA, 2)
        if cfg["lens"]["apex_exclude_um"] > 0:
            circle(img, s, cx, cy, cfg["lens"]["apex_exclude_um"], gui_steps.MAGENTA, 1)
    return img


def i_lens(s, cfg):
    if not s["lens_found"]:
        return "Lens NOT found: check the rim search range."
    return (f"Rim diameter {2 * s['rim_radius']:.1f} um, roundness {s['rim_roundness']:.1f} um (2-98 percentile)\n"
            f"Sag (apex - rim) {s['sag']:.2f} um\n"
            f"Analysed aperture (magenta) {2 * s['aperture']:.0f} um; usable pixels {s['fit_area'].sum()}")


def v_roc(s, cfg):
    return plots_rgb(s, cfg) if "zernike" in s else _roc_only(s, cfg)


def _roc_only(s, cfg):
    fig = Figure(figsize=(8, 5), dpi=90)
    a = fig.subplots()
    curve = [(f, rv) for f, rv in s["roc_curve"] if np.isfinite(rv)]
    if curve:
        f, rv = zip(*curve)
        a.plot(np.array(f) * 100, np.array(rv) / 1000, "o-")
    a.axhline(cfg["nominal_roc_um"] / 1000, color="k", ls="--", lw=1, label="nominal")
    a.set_xlabel("fitted aperture [% of aperture radius]")
    a.set_ylabel("R [mm]")
    a.legend()
    a.grid(alpha=0.3)
    fig.tight_layout()
    return figure_rgb(fig)


def v_fit_area(s, cfg):
    img = tint(height_rgb(s["z"]), ~s["fit_area"], (0, 0, 0), 0.6)
    if s["sphere"]:
        xc, yc, _, _ = s["sphere"]
        circle(img, s, xc, yc, cfg["sphere"]["fit_frac"] * s["aperture"], (255, 255, 255), 2)
    return img


def i_sphere(s, cfg):
    if not s["sphere"]:
        return "No lens, no fit."
    xc, yc, zc, R = s["sphere"]
    nominal = cfg["nominal_roc_um"]
    lines = [f"R = {R:.1f} um ({100 * (R - nominal) / nominal:+.3f}% vs nominal {nominal:.0f} um), "
             f"fitted to {cfg['sphere']['fit_frac']:.0%} of the aperture radius",
             f"Conic fit: R0 = {s['conic'][0]:.1f} um, k = {s['conic'][1]:+.4f}",
             "R vs aperture: " + "  ".join(f"{f:.0%} {rv:.0f}" for f, rv in s["roc_curve"])]
    return "\n".join(lines)


def v_residual(s, cfg):
    res = s["residual"]
    ok = np.isfinite(res)
    rng = cfg["output"]["residual_range_nm"] or (float(np.percentile(np.abs(res[ok]), 99)) if ok.any() else 1)
    return residual_rgb(res, (gray01(s["z"]) * 0.6).astype(np.uint8), max(rng, 1))


def v_midfreq(s, cfg):
    mid = s["midfreq"]
    if mid is None:
        return v_residual(s, cfg)
    ok = np.isfinite(mid)
    rng = float(np.percentile(np.abs(mid[ok]), 99)) if ok.any() else 1
    return residual_rgb(mid, (gray01(s["z"]) * 0.6).astype(np.uint8), max(rng, 1))


def i_shape(s, cfg):
    zc = s["zernike"]
    if not zc:
        return "No lens, no residual."
    res = s["residual"]
    ok = np.isfinite(res)
    mag = lambda *js: np.sqrt(sum(zc[j] ** 2 for j in js))
    return (f"Residual rms {np.sqrt(np.mean(res[ok] ** 2)):.0f} nm, PV {np.diff(np.percentile(res[ok], [0.1, 99.9]))[0]:.0f} nm"
            f" (0.1-99.9 percentile)\n"
            f"Astigmatism {mag(5, 6):.0f} nm, coma {mag(7, 8):.0f}, trefoil {mag(9, 10):.0f}, spherical {abs(zc[11]):.0f}\n"
            f"Without the Zernike terms: {np.sqrt(np.nanmean(s['midfreq'] ** 2)):.0f} nm rms (stitching, defects, noise)")


def v_stitch_map(s, cfg):
    if s["stitch_map"] is None:
        return v_midfreq(s, cfg)
    sm = s["stitch_map"]
    ok = np.isfinite(sm)
    rng = float(np.percentile(np.abs(sm[ok]), 99)) if ok.any() else 1
    return residual_rgb(sm, (gray01(s["z"]) * 0.6).astype(np.uint8), max(rng, 1))


def v_without_stitch(s, cfg):
    if s["stitch_map"] is None:
        return v_midfreq(s, cfg)
    d = s["midfreq"] - s["stitch_map"]
    ok = np.isfinite(d)
    rng = float(np.percentile(np.abs(d[ok]), 99)) if ok.any() else 1
    return residual_rgb(d, (gray01(s["z"]) * 0.6).astype(np.uint8), max(rng, 1))


def i_stitching(s, cfg):
    st, lay = s["stitch"], s["stitch_layout"]
    if not st:
        return "No stitching pattern (no lens)."
    src = (f"seeded from {lay['source']} ({lay['tiles'][0]} x {lay['tiles'][1]} tiles of {lay['tile_um'][0]:.1f} x "
           f"{lay['tile_um'][1]:.1f} um, {lay['overlap_pct']:.0f}% overlap)" if lay and not cfg["stitching"]["period_um"]
           else "from the config" if cfg["stitching"]["period_um"] else "searched 400-800 um (no matl.omp2info)")
    src = "from the autocorrelation, " + src if not cfg["stitching"]["period_um"] else src
    return (f"Tile period {st['period_um'][0]:.1f} x {st['period_um'][1]:.1f} um, {src}\n"
            f"Mean tile pattern: {st['rms_nm']:.1f} nm rms"
            + ("\nThe pattern is subtracted before looking for defects." if cfg["stitching"]["remove"] else ""))


def v_defects(s, cfg):
    img = v_without_stitch(s, cfg)
    outline(img, s["defect_lab"] > 0, (0, 0, 0), 2)
    return img


def v_tophat(s, cfg):
    t = np.maximum(s["tophat"], s["blackhat"])
    img = gui_steps.colormap(t / (2 * cfg["defect"]["min_height_nm"]))
    img[~s["fit_area"]] = (img[~s["fit_area"]] * 0.25).astype(np.uint8)
    return img


def i_defects(s, cfg):
    objs = s["objects"]
    b = [o for o in objs if o["class"] == "bump"]
    p = [o for o in objs if o["class"] == "pit"]
    return (f"Bumps: {len(b)}, highest {max((o['height_nm'] for o in b), default=0):.0f} nm\n"
            f"Pits: {len(p)}, deepest {max((-o['height_nm'] for o in p), default=0):.0f} nm\n"
            f"Total area {sum(o['area_um2'] for o in objs):.0f} um2")


def i_metrics(s, cfg):
    return gui_steps.format_metrics(s["metrics"], [
        "roc_um", "roc_dev_pct", "roc_spread_pct", "conic_roc_um", "conic_k", "lens_diameter_um",
        "rim_roundness_um", "sag_um", "residual_rms_nm", "residual_pv_nm", "astig_nm", "coma_nm", "trefoil_nm",
        "spherical_nm", "midfreq_rms_nm", "stitch_rms_nm", "bump_count", "pit_count", "defect_max_nm",
        "tilt_x_pct", "tilt_y_pct", "corner_rms_um", "cal_source"])


def step_ui():
    return [
        StepUI(
            "load", "Load the 3D scan",
            "The height map of the .lext file, with our calibration factors on top of the file's own scale: "
            "from the wafer folder's calibration CSV (a LEXT CSV export with 'X Calibration' in its header) if "
            "there is one, else the values below. The map is binned to keep the analysis fast. Pixels far "
            "off their neighbours (red, mostly the steep lens rim) are left out of everything.",
            [P(("calibration", "x_pct"), "Calibration X [%]", 95, 105, 0.000001,
               "Used when the wafer folder has no calibration CSV."),
             P(("calibration", "y_pct"), "Calibration Y [%]", 95, 105, 0.000001),
             P(("calibration", "z_pct"), "Calibration Z [%]", 95, 105, 0.000001),
             P(("bin_px",), "Binning [px]", 1, 16, 1, "Blocks of n x n pixels are averaged. 4 = ~3.8 um."),
             P(("noise", "max_step_um"), "Max. step to neighbours [um]", 0.1, 20, 0.1,
               "A pixel further than this from the median of its 3 x 3 neighbours is unreliable.")],
            {"Intensity (red = unreliable)": v_load,
             "Height": lambda s, c: height_rgb(s["z_raw"])},
            i_load),
        StepUI(
            "level", "Level on the corners",
            "A plane is fitted to the corners outside the lens (outside the circle below, around the scan "
            "centre), dropping points more than the clip limit off it, and subtracted from the whole map.",
            [P(("level", "corner_radius_um"), "Corners from radius [um]", 1500, 4000, 10,
               "Distance from the scan centre beyond which the surface counts as corner."),
             P(("level", "clip_sigma"), "Clip [sigma]", 1, 6, 0.5)],
            {"Leveled height": v_level,
             "Corners (green = used; colour = height)": v_corners},
            i_level),
        StepUI(
            "lens", "Find the lens",
            "The apex is the highest point near the centre; along 360 rays from it, the rim is the lowest "
            "point (yellow), and a circle is fitted to them (white). The analysed aperture (magenta) ends a "
            "margin inside the rim, where the surface gets too steep to measure well.",
            [P(("lens", "rim_search_um", 0), "Rim search from [um]", 500, 3000, 10),
             P(("lens", "rim_search_um", 1), "Rim search to [um]", 1000, 4000, 10),
             P(("lens", "edge_margin_um"), "Margin inside rim [um]", 0, 1000, 10),
             P(("lens", "apex_exclude_um"), "Leave out apex disk [um]", 0, 500, 5,
               "Radius around the apex left out of the fits (e.g. measurement rings at the apex).")],
            {"Rim fit": v_lens},
            i_lens),
        StepUI(
            "sphere", "Fit the sphere",
            "A sphere is fitted to the lens (shortest distance of each point to the sphere, robust to "
            "outliers) within the fit fraction of the aperture radius: its radius is R. For 'R vs aperture' "
            "the centre is kept and only R is refitted to growing parts of the lens, from the apex outward; "
            "a flat curve means a spherical lens. The conic fit allows an asphere (k = 0: sphere).",
            [P(("sphere", "fit_frac"), "Fit to fraction of aperture", 0.1, 1.0, 0.05),
             P(("nominal_roc_um",), "Nominal R [um]", 1000, 100000, 1)],
            {"R vs aperture": _roc_only,
             "Fit area": v_fit_area},
            i_sphere),
        StepUI(
            "shape", "Residual to the sphere",
            "Height difference to the sphere (normal to it; red above, blue below). Zernike polynomials over "
            "the aperture split it into astigmatism (saddle / warp), coma, trefoil, spherical aberration, "
            "and the rest: stitching, defects and noise ('Without Zernike terms').",
            [P(("zernike", "n_max"), "Zernike up to order", 2, 10, 1),
             P(("output", "residual_range_nm"), "Colour scale [nm] (0 = auto)", 0, 5000, 50)],
            {"Residual": v_residual,
             "Without Zernike terms": v_midfreq,
             "Plots": v_roc},
            i_shape),
        StepUI(
            "stitching", "Stitching pattern",
            "The scan is stitched from tiles, and the tiles leave a repeating pattern. Folding the residual "
            "(without Zernike terms) with the tile period averages all tiles into one mean tile, whose rms is "
            "the stitching measure: the part of the error every tile shares. Each tile also has its own "
            "offset and tilt, which this does not capture, so removing the mean tile only helps a little. "
            "The period is the strongest repeat of the residual's structure (2D autocorrelation), searched "
            "-15%..+20% around the tile step in the microscope's matl.omp2info, or 400-800 um without it; "
            "or set it here.",
            [P(("stitching", "period_um"), "Tile period [um] (0 = auto)", 0, 2000, 0.5),
             P(("stitching", "bins"), "Bins per tile", 8, 128, 4)],
            {"Stitch pattern": v_stitch_map,
             "Residual minus stitch pattern": v_without_stitch},
            i_stitching),
        StepUI(
            "defects", "Bumps and pits",
            "Local bumps (particles) and pits: the residual (without Zernike terms, and without the stitch "
            "pattern if set) minus its grey-level opening / closing over the window, above the minimum height.",
            [P(("defect", "window_um"), "Window [um]", 10, 400, 5,
               "Features smaller than this count; broader ones are shape, not defects."),
             P(("defect", "min_height_nm"), "Min. height / depth [nm]", 10, 2000, 10),
             P(("defect", "min_area_um2"), "Min. area [um2]", 0, 2000, 5)],
            {"Defects": v_defects,
             "Local height (top-hat)": v_tophat},
            i_defects),
        StepUI(
            "metrics", "Results",
            "Per-lens numbers, as written to results.csv by 'Run on all images'. The result image shows "
            "the residual to the sphere with the defects outlined (black = bump, white = pit).",
            [],
            {"Result": gui_steps.v_classes,
             "Plots": lambda s, c: s["analysis"].plots},
            i_metrics),
    ]
