"""P3 mirror: small particles and large defects (fibres, flakes, debris) on the cream mirror.

The mirror is the bright cream square on the dark background. The darker gold strip along its
left side has another colour (much less blue) and is left out.
"""
import cv2
import numpy as np
from skimage import measure

from . import detect, features, gui_steps, preprocess, report
from .gui_steps import P, StepUI, outline, tint
from .pipeline import Analysis, step_overlay
from .wafer import Metric

CLASSES = ("particle", "large_defect")


def cream_mask(rgb, cfg):
    """Pixels with the mirror's colour: bright red, and blue at least `min_blue_ratio` of red
    (the gold strip has far less blue, the background is dark)."""
    mc = cfg["mirror"]
    r = rgb[..., 0].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return (r >= mc["min_red"]) & (b >= mc["min_blue_ratio"] * np.maximum(r, 1))


def _longest_run(ok):
    """(first, last) index of the longest run of True, or None. Gaps up to 15 px (grid lines,
    dirt) are bridged."""
    ok = cv2.morphologyEx(np.asarray(ok, np.uint8)[None, :], cv2.MORPH_CLOSE, np.ones((1, 15), np.uint8))[0] > 0
    best, start = None, None
    for i, v in enumerate(np.append(ok, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - start > best[1] - best[0] + 1:
                best = (start, i - 1)
            start = None
    return best


def find_mirror(rgb, cfg):
    """The cream square as an axis-aligned rectangle (preprocess.Mirror), and the colour mask it
    came from. Its sides are where the share of mirror-coloured pixels per column / row drops below
    `min_line_frac`, so a pale patch on part of the gold strip doesn't widen it."""
    h, w = rgb.shape[:2]
    cream = cream_mask(rgb, cfg)
    frac = cfg["mirror"]["min_line_frac"]
    polygon = np.zeros((h, w), np.uint8)
    corners = np.zeros((4, 2))
    found = False
    rows = _longest_run(cream.mean(axis=1) >= frac * cream.mean(axis=1).max())
    if rows is not None:
        cols = _longest_run(cream[rows[0]:rows[1] + 1].mean(axis=0) >= frac)
        if cols is not None:   # refine the rows within those columns
            rows = _longest_run(cream[:, cols[0]:cols[1] + 1].mean(axis=1) >= frac) or rows
            x0, x1, y0, y1 = cols[0], cols[1] + 1, rows[0], rows[1] + 1
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
            polygon[y0:y1, x0:x1] = 1
            found = polygon.sum() > cfg["mirror"]["min_area_frac"] * h * w
    dist = cv2.distanceTransform(polygon, cv2.DIST_L2, 5)
    return preprocess.Mirror(corners, polygon > 0, dist, np.zeros((h, w), bool), bool(found), []), cream


# ---------------------------------------------------------------- pipeline steps

def step_region(s, cfg):
    px = cfg["um_per_px"]
    mirror, cream = find_mirror(s["rgb"], cfg)
    valid = mirror.dist > cfg["mirror"]["border_ignore_um"] / px
    inner = (mirror.dist >= cfg["edge_band_um"] / px) & valid
    return {"mirror": mirror, "cream": cream, "valid": valid, "inner": inner, "edge_band": valid & ~inner}


def step_lighting(s, cfg):
    ratio, bg = preprocess.flatten_illumination(s["gray"], s["mirror"].polygon, cfg)
    # the cream's red channel is clipped everywhere, so overexposure is judged on the grey value
    saturated = (s["gray"] >= cfg["saturation_level"]) & s["valid"]
    return {"ratio": ratio, "bg": bg, "saturated": saturated}


def step_dark(s, cfg):
    return {"dark": detect_dark_hysteresis(s["ratio"], s["valid"], cfg)}


def detect_dark_hysteresis(ratio, valid, cfg):
    """Dark objects (detect.detect_dark) grown into connected fainter pixels (ratio < t_ratio_low),
    so a faint fibre or a comet tail stays one object instead of a row of specks."""
    core = detect.detect_dark(ratio, valid, cfg)
    low = (ratio < cfg["dark"]["t_ratio_low"]) & valid
    lab = measure.label(low | core, connectivity=2)
    keep = np.unique(lab[core])
    return np.isin(lab, keep[keep > 0])


def classify(o, cfg):
    c = cfg["classify"]
    if o["eq_diam_um"] >= c["large_min_diam_um"] or o["feret_um"] >= c["large_min_length_um"]:
        return "large_defect"
    return "particle"


def step_classify(s, cfg):
    lab = measure.label(s["dark"], connectivity=2).astype(np.int32)
    objects = features.blob_features(lab, s["ratio"], s["rgb"], s["bg"], s["mirror"], s["saturated"], s["inner"], cfg)
    for o in objects:
        o["class"] = classify(o, cfg)
        o["chip_depth_um"] = np.nan
    a = Analysis(s["name"], s["rgb"], s["mirror"], s["inner"], s["edge_band"], s["saturated"], lab, objects,
                 um_per_px=cfg["um_per_px"], classes=CLASSES)
    return {"analysis": a}


def _size_um(mirror, px):
    c = mirror.corners
    w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2 * px
    h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2 * px
    return w, h


def image_metrics(a, cfg):
    px = cfg["um_per_px"]
    p = cfg["presence"]
    inner_n, edge_n = a.inner.sum(), a.edge_band.sum()
    dirt = a.objects_lab > 0
    by = {c: [o for o in a.objects if o["class"] == c] for c in CLASSES}
    large = by["large_defect"]
    large_mask = np.isin(a.objects_lab, [o["label_id"] for o in large]) if large else np.zeros_like(dirt)

    def pct(mask, region, n):
        return 100.0 * (mask & region).sum() / n if n else np.nan

    w, h = _size_um(a.mirror, px)
    d = [o["eq_diam_um"] for o in by["particle"]]
    return {
        "image": a.name,
        "um_per_px": px,
        "mirror_found": a.mirror.found,
        "mirror_width_um": w,
        "mirror_height_um": h,
        "mirror_area_mm2": (inner_n + edge_n) * px * px / 1e6,
        "dirt_coverage_inner_pct": pct(dirt, a.inner, inner_n),
        "dirt_coverage_edge_pct": pct(dirt, a.edge_band, edge_n),
        "dirt_coverage_total_pct": pct(dirt, a.inner | a.edge_band, inner_n + edge_n),
        "particle_count": len(d),
        "particle_max_diam_um": max(d, default=0.0),
        "particles_present": any(x >= p["particle_min_diam_um"] for x in d),
        "large_defect_count": len(large),
        "large_defect_max_um": max((o["feret_um"] for o in large), default=0.0),
        "large_defect_area_pct": pct(large_mask, a.inner | a.edge_band, inner_n + edge_n),
        "saturated_pct_inner": pct(a.saturated, a.inner, inner_n),
    }


def step_metrics(s, cfg):
    return {"metrics": image_metrics(s["analysis"], cfg)}


STEPS = [
    ("overlay", step_overlay),
    ("mirror", step_region),
    ("lighting", step_lighting),
    ("dark", step_dark),
    ("classify", step_classify),
    ("metrics", step_metrics),
]

METRICS = [
    Metric("dirt_coverage_inner_pct", "Dirt coverage inner [%]", 0.1, 0.5, 0.01),
    Metric("dirt_coverage_edge_pct", "Dirt coverage edge [%]", 0.5, 2, 0.05),
    Metric("particle_count", "Particle count", 20, 50, 1),
    Metric("particle_max_diam_um", "Largest particle [um]", 20, 35, 1),
    Metric("large_defect_count", "Large defects", 0, 2, 1),
    Metric("large_defect_max_um", "Largest defect [um]", 50, 150, 5),
]


# ---------------------------------------------------------------- tuning GUI

def v_region(s, cfg):
    m = s["mirror"]
    img = tint(s["rgb"], ~s["cream"] & (s["rgb"][..., 0] >= cfg["mirror"]["min_red"]), gui_steps.ORANGE, 0.45)
    outline(img, m.polygon, gui_steps.DARK_GREEN, 2)
    outline(img, s["valid"], gui_steps.GREEN)
    return outline(img, s["inner"], gui_steps.YELLOW)


def v_cream(s, cfg):
    img = (s["rgb"] * 0.35).astype(np.uint8)
    img[s["cream"]] = (255, 235, 200)
    return outline(img, s["mirror"].polygon, gui_steps.GREEN, 2)


def i_region(s, cfg):
    m, px = s["mirror"], cfg["um_per_px"]
    w, h = _size_um(m, px)
    a = px * px / 1e6
    return (f"Mirror found: {'yes' if m.found else 'NO'}\n"
            f"Fitted size: {w:.0f} x {h:.0f} um\n"
            f"Analysed area: {s['valid'].sum() * a:.3f} mm2  "
            f"(inner {s['inner'].sum() * a:.3f}, edge band {s['edge_band'].sum() * a:.3f})")


def v_saturated(s, cfg):
    img = gui_steps.hatch(s["rgb"], s["saturated"], gui_steps.CYAN)
    return outline(img, s["valid"], gui_steps.GREEN)


def i_lighting(s, cfg):
    poly, inner = s["mirror"].polygon, max(s["inner"].sum(), 1)
    if not poly.any():
        return "No mirror found."
    return (f"Overexposed: {100 * s['saturated'][s['inner']].sum() / inner:.1f}% of the inner area\n"
            f"Background brightness on mirror: {np.percentile(s['bg'][poly], 5):.0f}"
            f" - {np.percentile(s['bg'][poly], 95):.0f} (5th-95th percentile)")


def v_dark(s, cfg):
    img = outline(s["rgb"].copy(), s["valid"], gui_steps.GREEN)
    return outline(img, s["dark"], gui_steps.RED, 2)


def i_dark(s, cfg):
    return f"Dark blobs: {measure.label(s['dark'], connectivity=2).max()}"


def i_classify(s, cfg):
    objs = s["analysis"].objects
    lines = []
    for cls in CLASSES:
        o = [x for x in objs if x["class"] == cls]
        lines.append(f"{cls:13s} {len(o):3d}   max diam {max((x['eq_diam_um'] for x in o), default=0):.0f} um,"
                     f" max length {max((x['feret_um'] for x in o), default=0):.0f} um")
    return "\n".join(lines)


def i_metrics(s, cfg):
    return gui_steps.format_metrics(s["metrics"], [
        "mirror_width_um", "mirror_height_um", "dirt_coverage_inner_pct", "dirt_coverage_edge_pct",
        "particle_count", "particle_max_diam_um", "particles_present", "large_defect_count",
        "large_defect_max_um", "large_defect_area_pct", "saturated_pct_inner"])


def step_ui():
    return [
        gui_steps.overlay_ui(local=True),
        StepUI(
            "mirror", "Find the mirror",
            "The mirror is the bright cream square. Pixels count as mirror colour when red is above "
            "the minimum and blue is at least the given fraction of red: the gold strip on the left has "
            "far less blue (orange = bright but not mirror colour, left out) and the background is dark. "
            "The largest such area is fitted with a rectangle (dark green). Green = analysed area (minus "
            "the rim), yellow = inner / edge-band boundary.",
            [P(("mirror", "min_red"), "Min. red", 50, 250, 5,
               "Mirror pixels are at least this bright in red."),
             P(("mirror", "min_blue_ratio"), "Min. blue / red", 0.1, 0.9, 0.01,
               "Mirror ~0.6, gold strip < 0.3."),
             P(("mirror", "border_ignore_um"), "Rim ignored [um]", 0, 75, 2.5,
               "Strip along the edge never counted as dirt."),
             P(("edge_band_um",), "Edge band width [um]", 10, 200, 5,
               "Outer strip of the mirror reported separately from the inner area.")],
            {"Mirror areas": v_region,
             "Mirror colour mask": v_cream,
             "Image": lambda s, c: s["rgb"]},
            i_region),
        StepUI(
            "lighting", "Even out the lighting",
            "The background brightness is estimated inside the mirror by removing dark features (grey-level "
            "closing) and smoothing; every pixel is divided by it (1.0 = clean mirror). The red channel of "
            "the mirror is clipped everywhere, so overexposure is judged on the grey value (hatched).",
            [P(("illumination", "closing_um"), "Closing size [um]", 20, 400, 10,
               "Dark features smaller than this are removed from the background. Must exceed the largest defect width."),
             P(("illumination", "blur_um"), "Smoothing [um]", 5, 100, 5),
             P(("saturation_level",), "Overexposed grey level", 200, 256, 1,
               "Grey value (mean of R, G, B) at or above this counts as overexposed.")],
            {"Flattened (image / background)": gui_steps.v_ratio,
             "Background estimate": gui_steps.v_background,
             "Overexposed": v_saturated,
             "Image": lambda s, c: s["rgb"]},
            i_lighting),
        StepUI(
            "dark", "Dark objects",
            "Pixels clearly darker than the local background (flattened value below the threshold) are "
            "particles or defects (red).",
            [P(("dark", "t_ratio"), "Dark threshold", 0.2, 0.98, 0.01,
               "Pixel is dark if intensity / background is below this. Higher = fainter objects found."),
             P(("dark", "min_area_px"), "Min. blob size [px]", 1, 50, 1)],
            {"Detections": v_dark,
             "Below threshold": gui_steps.v_dark_threshold,
             "Image": lambda s, c: s["rgb"]},
            i_dark),
        StepUI(
            "classify", "Particles and large defects",
            "Every object is measured. Objects at least this wide (equivalent diameter) or this long "
            "(fibres, scratches, flakes) are large defects, the rest are particles.",
            [P(("classify", "large_min_diam_um"), "Large defect min. diameter [um]", 5, 300, 5),
             P(("classify", "large_min_length_um"), "Large defect min. length [um]", 10, 500, 5)],
            {"Classes": gui_steps.v_classes,
             "Image": lambda s, c: s["rgb"]},
            i_classify),
        StepUI(
            "metrics", "Results",
            "Per-image numbers. Dirt coverage = all detected objects as a share of the inner area and of "
            "the edge band. When you're happy, save the parameters and/or run them on all images.",
            [P(("presence", "particle_min_diam_um"), "Particle min. diameter [um]", 0, 100, 1,
               "particles_present: at least one particle this size.")],
            {"Classes": gui_steps.v_classes,
             "Image": lambda s, c: s["rgb"]},
            i_metrics),
    ]
