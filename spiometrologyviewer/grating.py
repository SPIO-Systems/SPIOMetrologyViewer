"""P3 grating ("Grading" images): particles, streaks (comets) and discoloured spots on the olive grating.

The grating is the large dark olive area; the light background on the left and the overexposed
white rectangle on the right are left out.
"""
import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, morphology

from . import detect, features, gui_steps, preprocess
from .gui_steps import P, StepUI, outline, tint
from .p3mirror import detect_dark_hysteresis
from .pipeline import Analysis, step_overlay
from .wafer import Metric

CLASSES = ("particle", "streak", "discolouration")


def find_grating(gray, cfg):
    """The grating as a mask: the largest dark area (grey below `max_gray`), holes filled."""
    gc = cfg["grating"]
    m = (gray < gc["max_gray"]).astype(np.uint8)
    k = 2 * (gc["open_px"] // 2) + 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    h, w = gray.shape
    if n <= 1:
        return np.zeros((h, w), bool)
    return ndimage.binary_fill_holes(lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))


def region(mask):
    """The grating mask as a preprocess.Mirror (bounding box as corners), so the shared steps work."""
    h, w = mask.shape
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(mask)
    corners = (np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], float)
               if len(xs) else np.zeros((4, 2)))
    return preprocess.Mirror(corners, mask, dist, np.zeros((h, w), bool), bool(mask.mean() > 0.1), [])


def chroma_difference(rgb, valid, cfg):
    """Colour difference from the grating's median colour, ignoring brightness (Lab a/b only, so
    shading doesn't count), smoothed over `smooth_um` so only broad blotches stand out."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    if not valid.any():
        return np.zeros(valid.shape, np.float32), (0.0, 0.0)
    am, bm = float(np.median(a[valid])), float(np.median(b[valid]))
    m = valid.astype(np.float32)
    sigma = cfg["discolour"]["smooth_um"] / cfg["um_per_px"]

    def blur(x):
        return cv2.GaussianBlur(x * m, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-3)

    return np.where(valid, np.hypot(blur(a) - am, blur(b) - bm), 0).astype(np.float32), (am, bm)


# ---------------------------------------------------------------- pipeline steps

def step_region(s, cfg):
    px = cfg["um_per_px"]
    g = region(find_grating(s["gray"], cfg))
    valid = g.dist > cfg["grating"]["border_ignore_um"] / px
    inner = (g.dist >= cfg["edge_band_um"] / px) & valid
    return {"mirror": g, "valid": valid, "inner": inner, "edge_band": valid & ~inner}


def step_lighting(s, cfg):
    ratio, bg = preprocess.flatten_illumination(s["gray"], s["mirror"].polygon, cfg)
    return {"ratio": ratio, "bg": bg, "saturated": (s["gray"] >= cfg["saturation_level"]) & s["valid"]}


def step_dark(s, cfg):
    dark = detect_dark_hysteresis(s["ratio"], s["valid"], cfg)
    # the shading along the grating's edge (next to the white rectangle) is not dirt
    rim = s["mirror"].polygon & (s["mirror"].dist <= cfg["grating"]["border_ignore_um"] / cfg["um_per_px"] + 2)
    return {"dark": detect.drop_touching(dark, rim)}


def step_discolour(s, cfg):
    dc = cfg["discolour"]
    de, ref = chroma_difference(s["rgb"], s["valid"], cfg)
    spots = (de > dc["min_delta_e"]) & s["valid"]
    spots = morphology.remove_small_objects(spots, max_size=int(dc["min_area_um2"] / cfg["um_per_px"] ** 2) - 1)
    return {"delta_e": de, "ref_ab": ref, "discolour": spots}


def classify(o, cfg):
    c = cfg["classify"]
    long_shape = o["elongation"] >= c["streak_min_elongation"] or o["thinness"] >= c["streak_min_thinness"]
    return "streak" if long_shape and o["feret_um"] >= c["streak_min_length_um"] else "particle"


def step_classify(s, cfg):
    lab = measure.label(s["dark"], connectivity=2).astype(np.int32)
    n = lab.max()
    spots = measure.label(s["discolour"], connectivity=2)   # whole spots; dark objects in them stay separate
    take = (spots > 0) & (lab == 0)
    lab[take] = spots[take] + n
    objects = features.blob_features(lab, s["ratio"], s["rgb"], s["bg"], s["mirror"], s["saturated"], s["inner"], cfg)
    for o in objects:
        if o["label_id"] > n:
            o["class"] = "discolouration"
            o["delta_e"] = float(s["delta_e"][lab == o["label_id"]].max())
        else:
            o["class"] = classify(o, cfg)
        o["chip_depth_um"] = np.nan
    a = Analysis(s["name"], s["rgb"], s["mirror"], s["inner"], s["edge_band"], s["saturated"], lab, objects,
                 um_per_px=cfg["um_per_px"], classes=CLASSES)
    return {"analysis": a}


def image_metrics(a, cfg, delta_e=None):
    px = cfg["um_per_px"]
    p = cfg["presence"]
    inner_n, edge_n = a.inner.sum(), a.edge_band.sum()
    by = {c: [o for o in a.objects if o["class"] == c] for c in CLASSES}
    ids = lambda cls: [o["label_id"] for o in a.objects if o["class"] in cls]
    dirt = np.isin(a.objects_lab, ids(("particle", "streak")))
    spots = np.isin(a.objects_lab, ids(("discolouration",)))

    def pct(mask, reg, n):
        return 100.0 * (mask & reg).sum() / n if n else np.nan

    d = [o["eq_diam_um"] for o in by["particle"]]
    sl = [o["feret_um"] for o in by["streak"]]
    return {
        "image": a.name,
        "um_per_px": px,
        "grating_found": a.mirror.found,
        "grating_area_mm2": (inner_n + edge_n) * px * px / 1e6,
        "dirt_coverage_inner_pct": pct(dirt, a.inner, inner_n),
        "dirt_coverage_edge_pct": pct(dirt, a.edge_band, edge_n),
        "particle_count": len(d),
        "particle_max_diam_um": max(d, default=0.0),
        "particles_present": any(x >= p["particle_min_diam_um"] for x in d),
        "streak_count": len(sl),
        "streak_max_length_um": max(sl, default=0.0),
        "streaks_present": any(x >= p["streak_min_length_um"] for x in sl),
        "discoloured_count": len(by["discolouration"]),
        "discoloured_area_pct": pct(spots, a.inner | a.edge_band, inner_n + edge_n),
        "discolour_max_delta_e": float(delta_e[a.inner | a.edge_band].max()) if delta_e is not None and (a.inner | a.edge_band).any() else np.nan,
    }


def step_metrics(s, cfg):
    return {"metrics": image_metrics(s["analysis"], cfg, s["delta_e"])}


STEPS = [
    ("overlay", step_overlay),
    ("grating", step_region),
    ("lighting", step_lighting),
    ("dark", step_dark),
    ("discolour", step_discolour),
    ("classify", step_classify),
    ("metrics", step_metrics),
]

METRICS = [
    Metric("dirt_coverage_inner_pct", "Dirt coverage inner [%]", 0.2, 1, 0.05),
    Metric("particle_count", "Particle count", 50, 100, 5),
    Metric("particle_max_diam_um", "Largest particle [um]", 25, 50, 1),
    Metric("streak_count", "Streak count", 10, 30, 1),
    Metric("streak_max_length_um", "Longest streak [um]", 100, 200, 5),
    Metric("discoloured_area_pct", "Discoloured area [%]", 0, 2, 0.1),
    Metric("discolour_max_delta_e", "Max. colour difference", 6, 12, 0.5),
]


# ---------------------------------------------------------------- tuning GUI

def v_region(s, cfg):
    img = tint(s["rgb"], ~s["mirror"].polygon, (40, 40, 40), 0.5)
    outline(img, s["mirror"].polygon, gui_steps.DARK_GREEN, 2)
    outline(img, s["valid"], gui_steps.GREEN)
    return outline(img, s["inner"], gui_steps.YELLOW)


def v_dark_mask(s, cfg):
    img = (s["rgb"] * 0.35).astype(np.uint8)
    img[s["gray"] < cfg["grating"]["max_gray"]] = (200, 190, 140)
    return outline(img, s["mirror"].polygon, gui_steps.GREEN, 2)


def i_region(s, cfg):
    px2 = cfg["um_per_px"] ** 2 / 1e6
    return (f"Grating found: {'yes' if s['mirror'].found else 'NO'}\n"
            f"Analysed area: {s['valid'].sum() * px2:.3f} mm2  "
            f"(inner {s['inner'].sum() * px2:.3f}, edge band {s['edge_band'].sum() * px2:.3f})")


def i_lighting(s, cfg):
    poly = s["mirror"].polygon
    if not poly.any():
        return "No grating found."
    return (f"Background brightness on the grating: {np.percentile(s['bg'][poly], 5):.0f}"
            f" - {np.percentile(s['bg'][poly], 95):.0f} (5th-95th percentile)")


def v_dark(s, cfg):
    img = outline(s["rgb"].copy(), s["valid"], gui_steps.GREEN)
    return outline(img, s["dark"], gui_steps.RED, 2)


def v_dark_low(s, cfg):
    img = gui_steps.dim_outside(gui_steps.gray(s["ratio"], 0, 1.4), s["mirror"].polygon)
    img = tint(img, (s["ratio"] < cfg["dark"]["t_ratio_low"]) & s["valid"], gui_steps.ORANGE, 0.6)
    img = tint(img, (s["ratio"] < cfg["dark"]["t_ratio"]) & s["valid"], gui_steps.RED, 0.8)
    return outline(img, s["valid"], gui_steps.GREEN)


def i_dark(s, cfg):
    return f"Dark objects: {measure.label(s['dark'], connectivity=2).max()}"


def v_delta_e(s, cfg):
    img = gui_steps.colormap(s["delta_e"] / (2 * cfg["discolour"]["min_delta_e"]))
    img[~s["valid"]] = (img[~s["valid"]] * 0.25).astype(np.uint8)
    return outline(img, s["discolour"], (255, 255, 255), 2)


def v_discolour(s, cfg):
    img = tint(s["rgb"], s["discolour"], (0, 160, 255), 0.35)
    return outline(img, s["discolour"], (0, 160, 255), 2)


def i_discolour(s, cfg):
    de, v = s["delta_e"], s["valid"]
    a, b = s["ref_ab"]
    if not v.any():
        return "No grating found."
    px2 = cfg["um_per_px"] ** 2
    n = measure.label(s["discolour"]).max()
    return (f"Grating colour (Lab a, b): {a:+.1f}, {b:+.1f}\n"
            f"Colour difference: median {np.median(de[v]):.1f}, max {de[v].max():.1f}"
            f"  (threshold = middle of the colour scale)\n"
            f"Discoloured spots: {n}, {s['discolour'].sum() * px2:.0f} um2"
            f" ({100 * s['discolour'].sum() / v.sum():.2f}% of the grating)")


def i_classify(s, cfg):
    objs = s["analysis"].objects
    lines = []
    for cls in CLASSES:
        o = [x for x in objs if x["class"] == cls]
        extra = (f"max diam {max((x['eq_diam_um'] for x in o), default=0):.0f} um" if cls == "particle" else
                 f"max length {max((x['feret_um'] for x in o), default=0):.0f} um" if cls == "streak" else
                 f"total {sum(x['area_um2'] for x in o):.0f} um2")
        lines.append(f"{cls:15s} {len(o):3d}   {extra}")
    return "\n".join(lines)


def i_metrics(s, cfg):
    return gui_steps.format_metrics(s["metrics"], [
        "dirt_coverage_inner_pct", "dirt_coverage_edge_pct", "particle_count", "particle_max_diam_um",
        "particles_present", "streak_count", "streak_max_length_um", "streaks_present", "discoloured_count",
        "discoloured_area_pct", "discolour_max_delta_e"])


def step_ui():
    return [
        gui_steps.overlay_ui(local=True),
        StepUI(
            "grating", "Find the grating",
            "The grating is the large dark olive area: pixels darker than the grey limit, opened to cut thin "
            "links, the largest such area with its holes (particles) filled. The light background and the "
            "overexposed white rectangle drop out. Dark green = grating, green = analysed area (minus the rim), "
            "yellow = inner / edge-band boundary.",
            [P(("grating", "max_gray"), "Max. grey", 60, 220, 5,
               "Grating pixels are darker than this (grating ~80, background ~190, white rectangle ~240)."),
             P(("grating", "open_px"), "Opening [px]", 1, 41, 2, "Cuts thin links to other dark areas."),
             P(("grating", "border_ignore_um"), "Rim ignored [um]", 0, 75, 2.5,
               "Strip along the edge never counted."),
             P(("edge_band_um",), "Edge band width [um]", 10, 200, 5)],
            {"Grating areas": v_region,
             "Dark mask": v_dark_mask,
             "Image": lambda s, c: s["rgb"]},
            i_region),
        StepUI(
            "lighting", "Even out the lighting",
            "The background brightness is estimated inside the grating by removing dark features (grey-level "
            "closing) and smoothing; every pixel is divided by it (1.0 = clean grating).",
            [P(("illumination", "closing_um"), "Closing size [um]", 20, 400, 10,
               "Dark features smaller than this are removed from the background."),
             P(("illumination", "blur_um"), "Smoothing [um]", 5, 100, 5)],
            {"Flattened (image / background)": gui_steps.v_ratio,
             "Background estimate": gui_steps.v_background,
             "Image": lambda s, c: s["rgb"]},
            i_lighting),
        StepUI(
            "dark", "Particles and streaks",
            "Pixels clearly darker than the background (below the threshold, red) are particle cores; each core "
            "is grown into the touching fainter pixels (below the second threshold, orange), so a comet keeps "
            "its tail. Objects touching the rim of the analysed area are dropped (shading along the edge).",
            [P(("dark", "t_ratio"), "Dark threshold", 0.2, 0.98, 0.01,
               "Core: intensity / background below this."),
             P(("dark", "t_ratio_low"), "Tail threshold", 0.5, 1.0, 0.01,
               "Cores are grown into touching pixels below this."),
             P(("dark", "min_area_px"), "Min. core size [px]", 1, 50, 1)],
            {"Detections": v_dark,
             "Thresholds": v_dark_low,
             "Image": lambda s, c: s["rgb"]},
            i_dark),
        StepUI(
            "discolour", "Discoloured spots",
            "Colour difference from the grating's median colour, in hue and saturation only (Lab a/b, so "
            "shading and brightness don't count), smoothed so only broad blotches stand out. Areas above the "
            "threshold and larger than the minimum area are discoloured spots.",
            [P(("discolour", "min_delta_e"), "Min. colour difference", 1, 30, 0.5,
               "Clean grating stays below ~5, the green / blue spots reach 15-30."),
             P(("discolour", "smooth_um"), "Smoothing [um]", 5, 200, 5),
             P(("discolour", "min_area_um2"), "Min. spot area [um2]", 0, 100000, 1000)],
            {"Colour difference": v_delta_e,
             "Spots": v_discolour,
             "Image": lambda s, c: s["rgb"]},
            i_discolour),
        StepUI(
            "classify", "Classify",
            "Dark objects that are long and thin (elongation or thinness above the limit, and at least the "
            "minimum length) are streaks / comets, the rest are particles. Discoloured spots keep their class.",
            [P(("classify", "streak_min_elongation"), "Streak min. elongation", 1, 15, 0.5),
             P(("classify", "streak_min_thinness"), "Streak min. thinness", 1, 30, 0.5),
             P(("classify", "streak_min_length_um"), "Streak min. length [um]", 0, 300, 5)],
            {"Classes": gui_steps.v_classes,
             "Image": lambda s, c: s["rgb"]},
            i_classify),
        StepUI(
            "metrics", "Results",
            "Per-image numbers. Dirt coverage = particles + streaks as a share of the inner area and of the "
            "edge band; discoloured area is reported separately.",
            [P(("presence", "particle_min_diam_um"), "Particle min. diameter [um]", 0, 100, 1),
             P(("presence", "streak_min_length_um"), "Streak min. length [um]", 0, 500, 5)],
            {"Classes": gui_steps.v_classes,
             "Image": lambda s, c: s["rgb"]},
            i_metrics),
    ]
