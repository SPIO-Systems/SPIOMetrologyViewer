"""P5 mirror: size of the mirror and the quality of its surface.

The mirror is a bright square on a dark background, at a varying position in one stitched colour image.
Its red and green channels are clipped, so the surface is judged on the blue channel only.
"""
import cv2
import numpy as np
from skimage import measure

from . import edges, features, gui_steps, preprocess
from .gui_steps import P, StepUI, outline, tint
from .p1opening import annotate_size, edge_params, rect_metrics
from .p3mirror import classify, detect_dark_hysteresis
from .pipeline import Analysis
from .wafer import Metric

CLASSES = ("particle", "large_defect", "stain", "edge_defect")


# ---------------------------------------------------------------- pipeline steps

def step_load(s, cfg):
    rgb = s["raw"]
    return {"rgb": rgb, "gray": rgb.mean(axis=2), "overlay": np.zeros(rgb.shape[:2], bool)}


def step_mirror(s, cfg):
    px = cfg["um_per_px"]
    rect = edges.fit_rectangle(s["gray"].astype(np.float32), s["overlay"], cfg["mirror"], px)
    polygon = rect.mask(s["gray"].shape)
    dist = cv2.distanceTransform(polygon.astype(np.uint8), cv2.DIST_L2, 5)
    corners = rect.corners if rect.found else np.zeros((4, 2))
    mirror = preprocess.Mirror(corners, polygon, dist, np.zeros(polygon.shape, bool), rect.found, [])
    lab, objects = edges.defects(rect, polygon.shape, px, cfg["mirror"])
    near = cv2.dilate((lab > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    valid = (dist > cfg["mirror"]["border_ignore_um"] / px) & ~near
    inner = (dist >= cfg["edge_band_um"] / px) & valid
    return {"rect": rect, "mirror": mirror, "edge_lab": lab, "edge_objects": objects, "valid": valid,
            "inner": inner, "edge_band": valid & ~inner}


def step_lighting(s, cfg):
    ratio, bg = edges.flatten(s["rgb"][..., 2], s["mirror"].polygon, cfg)
    clipped = s["rgb"][..., :2].min(axis=2) >= cfg["saturation_level"]
    return {"ratio": ratio, "bg": bg, "clipped": clipped, "saturated": np.zeros(ratio.shape, bool)}


def step_dark(s, cfg):
    return {"dark": detect_dark_hysteresis(s["ratio"], s["valid"], cfg)}


def classify_mirror(o, cfg):
    c = cfg["classify"]
    if o["eq_diam_um"] >= c["stain_min_diam_um"] and o["mean_ratio"] >= c["stain_min_ratio"]:
        return "stain"
    return classify(o, cfg)


def step_classify(s, cfg):
    lab = measure.label(s["dark"], connectivity=2).astype(np.int32)
    objects = features.blob_features(lab, s["ratio"], s["rgb"], s["bg"], s["mirror"], s["saturated"], s["inner"], cfg)
    for o in objects:
        o["class"] = classify_mirror(o, cfg)
        o["chip_depth_um"] = np.nan
    all_lab, all_objects = edges.merge(lab, objects, s["edge_lab"], s["edge_objects"])
    a = Analysis(s["name"], s["rgb"], s["mirror"], s["inner"], s["edge_band"], s["saturated"], all_lab, all_objects,
                 um_per_px=cfg["um_per_px"], classes=CLASSES)
    return {"analysis": a}


def image_metrics(s, cfg):
    px = cfg["um_per_px"]
    a = s["analysis"]
    inner_n, edge_n = a.inner.sum(), a.edge_band.sum()
    all_n = inner_n + edge_n
    surface = [o for o in a.objects if o["class"] != "edge_defect"]
    chips = [o for o in a.objects if o["class"] == "edge_defect"]
    by = {c: [o for o in surface if o["class"] == c] for c in ("particle", "large_defect", "stain")}
    d = [o["eq_diam_um"] for o in by["particle"]]

    def mask(objs):
        return np.isin(a.objects_lab, [o["label_id"] for o in objs])

    def pct(m, region, n):
        return 100.0 * (m & region).sum() / n if n else np.nan

    dirt = mask(surface)
    poly = a.mirror.polygon
    return {
        "image": a.name,
        "um_per_px": px,
        "mirror_found": s["rect"].found,
        **rect_metrics(s["rect"], px, cfg["mirror"]["size_um"]),
        "edge_defect_count": len(chips),
        "edge_defect_max_um": max((o["chip_depth_um"] for o in chips), default=0.0),
        "dirt_coverage_inner_pct": pct(dirt, a.inner, inner_n),
        "dirt_coverage_edge_pct": pct(dirt, a.edge_band, edge_n),
        "dirt_coverage_total_pct": pct(dirt, a.inner | a.edge_band, all_n),
        "particle_count": len(d),
        "particle_max_diam_um": max(d, default=0.0),
        "large_defect_count": len(by["large_defect"]),
        "large_defect_max_um": max((o["feret_um"] for o in by["large_defect"]), default=0.0),
        "stain_count": len(by["stain"]),
        "stain_area_pct": pct(mask(by["stain"]), a.inner | a.edge_band, all_n),
        "clipped_pct": 100.0 * (s["clipped"] & poly).sum() / poly.sum() if poly.any() else np.nan,
    }


def step_metrics(s, cfg):
    m = image_metrics(s, cfg)
    annotate_size(s["analysis"], m)
    return {"metrics": m}


STEPS = [
    ("load", step_load),
    ("mirror", step_mirror),
    ("lighting", step_lighting),
    ("dark", step_dark),
    ("classify", step_classify),
    ("metrics", step_metrics),
]

# Provisional: the size limits around the nominal and the rest from the spread on wafers 01-08.
METRICS = [
    Metric("width_dev_um", "|Width - nominal| [um]", 40, 75, 1, abs=True),
    Metric("length_dev_um", "|Length - nominal| [um]", 40, 75, 1, abs=True),
    Metric("dirt_coverage_inner_pct", "Dirt coverage inner [%]", 0.1, 0.5, 0.01),
    Metric("dirt_coverage_edge_pct", "Dirt coverage edge [%]", 0.5, 2, 0.05),
    Metric("particle_count", "Particle count", 20, 50, 1),
    Metric("particle_max_diam_um", "Largest particle [um]", 30, 60, 1),
    Metric("large_defect_count", "Large defects", 0, 2, 1),
    Metric("stain_area_pct", "Stains [%]", 0.5, 2, 0.05),
    Metric("edge_defect_count", "Edge defects", 0, 2, 1),
]


# ---------------------------------------------------------------- tuning GUI

def v_region(s, cfg):
    img = outline(s["rgb"].copy(), s["mirror"].polygon, gui_steps.DARK_GREEN, 2)
    img = tint(img, s["edge_lab"] > 0, gui_steps.RED, 0.7)
    outline(img, s["valid"], gui_steps.GREEN)
    return outline(img, s["inner"], gui_steps.YELLOW)


def v_blue(s, cfg):
    return np.repeat(s["rgb"][..., 2:], 3, axis=2)


def i_mirror(s, cfg):
    return edges.text(s["rect"], cfg["um_per_px"]) + f"\nEdge defects: {len(s['edge_objects'])}"


def i_lighting(s, cfg):
    poly = s["mirror"].polygon
    if not poly.any():
        return "No mirror found."
    return (f"Red and green clipped on {100 * s['clipped'][poly].mean():.0f}% of the mirror: only blue is usable\n"
            f"Blue background on the mirror: {np.percentile(s['bg'][poly], 5):.0f}"
            f" - {np.percentile(s['bg'][poly], 95):.0f} (5th-95th percentile)")


def v_dark(s, cfg):
    img = outline(s["rgb"].copy(), s["valid"], gui_steps.GREEN)
    return outline(img, s["dark"], gui_steps.RED, 2)


def i_dark(s, cfg):
    return f"Objects: {measure.label(s['dark'], connectivity=2).max()}"


def i_classify(s, cfg):
    objs = s["analysis"].objects
    return "\n".join(f"{c:13s} {sum(o['class'] == c for o in objs):3d}" for c in CLASSES)


def i_metrics(s, cfg):
    return gui_steps.format_metrics(s["metrics"], [
        "width_um", "length_um", "width_dev_um", "length_dev_um", "edge_rms_um", "edge_max_dev_um", "edge_defect_count",
        "dirt_coverage_inner_pct", "dirt_coverage_edge_pct", "particle_count", "particle_max_diam_um",
        "large_defect_count", "large_defect_max_um", "stain_count", "stain_area_pct", "clipped_pct"])


def step_ui():
    return [
        StepUI("load", "Image", "One stitched colour image per device. There is no grid in these images.", [],
               {"Image": lambda s, c: s["rgb"], "Blue channel": v_blue}),
        StepUI(
            "mirror", "Find the mirror",
            "The mirror is found anywhere in the image and its four sides are fitted (dark green). Green = analysed "
            "area (minus the rim), yellow = inner / edge-band boundary, red = where an edge leaves its line.",
            [*edge_params("mirror"),
             P(("mirror", "corner_px"), "Corners left out [px]", 2, 60, 1),
             P(("mirror", "border_ignore_um"), "Rim ignored [um]", 0, 75, 2.5,
               "Strip along the edge never counted as dirt."),
             P(("edge_band_um",), "Edge band width [um]", 10, 300, 5,
               "Outer strip of the mirror reported separately from the inner area.")],
            {"Mirror areas": v_region, "Image": lambda s, c: s["rgb"]},
            i_mirror),
        StepUI(
            "lighting", "Even out the lighting",
            "Red and green are clipped on the mirror, so the blue channel is used. Its background is estimated "
            "around dark features and smoothed; every pixel is divided by it (1.0 = clean mirror).",
            [P(("illumination", "closing_um"), "Closing size [um]", 20, 800, 10,
               "Features smaller than this are removed from the background. Must exceed the largest stain."),
             P(("illumination", "blur_um"), "Smoothing [um]", 5, 200, 5)],
            {"Flattened (blue / background)": gui_steps.v_ratio, "Background estimate": gui_steps.v_background,
             "Blue channel": v_blue},
            i_lighting),
        StepUI(
            "dark", "Dark objects",
            "Pixels clearly darker in blue than the local background (red).",
            [P(("dark", "t_ratio"), "Dark threshold", 0.2, 0.99, 0.01,
               "Pixel is dark if blue / background is below this."),
             P(("dark", "t_ratio_low"), "Grown down to", 0.5, 0.995, 0.005,
               "Dark objects grow into touching pixels below this."),
             P(("dark", "min_area_px"), "Min. blob size [px]", 1, 50, 1)],
            {"Detections": v_dark, "Below threshold": gui_steps.v_dark_threshold, "Image": lambda s, c: s["rgb"]},
            i_dark),
        StepUI(
            "classify", "Particles, large defects and stains",
            "Wide but faint objects are stains. Of the others, those at least this wide or this long are large "
            "defects, the rest particles.",
            [P(("classify", "stain_min_diam_um"), "Stain min. diameter [um]", 10, 500, 5),
             P(("classify", "stain_min_ratio"), "Stain min. mean level", 0.3, 0.99, 0.01,
               "Objects paler than this, on average, can be stains."),
             P(("classify", "large_min_diam_um"), "Large defect min. diameter [um]", 5, 300, 5),
             P(("classify", "large_min_length_um"), "Large defect min. length [um]", 10, 500, 5)],
            {"Classes": gui_steps.v_classes, "Image": lambda s, c: s["rgb"]},
            i_classify),
        StepUI(
            "metrics", "Results",
            "Per-image numbers. When you're happy, save the parameters and/or run them on all images.",
            [],
            {"Classes": gui_steps.v_classes, "Image": lambda s, c: s["rgb"]},
            i_metrics),
    ]
