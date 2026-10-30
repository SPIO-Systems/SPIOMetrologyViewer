"""P1 aperture and slit: size and edges of the opening in the coating, and dirt on the coating.

The opening is a rectangle at a varying position in the image. It is pale pink in the aperture images
and black in the slit images, against the blue-grey coating. The two differ by config only (signal,
pixel scale, nominal size).
"""
import cv2
import numpy as np
from skimage import measure, morphology

from . import detect, edges, features, gui_steps, preprocess, report
from .gui_steps import P, StepUI, outline, tint
from .p3mirror import classify, detect_dark_hysteresis
from .pipeline import Analysis, step_overlay
from .wafer import Metric

CLASSES = ("particle", "large_defect", "edge_defect")


def signal(raw, cfg):
    """A float image in which the opening is the bright part."""
    if cfg["opening"]["signal"] == "red_minus_blue":
        return raw[..., 0].astype(np.float32) - raw[..., 2].astype(np.float32)
    return 255.0 - raw.mean(axis=2).astype(np.float32)


# ---------------------------------------------------------------- pipeline steps

def step_opening(s, cfg):
    px = cfg["um_per_px"]
    rect = edges.fit_rectangle(signal(s["raw"], cfg), s["overlay"], cfg["opening"], px)
    opening = rect.mask(s["gray"].shape)
    coating = ~opening
    dist = cv2.distanceTransform(coating.astype(np.uint8), cv2.DIST_L2, 5)
    corners = rect.corners if rect.found else np.zeros((4, 2))
    region = preprocess.Mirror(corners, coating, dist, np.zeros(opening.shape, bool), rect.found, [])
    return {"rect": rect, "opening": opening, "mirror": region}


def step_edges(s, cfg):
    px = cfg["um_per_px"]
    lab, objects = edges.defects(s["rect"], s["opening"].shape, px, cfg["opening"])
    near = cv2.dilate((lab > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    d = s["mirror"].dist
    valid = (d > cfg["coating"]["rim_um"] / px) & ~near
    inner = (d >= cfg["coating"]["edge_band_um"] / px) & valid
    return {"edge_lab": lab, "edge_objects": objects, "valid": valid, "inner": inner, "edge_band": valid & ~inner}


def step_lighting(s, cfg):
    g = s["gray"].copy()
    grown = cv2.dilate(s["opening"].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    outside = ~grown
    if grown.any():
        g[grown] = np.median(g[outside])     # or the opening would leak into the background estimate
    ratio, bg = edges.flatten(g, outside, cfg)
    return {"ratio": ratio, "bg": bg, "saturated": np.zeros(g.shape, bool)}


def step_dirt(s, cfg):
    dark = detect_dark_hysteresis(s["ratio"], s["valid"], cfg)
    bright = (s["ratio"] > cfg["dark"]["t_bright"]) & s["valid"]
    bright = morphology.remove_small_objects(bright, max_size=cfg["dark"]["min_area_px"] - 1)
    return {"dark": dark | bright}


def step_classify(s, cfg):
    lab = measure.label(s["dark"], connectivity=2).astype(np.int32)
    objects = features.blob_features(lab, s["ratio"], s["rgb"], s["bg"], s["mirror"], s["saturated"], s["inner"], cfg)
    for o in objects:
        o["class"] = classify(o, cfg)
        o["chip_depth_um"] = np.nan
    all_lab, all_objects = edges.merge(lab, objects, s["edge_lab"], s["edge_objects"])
    a = Analysis(s["name"], s["rgb"], s["mirror"], s["inner"], s["edge_band"], s["saturated"], all_lab, all_objects,
                 um_per_px=cfg["um_per_px"], classes=CLASSES)
    return {"analysis": a}


def rect_metrics(rect, px, nominal):
    """Size columns shared with P5: width along x, length along y."""
    w, h = rect.width_px * px, rect.height_px * px
    return {
        "width_um": w, "length_um": h, "width_dev_um": w - nominal[0], "length_dev_um": h - nominal[1],
        "edge_rms_um": rect.rms_px * px, "edge_max_dev_um": rect.max_dev_px() * px,
        "centre_x_um": rect.centre[0] * px, "centre_y_um": rect.centre[1] * px,
        "angle_deg": rect.angle_deg, "parallel_deg": rect.parallel_deg,
        "edge_hidden": bool(rect.found and any(s.hidden for s in rect.sides.values())),
        **{f"edge_rms_{k}_um": rect.sides[k].rms_px * px if rect.found else np.nan for k in edges.SIDES},
    }


def image_metrics(s, cfg):
    px = cfg["um_per_px"]
    a = s["analysis"]
    inner_n, edge_n = a.inner.sum(), a.edge_band.sum()
    surface = [o for o in a.objects if o["class"] != "edge_defect"]
    chips = [o for o in a.objects if o["class"] == "edge_defect"]
    by = {c: [o for o in surface if o["class"] == c] for c in ("particle", "large_defect")}
    d = [o["eq_diam_um"] for o in by["particle"]]
    dirt = np.isin(a.objects_lab, [o["label_id"] for o in surface])

    def pct(region, n):
        return 100.0 * (dirt & region).sum() / n if n else np.nan

    return {
        "image": a.name,
        "um_per_px": px,
        "opening_found": s["rect"].found,
        **rect_metrics(s["rect"], px, cfg["opening"]["size_um"]),
        "edge_defect_count": len(chips),
        "edge_defect_max_um": max((o["chip_depth_um"] for o in chips), default=0.0),
        "dirt_coverage_inner_pct": pct(a.inner, inner_n),
        "dirt_coverage_edge_pct": pct(a.edge_band, edge_n),
        "particle_count": len(d),
        "particle_max_diam_um": max(d, default=0.0),
        "large_defect_count": len(by["large_defect"]),
        "large_defect_max_um": max((o["feret_um"] for o in by["large_defect"]), default=0.0),
    }


def annotate_size(a, m):
    def annotate(bgr):
        if not np.isfinite(m["width_um"]):
            report.outlined_text(bgr, "Not found", (10, bgr.shape[0] - 12))
            return
        lines = [f"{m['width_um']:.2f} x {m['length_um']:.2f} um  ({m['width_dev_um']:+.2f}, {m['length_dev_um']:+.2f})",
                 f"edge scatter {m['edge_rms_um']:.2f} um, furthest {m['edge_max_dev_um']:.2f} um"
                 + ("  (edge under grid line)" if m["edge_hidden"] else "")]
        y = bgr.shape[0] - 12 - 22 * (len(lines) - 1)
        for t in lines:
            report.outlined_text(bgr, t, (10, y))
            y += 22
    a.annotate = annotate


def step_metrics(s, cfg):
    m = image_metrics(s, cfg)
    annotate_size(s["analysis"], m)
    return {"metrics": m}


STEPS = [
    ("overlay", step_overlay),
    ("opening", step_opening),
    ("edges", step_edges),
    ("lighting", step_lighting),
    ("dirt", step_dirt),
    ("classify", step_classify),
    ("metrics", step_metrics),
]

_SURFACE = [
    Metric("dirt_coverage_inner_pct", "Coating dirt [%]", 0.25, 1, 0.01),
    Metric("particle_count", "Particle count", 50, 150, 1),
    Metric("particle_max_diam_um", "Largest particle [um]", 25, 30, 1),
    Metric("large_defect_count", "Large defects", 0, 2, 1),
]

# Provisional: the size limits around the nominal and the rest from the spread on wafers 00, 03, 07.
METRICS = {
    "p1_aperture": [
        Metric("width_dev_um", "|Width - nominal| [um]", 5, 10, 0.5, abs=True),
        Metric("length_dev_um", "|Length - nominal| [um]", 5, 10, 0.5, abs=True),
        Metric("edge_rms_um", "Edge scatter [um]", 0.5, 1, 0.05),
        Metric("edge_max_dev_um", "Largest edge deviation [um]", 6, 12, 0.5),
        Metric("edge_defect_count", "Edge defects", 0, 2, 1),
        *_SURFACE,
    ],
    "p1_slit": [
        Metric("width_dev_um", "|Width - nominal| [um]", 1, 2, 0.1, abs=True),
        Metric("length_dev_um", "|Length - nominal| [um]", 2, 4, 0.1, abs=True),
        Metric("edge_rms_um", "Edge scatter [um]", 0.15, 0.3, 0.01),
        Metric("edge_max_dev_um", "Largest edge deviation [um]", 1, 3, 0.1),
        Metric("edge_defect_count", "Edge defects", 0, 2, 1),
        *_SURFACE,
    ],
}


# ---------------------------------------------------------------- tuning GUI

def v_opening(s, cfg):
    return outline(s["rgb"].copy(), s["opening"], gui_steps.GREEN)


def v_signal(s, cfg):
    return gui_steps.gray(signal(s["raw"], cfg), -128, 255)


def i_opening(s, cfg):
    return edges.text(s["rect"], cfg["um_per_px"])


def v_edges(s, cfg):
    img = tint(s["rgb"], s["edge_lab"] > 0, gui_steps.RED, 0.7)
    outline(img, s["opening"], gui_steps.GREEN)
    return outline(img, s["valid"], gui_steps.YELLOW)


def i_edges(s, cfg):
    o = s["edge_objects"]
    return "\n".join([f"Edge defects: {len(o)}"] + [
        f"{x['side']:7s} {x['chip_depth_um']:6.2f} um deep, {x['length_um']:6.1f} um long, "
        f"{'material in the opening' if x['coating'] else 'coating missing'}" for x in o])


def v_dirt(s, cfg):
    img = outline(s["rgb"].copy(), s["valid"], gui_steps.GREEN)
    return outline(img, s["dark"], gui_steps.RED, 2)


def i_dirt(s, cfg):
    return f"Objects on the coating: {measure.label(s['dark'], connectivity=2).max()}"


def i_classify(s, cfg):
    objs = s["analysis"].objects
    return "\n".join(f"{c:13s} {sum(o['class'] == c for o in objs):3d}" for c in CLASSES)


def i_metrics(s, cfg):
    return gui_steps.format_metrics(s["metrics"], [
        "width_um", "length_um", "width_dev_um", "length_dev_um", "edge_rms_um", "edge_max_dev_um",
        "edge_hidden", "edge_defect_count", "edge_defect_max_um", "angle_deg", "parallel_deg",
        "dirt_coverage_inner_pct", "dirt_coverage_edge_pct", "particle_count", "particle_max_diam_um",
        "large_defect_count", "large_defect_max_um"])


def edge_params(section):
    return [P((section, "defect_um"), "Edge defect from [um]", 0.2, 30, 0.1,
              "An edge further than this from its fitted line is a defect."),
            P((section, "defect_min_um"), "Edge defect min. length [um]", 0, 50, 0.5,
              "Shorter deviations are left out.")]


def step_ui():
    return [
        gui_steps.overlay_ui(),
        StepUI(
            "opening", "Find the opening",
            "The opening is found anywhere in the image. Each side is followed line by line, to where the "
            "signal is halfway between the levels just outside and just inside, and a straight line is fitted "
            "to those points (green). Grid lines are bridged: a side under one is less certain.",
            [P(("opening", "corner_px"), "Corners left out [px]", 2, 60, 1,
               "Points this close to the corners don't count for the fit."),
             P(("opening", "min_contrast"), "Min. contrast", 5, 100, 1,
               "Less than this between opening and coating: not found.")],
            {"Fitted opening": v_opening, "Signal": v_signal, "Image": lambda s, c: s["rgb"]},
            i_opening),
        StepUI(
            "edges", "Edge quality",
            "The scatter of the points around the fitted lines is the edge roughness. Where an edge leaves its "
            "line by more than the limit, it is an edge defect (red). Yellow = the coating that is analysed: "
            "everything except the rim along the opening.",
            [*edge_params("opening"),
             P(("coating", "rim_um"), "Rim ignored [um]", 0, 50, 0.5, "Strip along the opening, never counted as dirt."),
             P(("coating", "edge_band_um"), "Edge band width [um]", 5, 200, 5,
               "Coating this close to the opening is reported separately.")],
            {"Edge defects": v_edges, "Image": lambda s, c: s["rgb"]},
            i_edges),
        StepUI(
            "lighting", "Even out the lighting",
            "The brightness of the coating is estimated around dark features and smoothed; every pixel is "
            "divided by it (1.0 = clean coating).",
            [P(("illumination", "closing_um"), "Closing size [um]", 10, 400, 5,
               "Dark features smaller than this are removed from the background."),
             P(("illumination", "blur_um"), "Smoothing [um]", 2, 100, 1)],
            {"Flattened (image / background)": gui_steps.v_ratio, "Background estimate": gui_steps.v_background,
             "Image": lambda s, c: s["rgb"]}),
        StepUI(
            "dirt", "Dirt on the coating",
            "Pixels clearly darker or brighter than the local background (red).",
            [P(("dark", "t_ratio"), "Dark threshold", 0.2, 0.98, 0.01,
               "Pixel is dark if intensity / background is below this."),
             P(("dark", "t_ratio_low"), "Grown down to", 0.5, 0.99, 0.01,
               "Dark objects grow into touching pixels below this."),
             P(("dark", "t_bright"), "Bright threshold", 1.02, 2, 0.01,
               "Pixel is bright if intensity / background is above this."),
             P(("dark", "min_area_px"), "Min. blob size [px]", 1, 50, 1)],
            {"Detections": v_dirt, "Below threshold": gui_steps.v_dark_threshold, "Image": lambda s, c: s["rgb"]},
            i_dirt),
        StepUI(
            "classify", "Particles and large defects",
            "Objects at least this wide or this long are large defects, the rest are particles.",
            [P(("classify", "large_min_diam_um"), "Large defect min. diameter [um]", 2, 300, 1),
             P(("classify", "large_min_length_um"), "Large defect min. length [um]", 5, 500, 5)],
            {"Classes": gui_steps.v_classes, "Image": lambda s, c: s["rgb"]},
            i_classify),
        StepUI(
            "metrics", "Results",
            "Per-image numbers. When you're happy, save the parameters and/or run them on all images.",
            [],
            {"Classes": gui_steps.v_classes, "Image": lambda s, c: s["rgb"]},
            i_metrics),
    ]
