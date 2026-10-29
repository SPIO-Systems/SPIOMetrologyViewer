"""The analysis as a list of steps. Each step takes the state so far (a dict) and returns new keys.

The CLI runs all steps in one go; the GUI runs them one at a time so each can be tuned.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage import measure

from . import classify, detect, features, io, metrics, preprocess


@dataclass
class Analysis:
    name: str
    rgb: np.ndarray           # overlay removed
    mirror: preprocess.Mirror
    inner: np.ndarray
    edge_band: np.ndarray
    saturated: np.ndarray
    objects_lab: np.ndarray   # label image, ids match objects[i]["label_id"]
    objects: list = field(default_factory=list)
    residue_threshold: float = 0.0
    alignment: dict = field(default_factory=dict)
    um_per_px: float = 0.0
    classes: tuple = None      # classes in the overlay legend (None: report.CLASS_COLORS order)
    annotate: object = None    # fn(BGR image) drawing extra things on the result image
    result_rgb: np.ndarray = None   # base of the result image if not `rgb` (lens3d: the residual map)
    plots: np.ndarray = None   # RGB image of per-image plots, written to plots/ by the batch run


def step_overlay(s, cfg):
    overlay = preprocess.overlay_mask(s["raw"], cfg)
    rgb = preprocess.remove_overlay(s["raw"], overlay)
    return {"overlay": overlay, "rgb": rgb, "gray": rgb.mean(axis=2)}


def step_mirror(s, cfg):
    px = cfg["um_per_px"]
    mirror = preprocess.find_mirror(s["gray"], cfg, ignore=s["overlay"])
    valid = (mirror.dist > cfg["mirror"]["border_ignore_um"] / px) & ~mirror.corner_zone
    inner = (mirror.dist >= cfg["edge_band_um"] / px) & valid
    return {"mirror": mirror, "valid": valid, "inner": inner, "edge_band": valid & ~inner}


def step_lens(s, cfg):
    lens = preprocess.find_lens(s["gray"], cfg, s["mirror"].corners.mean(axis=0), ignore=s["overlay"])
    return {"lens": lens, "alignment": preprocess.alignment(s["mirror"], lens, cfg)}


def step_lens_region(s, cfg):
    """Images without a mirror: the lens is found from the image centre, and the disk inside its
    bright ring is analysed. Stored as "mirror" so the later steps work unchanged."""
    px = cfg["um_per_px"]
    h, w = s["gray"].shape
    lens = preprocess.find_lens(s["gray"], cfg, (w / 2, h / 2), ignore=s["overlay"])
    region = preprocess.lens_region(lens, (h, w), cfg)
    valid = region.polygon.copy()
    inner = (region.dist >= cfg["edge_band_um"] / px) & valid
    return {"lens": lens, "mirror": region, "valid": valid, "inner": inner, "edge_band": valid & ~inner,
            "alignment": preprocess.lens_info(lens, cfg)}


def step_lighting(s, cfg):
    mirror = s["mirror"]
    ratio, bg = preprocess.flatten_illumination(s["gray"], mirror.polygon, cfg)
    sat_all = preprocess.saturation_mask(s["rgb"], cfg)
    glare = preprocess.glare_mask(sat_all, mirror, cfg)
    valid = s["valid"] & ~glare
    # Anything dark touching a reflection arc or a corner is (arc) shadow, not dirt or a chip.
    shadow_zone = cv2.dilate((glare | mirror.corner_zone).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return {"ratio": ratio, "bg": bg, "sat_all": sat_all, "glare": glare, "valid": valid,
            "inner": s["inner"] & valid, "edge_band": s["edge_band"] & valid,
            "saturated": sat_all & valid, "shadow_zone": shadow_zone}


def step_dark(s, cfg):
    if is_lens(cfg):  # no mirror edge, so no edge chips
        chip_lab, chip_info = np.zeros(s["ratio"].shape, np.int32), []
    else:
        chip_lab, chip_info = detect.detect_edge_chips(s["ratio"], s["mirror"], cfg, exclude=s["shadow_zone"])
    dark = detect.detect_dark(s["ratio"], s["valid"] & (chip_lab == 0), cfg)
    dark = detect.drop_touching(dark, s["shadow_zone"])
    return {"chip_lab": chip_lab, "chip_info": chip_info, "dark": dark}


def step_residue(s, cfg):
    dark = s["dark"]
    residue, thr, energy = detect.detect_residue(s["ratio"], s["valid"], dark, s["saturated"], s["mirror"].dist, cfg)
    residue = detect.drop_touching(residue, s["shadow_zone"])
    clusters = detect.dense_clusters(dark, s["valid"], cfg)
    residue = residue | clusters

    # Dark blobs sitting in a residue patch are part of that residue.
    dark_lab = measure.label(dark, connectivity=2)
    residue_near = cv2.dilate(residue.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    absorbed = np.unique(dark_lab[residue_near & (dark_lab > 0)])
    residue |= np.isin(dark_lab, absorbed) & (dark_lab > 0)
    return {"residue": residue, "dark": dark & ~residue, "clusters": clusters,
            "residue_threshold": thr, "texture_energy": energy}


def step_classify(s, cfg):
    # One label image for everything; classes stored alongside.
    objects_lab = np.zeros(s["gray"].shape, np.int32)
    classes = {}
    next_id = 1
    for r in measure.regionprops(measure.label(s["dark"], connectivity=2)):
        objects_lab[r.coords[:, 0], r.coords[:, 1]] = next_id
        classes[next_id] = None
        next_id += 1
    for r in measure.regionprops(measure.label(s["residue"], connectivity=2)):
        objects_lab[r.coords[:, 0], r.coords[:, 1]] = next_id
        classes[next_id] = "residue"
        next_id += 1
    chip_depth = {}
    for i, info in enumerate(s["chip_info"], start=1):
        objects_lab[s["chip_lab"] == i] = next_id
        classes[next_id] = "edge_chip"
        chip_depth[next_id] = info["depth_um"]
        next_id += 1

    objects = features.blob_features(objects_lab, s["ratio"], s["rgb"], s["bg"], s["mirror"],
                                     s["saturated"], s["inner"], cfg)
    for o in objects:
        o["class"] = classes[o["label_id"]] or classify.classify_dark(o, cfg)
        o["chip_depth_um"] = chip_depth.get(o["label_id"], np.nan)

    a = Analysis(s["name"], s["rgb"], s["mirror"], s["inner"], s["edge_band"], s["saturated"],
                 objects_lab, objects, s["residue_threshold"], s["alignment"], cfg["um_per_px"])
    return {"analysis": a}


def step_metrics(s, cfg):
    return {"metrics": metrics.image_metrics(s["analysis"], cfg)}


STEPS = [
    ("overlay", step_overlay),
    ("mirror", step_mirror),
    ("lens", step_lens),
    ("lighting", step_lighting),
    ("dark", step_dark),
    ("residue", step_residue),
    ("classify", step_classify),
    ("metrics", step_metrics),
]
# Images without a mirror (config `region: lens`): the lens surface is analysed instead.
LENS_STEPS = [("overlay", step_overlay), ("lens_region", step_lens_region), *STEPS[3:]]


def is_lens(cfg):
    return cfg.get("region", "mirror") == "lens"


def steps_for(cfg):
    from . import analyses
    return analyses.steps(cfg)


def initial_state(path, cfg=None):
    """The loaded image. A LEXT scan (P2 with mirror: its laser intensity) is binned by the config's
    input.bin_px, and its scale checked against um_per_px."""
    raw = io.load_image(path)
    inp = (cfg or {}).get("input", {})
    if "bin_px" not in inp:
        return {"name": Path(path).stem, "path": Path(path), "raw": raw}
    b = int(inp["bin_px"])
    if b > 1:
        h, w = raw.shape[:2]
        raw = cv2.resize(raw, (w // b, h // b), interpolation=cv2.INTER_AREA)
    if Path(path).suffix.lower() == ".lext":
        file_px = io.load_lext(path)["um_per_px"][0] * b
        if abs(file_px / cfg["um_per_px"] - 1) > 0.01:
            raise ValueError(f"{Path(path).name} has {file_px:.4f} um/px after binning by {b}, the config says "
                             f"um_per_px {cfg['um_per_px']}: another objective? Set um_per_px (or input.bin_px).")
    return {"name": Path(path).stem, "path": Path(path), "raw": raw}


def run_steps(state, cfg, steps=None):
    for _, fn in steps or steps_for(cfg):
        state = {**state, **fn(state, cfg)}
    return state


def analyse_image(path, cfg):
    return run_steps(initial_state(path, cfg), cfg)["analysis"]


def analyse(path, cfg):
    """(Analysis, results row) for one image."""
    s = run_steps(initial_state(path, cfg), cfg)
    return s["analysis"], s["metrics"]
