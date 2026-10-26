"""What the tuning GUI shows for each pipeline step: its parameters, its views and a short summary.

Views are plain functions (state, cfg) -> RGB uint8 image, so they can be tested without a window.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import report

RED, GREEN, YELLOW, ORANGE = (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 150, 0)
MAGENTA, CYAN, DARK_GREEN = (255, 0, 255), (0, 220, 255), (0, 120, 0)


@dataclass
class P:
    """A tunable parameter. `path` = config keys (an int indexes into a list); an int `step` makes it an int."""
    path: tuple
    label: str
    lo: float
    hi: float
    step: float
    help: str = ""

    @property
    def integer(self):
        return isinstance(self.step, int)


@dataclass
class StepUI:
    key: str        # matches pipeline.STEPS
    title: str
    text: str       # what the step does
    params: list
    views: dict     # view name -> fn(state, cfg) -> RGB image
    info: callable = None
    notes: list = field(default_factory=list)


def get_param(cfg, path):
    for k in path:
        cfg = cfg[k]
    return cfg


def set_param(cfg, path, value):
    for k in path[:-1]:
        cfg = cfg[k]
    cfg[path[-1]] = value


# ---------------------------------------------------------------- drawing helpers

def tint(img, mask, color, alpha=0.4):
    out = img.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def outline(img, mask, color, thickness=1):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img


def hatch(img, mask, color, period=8):
    yy, xx = np.indices(mask.shape)
    return tint(img, mask & ((xx + yy) % period < 2), color, 0.7)


def gray(x, lo, hi):
    g = (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return np.dstack([g] * 3)


def colormap(x01):
    return cv2.cvtColor(cv2.applyColorMap((np.clip(x01, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
                        cv2.COLOR_BGR2RGB)


def dim_outside(img, region, f=0.3):
    out = img.copy()
    out[~region] = (out[~region] * f).astype(np.uint8)
    return out


# ---------------------------------------------------------------- 1. overlay

def v_overlay_mask(s, cfg):
    return tint(s["raw"], s["overlay"], MAGENTA, 0.8)


def i_overlay(s, cfg):
    return f"Grid / scale bar pixels: {100 * s['overlay'].mean():.2f}% of the image"


# ---------------------------------------------------------------- 2. mirror

def v_mirror(s, cfg):
    m = s["mirror"]
    border = cfg["mirror"]["border_ignore_um"] / cfg["um_per_px"]
    img = tint(s["rgb"], m.corner_zone, ORANGE, 0.35)
    img = tint(img, m.polygon & (m.dist <= border) & ~m.corner_zone, RED, 0.5)
    outline(img, m.polygon, DARK_GREEN)
    outline(img, s["valid"], GREEN)
    outline(img, s["inner"], YELLOW)
    for x, y in m.corners:
        cv2.circle(img, (int(round(x)), int(round(y))), 6, RED, 2)
    return img


def v_edges(s, cfg):
    L = np.log1p(cv2.GaussianBlur(s["gray"].astype(np.float32), (0, 0), 1.5))
    mag = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    img = colormap(mag / max(np.percentile(mag, 99.5), 1e-6))
    return outline(img, s["mirror"].polygon, (255, 255, 255))


def i_mirror(s, cfg):
    m, px = s["mirror"], cfg["um_per_px"]
    c = m.corners
    w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2 * px
    h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2 * px
    a = px * px / 1e6
    exp_w, exp_h = cfg["mirror"]["size_um"]
    return (f"Mirror found: {'yes' if m.found else 'NO'}\n"
            f"Fitted size: {w:.0f} x {h:.0f} um   (expected {exp_w} x {exp_h})\n"
            f"Sides placed from expected size: {', '.join(m.corrected) or 'none'}\n"
            f"Analysed area: {s['valid'].sum() * a:.3f} mm2  "
            f"(inner {s['inner'].sum() * a:.3f}, edge band {s['edge_band'].sum() * a:.3f})")


# ---------------------------------------------------------------- 3. lens / alignment

def _cross(img, xy, color, size=12, thickness=2):
    x, y = (int(round(v)) for v in xy)
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def v_lens(s, cfg):
    lens, m = s["lens"], s["mirror"]
    img = s["rgb"].copy()
    if np.isfinite(lens.radius):
        cv2.circle(img, tuple(int(round(v)) for v in lens.centre), int(round(lens.radius)), CYAN, 1)
    if np.isfinite(lens.ring_radius):
        cv2.circle(img, tuple(int(round(v)) for v in lens.ring_centre), int(round(lens.ring_radius)), MAGENTA, 1)
    for x, y in lens.edge_points:
        img[int(y), int(x)] = YELLOW
    for x, y in lens.ring_points:
        img[int(y), int(x)] = YELLOW
    cv2.polylines(img, [np.round(m.corners).astype(np.int32)], True, GREEN, 1)
    mc = m.corners.mean(axis=0)
    _cross(img, lens.centre, CYAN)
    _cross(img, mc, GREEN)
    cv2.line(img, tuple(int(round(v)) for v in lens.centre), tuple(int(round(v)) for v in mc), RED, 2)
    return img


def i_lens(s, cfg):
    a = s["alignment"]
    return (f"Lens found: {'yes' if a['lens_found'] else 'NO'}   diameter {a['lens_diameter_um']:.0f} um\n"
            f"Mirror centre - lens centre (+x right, +y down):\n"
            f"  x {a['offset_x_um']:+.1f} um   y {a['offset_y_um']:+.1f} um   (total {a['offset_um']:.1f} um)\n"
            f"Mirror rotation vs image axes: {a['mirror_angle_deg']:+.2f} deg\n"
            f"Outer edge vs inner ring centre: {a['lens_ring_offset_um']:.1f} um (should be ~0-3)")


# ---------------------------------------------------------------- 4. lighting

def v_ratio(s, cfg):
    return dim_outside(gray(s["ratio"], 0, 1.4), s["mirror"].polygon)


def v_background(s, cfg):
    return dim_outside(gray(s["bg"], 0, 255), s["mirror"].polygon)


def v_saturation(s, cfg):
    img = hatch(s["rgb"], s["sat_all"] & s["mirror"].polygon, CYAN)
    img = tint(img, s["glare"] & s["mirror"].polygon, RED, 0.45)
    outline(img, s["shadow_zone"] & s["mirror"].polygon, ORANGE)
    return outline(img, s["valid"], GREEN)


def i_lighting(s, cfg):
    inner_n = max(s["inner"].sum(), 1)
    a = cfg["um_per_px"] ** 2 / 1e6
    poly = s["mirror"].polygon
    return (f"Overexposed: {100 * s['saturated'][s['inner']].sum() / inner_n:.1f}% of the inner area\n"
            f"Reflection arcs removed from analysis: {(s['glare'] & poly).sum() * a:.3f} mm2\n"
            f"Background brightness on mirror: {np.percentile(s['bg'][poly], 5):.0f}"
            f" - {np.percentile(s['bg'][poly], 95):.0f} (5th-95th percentile)")


# ---------------------------------------------------------------- 5. dark objects + chips

def v_dark(s, cfg):
    img = tint(s["rgb"], s["shadow_zone"] & s["mirror"].polygon, (128, 128, 128), 0.5)
    outline(img, s["valid"], GREEN)
    outline(img, s["dark"], RED, 2)
    return outline(img, s["chip_lab"] > 0, ORANGE, 2)


def v_dark_threshold(s, cfg):
    img = dim_outside(gray(s["ratio"], 0, 1.4), s["mirror"].polygon)
    img = tint(img, (s["ratio"] < cfg["dark"]["t_ratio"]) & s["mirror"].polygon, RED, 0.7)
    return outline(img, s["valid"], GREEN)


def i_dark(s, cfg):
    from skimage import measure
    n = measure.label(s["dark"], connectivity=2).max()
    if cfg.get("region", "mirror") == "lens":
        return f"Dark blobs: {n}"
    depths = ", ".join(f"{c['depth_um']:.0f}" for c in s["chip_info"])
    return f"Dark blobs: {n}\nEdge chips: {len(s['chip_info'])}" + (f"  (depth um: {depths})" if depths else "")


# ---------------------------------------------------------------- 6. residue

def v_residue(s, cfg):
    img = s["rgb"].copy()
    outline(img, s["valid"], GREEN)
    outline(img, s["dark"], RED, 1)
    return outline(img, s["residue"], MAGENTA, 2)


def v_energy(s, cfg):
    usable = s["valid"] & ~s["saturated"]
    img = colormap(s["texture_energy"] / (2 * s["residue_threshold"]))
    img[~usable] = (img[~usable] * 0.25).astype(np.uint8)
    return outline(img, s["residue"], (255, 255, 255), 1)


def i_residue(s, cfg):
    from skimage import measure
    lab = measure.label(s["residue"], connectivity=2)
    a = cfg["um_per_px"] ** 2
    return (f"Texture threshold: {s['residue_threshold']:.4f}"
            f"  (= middle of the colour scale in 'Texture energy')\n"
            f"Residue patches: {lab.max()}, total {s['residue'].sum() * a:.0f} um2"
            f"  (of which dense dark-blob clusters: {s['clusters'].sum() * a:.0f} um2)\n"
            f"Dark blobs left as particles/fibres: {measure.label(s['dark'], connectivity=2).max()}")


# ---------------------------------------------------------------- 7./8. classify + results

def v_classes(s, cfg):
    return cv2.cvtColor(report.draw_overlay(s["analysis"]), cv2.COLOR_BGR2RGB)


def i_classify(s, cfg):
    objs = s["analysis"].objects
    lines = []
    for cls in report.CLASS_COLORS:
        if cls == "edge_chip" and cfg.get("region", "mirror") == "lens":
            continue
        o = [x for x in objs if x["class"] == cls]
        if cls in ("particle", "unknown"):
            extra = f"max diam {max((x['eq_diam_um'] for x in o), default=0):.0f} um"
        elif cls == "fibre":
            extra = f"max length {max((x['length_um'] for x in o), default=0):.0f} um"
        elif cls == "residue":
            extra = f"total {sum(x['area_um2'] for x in o):.0f} um2"
        else:
            extra = f"max depth {max((x['chip_depth_um'] for x in o), default=0):.0f} um"
        lines.append(f"{cls:10s} {len(o):3d}   {extra}")
    return "\n".join(lines)


def i_metrics(s, cfg):
    m = s["metrics"]
    keys = ["offset_x_um", "offset_y_um", "dirt_coverage_inner_pct", "dirt_coverage_edge_pct", "dirt_coverage_total_pct",
            "particles_present", "fibres_present", "residue_present", "edge_chips_present",
            "particle_count", "fibre_count", "edge_chip_count", "unknown_count", "saturated_pct_inner"]
    fmt = lambda v: ("yes" if v else "no") if isinstance(v, (bool, np.bool_)) else (
        f"{v:.2f}" if isinstance(v, float) else str(v))
    return "\n".join(f"{k:26s} {fmt(m[k])}" for k in keys if k in m)


STEP_UI = [
    StepUI(
        "overlay", "Remove grid overlay",
        "The microscope burns a green/yellow grid and a scale bar into the pixels. They are found "
        "by colour (green or yellow clearly above the other channels), widened a little and filled "
        "in from the surrounding pixels (inpainting).",
        [P(("um_per_px",), "Pixel size [um/px]", 1.5, 4.0, 0.01,
           "Measured from the 500 um scale bar. Every um setting below is converted with this."),
         P(("overlay", "color_margin"), "Colour margin", 10, 150, 5,
           "How much G (green) or R+G (yellow) must exceed the other channels. Lower = more pixels masked."),
         P(("overlay", "dilate_px"), "Widen mask [px]", 0, 5, 1,
           "Grow the mask to catch the anti-aliased rim of the grid lines.")],
        {"Overlay mask (magenta)": v_overlay_mask,
         "After removal": lambda s, c: s["rgb"],
         "Original": lambda s, c: s["raw"]},
        i_overlay),

    StepUI(
        "mirror", "Find the mirror",
        "Edges are found in the log-brightness (so the dark, vignetted bottom still has edges). "
        "For each side, the first strong edge going outward from the image centre is the mirror; "
        "it is then refined per row/column and fitted with a straight line. A side that disagrees "
        "with the known mirror size is searched for again at the expected distance. "
        "Dark green = fitted outline, red = bevel rim (ignored), orange = corners (ignored), "
        "green = analysed area, yellow = inner / edge-band boundary.",
        [P(("mirror", "size_um", 0), "Mirror width [um]", 1000, 1500, 1),
         P(("mirror", "size_um", 1), "Mirror height [um]", 1300, 1900, 1),
         P(("mirror", "size_tol_um"), "Size tolerance [um]", 5, 150, 5,
           "If a side pair is further from the expected size than this, the weaker side is re-placed."),
         P(("mirror", "rough_step_frac"), "Edge strength fraction", 0.05, 0.95, 0.05,
           "First edge (walking outward) at least this strong relative to the strongest one is taken."),
         P(("mirror", "edge_search_px"), "Edge refine window [px]", 3, 40, 1,
           "Search window around the rough edge when refining each row / column."),
         P(("mirror", "corner_exclude_um"), "Corner exclusion [um]", 0, 500, 5,
           "Chamfered corners catch reflections: a 45 deg triangle this size is not analysed."),
         P(("mirror", "border_ignore_um"), "Bevel rim ignored [um]", 0, 75, 2.5,
           "Strip along the edge (bevel brightness ramp) never counted as dirt."),
         P(("edge_band_um",), "Edge band width [um]", 10, 200, 5,
           "Outer strip of the mirror reported separately from the inner area.")],
        {"Mirror areas": v_mirror,
         "Edge strength": v_edges,
         "Image": lambda s, c: s["rgb"]},
        i_mirror),

    StepUI(
        "lens", "Lens and alignment",
        "The round lens around the mirror is found along 720 rays from the mirror centre: on each "
        "ray the strongest dark-to-bright step going outward is the outer edge of the dark lens "
        "(yellow dots on the cyan circle). A circle is fitted, ignoring points far off it "
        "(reflections, dirt, cables), and the search is repeated around that circle. The thin bright "
        "ring just inside (magenta) is fitted separately as a cross-check; both centres should agree. "
        "Offset = mirror centre (green cross) minus lens centre (cyan cross), red line.",
        [P(("lens", "search_um", 0), "Edge search from [um]", 500, 1500, 10,
           "Radius range (from the mirror centre) in which the outer lens edge is searched ..."),
         P(("lens", "search_um", 1), "Edge search to [um]", 800, 1800, 10, "... up to this radius."),
         P(("lens", "ring_search_um", 0), "Ring search from [um]", 0, 300, 5,
           "The inner ring is searched between these distances inside the outer edge."),
         P(("lens", "ring_search_um", 1), "Ring search to [um]", 20, 500, 5)],
        {"Lens fit": v_lens,
         "Image": lambda s, c: s["rgb"]},
        i_lens),

    StepUI(
        "lighting", "Even out the lighting",
        "The background brightness is estimated by removing dark features (grey-level closing) "
        "and smoothing, using only mirror pixels. Every pixel is then divided by it, so dirt "
        "looks the same in the bright centre and the dark corners (1.0 = clean mirror). "
        "Overexposed pixels are marked; overexposed reflection arcs crossing the outline are "
        "removed from the analysis together with their surroundings (red), and dark things "
        "touching them or the corners (orange outline) are treated as shadow.",
        [P(("illumination", "closing_um"), "Closing size [um]", 20, 300, 5,
           "Dark features smaller than this are removed from the background. Must exceed the largest particle."),
         P(("illumination", "blur_um"), "Smoothing [um]", 5, 100, 5,
           "Gaussian smoothing of the background estimate."),
         P(("saturation_level",), "Overexposed level", 200, 255, 1,
           "Any channel at or above this counts as overexposed."),
         P(("glare_margin_um",), "Reflection margin [um]", 0, 150, 5,
           "Exclusion zone grown around reflection arcs to cover their shadow.")],
        {"Flattened (image / background)": v_ratio,
         "Background estimate": v_background,
         "Overexposed + reflections": v_saturation,
         "Image": lambda s, c: s["rgb"]},
        i_lighting),

    StepUI(
        "dark", "Dark objects and edge chips",
        "Pixels clearly darker than the local background (flattened value below the threshold) are "
        "opaque dirt: particles and fibres (red). Dark notches cutting into the mirror outline are "
        "edge chips (orange); they are searched closer to the edge than the dirt. Grey = shadow "
        "zone, objects touching it are dropped. The 'Below threshold' view shows the raw threshold "
        "before the size and zone filters.",
        [P(("dark", "t_ratio"), "Dark threshold", 0.2, 0.95, 0.01,
           "Pixel is dark if intensity / background is below this. Higher = fainter objects found."),
         P(("dark", "min_area_px"), "Min. blob size [px]", 1, 50, 1,
           "Smaller dark specks are ignored (noise)."),
         P(("edge_chip", "start_um"), "Chip search start [um]", 0, 40, 2.5,
           "Chips are searched from this far inside the fitted edge (edge blur)."),
         P(("edge_chip", "min_depth_um"), "Chip min. depth [um]", 5, 100, 2.5,
           "A notch must reach at least this deep from the fitted edge.")],
        {"Detections": v_dark,
         "Below threshold": v_dark_threshold,
         "Image": lambda s, c: s["rgb"]},
        i_dark),

    StepUI(
        "residue", "Residue / stains",
        "Residue is faint relief (fern patterns, droplet rings) rather than dark material. A band-pass "
        "filter (difference of Gaussians) picks up fine structure while ignoring smooth lighting; its "
        "local average is the texture energy. The threshold is set per image from the clean mirror "
        "(median + k x MAD). Areas next to solid particles, thin strips along the bevel and overexposed "
        "areas are skipped. Crowded fields of small dark blobs also count as residue.",
        [P(("residue", "dog_sigmas", 0), "Band-pass fine [px]", 0.5, 5, 0.5,
           "Small Gaussian of the difference of Gaussians."),
         P(("residue", "dog_sigmas", 1), "Band-pass coarse [px]", 1.0, 15, 0.5,
           "Large Gaussian; structure coarser than this is ignored."),
         P(("residue", "window_px"), "Energy window [px]", 3, 31, 1,
           "Local averaging of the band-pass magnitude."),
         P(("residue", "k_mad"), "Threshold k (x MAD)", 1, 20, 0.5,
           "Threshold = median + k x MAD of the energy on clean mirror. Lower = more residue."),
         P(("residue", "min_energy"), "Min. threshold", 0.0, 0.2, 0.005,
           "Absolute floor on the threshold (for very clean images)."),
         P(("residue", "particle_halo_px"), "Particle halo [px]", 0, 40, 1,
           "Texture this close to a solid dark particle is ignored."),
         P(("residue", "closing_px"), "Gap closing [px]", 1, 31, 1,
           "Joins nearby texture pixels into patches."),
         P(("residue", "min_area_um2"), "Min. patch area [um2]", 0, 20000, 250),
         P(("residue", "bevel_strip_elongation"), "Bevel strip elongation", 2, 20, 0.5,
           "Patches at least this elongated lying entirely in the edge band are bevel artefacts."),
         P(("residue", "cluster", "link_um"), "Cluster link distance [um]", 0, 150, 5,
           "Dark blobs closer than this are linked into a cluster."),
         P(("residue", "cluster", "min_blobs"), "Cluster min. blobs", 2, 30, 1,
           "A linked group of at least this many blobs becomes residue.")],
        {"Detections": v_residue,
         "Texture energy": v_energy,
         "Image": lambda s, c: s["rgb"]},
        i_residue),

    StepUI(
        "classify", "Classify",
        "Every object is measured (size, length, elongation, solidity, colour, ...) and the "
        "remaining dark objects are sorted by rules: long and thin = fibre, very ragged = unknown, "
        "otherwise particle. Residue and edge chips keep the class from their detector.",
        [P(("classify", "fibre_min_elongation"), "Fibre min. elongation", 1, 15, 0.5,
           "Major / minor axis. Fibre if elongation OR thinness is above its limit ..."),
         P(("classify", "fibre_min_thinness"), "Fibre min. thinness", 1, 30, 0.5,
           "Skeleton length^2 / area (about length / width); catches curved fibres."),
         P(("classify", "fibre_min_length_um"), "Fibre min. length [um]", 0, 200, 5,
           "... AND it is at least this long."),
         P(("classify", "unknown_max_solidity"), "Unknown max. solidity", 0.0, 1.0, 0.05,
           "Non-fibre blobs less solid (area / convex hull) than this are 'unknown'.")],
        {"Classes": v_classes,
         "Image": lambda s, c: s["rgb"]},
        i_classify),

    StepUI(
        "metrics", "Results",
        "Per-image numbers. Dirt coverage = particles + fibres + residue + unknown, as a share of "
        "the inner area and of the edge band. A class counts as present when at least one object "
        "(or for residue the total area) reaches the minimum size below. When you're happy, save "
        "the parameters and/or run them on all images.",
        [P(("presence", "particle_min_diam_um"), "Particle min. diameter [um]", 0, 100, 1),
         P(("presence", "fibre_min_length_um"), "Fibre min. length [um]", 0, 500, 5),
         P(("presence", "residue_min_total_area_um2"), "Residue min. total area [um2]", 0, 50000, 500),
         P(("presence", "edge_chip_min_depth_um"), "Edge chip min. depth [um]", 0, 100, 2.5)],
        {"Classes": v_classes,
         "Image": lambda s, c: s["rgb"]},
        i_metrics),
]


# ---------------------------------------------------------------- images without a mirror (region: lens)

def v_lens_region(s, cfg):
    lens = s["lens"]
    img = s["rgb"].copy()
    if np.isfinite(lens.radius):
        cv2.circle(img, tuple(int(round(v)) for v in lens.centre), int(round(lens.radius)), CYAN, 1)
    if np.isfinite(lens.ring_radius):
        cv2.circle(img, tuple(int(round(v)) for v in lens.ring_centre), int(round(lens.ring_radius)), MAGENTA, 1)
    for x, y in lens.edge_points:
        img[int(y), int(x)] = YELLOW
    outline(img, s["valid"], GREEN)
    outline(img, s["inner"], YELLOW)
    _cross(img, lens.centre, CYAN)
    return img


def i_lens_region(s, cfg):
    a, px2 = s["alignment"], cfg["um_per_px"] ** 2 / 1e6
    return (f"Lens found: {'yes' if a['lens_found'] else 'NO'}   diameter {a['lens_diameter_um']:.0f} um,"
            f" bright ring {a['lens_ring_diameter_um']:.0f} um\n"
            f"Outer edge vs inner ring centre: {a['lens_ring_offset_um']:.1f} um (should be ~0-3)\n"
            f"Analysed area: {s['valid'].sum() * px2:.3f} mm2  "
            f"(inner {s['inner'].sum() * px2:.3f}, edge band {s['edge_band'].sum() * px2:.3f})")


LENS_REGION_UI = StepUI(
    "lens_region", "Find the lens",
    "There is no mirror on these images, so the lens surface itself is analysed. The outer edge of "
    "the dark lens (cyan, from the yellow dots) is found on 720 rays from the image centre, and the "
    "thin bright ring inside it (magenta) is fitted separately. The analysed area (green) is the "
    "disk inside the bright ring, minus a margin for the dark band along it; yellow = inner / "
    "edge-band boundary.",
    [P(("lens", "search_um", 0), "Edge search from [um]", 500, 1500, 10,
       "Radius range (from the image centre) in which the outer lens edge is searched ..."),
     P(("lens", "search_um", 1), "Edge search to [um]", 800, 1800, 10, "... up to this radius."),
     P(("lens", "ring_search_um", 0), "Ring search from [um]", 0, 300, 5,
       "The bright ring is searched between these distances inside the outer edge."),
     P(("lens", "ring_search_um", 1), "Ring search to [um]", 20, 500, 5),
     P(("lens", "inner_margin_um"), "Margin inside ring [um]", 0, 200, 5,
       "The analysed disk ends this far inside the bright ring (dark band and its shading)."),
     P(("edge_band_um",), "Edge band width [um]", 10, 200, 5,
       "Outer strip of the disk reported separately from the inner area.")],
    {"Lens areas": v_lens_region,
     "Image": lambda s, c: s["rgb"]},
    i_lens_region)


def steps_for(cfg):
    """The steps the tuning GUI shows, matching pipeline.steps_for(cfg)."""
    from . import analyses
    return analyses.step_ui(cfg)


def lens_steps():
    """Tuning steps for images without a mirror (analysis "lens")."""
    by_key = {ui.key: ui for ui in STEP_UI}
    dark = by_key["dark"]
    dark = StepUI(dark.key, "Dark objects", dark.text.split(" Dark notches")[0] + " Grey = shadow zone.",
                  [p for p in dark.params if p.path[0] != "edge_chip"], dark.views, dark.info)
    res = by_key["metrics"]
    res = StepUI(res.key, res.title, res.text, [p for p in res.params if "edge_chip" not in p.path[-1]],
                 res.views, res.info)
    return [by_key["overlay"], LENS_REGION_UI, by_key["lighting"], dark, by_key["residue"], by_key["classify"], res]


# ---------------------------------------------------------------- shared by the P3 analyses

def overlay_ui(local=False):
    """The overlay step; with `local`, the grid is found by its colour against the surroundings
    (for yellowish images, see preprocess.overlay_mask)."""
    ov = STEP_UI[0]
    if not local:
        return ov
    return StepUI(
        ov.key, ov.title,
        ov.text + " Here the image itself is yellowish, so a pixel counts when it is clearly greener or "
                  "yellower than the median of its surroundings (window size below).",
        [*ov.params,
         P(("overlay", "local_px"), "Local window [px]", 3, 41, 2,
           "Median window the colour is compared with. Must be wider than the grid lines.")],
        ov.views, ov.info)


def format_metrics(m, keys):
    fmt = lambda v: ("yes" if v else "no") if isinstance(v, (bool, np.bool_)) else (
        f"{v:.2f}" if isinstance(v, float) else str(v))
    return "\n".join(f"{k:26s} {fmt(m[k])}" for k in keys if k in m)
