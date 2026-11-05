"""Full stack: thickness of every layer of a glued stack (P1..P6 glass, glue and polymer) from a LEXT
cross-section image, one tall stitched image per device.

Ported from ThicknessImageProcessor (thickness/steps.py); the algorithms are in fullstack_profiles.
The image is levelled, the glass plates are found from the texture, and inside each interface the
polymer and glue layers from the thin dark lines between them. Each glass runs between the layers
next to it. Positions and thicknesses are in px of the levelled image; with `um_per_px` set the
results also get um columns, and with a design thickness in `nominal_um` (project.yaml full_stack)
the deviation from it, which the tolerances look at.
"""
import copy
from pathlib import Path

import cv2
import numpy as np
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from . import fullstack_profiles as pr, preprocess
from .gui_steps import P, StepUI
from .pipeline import Analysis
from .wafer import Metric

CYAN, YELLOW = (0, 220, 255), (255, 230, 0)
LAYER_COLORS = {"polymer": (60, 200, 90), "glue": (255, 150, 0), "mixed": (190, 80, 230)}
PLATE_COLORS = [(66, 135, 245), (60, 200, 90), (245, 160, 40), (200, 80, 220), (240, 70, 70), (40, 200, 200)]
GLASS_COLORS = [(66, 135, 245), (40, 190, 215)]   # alternating blues: no other layer type is blue
TEMPLATE_CONFIG = Path(__file__).resolve().parents[1] / "project_template" / "configs" / "FS_thickness.yaml"


def glass_names(cfg):
    return [lay["name"] for lay in cfg["layers"] if lay["type"] == "glass"]


def measured(cfg):
    """Names of the layers that are measured (glass and every layer with extract: true)."""
    return [lay["name"] for lay in cfg["layers"] if lay["type"] == "glass" or lay.get("extract", True)]


def plate_color(k, scale=1):
    return np.array(PLATE_COLORS[k % len(PLATE_COLORS)]) / scale


# ---------------------------------------------------------------- pipeline steps

def _hover(s, x, y):
    text = f"x {x:4d}  y {y:5d} px"
    um = s.get("um_per_px")
    if um:
        text += f"  ({x * um:6.0f}, {y * um:7.0f} um)"
    if y < s["level"].shape[0] and x < s["level"].shape[1]:
        text += f"   intensity {s['level'][y, x]:5.0f}"
    if "plates" in s and y < len(s["plates"].texture):
        text += f"   texture {s['plates'].texture[y]:5.2f}"
    return text


def step_level(s, cfg):
    t = cfg["tilt"]
    gray = pr.to_gray(s["raw"])
    est, angles, sharp = pr.estimate_tilt(gray, t["range_deg"])
    angle = est if t["auto"] else t["manual_deg"]
    return {"angle": angle, "angle_est": est, "angles": angles, "sharp": sharp, "level": pr.deskew(gray, angle),
            "um_per_px": cfg.get("um_per_px"), "hover": _hover}


def step_plates(s, cfg):
    t = cfg["texture"]
    level = s["level"]
    texture = pr.texture_profile(level, t["measure"], t["blur_sigma"], t["highpass_sigma"], t["smooth_sigma"])
    top, bottom = pr.air_edges(pr.project(level), t["air_threshold"])
    plates = pr.find_glass_plates(texture, top, bottom, glass_names(cfg), t["threshold"], t["extend"],
                                  t["min_glass"], t["edge_margin"], t["glass_percentile"], t["level_window"])
    return {"texture": texture, "plates": plates}


def step_interfaces(s, cfg):
    return {"crops": pr.interface_crops(s["level"], s["plates"], cfg["interfaces"]["margin"])}


def step_polymers(s, cfg):
    c = cfg["lines"]
    _, _, depth = pr.relative_depth(pr.project(s["level"]), c["line_sigma"], c["bg_window"])
    windows = pr.interface_windows(depth, s["plates"], cfg["layers"], c["search"], c["min_depth"],
                                   c["min_steepness"])
    return {"depth": depth, "windows": windows, "polymers": pr.polymer_layers(windows)}


def step_glues(s, cfg):
    g = cfg["glue"]
    rel, grad = pr.gradient_profile(s["level"], g["blur_sigma"], cfg["lines"]["bg_window"])
    windows = copy.deepcopy(s["windows"])   # glue_layers marks the edges it finds
    glues = pr.glue_layers(windows, rel, grad, s["polymers"], 1 - cfg["lines"]["min_depth"], g["flank"],
                           g["max_thickness"], g["min_contrast"], g["max_band"])
    return {"rel": rel, "grad": grad, "glue_windows": windows, "glues": glues}


def step_stack(s, cfg):
    mixed = pr.mixed_layers(s["windows"])
    found = {**s["polymers"], **s["glues"], **mixed}
    return {"mixed": mixed, "stack": pr.stack_layers(cfg["layers"], s["plates"], found)}


def _finite(lay):
    return lay is not None and np.isfinite(lay.top) and np.isfinite(lay.bottom)


def image_metrics(s, cfg):
    """One results.csv row: per layer its thickness in px (and um, and the deviation from the nominal)."""
    um = cfg.get("um_per_px") or None
    nominal = cfg.get("nominal_um") or {}
    stack = {lay.name: lay for lay in s["stack"]}
    names = measured(cfg)
    found = [lay for lay in s["stack"] if _finite(lay)]
    total = found[-1].bottom - found[0].top if found else np.nan
    missing = [n for n in names if not _finite(stack.get(n))]
    row = {"image": s["name"], "um_per_px": um if um else np.nan, "tilt_deg": s["angle"], "stack_px": total,
           "missing_layers": len(missing), "missing": ";".join(missing), "notes": ";".join(s["plates"].notes)}
    if um:
        row["stack_um"] = total * um
    for n in names:
        lay = stack.get(n)
        t = lay.thickness if _finite(lay) else np.nan
        row[f"{n}_px"] = t
        row[f"{n}_method"] = lay.method if lay is not None else ""
        if um:
            row[f"{n}_um"] = t * um
            nom = nominal.get(n)
            row[f"{n}_dev_um"] = t * um - float(nom) if nom is not None else np.nan
    return row


def stack_diagram(s, cfg, size):
    """Thumbnail of the results: one band per layer (equal heights, coloured by type), with its
    name and thickness. Missing layers are dark red."""
    names = measured(cfg)
    stack = {lay.name: lay for lay in s["stack"]}
    um = cfg.get("um_per_px")
    img = np.full((size, size, 3), 255, np.uint8)
    band = size / max(len(names), 1)
    g = 0
    scale = max(0.28, min(0.5, band / 40))
    for k, n in enumerate(names):
        lay = stack.get(n)
        y0, y1 = int(round(k * band)), int(round((k + 1) * band))
        if _finite(lay):
            col = stack_color(cfg, lay, g)
            value = f"{lay.thickness * um:.1f} um" if um else f"{lay.thickness:.0f} px"
        else:
            col, value = np.array((140, 30, 30)), "missing"
        g += {l["name"]: l["type"] for l in cfg["layers"]}.get(n) == "glass"
        img[y0:y1] = (col * 0.55 + 255 * 0.45).astype(np.uint8)
        img[y1 - 1:y1] = 255
        yt = int((y0 + y1) / 2 + 4 * scale * 2)
        cv2.putText(img, n.replace("_", " "), (3, yt), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(img, value, (size - tw - 3, yt), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def step_metrics(s, cfg):
    row = image_metrics(s, cfg)
    # The result image at half width (full height for the thin layers): a full-size one is ~30 MB as PNG
    level = s["level"]
    half = cv2.resize(level, (max(level.shape[1] // 2, 1), level.shape[0]), interpolation=cv2.INTER_AREA)
    result = view_stack_labelled({**s, "level": half}, cfg)
    rgb = to_rgb(half)
    shape = rgb.shape[:2]
    zero = np.zeros(shape, bool)
    region = preprocess.Mirror(np.zeros((4, 2)), zero, np.zeros(shape, np.float32), zero, True, [])

    def annotate(bgr):
        lines = [f"tilt {row['tilt_deg']:.3f} deg",
                 f"stack {row['stack_px']:.0f} px" + (f" = {row['stack_um']:.1f} um" if "stack_um" in row else "")]
        if row["missing"]:
            lines.append(f"missing: {row['missing'].replace(';', ', ')}")
        for i, t in enumerate(lines):
            text(bgr, t, (12, 40 + 32 * i), 0.8, (255, 255, 0))

    size = int(cfg.get("output", {}).get("thumb_px", 256))
    raw_thumb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)   # squeezed: the layers show as bands
    a = Analysis(s["name"], rgb, region, zero, zero, zero, np.zeros(shape, np.int32), [],
                 um_per_px=cfg.get("um_per_px") or 0.0, classes=(), annotate=annotate,
                 result_rgb=result, plots=view_depth_profiles(s, cfg),
                 thumbs=(raw_thumb, stack_diagram(s, cfg, size)), overlay_jpg=True)
    return {"metrics": row, "analysis": a}


STEPS = [("level", step_level), ("plates", step_plates), ("interfaces", step_interfaces),
         ("polymers", step_polymers), ("glues", step_glues), ("stack", step_stack), ("metrics", step_metrics)]


def _metrics():
    """Per layer: |thickness - nominal| in um (off until the calibration and nominals are set) and
    the thickness in px to look at; and the number of layers not found."""
    try:
        cfg = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    except OSError:
        return [Metric("missing_layers", "Missing layers", 0, 1, 1, decimals=0)]
    out = [Metric("missing_layers", "Missing layers", 0, 1, 1, decimals=0)]
    for n in measured(cfg):
        out.append(Metric(f"{n}_dev_um", f"|{n} - nominal| [um]", 1, 3, 0.1, abs=True, enabled=False))
        out.append(Metric(f"{n}_px", f"{n} [px]", 0, 0, 1, enabled=False, follows=f"{n}_dev_um", decimals=1))
    return out


METRICS = _metrics()


# ---------------------------------------------------------------- drawing helpers

def to_rgb(gray, lo_pct=0.5, hi_pct=99.5):
    """Contrast-stretched uint8 RGB (the images differ a lot in exposure)."""
    sub = gray[::7, ::7]
    lo, hi = np.percentile(sub, lo_pct), np.percentile(sub, hi_pct)
    g = (np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return np.dstack([g] * 3)


def lw(img):
    """Line width that stays visible when the tall image is zoomed out."""
    return max(1, img.shape[0] // 4000)


def text(img, s, org, scale=1.2, color=YELLOW):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(2, int(4 * scale)), cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, tuple(int(c) for c in color),
                max(1, int(2 * scale)), cv2.LINE_AA)


def figure(w_px, h_px, dpi=100):
    fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    return fig


def render(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def draw_plates(img, plates, scale=1.0, label=True):
    """Tint each glass plate in its own colour, mark its edges and write its name and height.
    `scale` = image px per levelled-image px (to draw on a downscaled image)."""
    out = img.copy()
    w = out.shape[1]
    for k, (name, a, b) in enumerate(plates.plates):
        sl = slice(int(round(a * scale)), int(round(b * scale)) + 1)
        out[sl] = (out[sl] * 0.7 + plate_color(k) * 0.3).astype(np.uint8)
        for y in (a, b):
            cv2.line(out, (0, int(round(y * scale))), (w, int(round(y * scale))), CYAN, lw(out))
    if label:
        for k, (name, a, b) in enumerate(plates.plates):
            text(out, f"{name}  {b - a:.0f} px", (20, int((a + b) / 2 * scale)), 2.0, plate_color(k))
    return out


# ---------------------------------------------------------------- 1. level

def view_level(s, cfg):
    out = to_rgb(s["level"])
    for y in range(0, out.shape[0], 250):
        cv2.line(out, (0, y), (out.shape[1], y), CYAN, lw(out))
    return out


def view_sharpness(s, cfg):
    fig = figure(900, 500)
    ax = fig.add_subplot()
    ax.plot(s["angles"], s["sharp"] / s["sharp"].max(), "-o", ms=2)
    ax.axvline(s["angle_est"], color="g", label=f"estimate {s['angle_est']:.3f}°")
    if s["angle"] != s["angle_est"]:
        ax.axvline(s["angle"], color="r", label=f"used {s['angle']:.3f}°")
    ax.set_xlabel("tilt [°]")
    ax.set_ylabel("projection sharpness (relative)")
    ax.legend()
    ax.grid(alpha=0.3)
    return render(fig)


def info_level(s, cfg):
    h, w = s["raw"].shape[:2]
    return (f"{s['name']}: {w} x {h} px\n"
            f"Estimated tilt {s['angle_est']:.3f}°, used {s['angle']:.3f}°\n"
            f"Over the width that is {np.tan(np.radians(s['angle'])) * w:.1f} px")


# ---------------------------------------------------------------- 2. glass plates

def texture_axes(ax, s, cfg, vertical):
    """The normalised texture with threshold, extend level, plates and non-glass runs."""
    p, t = s["plates"], cfg["texture"]
    y = np.arange(len(p.texture))
    level = ax.axvline if vertical else ax.axhline
    row = ax.axhline if vertical else ax.axvline
    span = ax.axhspan if vertical else ax.axvspan
    ax.plot(*((p.texture, y) if vertical else (y, p.texture)), lw=0.6, color="k")
    level(t["threshold"], color="r", lw=0.8, label=f"threshold {t['threshold']}")
    if t["extend"] > t["threshold"]:
        level(t["extend"], color="orange", lw=0.8, ls="--", label=f"extend {t['extend']}")
    for k, (name, a, b) in enumerate(p.plates):
        span(a, b, color=plate_color(k, 255), alpha=0.18)
    for a, b in p.nonglass:
        span(a, b, color="r", alpha=0.2)
    for e in (p.top, p.bottom):
        row(e, color="c", lw=0.8)
    (ax.set_xlim if vertical else ax.set_ylim)(0.3, 1.4)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")


def view_plates_overview(s, cfg):
    """The levelled image with the plates labelled, next to the texture profile on the same y axis."""
    img, p = s["level"], s["plates"]
    h = img.shape[0]
    fig = figure(1100, 2400)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.04)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(draw_plates(to_rgb(img[::4, ::4]), p, scale=0.25, label=False), aspect="auto",
               extent=(0, img.shape[1], h, 0))
    for k, (name, a, b) in enumerate(p.plates):
        ax0.text(img.shape[1] * 0.03, (a + b) / 2, f"{name}\n{b - a:.0f} px", va="center", fontsize=11,
                 color="white", fontweight="bold", bbox=dict(facecolor=plate_color(k, 255), alpha=0.85, lw=0))
    ax0.set_ylabel("y [px]")
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    texture_axes(ax1, s, cfg, vertical=True)
    ax1.tick_params(labelleft=False)
    ax1.set_title("texture / glass level", fontsize=10)
    ax0.set_ylim(h, 0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.02)
    return render(fig)


def view_plates_image(s, cfg):
    return draw_plates(to_rgb(s["level"]), s["plates"])


def view_texture(s, cfg):
    fig = figure(1900, 650)
    ax = fig.add_subplot()
    texture_axes(ax, s, cfg, vertical=False)
    for k, (name, a, b) in enumerate(s["plates"].plates):
        ax.text((a + b) / 2, 1.33, name, ha="center", fontsize=9, color=plate_color(k, 255))
    ax.set_xlabel("y [px]")
    ax.set_ylabel("texture / glass level")
    ax.set_title("Red: rows below the threshold (extended to the extend level); coloured: glass plates")
    fig.tight_layout()
    return render(fig)


def info_plates(s, cfg):
    p = s["plates"]
    lines = [f"Stack {p.top:.0f} .. {p.bottom:.0f} px ({p.bottom - p.top:.0f} px)", "Glass plates [px]:"]
    lines += [f"  {name:<10} {a:7.0f} .. {b:7.0f}  {b - a:6.0f}" for name, a, b in p.plates]
    lines += ["Interfaces [px]:"]
    lines += [f"  {up}|{down}: {a:.0f} .. {b:.0f} ({b - a:.0f})" for a, b, up, down in p.interfaces]
    return "\n".join(lines + p.notes)


# ---------------------------------------------------------------- 3. interfaces

def view_crops(s, cfg):
    """The interface crops at full resolution, stacked top to bottom, each under a label strip.
    The margin rows (into the glass) are marked with short ticks at both sides."""
    crops = s["crops"]
    if not crops:
        return view_plates_image(s, cfg)
    rgb = to_rgb(s["level"])
    m = cfg["interfaces"]["margin"]
    w, head = rgb.shape[1], 44
    parts = []
    for name, r0, r1, _ in crops:
        strip = np.full((head, w, 3), 255, np.uint8)
        cv2.putText(strip, f"{name}   rows {r0}..{r1} ({r1 - r0} px)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2, cv2.LINE_AA)
        crop = rgb[r0:r1].copy()
        for y in (m, crop.shape[0] - m - 1):
            if 0 <= y < crop.shape[0]:
                crop[y, :25] = crop[y, -25:] = CYAN
        parts += [strip, crop, np.full((16, w, 3), 40, np.uint8)]
    return np.concatenate(parts[:-1])


def view_crop_boxes(s, cfg):
    out = draw_plates(to_rgb(s["level"]), s["plates"], label=False)
    w = out.shape[1]
    for name, r0, r1, _ in s["crops"]:
        cv2.rectangle(out, (2, r0), (w - 3, r1), YELLOW, lw(out) + 2)
        text(out, name, (20, max(r0 - 15, 40)), 1.6)
    return out


def info_interfaces(s, cfg):
    m = cfg["interfaces"]["margin"]
    lines = [f"Interfaces with {m} px into the glass on both sides:"]
    lines += [f"  {name:<20} rows {r0:6d} .. {r1:6d}  ({r1 - r0} px)" for name, r0, r1, _ in s["crops"]]
    return "\n".join(lines + s["plates"].notes)


# ---------------------------------------------------------------- 4. / 5. layers from the dark lines

def layer_color(cfg, name, scale=1):
    t = {lay["name"]: lay["type"] for lay in cfg["layers"]}.get(name, "glue")
    return np.array(LAYER_COLORS.get(t, LAYER_COLORS["glue"])) / scale


def closeups(s, cfg, windows, layers, faint=()):
    """Each window at full resolution with its layers tinted and boundaries drawn, next to its depth
    profile: every dark line is a dot with its depth, red if used as a boundary. Layers in `faint`
    (found in an earlier step) are tinted lightly."""
    if not windows:
        return view_crops(s, cfg)
    img, depth = s["level"], s["depth"]
    layers, faint = list(layers), list(faint)
    fig = figure(560 * len(windows), 1250)
    gs = fig.add_gridspec(1, 2 * len(windows), width_ratios=[1.4, 1] * len(windows), wspace=0.06)
    for k, w in enumerate(windows):
        ax = fig.add_subplot(gs[2 * k])
        ax.imshow(to_rgb(img[w.lo:w.hi]), aspect="auto", extent=(0, img.shape[1], w.hi, w.lo))
        axp = fig.add_subplot(gs[2 * k + 1], sharey=ax)
        axp.plot(depth[w.lo:w.hi], np.arange(w.lo, w.hi), lw=0.7)
        axp.axvline(cfg["lines"]["min_depth"], color="gray", ls="--", lw=0.6)
        for i, (p, d) in enumerate(zip(w.lines.pos, w.lines.depth)):
            col = "r" if i in w.used else "0.55"
            axp.plot(d, p, "o", color=col, ms=4)
            axp.text(d + 0.03, p, f"{d:.2f}", fontsize=7, va="center", color=col)
        for lay in layers + faint:
            if lay.interface != w.label:
                continue
            for a in (ax, axp):
                a.axhspan(lay.top, lay.bottom, color=layer_color(cfg, lay.name, 255),
                          alpha=0.12 if lay in faint else 0.35)
            if lay in faint:
                continue
            for b in (lay.top, lay.bottom):
                ax.axhline(b, color="r", lw=0.8)
            label = f"{lay.name}\n{lay.thickness:.1f} px" + (f"\n({lay.method})" if lay.method != "lines" else "")
            axp.text(0.97, (lay.top + lay.bottom) / 2, label, transform=axp.get_yaxis_transform(), ha="right",
                     va="center", fontsize=8, color=layer_color(cfg, lay.name, 255) * 0.7)
        ax.set_ylim(w.hi, w.lo)
        ax.set_title(w.label, fontsize=9)
        axp.set_xlim(right=max(depth[w.lo:w.hi].max(), cfg["lines"]["min_depth"]) + 0.25)   # room for the labels
        axp.tick_params(labelleft=False)
        axp.set_xlabel("depth", fontsize=8)
        axp.grid(alpha=0.3)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.05)
    return render(fig)


def draw_layers(img, layers, cfg):
    """Tint the layers, draw their boundaries and write name and thickness above each."""
    out = img.copy()
    w = out.shape[1]
    for lay in layers:
        sl = slice(int(round(lay.top)), int(round(lay.bottom)) + 1)
        out[sl] = (out[sl] * 0.55 + layer_color(cfg, lay.name) * 0.45).astype(np.uint8)
        for b in (lay.top, lay.bottom):
            cv2.line(out, (0, int(round(b))), (w, int(round(b))), (255, 40, 40), lw(out))
        text(out, f"{lay.name}  {lay.thickness:.1f} px", (20, max(int(lay.top) - 12, 30)), 1.4,
             layer_color(cfg, lay.name))
    return out


def layer_table(layers):
    return [f"  {lay.name:<11} {lay.top:8.1f} .. {lay.bottom:8.1f}  {lay.thickness:6.1f} px"
            + (f"  {lay.method}" if lay.method != "lines" else "") for lay in layers]


def missing(cfg, kind, found):
    """Layers of type `kind` to be measured that were not found."""
    want = [lay["name"] for lay in cfg["layers"] if lay["type"] == kind and lay.get("extract", True)]
    return [n for n in want if n not in found]


def polymer_windows(s):
    return [w for w in s["windows"] if w.extract and w.extract[-1]["type"] == "polymer"]


def view_polymer_closeups(s, cfg):
    return closeups(s, cfg, polymer_windows(s), s["polymers"].values())


def view_polymers_image(s, cfg):
    return draw_layers(to_rgb(s["level"]), s["polymers"].values(), cfg)


def view_depth_profiles(s, cfg):
    windows = s["windows"]
    fig = figure(1900, 300 * max(len(windows), 1) + 80)
    for k, w in enumerate(windows):
        ax = fig.add_subplot(len(windows), 1, k + 1)
        ax.plot(np.arange(w.lo, w.hi), s["depth"][w.lo:w.hi], lw=0.7)
        ax.axhline(cfg["lines"]["min_depth"], color="gray", ls="--", lw=0.6)
        ax.plot(w.lines.pos, w.lines.depth, "o", color="r", ms=4)
        ax.set_title(f"{w.label}: {len(w.lines)} dark lines", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return render(fig)


def info_polymers(s, cfg):
    lines = ["Polymer layers [px]:"] + layer_table(s["polymers"].values())
    gone = missing(cfg, "polymer", s["polymers"])
    lines += [f"Not found: {', '.join(gone)} (too few dark lines: lower min depth?)"] if gone else []
    lines += ["Dark lines per interface (row, depth):"]
    for w in s["windows"]:
        lines.append(f"  {w.label}: " + ", ".join(f"{p:.0f} ({d:.2f})" for p, d in zip(w.lines.pos, w.lines.depth)))
    return "\n".join(lines + s["plates"].notes)


def glue_windows(s):
    return [w for w in s["glue_windows"] if any(lay["type"] == "glue" for lay in w.extract)]


def view_glue_closeups(s, cfg):
    """Each glue window at full resolution with the glue tinted, next to the relative intensity (blue)
    and its vertical gradient (red): the glue runs from the rising edge (green) to the falling edge
    (red), each next to a dark line below the darkness limit (dashed)."""
    windows = glue_windows(s)
    if not windows:
        return view_crops(s, cfg)
    img, rel, grad = s["level"], s["rel"], s["grad"]
    dark = 1 - cfg["lines"]["min_depth"]
    glues = {lay.interface: lay for lay in s["glues"].values()}
    fig = figure(560 * len(windows), 1250)
    gs = fig.add_gridspec(1, 2 * len(windows), width_ratios=[1.4, 1] * len(windows), wspace=0.06)
    for k, w in enumerate(windows):
        lo, hi = w.search or (w.lo, w.hi)
        ax = fig.add_subplot(gs[2 * k])
        ax.imshow(to_rgb(img[lo:hi]), aspect="auto", extent=(0, img.shape[1], hi, lo))
        axp = fig.add_subplot(gs[2 * k + 1], sharey=ax)
        y = np.arange(lo, hi)
        axp.plot(rel[lo:hi], y, lw=0.8, color="tab:blue")
        axp.axvline(dark, color="gray", ls="--", lw=0.6)
        axg = axp.twiny()
        axg.plot(grad[lo:hi], y, lw=0.6, color="r", alpha=0.7)
        axg.axvline(0, color="r", lw=0.4, alpha=0.5)
        axg.tick_params(labelsize=7, colors="r")
        lay = glues.get(w.label)
        if lay is not None:
            for a in (ax, axp):
                a.axhspan(lay.top, lay.bottom, color=layer_color(cfg, lay.name, 255), alpha=0.3)
            for b, col in ((lay.top, "lime"), (lay.bottom, "r")):
                ax.axhline(b, color=col, lw=1)
                axp.axhline(b, color=col, lw=0.8)
            axp.text(0.03, (lay.top + lay.bottom) / 2, f"{lay.name}\n{lay.thickness:.1f} px\n({lay.method})",
                     transform=axp.get_yaxis_transform(), ha="left", va="center", fontsize=8,
                     color=layer_color(cfg, lay.name, 255) * 0.7)
        ax.set_ylim(hi, lo)
        ax.set_title(w.label, fontsize=9)
        axp.tick_params(labelleft=False, labelsize=7)
        axp.set_xlabel("intensity / background", fontsize=8)
        axp.grid(alpha=0.3)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.05)
    return render(fig)


def view_layers_image(s, cfg):
    return draw_layers(to_rgb(s["level"]), [*s["polymers"].values(), *s["glues"].values()], cfg)


def info_glues(s, cfg):
    lines = ["Glue layers [px]:"] + layer_table(s["glues"].values())
    gone = missing(cfg, "glue", s["glues"])
    lines += [f"Not found: {', '.join(gone)}"] if gone else []
    skipped = [lay["name"] for lay in cfg["layers"] if not lay.get("extract", True)]
    lines += [f"Not measured (extract: false): {', '.join(skipped)}"] if skipped else []
    lines += ["Polymer layers [px]:"] + layer_table(s["polymers"].values())
    return "\n".join(lines)


# ---------------------------------------------------------------- 6. the full stack

def stack_color(cfg, layer, glass_index):
    t = {lay["name"]: lay["type"] for lay in cfg["layers"]}.get(layer.name, "glass")
    return np.array(GLASS_COLORS[glass_index % len(GLASS_COLORS)] if t == "glass" else LAYER_COLORS[t])


def view_full_image(s, cfg):
    return to_rgb(s["level"])


def view_stack_labelled(s, cfg):
    """The levelled image with every layer of the stack tinted in its own colour, its boundaries
    drawn and its name and thickness written on it (thin layers labelled to the right)."""
    out = to_rgb(s["level"])
    w = out.shape[1]
    tint = out.copy()
    um = cfg.get("um_per_px")
    g = 0
    colors = []
    for lay in s["stack"]:
        col = stack_color(cfg, lay, g)
        g += lay.method == "glass"
        colors.append(col)
        a, b = int(round(lay.top)) if np.isfinite(lay.top) else 0, int(round(lay.bottom)) if np.isfinite(lay.bottom) else 0
        if _finite(lay) and b > a:
            tint[a:b + 1] = col
    out = (out * 0.6 + tint * 0.4).astype(np.uint8)
    for lay, col in zip(s["stack"], colors):
        for y in (lay.top, lay.bottom):
            if np.isfinite(y):
                cv2.line(out, (0, int(round(y))), (w, int(round(y))), tuple(int(c) for c in col * 0.6), lw(out))
    thin = 0
    f = min(1.0, w / 991)   # the batch run's result image is half as wide
    for lay, col in zip(s["stack"], colors):
        if not _finite(lay):
            continue
        mid = int((lay.top + lay.bottom) / 2)
        label = f"{lay.name}  {lay.thickness:.1f} px" + (f" = {lay.thickness * um:.2f} um" if um else "")
        if lay.thickness >= 120:
            text(out, label, (int(20 * f), mid + 15), max(1.2, 2.0 * f), col)
        elif f < 1:   # narrow: thin layers labelled at the left, just below the line
            text(out, label, (12, mid + 8), 0.8, col)
        else:   # thin: alternate the text position so neighbouring labels don't overlap
            text(out, label, (int(w * (0.35 + 0.3 * (thin % 2))) - 150, mid + 10), 1.1, col)
            thin += 1
    return out


def info_stack(s, cfg):
    um = cfg.get("um_per_px")
    head = f"{'layer':<21} {'top':>8} {'bottom':>8} {'px':>7}" + (f" {'um':>8}" if um else "")
    rows = [head]
    for lay in s["stack"]:
        row = f"{lay.name:<21} {lay.top:8.1f} {lay.bottom:8.1f} {lay.thickness:7.1f}"
        rows.append(row + (f" {lay.thickness * um:8.2f}" if um else ""))
    total = s["stack"][-1].bottom - s["stack"][0].top
    gaps = sum(b.top - a.bottom for a, b in zip(s["stack"], s["stack"][1:]) if b.top > a.bottom)
    rows.append(f"Stack {total:.1f} px; between layers (dark boundary lines) {gaps:.1f} px")
    gone = [lay["name"] for lay in cfg["layers"] if lay.get("extract", True)
            and lay["name"] not in {l.name for l in s["stack"]}]
    rows += [f"Not found: {', '.join(gone)}"] if gone else []
    if not um:
        rows.append("um_per_px is not set: thicknesses in px only")
    return "\n".join(rows)


def info_metrics(s, cfg):
    m = s["metrics"]
    nominal = cfg.get("nominal_um") or {}
    um = cfg.get("um_per_px")
    rows = [f"{'layer':<21} {'px':>8}" + (f" {'um':>8} {'nominal':>8} {'dev':>7}" if um else "")]
    for n in measured(cfg):
        row = f"{n:<21} {m[f'{n}_px']:8.1f}"
        if um:
            nom = nominal.get(n)
            row += f" {m[f'{n}_um']:8.2f} " + (f"{nom:8.2f} {m[f'{n}_dev_um']:+7.2f}" if nom is not None else f"{'-':>8}")
        rows.append(row)
    rows.append(f"Missing layers: {m['missing_layers']}" + (f" ({m['missing'].replace(';', ', ')})" if m["missing"] else ""))
    if not um:
        rows.append("Set um_per_px (LEXT objective) for um values, and the nominals in project.yaml (full_stack) "
                    "for the deviations the tolerances look at.")
    return "\n".join(rows)


# ---------------------------------------------------------------- tuning GUI

def step_ui():
    return [
        StepUI("level", "Level the layers",
               "The image is converted to gray (the LEXT TIFFs are gray stored as RGB) and used at its full "
               "width. Projecting along x only keeps thin layers sharp if they are horizontal: the tilt is "
               "the shear angle that gives the sharpest projected profile. Use 'Fit width' to scroll down the "
               "tall image.",
               [P(("tilt", "auto"), "Automatic tilt (1 = on)", 0, 1, 1),
                P(("tilt", "range_deg"), "Search range [°]", 0.1, 3, 0.1),
                P(("tilt", "manual_deg"), "Manual tilt [°]", -3, 3, 0.01, "Used when automatic tilt is off")],
               {"Levelled image (guides every 250 px)": view_level, "Sharpness vs angle": view_sharpness}, info_level),
        StepUI("plates", "Glass plates from the texture",
               "The glass is speckled with vertical streaks; glue and polymer are smoother. The texture per "
               "row (full width) is divided by the local glass level. Rows below the threshold mark an "
               "interface, which extends while the texture stays below the extend level. Interfaces closer "
               "than the minimum glass thickness are joined (a thick polymer band can look glass-like in its "
               "middle), and the deepest ones, one fewer than the glass plates, are kept.",
               [P(("texture", "measure"), "Measure (0 = anisotropy, 1 = high-pass)", 0, 1, 1,
                  "Anisotropy = mean |d/dx| / mean |d/dy|: flat in the glass, scratches hardly show. "
                  "High-pass = speckle energy; dips at scratches too, needs a lower threshold"),
                P(("texture", "blur_sigma"), "Blur before the gradients [px]", 0, 5, 0.5, "Anisotropy only"),
                P(("texture", "highpass_sigma"), "High-pass sigma [px]", 1, 15, 0.5, "High-pass only"),
                P(("texture", "smooth_sigma"), "Smoothing along y [px]", 2, 60, 1),
                P(("texture", "threshold"), "Threshold (x glass level)", 0.3, 1.0, 0.01),
                P(("texture", "extend"), "Extend while below (x glass level)", 0.3, 1.0, 0.01,
                  "Equal to the threshold: no extension"),
                P(("texture", "min_glass"), "Min glass thickness [px]", 100, 2000, 10,
                  "Interfaces closer than this are joined"),
                P(("texture", "level_window"), "Glass level window [px]", 0, 5001, 50,
                  "Running 75th percentile; 0 = one level for the whole stack"),
                P(("texture", "edge_margin"), "Ignore next to the air edges [px]", 0, 800, 10),
                P(("texture", "air_threshold"), "Air threshold [intensity]", 5, 100, 1)],
               {"Plates + texture": view_plates_overview, "Plates on the image": view_plates_image,
                "Texture profile": view_texture}, info_plates),
        StepUI("interfaces", "Interfaces between the glass plates",
               "Each interface between two glass plates, with a margin into the glass above and below.",
               [P(("interfaces", "margin"), "Margin into the glass [px]", 0, 300, 5)],
               {"Interface crops": view_crops, "Crops on the image": view_crop_boxes}, info_interfaces),
        StepUI("polymers", "Polymer layers (P2, P4)",
               "Inside the interfaces every layer boundary is a thin dark line. The projected intensity "
               "divided by a running median gives a depth profile; its peaks are the lines. A polymer on "
               "top of the glass below lies between the two lowest of the deepest lines (P1|P2: 3 lines, "
               "for the P1 edge, glue|polymer and polymer|P2; P3|P4: 3 lines, for the P3 edge, the top of "
               "P4_Polymer and the P4 edge). These line parameters also apply to the glue step.",
               [P(("lines", "min_depth"), "Min line depth", 0.02, 0.6, 0.01,
                  "How much darker than the background a line must be"),
                P(("lines", "min_steepness"), "Min line steepness [depth / px]", 0, 0.03, 0.0005,
                  "Depth divided by the width at half depth; drops broad brightness waves in the glass"),
                P(("lines", "line_sigma"), "Smoothing along y [px]", 0.5, 6, 0.5),
                P(("lines", "bg_window"), "Background window [px]", 51, 1001, 10,
                  "Running median; much wider than a line, wider than a glue layer"),
                P(("lines", "search"), "Search beyond the interface [px]", 0, 400, 10,
                  "The texture interface can stop inside a glass-like polymer")],
               {"Polymer close-ups": view_polymer_closeups, "Polymers on the image": view_polymers_image,
                "Depth profiles": view_depth_profiles}, info_polymers),
        StepUI("glues", "Glue layers (P1P2, P2P3, P4P5, P5P6)",
               "A glue layer is light between two dark boundary lines. The vertical intensity gradient "
               "(Sobel, averaged over the width, divided by the background) has a rising edge where the "
               "glue starts (green) and a falling edge where it ends (red). The rising edge must come out "
               "of a dark line and the falling edge go into one (darker than the dashed limit = 1 - min "
               "line depth of the polymer step). A glue too thin to show a light interior is measured as "
               "the dark band itself. P1P2_Glue is searched above P2_Polymer.",
               [P(("glue", "blur_sigma"), "Blur before the gradient [px]", 0.5, 5, 0.5),
                P(("glue", "flank"), "Dark line within [px] of an edge", 3, 40, 1,
                  "The rising edge needs a dark line this close above it, the falling edge below it"),
                P(("glue", "max_thickness"), "Max glue thickness [px]", 20, 300, 5),
                P(("glue", "min_contrast"), "Min interior contrast", 0, 0.3, 0.01,
                  "How much lighter the glue must be than the dark lines around it"),
                P(("glue", "max_band"), "Max dark band [px]", 5, 150, 5,
                  "Thin glue without a light interior: the widest dark band accepted")],
               {"Glue close-ups": view_glue_closeups, "All layers on the image": view_layers_image}, info_glues),
        StepUI("stack", "All layers",
               "Every layer of the stack. Polymers, glues and P3_Polymer+P3P4_Glue (measured together, "
               "from the P3 edge line to the top of P4_Polymer) come from the previous steps; each glass "
               "runs between the measured layers next to it (the air edges at the top and bottom). The "
               "dark boundary lines between a glue edge and the next layer are left uncoloured.",
               [],
               {"Layers labelled": view_stack_labelled, "Full resolution image": view_full_image}, info_stack),
        StepUI("metrics", "Results",
               "Per-image numbers: each layer's thickness in px, and in um with the deviation from the "
               "nominal when um_per_px and the nominal thicknesses (project.yaml, full_stack) are set. "
               "When you're happy, save the parameters and/or run them on all images.",
               [],
               {"Stack diagram": lambda s, c: stack_diagram(s, c, 640), "Layers labelled": view_stack_labelled},
               info_metrics),
    ]
