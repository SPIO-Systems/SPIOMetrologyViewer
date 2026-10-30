import cv2
import numpy as np

# BGR
CLASS_COLORS = {
    "particle": (0, 0, 255),
    "fibre": (255, 128, 0),
    "residue": (255, 0, 255),
    "edge_chip": (0, 165, 255),
    "unknown": (255, 255, 0),
    # P3
    "large_defect": (255, 0, 128),
    "streak": (0, 215, 255),
    "discolouration": (255, 160, 0),
    "ring": (180, 105, 255),
    "stain": (30, 60, 140),
    # P4 3D profile
    "bump": (0, 0, 0),
    "pit": (255, 255, 255),
    # P1, P5
    "edge_defect": (0, 100, 255),
}


def _outline(img, mask, color, thickness=1):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, color, thickness)


def outlined_text(img, text, org, scale=0.55):
    """White text with a black outline. The outline is the thin text drawn around it, not thick text
    below it: in OpenCV 5 thickness changes the letter spacing, so the two would drift apart."""
    for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
        cv2.putText(img, text, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(a):
    img = cv2.cvtColor(a.rgb if a.result_rgb is None else a.result_rgb, cv2.COLOR_RGB2BGR)

    # Hatch overexposed area: residue can't be seen there.
    h, w = a.saturated.shape
    yy, xx = np.mgrid[:h, :w]
    hatch = a.saturated & ((xx + yy) % 10 < 2)
    img[hatch] = (img[hatch] * 0.4 + np.array([255, 200, 0]) * 0.6).astype(np.uint8)

    _outline(img, a.mirror.polygon, (0, 120, 0))       # fitted mirror outline
    _outline(img, a.inner | a.edge_band, (0, 255, 0))  # analysed area (minus bevel rim and corners)
    _outline(img, a.inner, (0, 255, 255))              # inner / edge-band boundary

    for o in a.objects:
        _outline(img, a.objects_lab == o["label_id"], CLASS_COLORS[o["class"]], 2)

    al = a.alignment
    has_mirror = "offset_x_um" in al   # images without a mirror only have the lens columns
    if al and np.isfinite(al["lens_diameter_um"]):  # lens circle, lens centre (cyan) and mirror centre (green)
        lc = (int(round(al["lens_cx_px"])), int(round(al["lens_cy_px"])))
        r = int(round(al["lens_diameter_um"] / 2 / a.um_per_px)) if a.um_per_px else 0
        if r > 0:
            cv2.circle(img, lc, r, (255, 255, 0), 1)
        cv2.drawMarker(img, lc, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        if has_mirror:
            mc = (int(round(al["mirror_cx_px"])), int(round(al["mirror_cy_px"])))
            cv2.drawMarker(img, mc, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.line(img, lc, mc, (0, 0, 255), 2)

    y = 22
    legend = a.classes or [c for c in CLASS_COLORS if c in ("particle", "fibre", "residue", "unknown")
                           or (c == "edge_chip" and has_mirror)]
    for cls in legend:
        color = CLASS_COLORS[cls]
        n = sum(o["class"] == cls for o in a.objects)
        cv2.putText(img, f"{cls}: {n}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 24
    cv2.putText(img, "hatched = overexposed", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
    if has_mirror:
        cv2.putText(img, f"mirror - lens centre: x {al['offset_x_um']:+.1f} um, y {al['offset_y_um']:+.1f} um",
                    (10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    if a.annotate is not None:
        a.annotate(img)
    return img


def crop(a, o, size):
    """Square crop centred on the object, at least `size` px, resized to `size`."""
    r0, c0, r1, c1 = o["bbox"]
    side = int(max(size, 1.5 * max(r1 - r0, c1 - c0)))
    cy, cx = int(o["cy"]), int(o["cx"])
    h, w = a.objects_lab.shape
    y0, x0 = cy - side // 2, cx - side // 2
    out = np.zeros((side, side, 3), np.uint8)
    sy0, sx0, sy1, sx1 = max(y0, 0), max(x0, 0), min(y0 + side, h), min(x0 + side, w)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a.rgb[sy0:sy1, sx0:sx1]
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_NEAREST)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
