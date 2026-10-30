"""Size and edge quality of a rectangle (P1 aperture and slit, P5 mirror).

The rectangle is found anywhere in the image. Each of its sides is then followed with sub-pixel
precision: on every line across the side, the edge is where the signal crosses halfway between the
levels just outside and just inside. A straight line fitted to those points gives the size, and what
is left over gives the edge quality. Burned-in grid pixels (`ignore`) are bridged, not trusted.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from . import preprocess

SIDES = ("left", "right", "top", "bottom")


@dataclass
class Side:
    name: str
    lines: np.ndarray    # row (left, right) or column (top, bottom) of each point
    pos: np.ndarray      # edge position there, NaN if that line is under the grid
    res: np.ndarray      # pos - fitted line [px], + = into the rectangle
    slope: float
    offset: float        # fitted position = slope * line + offset
    rms_px: float        # scatter around the line, without defects
    hidden: bool         # most of the side lies under a grid line: its position is less certain

    def at(self, line):
        return self.slope * line + self.offset


@dataclass
class Rect:
    found: bool
    sides: dict = None
    corners: np.ndarray = None   # (4, 2) x, y: top-left, top-right, bottom-right, bottom-left
    width_px: float = np.nan     # along x
    height_px: float = np.nan    # along y
    centre: tuple = (np.nan, np.nan)
    angle_deg: float = np.nan
    parallel_deg: float = np.nan

    def mask(self, shape):
        m = np.zeros(shape, np.uint8)
        if self.found:
            cv2.fillPoly(m, [np.round(self.corners).astype(np.int32)], 1)
        return m > 0

    @property
    def rms_px(self):
        return float(np.mean([s.rms_px for s in self.sides.values()])) if self.found else np.nan

    def max_dev_px(self):
        """Largest distance of any side's edge from its fitted line."""
        if not self.found:
            return np.nan
        return max((float(np.nanmax(np.abs(s.res))) for s in self.sides.values()), default=np.nan)


def rough_box(sig, ignore, cfg, nominal_px):
    """(x0, x1, y0, y1) of the bright area of `sig` closest to `nominal_px` (width, height), or None:
    a big flake can be larger than a slit. Taken as the median outline, so a defect at an edge doesn't
    move it."""
    outside, inside = np.median(sig), np.percentile(sig, cfg["inside_percentile"])
    if inside - outside < cfg["min_contrast"]:
        return None
    m = ((sig > (outside + inside) / 2) & ~ignore).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))   # thin fibres touching it
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    misfit = np.abs(np.log(st[1:, cv2.CC_STAT_WIDTH] / nominal_px[0])) + np.abs(np.log(st[1:, cv2.CC_STAT_HEIGHT] / nominal_px[1]))
    i = 1 + int(np.argmin(misfit))
    blob = lab == i
    rows = np.flatnonzero(blob.any(axis=1))
    x0 = int(np.median(blob[rows].argmax(axis=1)))
    x1 = sig.shape[1] - int(np.median(blob[rows, ::-1].argmax(axis=1)))
    cols = blob[:, x0:x1].any(axis=0)
    inner = blob[:, x0:x1][:, cols]
    if not inner.size:
        return None
    y0 = int(np.median(inner.argmax(axis=0)))
    y1 = sig.shape[0] - int(np.median(inner[::-1].argmax(axis=0)))
    return x0, x1, y0, y1


def _follow(sig, ignore, lines, c0, sign, reach_in, cfg):
    """Edge position per line of `sig` (lines = rows), the inside lying at `sign` from `c0`."""
    out, inset, plateau = cfg["search_out_px"], cfg["inset_px"], cfg["plateau_px"]
    idx = c0 + sign * np.arange(-out, reach_in + 1)
    keep = (idx >= 0) & (idx < sig.shape[1])
    idx, n0 = idx[keep], int((~keep[:out]).sum())     # n0: positions lost outside the image
    p = sig[lines][:, idx].astype(np.float32)
    bad = ignore[lines][:, idx]
    skip = bad.mean(axis=1) > 0.5                     # the line runs along a grid line
    x = np.arange(p.shape[1])
    for i in np.flatnonzero(bad.any(axis=1) & ~skip):
        p[i, bad[i]] = np.interp(x[bad[i]], x[~bad[i]], p[i, ~bad[i]])
    p = cv2.GaussianBlur(p, (5, 1), 0)

    c = out - n0                                      # c0 in `p`
    level = lambda a, b: ndimage.median_filter(np.median(p[:, max(a, 0):max(b, 1)], axis=1), size=cfg["level_lines"],
                                               mode="nearest")
    thr = (level(c - inset - plateau, c - inset) + level(c + inset, c + inset + plateau)) / 2
    pos, hidden = np.full(len(lines), np.nan), np.zeros(len(lines), bool)
    for i in np.flatnonzero(~skip):
        # walk out of the rectangle: a speck on the coating further out doesn't count as the edge
        below = np.flatnonzero(p[i] < thr[i])
        k = below[-1] if len(below) else 0
        k = min(k, p.shape[1] - 2)
        frac = (thr[i] - p[i, k]) / (p[i, k + 1] - p[i, k]) if p[i, k + 1] != p[i, k] else 0.0
        pos[i] = idx[k] + sign * float(np.clip(frac, 0, 1))
        hidden[i] = bad[i, max(k - 1, 0):k + 3].any()
    return pos, hidden


def _fit(lines, pos, cfg):
    """(slope, offset, inlier mask): a straight line that isn't pulled by defects."""
    ok = np.isfinite(pos)
    keep = ok.copy()
    slope, offset = 0.0, float(np.nanmedian(pos))
    for _ in range(6):
        slope, offset = np.polyfit(lines[keep], pos[keep], 1)
        res = pos - (slope * lines + offset)
        spread = 1.4826 * np.nanmedian(np.abs(res[keep]))
        new = ok & (np.abs(np.nan_to_num(res)) <= max(cfg["clip_sigma"] * spread, 0.3))
        if new.sum() < 2 or (new == keep).all():
            break
        keep = new
    return float(slope), float(offset), keep


def _side(name, sig, ignore, lines, c0, sign, reach_in, cfg):
    pos, hidden = _follow(sig, ignore, lines, c0, sign, reach_in, cfg)
    if np.isfinite(pos).sum() < 0.3 * len(lines):
        return None
    slope, offset, keep = _fit(lines, pos, cfg)
    res = (pos - (slope * lines + offset)) * sign
    return Side(name, lines, pos, res, slope, offset, float(np.std(res[keep])), bool(hidden.mean() > 0.5))


def fit_rectangle(sig, ignore, cfg, px):
    """The rectangle (Rect) of the bright area in `sig` of about the nominal `size_um` in `cfg`; pixels
    in `ignore` aren't used."""
    box = rough_box(sig, ignore, cfg, [v / px for v in cfg["size_um"]])
    h, w = sig.shape
    if box is None:
        return Rect(False)
    x0, x1, y0, y1 = box
    trim = cfg["corner_px"]
    if x1 - x0 <= 2 * trim + 4 or y1 - y0 <= 2 * trim + 4:
        return Rect(False)
    rows, cols = np.arange(y0 + trim, y1 - trim), np.arange(x0 + trim, x1 - trim)
    reach_x = min(cfg["search_in_px"], (x1 - x0) // 2 - 2)
    reach_y = min(cfg["search_in_px"], (y1 - y0) // 2 - 2)
    # top and bottom: the same on the transposed image
    st, it = sig.T, ignore.T
    sides = {
        "left": _side("left", sig, ignore, rows, x0, 1, reach_x, cfg),
        "right": _side("right", sig, ignore, rows, x1 - 1, -1, reach_x, cfg),
        "top": _side("top", st, it, cols, y0, 1, reach_y, cfg),
        "bottom": _side("bottom", st, it, cols, y1 - 1, -1, reach_y, cfg),
    }
    if any(s is None for s in sides.values()):
        return Rect(False)

    def corner(v, u):          # v: left / right (x from y), u: top / bottom (y from x)
        y = (u.slope * v.offset + u.offset) / (1 - u.slope * v.slope)
        return v.at(y), y

    l, r, t, b = (sides[k] for k in SIDES)
    corners = np.array([corner(l, t), corner(r, t), corner(r, b), corner(l, b)])
    cx, cy = corners.mean(axis=0)
    # the slopes are of opposite sense: a turned rectangle has l.slope = -t.slope
    angles = np.degrees(np.arctan([l.slope, r.slope, -t.slope, -b.slope]))
    # from the longer pair of sides: on the slit the ends are too short for a slope
    pair = angles[:2] if len(l.lines) >= len(t.lines) else angles[2:]
    return Rect(True, sides, corners, r.at(cy) - l.at(cy), b.at(cx) - t.at(cx), (float(cx), float(cy)),
                float(pair.mean()), float(abs(pair[0] - pair[1])))


def defects(rect, shape, px, cfg):
    """Where an edge leaves its fitted line by more than `defect_um` over at least `defect_min_um`.
    Returns (label image, list of object dicts in the form of features.blob_features). `coating`:
    True if the edge moved into the rectangle (material sticking in), False if away from it (a chip)."""
    lab = np.zeros(shape, np.int32)
    objects = []
    if not rect.found:
        return lab, objects
    h, w = shape
    for name, s in rect.sides.items():
        res = np.nan_to_num(s.res)
        # bridged, so a grid line across a defect doesn't make two of it
        marked, n = ndimage.label(ndimage.binary_closing(np.abs(res) > cfg["defect_um"] / px, np.ones(7)))
        for i in range(1, n + 1):
            run = np.flatnonzero(marked == i)
            if len(run) * px < cfg["defect_min_um"]:
                continue
            ident = len(objects) + 1
            pts = []
            for k in run:
                if not np.isfinite(s.pos[k]):
                    continue
                a, b = sorted((s.at(s.lines[k]), s.pos[k]))
                across = np.arange(int(np.floor(a)), int(np.ceil(b)) + 1)
                line = np.full(across.shape, s.lines[k])
                y, x = (line, across) if name in ("left", "right") else (across, line)
                in_image = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                lab[y[in_image], x[in_image]] = ident
                pts += zip(y[in_image], x[in_image])
            if not pts:
                continue
            pts = np.array(pts)
            depth = float(np.abs(res[run]).max()) * px
            length = len(run) * px
            area = len(pts) * px * px
            objects.append({
                "label_id": ident, "class": "edge_defect", "side": name,
                "cy": float(pts[:, 0].mean()), "cx": float(pts[:, 1].mean()),
                "bbox": (int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()) + 1, int(pts[:, 1].max()) + 1),
                "area_um2": area, "eq_diam_um": float(2 * np.sqrt(area / np.pi)), "length_um": length,
                "feret_um": float(max(length, depth)), "chip_depth_um": depth, "edge_dist_um": 0.0, "in_inner": False,
                "coating": bool(res[run[np.argmax(np.abs(res[run]))]] > 0),
            })
    return lab, objects


def flatten(gray, region, cfg):
    """preprocess.flatten_illumination, scaled so the typical pixel of `region` is 1.0 (noise lifts the
    estimated background, which would put a clean surface just below the thresholds)."""
    ratio, bg = preprocess.flatten_illumination(gray, region, cfg)
    if region.any():
        level = float(np.median(ratio[region]))
        ratio, bg = np.where(region, ratio / level, 1.0).astype(np.float32), bg * level
    return ratio, bg


def merge(surface_lab, surface_objects, edge_lab, edge_objects):
    """One label image for the coating or mirror objects and the edge defects (ids kept apart)."""
    lab = surface_lab.astype(np.int32).copy()
    shift = int(lab.max())
    lab[edge_lab > 0] = edge_lab[edge_lab > 0] + shift
    moved = [{**o, "label_id": o["label_id"] + shift} for o in edge_objects]
    return lab, surface_objects + moved


def text(rect, px):
    """Summary of a fit for the tuning GUI."""
    if not rect.found:
        return "Not found."
    lines = [f"Size: {rect.width_px * px:.2f} x {rect.height_px * px:.2f} um   centre ({rect.centre[0] * px:.0f}, "
             f"{rect.centre[1] * px:.0f}) um   turned {rect.angle_deg:+.3f} deg"]
    for s in rect.sides.values():
        lines.append(f"{s.name:7s} scatter {s.rms_px * px:5.2f} um, furthest {np.nanmax(np.abs(s.res)) * px:6.2f} um"
                     + ("   under a grid line" if s.hidden else ""))
    return "\n".join(lines)
