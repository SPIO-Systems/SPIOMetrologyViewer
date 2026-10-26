from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import label as ndimage_label


def overlay_mask(rgb, cfg):
    """Pixels belonging to the burned-in green/yellow grid and scale bar.

    With `overlay.local_px` the greenness / yellowness is compared with the median over that
    window instead of with the other channels at the same pixel, for images whose own colour is
    yellowish (P3 mirror, grating): the thin grid lines stand out from their surroundings."""
    oc = cfg["overlay"]
    m = oc["color_margin"]
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    if oc.get("local_px"):
        k = 2 * (oc["local_px"] // 2) + 1
        mask = np.zeros(r.shape, bool)
        for score in (g - np.maximum(r, b), np.minimum(r, g) - b):     # green, yellow
            local = cv2.medianBlur(np.clip(score + 128, 0, 255).astype(np.uint8), k).astype(np.int16) - 128
            mask |= (score - local > m) & (score > 20)
        mask = mask.astype(np.uint8)
    else:
        green = (g - r > m) & (g - b > m)
        yellow = (r - b > m) & (g - b > m) & (np.abs(r - g) < m)
        mask = (green | yellow).astype(np.uint8)
    k = 2 * cfg["overlay"]["dilate_px"] + 1
    return cv2.dilate(mask, np.ones((k, k), np.uint8)) > 0


def remove_overlay(rgb, mask):
    return cv2.inpaint(rgb, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)


@dataclass
class Mirror:
    corners: np.ndarray      # (4, 2) float, x/y, order: top-left, top-right, bottom-right, bottom-left
    polygon: np.ndarray      # bool mask of the ideal (fitted) rectangle
    dist: np.ndarray         # distance in px from the polygon border, 0 outside
    corner_zone: np.ndarray  # bool mask near the rounded corners
    found: bool
    corrected: list          # sides re-placed from the expected mirror size


def _fit_line(along, across, max_resid=2.0, iters=3):
    """Fit across = a * along + b, dropping outliers (chips, particles on the edge)."""
    keep = np.ones(len(along), bool)
    a, b = 0.0, float(np.median(across))
    for _ in range(iters):
        if keep.sum() < 10:
            break
        a, b = np.polyfit(along[keep], across[keep], 1)
        keep = np.abs(across - (a * along + b)) < max_resid
    return a, b


def _intersect(v_line, h_line):
    """v_line: x = a*y + b, h_line: y = c*x + d."""
    a, b = v_line
    c, d = h_line
    y = (c * b + d) / (1 - a * c)
    return a * y + b, y


def _first_step(profile, frac):
    """Index of the first local maximum of `profile` reaching `frac` of its global maximum.

    Used walking outward from the image centre: the mirror is the innermost edge, while
    the housing rings further out can give stronger steps when the mirror edge is in shadow.
    """
    thr = frac * profile.max()
    for i in range(1, len(profile) - 1):
        if profile[i] >= thr and profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]:
            return i
    return int(np.argmax(profile))


def find_mirror(gray, cfg, ignore=None):
    """Fit the mirror rectangle from log-intensity edges (robust to the strong vignetting).

    `ignore`: pixels (e.g. the inpainted grid overlay) whose gradients are not trusted.
    """
    h, w = gray.shape
    L = np.log1p(cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5))
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    if ignore is not None:
        bad = cv2.dilate(ignore.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        gx[bad] = 0
        gy[bad] = 0

    # Rough edges: first strong step walking outward from the centre, averaged over a central band.
    cy, cx, band = h // 2, w // 2, h // 10
    frac = cfg["mirror"]["rough_step_frac"]
    row_d = gx[cy - band:cy + band].mean(0)
    col_d = gy[:, cx - band:cx + band].mean(1)
    left = cx - 1 - _first_step(row_d[:cx][::-1], frac)
    right = cx + _first_step(-row_d[cx:], frac)
    top = cy - 1 - _first_step(col_d[:cy][::-1], frac)
    bottom = cy + _first_step(-col_d[cy:], frac)

    s = cfg["mirror"]["edge_search_px"]
    excl = int(cfg["mirror"]["corner_exclude_um"] / cfg["um_per_px"])
    ys = np.arange(top + excl, bottom - excl)
    xs = np.arange(left + excl, right - excl)

    def refine(side, centre, half):
        """Best edge per row/column within +-half of `centre` (scalar or one per position).

        Returns the edge positions and the median step strength along them."""
        grad, positions, sign, vertical = sides[side]
        centre = np.broadcast_to(np.round(centre).astype(int), positions.shape)
        idx = np.clip(centre[:, None] + np.arange(-half, half + 1)[None, :], 0, (w if vertical else h) - 1)
        vals = sign * (grad[positions[:, None], idx] if vertical else grad[idx, positions[:, None]])
        best = np.argmax(vals, axis=1)
        rows = np.arange(len(best))
        return idx[rows, best], float(np.median(vals[rows, best]))

    sides = {"left": (gx, ys, 1, True), "right": (gx, ys, -1, True),
             "top": (gy, xs, 1, False), "bottom": (gy, xs, -1, False)}
    rough = {"left": left, "right": right, "top": top, "bottom": bottom}
    lines = {name: _fit_line(sides[name][1], refine(name, rough[name], s)[0]) for name in sides}

    # The mirror is the same part in every image. If a side pair is the wrong distance apart
    # (one edge lost in shadow, a housing ring picked instead), try both hypotheses: keep one
    # side and look for the other at the expected distance. The hypothesis that finds a real
    # edge there wins.
    px = cfg["um_per_px"]
    tol = int(cfg["mirror"]["size_tol_um"] / px)
    corrected = []
    for (a, b), size_um, mid in [(("left", "right"), cfg["mirror"]["size_um"][0], cy),
                                 (("top", "bottom"), cfg["mirror"]["size_um"][1], cx)]:
        expected = size_um / px
        (sa, ia), (sb, ib) = lines[a], lines[b]
        if abs((sb * mid + ib) - (sa * mid + ia) - expected) <= tol:
            continue
        best = None
        for keep, other, offset in ((a, b, expected), (b, a, -expected)):
            slope, icpt = lines[keep]
            pos = sides[other][1]
            edge, strength = refine(other, slope * pos + icpt + offset, tol)
            if best is None or strength > best[0]:
                best = (strength, other, _fit_line(pos, edge))
        _, other, line = best
        lines[other] = line
        corrected.append(other)
    corners = np.array([
        _intersect(lines["left"], lines["top"]),
        _intersect(lines["right"], lines["top"]),
        _intersect(lines["right"], lines["bottom"]),
        _intersect(lines["left"], lines["bottom"]),
    ])

    polygon = np.zeros((h, w), np.uint8)
    cv2.fillPoly(polygon, [np.round(corners).astype(np.int32)], 1)
    dist = cv2.distanceTransform(polygon, cv2.DIST_L2, 5)

    # Corners are chamfered and catch reflections from the housing: exclude a 45-degree
    # triangle at each corner (distance to edge 1 + distance to edge 2 < excl).
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    edge_dist = []
    centre = corners.mean(axis=0)
    for i in range(4):
        (x1, y1), (x2, y2) = corners[i], corners[(i + 1) % 4]
        nx, ny = y2 - y1, -(x2 - x1)
        norm = np.hypot(nx, ny)
        d = ((xx - x1) * nx + (yy - y1) * ny) / norm
        if ((centre[0] - x1) * nx + (centre[1] - y1) * ny) < 0:
            d = -d
        edge_dist.append(d)
    corner_zone = np.zeros((h, w), bool)
    for i in range(4):
        corner_zone |= edge_dist[i - 1] + edge_dist[i] < excl
    corner_zone &= polygon > 0

    width, height = right - left, bottom - top
    found = 0.2 * w < width < 0.9 * w and 0.2 * h < height < 0.95 * h
    return Mirror(corners, polygon > 0, dist, corner_zone, found, corrected)


def flatten_illumination(gray, region, cfg):
    """Return intensity / local background, with background estimated only from inside `region`."""
    px = cfg["um_per_px"]
    k = int(cfg["illumination"]["closing_um"] / px) | 1
    g = gray.astype(np.float32)
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    sigma = cfg["illumination"]["blur_um"] / px
    m = region.astype(np.float32)
    bg = cv2.GaussianBlur(closed * m, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-3)
    bg = np.maximum(bg, 1.0)
    return np.where(region, g / bg, 1.0).astype(np.float32), bg


def saturation_mask(rgb, cfg):
    return rgb.max(axis=2) >= cfg["saturation_level"]


def glare_mask(saturated, mirror, cfg):
    """Reflection arcs from the housing: saturated spots crossing the mirror outline,
    other than the central hotspot, widened by `glare_margin_um` to cover their shadow."""
    lab, n = ndimage_label(saturated)
    if n <= 1:
        return np.zeros_like(saturated)
    sizes = np.bincount(lab.ravel())[1:]
    crossing = np.intersect1d(np.unique(lab[mirror.polygon]), np.unique(lab[~mirror.polygon]))
    glare = np.isin(lab, crossing[crossing > 0]) & (lab != 1 + int(np.argmax(sizes)))
    k = 2 * int(cfg["glare_margin_um"] / cfg["um_per_px"]) + 1
    return cv2.dilate(glare.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


@dataclass
class Lens:
    centre: np.ndarray        # (x, y) px, circle fitted to the outer edge of the dark round lens
    radius: float             # px
    ring_centre: np.ndarray   # thin bright ring just inside, fitted on its own (consistency check)
    ring_radius: float
    found: bool
    edge_points: np.ndarray   # (n, 2) x/y points used for the outer circle (inliers), for display
    ring_points: np.ndarray


def _fit_circle(pts, max_resid=1.5, iters=6):
    """Least-squares circle through `pts` (n, 2), repeatedly dropping points far off the circle
    (reflections, particles, the grid). Returns centre, radius and the inlier mask."""
    keep = np.ones(len(pts), bool)
    cx = cy = r = np.nan
    for _ in range(iters):
        if keep.sum() < 10:
            break
        x, y = pts[keep].T
        sol = np.linalg.lstsq(np.c_[x, y, np.ones_like(x)], x * x + y * y, rcond=None)[0]
        cx, cy = sol[0] / 2, sol[1] / 2
        r = np.sqrt(sol[2] + cx * cx + cy * cy)
        resid = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        keep = resid < max(max_resid, 2.5 * np.median(resid[keep]))
    return np.array([cx, cy]), r, keep


def _polar(img, centre, radii, angles):
    """Sample `img` along rays from `centre`: rows = angles, columns = radii. Outside = NaN."""
    h, w = img.shape
    xs = (centre[0] + radii[None, :] * np.cos(angles)[:, None]).astype(np.float32)
    ys = (centre[1] + radii[None, :] * np.sin(angles)[:, None]).astype(np.float32)
    out = cv2.remap(img.astype(np.float32), xs, ys, cv2.INTER_LINEAR)
    out[(xs < 0) | (xs > w - 1) | (ys < 0) | (ys > h - 1)] = np.nan
    return out, xs, ys


def _peak_points(score, radii, angles, centre):
    """Strongest radius per angle (sub-pixel), as x/y points. Rows without a usable peak are dropped."""
    score = np.where(np.isfinite(score), score, -np.inf)
    i = np.argmax(score, axis=1)
    rows = np.arange(len(i))
    ok = np.isfinite(score[rows, i]) & (score[rows, i] > 0) & (i > 0) & (i < score.shape[1] - 1)
    i, rows = i[ok], rows[ok]
    a, b, c = score[rows, i - 1], score[rows, i], score[rows, i + 1]
    den = a - 2 * b + c
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(np.isfinite(den) & (den < 0), 0.5 * (a - c) / np.where(den < 0, den, -1), 0)
    r = radii[i] + np.clip(delta, -0.5, 0.5) * (radii[1] - radii[0])
    return np.c_[centre[0] + r * np.cos(angles[rows]), centre[1] + r * np.sin(angles[rows])]


def find_lens(gray, cfg, centre, ignore=None):
    """Fit the round lens around the mirror.

    The outer edge of the dark lens (dark -> bright going outward) is found on rays from
    `centre` (the mirror centre) and fitted with a circle; the search is repeated once around
    that fit. The thin bright ring just inside is fitted separately as a cross-check."""
    px = cfg["um_per_px"]
    lc = cfg["lens"]
    L = np.log1p(cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0))
    if ignore is not None:
        L[cv2.dilate(ignore.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = np.nan
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    step = 0.5

    centre = np.asarray(centre, float)
    radius = np.nan
    pts = np.zeros((0, 2))
    keep = np.zeros(0, bool)
    for _ in range(2):
        radii = np.arange(lc["search_um"][0] / px, lc["search_um"][1] / px, step)
        P, _, _ = _polar(L, centre, radii, angles)
        d = np.full_like(P, np.nan)
        k = int(round(2 / step))
        d[:, k:-k] = P[:, 2 * k:] - P[:, :-2 * k]      # outward brightness step over 4 px
        pts = _peak_points(d, radii, angles, centre)
        if len(pts) < 20:
            break
        centre, radius, keep = _fit_circle(pts)

    # Thin bright ring inside the outer edge: a narrow peak above its surroundings.
    ring_c, ring_r, ring_pts = np.array([np.nan, np.nan]), np.nan, np.zeros((0, 2))
    if np.isfinite(radius):
        lo, hi = lc["ring_search_um"]
        radii = np.arange(radius - hi / px, radius - lo / px, step)
        P, _, _ = _polar(L, centre, radii, angles)
        k = int(round(4 / step))
        peak = np.full_like(P, np.nan)
        peak[:, k:-k] = P[:, k:-k] - 0.5 * (P[:, :-2 * k] + P[:, 2 * k:])
        rp = _peak_points(peak, radii, angles, centre)
        if len(rp) >= 20:
            ring_c, ring_r, rk = _fit_circle(rp)
            ring_pts = rp[rk]

    h, w = gray.shape
    found = bool(np.isfinite(radius) and keep.sum() > 0.3 * len(angles)
                 and lc["search_um"][0] / px < radius < lc["search_um"][1] / px)
    return Lens(centre, float(radius), ring_c, float(ring_r), found, pts[keep] if len(keep) else pts, ring_pts)


def alignment(mirror, lens, cfg):
    """Mirror centre relative to the lens centre (image axes: +x right, +y down), in um."""
    px = cfg["um_per_px"]
    c = mirror.corners
    mc = c.mean(axis=0)
    off = (mc - lens.centre) * px
    # Rotation of the mirror against the image axes, from its top and bottom edges.
    angle = np.degrees(np.mean([np.arctan2(*(c[1] - c[0])[::-1]), np.arctan2(*(c[2] - c[3])[::-1])]))
    return {
        "lens_found": lens.found,
        "lens_cx_px": float(lens.centre[0]),
        "lens_cy_px": float(lens.centre[1]),
        "lens_diameter_um": 2 * lens.radius * px,
        "mirror_cx_px": float(mc[0]),
        "mirror_cy_px": float(mc[1]),
        "offset_x_um": float(off[0]),
        "offset_y_um": float(off[1]),
        "offset_um": float(np.hypot(*off)),
        "mirror_angle_deg": float(angle),
        "lens_ring_offset_um": float(np.hypot(*(lens.ring_centre - lens.centre)) * px),
    }


def lens_region(lens, shape, cfg):
    """For images without a mirror: the lens surface as the analysed region, i.e. the disk inside
    the thin bright ring, minus `lens.inner_margin_um`. Returned as a `Mirror` (without corners
    to exclude), so the later steps don't care which one they get."""
    h, w = shape
    ring = lens.ring_radius if np.isfinite(lens.ring_radius) else lens.radius
    radius = ring - cfg["lens"]["inner_margin_um"] / cfg["um_per_px"]
    polygon = np.zeros((h, w), np.uint8)
    if np.isfinite(radius) and radius > 0:
        cv2.circle(polygon, tuple(int(round(v)) for v in lens.centre), int(round(radius)), 1, -1)
    dist = cv2.distanceTransform(polygon, cv2.DIST_L2, 5)
    cx, cy = lens.centre
    r = radius if np.isfinite(radius) else 0.0
    corners = np.array([[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]], float)
    return Mirror(corners, polygon > 0, dist, np.zeros((h, w), bool), lens.found, [])


def lens_info(lens, cfg):
    """The lens columns of `alignment`, for images without a mirror (so no offset)."""
    px = cfg["um_per_px"]
    return {
        "lens_found": lens.found,
        "lens_cx_px": float(lens.centre[0]),
        "lens_cy_px": float(lens.centre[1]),
        "lens_diameter_um": 2 * lens.radius * px,
        "lens_ring_diameter_um": 2 * lens.ring_radius * px,
        "lens_ring_offset_um": float(np.hypot(*(lens.ring_centre - lens.centre)) * px),
    }
