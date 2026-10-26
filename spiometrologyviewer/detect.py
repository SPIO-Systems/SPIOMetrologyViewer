import cv2
import numpy as np
from skimage import measure, morphology


def _masked_blur(img, mask, sigma):
    m = mask.astype(np.float32)
    return cv2.GaussianBlur(img * m, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-3)


def _texture_energy(img, mask, sigmas, win):
    """Local mean of |difference of Gaussians|, using only pixels inside `mask`."""
    dog = np.abs(_masked_blur(img, mask, sigmas[0]) - _masked_blur(img, mask, sigmas[1]))
    m = mask.astype(np.float32)
    n = np.maximum(cv2.boxFilter(m, -1, (win, win)), 1e-3)
    return cv2.boxFilter(dog * m, -1, (win, win)) / n


def detect_dark(ratio, valid, cfg):
    """Opaque material (particles, fibres, chips): clearly darker than the local background."""
    dark = (ratio < cfg["dark"]["t_ratio"]) & valid
    return morphology.remove_small_objects(dark, max_size=cfg["dark"]["min_area_px"] - 1)


def detect_residue(ratio, valid, dark, saturated, dist, cfg):
    """Faint textured films (dried liquid, ferns, droplet rings) in non-saturated areas.

    Returns the residue mask, the texture threshold that was used and the texture energy map.
    """
    rc = cfg["residue"]
    usable = valid & ~saturated
    energy = _texture_energy(np.clip(ratio, 0, 2), usable, rc["dog_sigmas"], rc["window_px"])

    # Solid particles leave a band-pass halo around themselves; that's not residue.
    k = 2 * rc["particle_halo_px"] + 1
    near_dark = cv2.dilate(dark.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    clean = usable & ~near_dark
    vals = energy[clean]
    med = np.median(vals) if vals.size else 0.0
    mad = np.median(np.abs(vals - med)) if vals.size else 0.0
    thr = max(med + rc["k_mad"] * 1.4826 * mad, rc["min_energy"])

    tex = (energy > thr) & usable & ~near_dark
    k = rc["closing_px"]
    tex = cv2.morphologyEx(tex.astype(np.uint8), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    tex &= valid
    min_px = int(rc["min_area_um2"] / cfg["um_per_px"] ** 2)
    tex = morphology.remove_small_objects(tex, max_size=min_px - 1)

    # A long thin strip hugging the edge is the bevel, not residue.
    band = cfg["edge_band_um"] / cfg["um_per_px"]
    for r in measure.regionprops(measure.label(tex)):
        rr, cc = r.coords[:, 0], r.coords[:, 1]
        thin = r.axis_major_length > rc["bevel_strip_elongation"] * max(r.axis_minor_length, 1)
        if thin and dist[rr, cc].max() < band:
            tex[rr, cc] = False
    return tex, thr, energy


def dense_clusters(dark, valid, cfg):
    """Regions where many small dark blobs crowd together (droplet fields, spatter).

    Blobs are linked by a closing of `link_um`; a linked group of at least `min_blobs`
    blobs becomes one residue region (its closed hull).
    """
    cc = cfg["residue"]["cluster"]
    k = 2 * int(cc["link_um"] / cfg["um_per_px"] / 2) + 1
    linked = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    linked &= valid
    blobs = measure.label(dark, connectivity=2)
    out = np.zeros_like(dark)
    for r in measure.regionprops(measure.label(linked, connectivity=2)):
        rr, cc_ = r.coords[:, 0], r.coords[:, 1]
        n = len(np.unique(blobs[rr, cc_])) - (blobs[rr, cc_] == 0).any()
        if n >= cc["min_blobs"]:
            out[rr, cc_] = True
    return out


def drop_touching(mask, zone):
    """Remove connected components of `mask` that touch `zone`."""
    lab = measure.label(mask, connectivity=2)
    bad = np.unique(lab[zone & (lab > 0)])
    return mask & ~np.isin(lab, bad)


def detect_edge_chips(ratio, mirror, cfg, exclude=None):
    """Dark notches eating into the mirror outline.

    Looks closer to the edge than the dirt detectors (from `start_um` instead of the
    bevel rim). A dark blob touching that start line, reaching at least `min_depth_um`
    from the fitted edge, and not centred in a (chamfered) corner counts as a chip.
    Returns (chip label image, list of chip dicts).
    """
    px = cfg["um_per_px"]
    ec = cfg["edge_chip"]
    start = ec["start_um"] / px
    dark = (ratio < cfg["dark"]["t_ratio"]) & (mirror.dist > start)
    if exclude is not None:
        dark = drop_touching(dark, exclude)
    touch = mirror.dist <= start + 1.5

    chips = np.zeros(ratio.shape, np.int32)
    info = []
    for r in measure.regionprops(measure.label(dark, connectivity=2)):
        rr, cc = r.coords[:, 0], r.coords[:, 1]
        cy, cx = (int(round(v)) for v in r.centroid)
        if not touch[rr, cc].any() or mirror.corner_zone[cy, cx]:
            continue
        depth = float(mirror.dist[rr, cc].max()) * px
        if depth >= ec["min_depth_um"]:
            chips[rr, cc] = len(info) + 1
            info.append({"depth_um": depth})
    return chips, info
