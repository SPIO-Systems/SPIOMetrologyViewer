import numpy as np
from skimage import measure, morphology


def blob_features(label_img, ratio, rgb, bg, mirror, saturated, inner, cfg):
    """One feature dict per labelled blob. Lengths in um, areas in um^2."""
    px = cfg["um_per_px"]
    rows = []
    for r in measure.regionprops(label_img, intensity_image=ratio):
        rr, cc = r.coords[:, 0], r.coords[:, 1]
        skel_len = float(morphology.skeletonize(r.image).sum())
        major = r.axis_major_length
        minor = max(r.axis_minor_length, 1.0)
        rgb_rel = rgb[rr, cc].astype(np.float32) / bg[rr, cc, None]
        rows.append({
            "label_id": r.label,
            "cy": r.centroid[0], "cx": r.centroid[1],
            "bbox": r.bbox,
            "area_um2": r.area * px * px,
            "eq_diam_um": r.equivalent_diameter_area * px,
            "length_um": max(major, skel_len) * px,
            "feret_um": float(r.feret_diameter_max) * px,   # largest extent (a ragged blob's skeleton zigzags)
            "elongation": major / minor,
            "thinness": skel_len ** 2 / r.area,
            "solidity": r.solidity,
            "mean_ratio": float(r.intensity_mean),
            "min_ratio": float(r.intensity_min),
            "rel_r": float(rgb_rel[:, 0].mean()),
            "rel_g": float(rgb_rel[:, 1].mean()),
            "rel_b": float(rgb_rel[:, 2].mean()),
            "bright_frac": float((ratio[rr, cc] > 1.1).mean()),
            "edge_dist_um": float(mirror.dist[rr, cc].min()) * px,
            "in_inner": bool(inner[rr, cc].mean() > 0.5),
            "saturated_frac": float(saturated[rr, cc].mean()),
        })
    return rows
