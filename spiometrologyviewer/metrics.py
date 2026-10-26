import numpy as np

DIRT_CLASSES = ("particle", "fibre", "residue", "unknown")


def class_mask(a, cls):
    ids = [o["label_id"] for o in a.objects if o["class"] in cls]
    return np.isin(a.objects_lab, ids) if ids else np.zeros(a.objects_lab.shape, bool)


def image_metrics(a, cfg):
    p = cfg["presence"]
    px2 = cfg["um_per_px"] ** 2
    by = {c: [o for o in a.objects if o["class"] == c] for c in ("particle", "fibre", "residue", "edge_chip", "unknown")}

    dirt = class_mask(a, DIRT_CLASSES)
    residue = class_mask(a, ("residue",))
    inner_n, edge_n = a.inner.sum(), a.edge_band.sum()
    analysed = inner_n + edge_n

    def pct(mask, region, n):
        return 100.0 * (mask & region).sum() / n if n else np.nan

    particle_d = [o["eq_diam_um"] for o in by["particle"]]
    fibre_l = [o["length_um"] for o in by["fibre"]]
    residue_a = [o["area_um2"] for o in by["residue"]]
    chip_d = [o["chip_depth_um"] for o in by["edge_chip"]]

    lens = cfg.get("region", "mirror") == "lens"   # no mirror: the lens surface was analysed
    region = {"lens_area_mm2": analysed * px2 / 1e6} if lens else {
        "mirror_found": a.mirror.found,
        "mirror_sides_from_size": "+".join(a.mirror.corrected),
        "mirror_area_mm2": analysed * px2 / 1e6,
    }
    chips = {} if lens else {
        "edge_chip_count": len(chip_d),
        "edge_chip_max_depth_um": max(chip_d, default=0.0),
        "edge_chips_present": any(d >= p["edge_chip_min_depth_um"] for d in chip_d),
    }
    return {
        "image": a.name,
        "um_per_px": cfg["um_per_px"],
        **region,
        **a.alignment,
        "dirt_coverage_inner_pct": pct(dirt, a.inner, inner_n),
        "dirt_coverage_edge_pct": pct(dirt, a.edge_band, edge_n),
        "dirt_coverage_total_pct": pct(dirt, a.inner | a.edge_band, analysed),
        "particle_count": len(particle_d),
        "particle_max_diam_um": max(particle_d, default=0.0),
        "particles_present": any(d >= p["particle_min_diam_um"] for d in particle_d),
        "fibre_count": len(fibre_l),
        "fibre_max_length_um": max(fibre_l, default=0.0),
        "fibres_present": any(l >= p["fibre_min_length_um"] for l in fibre_l),
        "residue_area_pct": pct(residue, a.inner | a.edge_band, analysed),
        "residue_present": sum(residue_a) >= p["residue_min_total_area_um2"],
        **chips,
        "unknown_count": len(by["unknown"]),
        "saturated_pct_inner": pct(a.saturated, a.inner, inner_n),
    }
