"""Rule-based classes. Once enough labels exist, swap `classify_dark` for a trained model
that takes the same feature dicts."""

CLASSES = ["particle", "fibre", "residue", "edge_chip", "unknown"]


def classify_dark(feat, cfg):
    c = cfg["classify"]
    long_shape = feat["elongation"] > c["fibre_min_elongation"] or feat["thinness"] > c["fibre_min_thinness"]
    if long_shape and feat["length_um"] >= c["fibre_min_length_um"]:
        return "fibre"
    if feat["solidity"] < c["unknown_max_solidity"]:
        return "unknown"
    return "particle"
