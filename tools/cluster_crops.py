"""Group detected objects by their features, to discover categories and speed up labelling.

    python tools/cluster_crops.py output/ --k 8

Writes output/clusters/cluster_XX.png (a grid of crops per cluster) and
output/objects_clustered.csv (objects.csv + a `cluster` column). Look at the grids, name
the groups that make sense, and copy those names into the `label` column of objects.csv.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ["area_um2", "eq_diam_um", "length_um", "elongation", "thinness", "solidity",
            "mean_ratio", "min_ratio", "rel_r", "rel_g", "rel_b", "bright_frac", "edge_dist_um"]
LOG_FEATURES = ["area_um2", "eq_diam_um", "length_um", "elongation", "thinness"]


def grid(paths, tile, cols=10, max_n=100):
    tiles = [cv2.resize(cv2.imread(str(p)), (tile, tile)) for p in paths[:max_n]]
    while len(tiles) % cols:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="output folder of spiometrologyviewer.cli (with objects.csv and crops/)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tile", type=int, default=96)
    args = ap.parse_args()

    df = pd.read_csv(args.out / "objects.csv")
    X = df[FEATURES].copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    X = StandardScaler().fit_transform(X.fillna(0))
    df["cluster"] = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(X)

    (args.out / "clusters").mkdir(exist_ok=True)
    for c, g in df.groupby("cluster"):
        g = g.sort_values("area_um2", ascending=False)
        paths = [args.out / "crops" / f"{oid}.png" for oid in g["object_id"]]
        cv2.imwrite(str(args.out / "clusters" / f"cluster_{c:02d}.png"), grid(paths, args.tile))
        classes = g["class"].value_counts().to_dict()
        print(f"cluster {c:2d}: n={len(g):4d}  median diam {g['eq_diam_um'].median():6.1f} um  rule classes {classes}")
    df.to_csv(args.out / "objects_clustered.csv", index=False)


if __name__ == "__main__":
    main()
