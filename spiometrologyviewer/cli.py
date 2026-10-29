"""Analyse a folder of mirror images, or a folder of wafer folders.

    python -m spiometrologyviewer.cli --measurement "P2/With mirror"   # a measurement type of the current project
    python -m spiometrologyviewer.cli --all                            # every measurement type in it
    python -m spiometrologyviewer.cli --project 17013_2 --all          # of another project (see projects.yaml)
    python -m spiometrologyviewer.cli some/wafer/folder --out output/some_wafer
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

from . import __version__, inventory, io, measurements, metrics, pipeline, report, tuning, workspace

FEATURE_COLUMNS = [
    "area_um2", "eq_diam_um", "length_um", "elongation", "thinness", "solidity",
    "mean_ratio", "min_ratio", "rel_r", "rel_g", "rel_b", "bright_frac",
    "edge_dist_um", "in_inner", "saturated_frac", "chip_depth_um",
    "height_nm",   # P4 3D profile: bump height / pit depth
]


def _input(cfg):
    i = cfg.get("input", {})
    return i.get("image_glob"), i.get("min_images", 10)


def _thumb(bgr, size):
    h, w = bgr.shape[:2]
    f = size / max(h, w)
    return cv2.resize(bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)


def run_batch(images, out, cfg, crops=True, progress=print, tuning_name=None, template=io.DEFAULT_CONFIG):
    """Analyse every image in `images`, writing results/objects CSVs, overlays, thumbnails and
    crops to `out`.

    The parameters used are saved next to the results as config_used.yaml, and `tuning_name`
    (see tuning.register) goes into results.csv as the `tuning` column, the code version as
    `code_version`. Image paths are stored relative to the workspace (see workspace.relative)."""
    out = Path(out)
    for sub in ("overlays", "thumbs", "thumbs_result", "plots"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if crops:
        (out / "crops").mkdir(exist_ok=True)
    thumb_px = cfg["output"].get("thumb_px", 256)

    paths = io.list_images(images, _input(cfg)[0])
    results, objects = [], []
    for n, path in enumerate(paths, start=1):
        a, m = pipeline.analyse(path, cfg)
        m["image_path"] = workspace.relative(path)
        if tuning_name:
            m["tuning"] = tuning_name
        m["code_version"] = __version__
        results.append(m)
        overlay = report.draw_overlay(a)
        cv2.imwrite(str(out / "overlays" / f"{a.name}.png"), overlay)
        cv2.imwrite(str(out / "thumbs" / f"{a.name}.jpg"), _thumb(a.rgb[..., ::-1], thumb_px))
        cv2.imwrite(str(out / "thumbs_result" / f"{a.name}.jpg"), _thumb(overlay, thumb_px))
        if a.plots is not None:
            cv2.imwrite(str(out / "plots" / f"{a.name}.png"), a.plots[..., ::-1])

        for i, o in enumerate(a.objects):
            oid = f"{a.name}_{i:04d}"
            row = {"object_id": oid, "image": a.name, "class": o["class"], "label": "",
                   "cx": round(o["cx"], 1), "cy": round(o["cy"], 1)}
            row.update({k: o.get(k, float("nan")) for k in FEATURE_COLUMNS})
            objects.append(row)
            if crops:
                cv2.imwrite(str(out / "crops" / f"{oid}.png"), report.crop(a, o, cfg["output"]["crop_px"]))

        counts = {}
        for o in a.objects:
            counts[o["class"]] = counts.get(o["class"], 0) + 1
        dirt = next((m[k] for k in ("dirt_coverage_inner_pct", "dirt_coverage_total_pct") if k in m), None)
        progress(f"[{n}/{len(paths)}] {a.name}: "
                 + (f"dirt {dirt:.2f}%  " if dirt is not None else "")
                 + (f"R {m['roc_um'] / 1000:.4f} mm  " if "roc_um" in m else "")
                 + " ".join(f"{c}={k}" for c, k in sorted(counts.items()))
                 + (f"  offset x {m['offset_x_um']:+.1f} y {m['offset_y_um']:+.1f} um" if "offset_x_um" in m else ""))

    for name, rows in (("results.csv", results), ("objects.csv", objects)):
        try:
            pd.DataFrame(rows).round(3).to_csv(out / name, index=False)
        except PermissionError:
            raise PermissionError(f"Can't write {out / name}: is it open in Excel? Close it and run again.") from None
    io.save_config(out / "config_used.yaml", cfg, template=template)
    progress(f"Wrote {len(results)} images, {len(objects)} objects to {out}")


def run_wafers(root, out, cfg, crops=True, skip_done=False, progress=print, wafers=None, tuning_name=None,
               template=io.DEFAULT_CONFIG):
    """Run the batch on every wafer folder under `root` (see io.find_wafers), or on `wafers`
    ({name: image folder}) if given, each into out/<wafer name>/. If `root` itself holds the
    images, its results go straight into `out`.

    Writes out/wafers.csv (wafer, folder, images) and returns {wafer name: output folder}."""
    out = Path(out)
    pattern, min_images = _input(cfg)
    if wafers is None:
        wafers = io.find_wafers(root, pattern, min_images)
        if list(wafers.values()) == [Path(root)]:
            run_batch(root, out, cfg, crops, progress, tuning_name, template)
            return {Path(root).name: out}
    if not wafers:
        raise FileNotFoundError(f"No folder under {root} holds {min_images} or more {pattern or 'images'}")

    done, rows = {}, []
    for n, (name, folder) in enumerate(wafers.items(), start=1):
        dest = out / name
        rows.append({"wafer": name, "folder": workspace.relative(folder), "images": len(io.list_images(folder, pattern))})
        if skip_done and (dest / "results.csv").exists():
            progress(f"=== Wafer {n}/{len(wafers)} {name}: already done, skipped")
        else:
            progress(f"=== Wafer {n}/{len(wafers)} {name} ({folder})")
            t = time.time()
            run_batch(folder, dest, cfg, crops, lambda msg: progress(f"{name} {msg}"), tuning_name, template)
            progress(f"=== {name} took {time.time() - t:.0f} s")
        done[name] = dest
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "wafers.csv", index=False)
    return done


def run_measurement(m, crops=True, skip_done=False, progress=print):
    """Run the batch for one measurement type (measurements.Measurement) with its own config,
    into its output folder."""
    busy = workspace.lock_info(m.output)
    if busy:
        raise RuntimeError(f"{m.key} is being run already ({busy}); delete {m.output / workspace.LOCK_NAME} "
                           "if that run has stopped")
    cfg = m.load_config()
    if m.is_browse(cfg):
        raise ValueError(f"{m.key} has no analysis (browse only)")
    name = tuning.register(cfg, tuning.tunings_dir(m.config), template=m.config)
    progress(f"Tuning: {name}")
    with workspace.Lock(m.output):
        done = run_wafers(m.images, m.output, cfg, crops, skip_done, progress, wafers=m.wafers(cfg),
                          tuning_name=name, template=m.config)
    new = inventory.sync()
    if new:
        progress(f"New in the inventory: {', '.join(new)}")
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path, nargs="?", help="a wafer folder, or a folder of wafer folders")
    ap.add_argument("--project", help=f"a project of the workspace (default: this PC's current one, "
                                      f"{workspace.PROJECT_NAME})")
    ap.add_argument("--measurement", help='a measurement type of the project, e.g. "P2/With mirror"')
    ap.add_argument("--all", action="store_true", help="every measurement type of the project")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--config", type=Path, default=io.DEFAULT_CONFIG)
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="skip wafers whose results.csv already exists")
    args = ap.parse_args(argv)
    if args.project and args.project != workspace.PROJECT_NAME:
        # the project is read at import: run again with it
        if args.project not in workspace.projects():
            ap.error(f"no project {args.project!r} in {workspace.ROOT / workspace.PROJECTS_NAME}")
        rest = list(sys.argv[1:] if argv is None else argv)
        i = next(k for k, a in enumerate(rest) if a.startswith("--project"))
        del rest[i:i + (1 if "=" in rest[i] else 2)]
        sys.exit(subprocess.call([sys.executable, "-m", "spiometrologyviewer.cli", *rest], cwd=workspace.REPO,
                                 env={**os.environ, workspace.PROJECT_ENV: args.project}))
    if args.all or args.measurement:
        todo = ([m for ms in measurements.load().values() for m in ms if not m.is_browse()] if args.all
                else [measurements.find(args.measurement)])
        for m in todo:
            print(f"##### {m.key} -> {m.output}")
            run_measurement(m, crops=not args.no_crops, skip_done=args.skip_done)
    elif args.images:
        cfg = io.load_config(args.config)
        name = tuning.register(cfg, tuning.tunings_dir(args.config), template=args.config)
        print(f"Tuning: {name}")
        run_wafers(args.images, args.out, cfg, crops=not args.no_crops, skip_done=args.skip_done, tuning_name=name)
    else:
        ap.error("give an image folder, --measurement or --all")


if __name__ == "__main__":
    main()
