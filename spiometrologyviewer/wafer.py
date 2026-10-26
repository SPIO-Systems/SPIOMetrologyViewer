"""Devices on the wafer: positions, batch results per device and pass / warn / fail tolerances."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import io, tuning, workspace

ROOT = workspace.ROOT
DEFAULT_POSITIONS = workspace.file("DevicePositionMap.xlsx")
DEFAULT_TOLERANCES = ROOT / "tolerances.yaml"

SEVERITY = {"nodata": 0, "pass": 1, "warn": 2, "fail": 3}


@dataclass
class Metric:
    """One results.csv column with its tolerances: pass if value <= pass_max, fail if > fail_min,
    warn in between. With `abs`, |value| is compared (offsets, rotation)."""
    column: str
    label: str
    pass_max: float
    fail_min: float
    step: float = 1.0      # spin box increment in the GUI
    abs: bool = False
    enabled: bool = True

    def values(self, df):
        v = pd.to_numeric(df[self.column], errors="coerce") if self.column in df else pd.Series(np.nan, df.index)
        return v.abs() if self.abs else v

    def value(self, row):
        try:
            v = float(row.get(self.column, np.nan))
        except (TypeError, ValueError):
            return np.nan
        return abs(v) if self.abs else v

    def status(self, v):
        if v is None or not np.isfinite(v):
            return "nodata"
        if v <= self.pass_max:
            return "pass"
        return "fail" if v > self.fail_min else "warn"


DEFAULT_METRICS = [
    Metric("dirt_coverage_inner_pct", "Dirt coverage inner [%]", 0.5, 2, 0.05),
    Metric("dirt_coverage_edge_pct", "Dirt coverage edge [%]", 1, 5, 0.1),
    Metric("particle_count", "Particle count", 20, 50, 1),
    Metric("particle_max_diam_um", "Largest particle [um]", 35, 50, 1),
    Metric("fibre_max_length_um", "Longest fibre [um]", 50, 100, 5),
    Metric("residue_area_pct", "Residue area [%]", 0.25, 1, 0.05),
    Metric("edge_chip_max_depth_um", "Deepest edge chip [um]", 25, 50, 1),
    Metric("offset_x_um", "|Offset x| [um]", 50, 100, 5, abs=True),
    Metric("offset_y_um", "|Offset y| [um]", 50, 100, 5, abs=True),
    Metric("offset_um", "Total offset [um]", 75, 125, 5),
    Metric("mirror_angle_deg", "|Mirror rotation| [deg]", 0.5, 1, 0.05, abs=True),
]
# Only measured when there is a mirror (config region "mirror")
MIRROR_ONLY = {"edge_chip_max_depth_um", "offset_x_um", "offset_y_um", "offset_um", "mirror_angle_deg"}


def default_metrics(analysis="mirror"):
    """Default tolerance metrics for an analysis type (see analyses.py)."""
    if analysis not in ("mirror", "lens"):
        from . import analyses
        return [replace(m) for m in analyses.default_metrics(analysis)]
    return [replace(m) for m in DEFAULT_METRICS if analysis == "mirror" or m.column not in MIRROR_ONLY]


def device_number(image_name):
    """Last purely numeric '_'-separated part of the image name: 17013_2_P2_001_COLOR2D -> 1."""
    nums = [t for t in Path(str(image_name)).stem.split("_") if t.isdigit()]
    return int(nums[-1]) if nums else None


def load_positions(path=DEFAULT_POSITIONS):
    """{device: (x, y)} from a sheet with columns 'Position', 'X ...', 'Y ...'."""
    df = pd.read_excel(path)
    cols = {str(c).strip().lower().split()[0]: c for c in df.columns}
    df = df[[cols["position"], cols["x"], cols["y"]]].dropna()
    return {int(p): (float(x), float(y)) for p, x, y in df.itertuples(index=False)}


def find_results(root):
    """{wafer name: folder} for every subfolder of `root` with a results.csv (one per wafer, as
    written by cli.run_wafers), plus `root` itself if it holds one."""
    root = Path(root)
    found = {}
    if root.is_dir():
        found = {p.name: p for p in sorted(root.iterdir()) if p.is_dir() and (p / "results.csv").exists()}
    if (root / "results.csv").exists():
        found[f"{root.name} (top level)" if found else root.name] = root
    return found


def load_results(out_dir, positions, tunings=None):
    """Batch results joined to the wafer positions, one row per device.

    Devices on the map without an image get NaN results; images whose device is not on the map
    get NaN x/y. Every image gets a `tuning` (tuning.of_results; `tunings` = the folder of tunings
    to match older runs against). Returns (devices, objects or None)."""
    out_dir = Path(out_dir)
    res = pd.read_csv(out_dir / "results.csv")
    fallback = tuning.of_results(out_dir, None, tunings)
    res["tuning"] = res["tuning"].fillna(fallback) if "tuning" in res else fallback
    res["code_version"] = res["code_version"].astype(str) if "code_version" in res else "?"
    res["device"] = res["image"].map(device_number)
    res = res.dropna(subset=["device"]).drop_duplicates("device", keep="last")
    res["device"] = res["device"].astype(int)
    if not positions:   # no position map (yet): the devices in a row, by number
        positions = {d: (float(d), 0.0) for d in sorted(res["device"].unique())}
    pos = pd.DataFrame([(d, x, y) for d, (x, y) in positions.items()], columns=["device", "x", "y"])
    devices = pos.merge(res, on="device", how="outer").sort_values("device").reset_index(drop=True)
    objects_path = out_dir / "objects.csv"
    objects = pd.read_csv(objects_path, keep_default_na=False, na_values=[""]) if objects_path.exists() else None
    return devices, objects


def overall(row, metrics):
    """Worst status over the enabled metrics (metrics without data are skipped)."""
    worst = "nodata"
    for m in metrics:
        if m.enabled:
            s = m.status(m.value(row))
            if SEVERITY[s] > SEVERITY[worst]:
                worst = s
    return worst


def stats(df, metric):
    v = metric.values(df).dropna()
    counts = {s: 0 for s in SEVERITY}
    for x in v:
        counts[metric.status(x)] += 1
    counts["nodata"] = len(df) - len(v)
    if v.empty:
        return {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, **counts}
    return {"n": len(v), "min": v.min(), "max": v.max(), "mean": v.mean(), "median": v.median(), **counts}


def load_tolerances(path=DEFAULT_TOLERANCES, analysis="mirror"):
    """Default metrics (for the config's analysis type), overridden by the values in `path` (if it exists)."""
    metrics = default_metrics(analysis)
    try:
        saved = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except OSError:
        return metrics
    by_col = {m.column: m for m in metrics}
    for col, vals in saved.items():
        m = by_col.get(col)
        if m is None:  # a column added by hand in the file
            m = Metric(col, vals.get("label", col), 0, 0, abs=bool(vals.get("abs", False)))
            metrics.append(m)
        for k in ("pass_max", "fail_min", "enabled"):
            if k in vals:
                setattr(m, k, vals[k])
    return metrics


_HEADER = """\
# Wafer map tolerances, per results.csv column (edited and saved from the GUI's Wafer map tab).
#   green / pass : value <= pass_max
#   yellow / warn: pass_max < value <= fail_min
#   red / fail   : value > fail_min
# Offsets and rotation are compared as absolute values. enabled: false leaves a metric out of
# the overall (worst-of) colour.
"""


def save_tolerances(metrics, path=DEFAULT_TOLERANCES):
    lines = [_HEADER]
    for m in metrics:
        lines.append(f"{m.column}:  # {m.label}")
        lines.append(f"  pass_max: {m.pass_max}")
        lines.append(f"  fail_min: {m.fail_min}")
        lines.append(f"  enabled: {'true' if m.enabled else 'false'}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- several wafers

OVERALL = "Overall (worst of enabled)"
_NOT_PARAMETERS = {"device", "x", "y", "um_per_px", "out_dir"}


@dataclass
class Param:
    """Something to plot: a tolerance metric (with pass / warn / fail), a plain results column,
    or the overall status (column None, metric None)."""
    label: str
    column: str = None
    metric: Metric = None

    def values(self, df):
        if self.metric is not None:
            return self.metric.values(df)
        if self.column is None or self.column not in df:
            return pd.Series(np.nan, df.index)
        # True / False flags count as 1 / 0, so their mean is the share of devices with it
        return df[self.column].map(lambda v: float(v) if isinstance(v, (bool, np.bool_, int, float, np.integer,
                                                                           np.floating)) else np.nan)

    def statuses(self, df, metrics):
        """pass / warn / fail / nodata per row, or None if this parameter has no tolerances."""
        if self.metric is not None:
            return self.values(df).map(self.metric.status)
        if self.column is None:
            return df.apply(lambda r: overall(r, metrics), axis=1)
        return None


def parameters(df, metrics, with_overall=False):
    """The tolerance metrics first (by label), then every other numeric results column."""
    params = [Param(OVERALL)] if with_overall else []
    params += [Param(m.label, m.column, m) for m in metrics]
    # abs metrics (offsets, rotation) also with their sign, e.g. for an offset x vs offset y plot
    params += [Param(f"{m.column} (signed)", m.column) for m in metrics if m.abs and m.column in df]
    used = {m.column for m in metrics}
    for col in df.columns:
        if col in used or col in _NOT_PARAMETERS or col.endswith(("_px", "_found")):
            continue
        if Param(col, col).values(df).notna().any():
            params.append(Param(col, col))
    return params


def tuning_of(devices):
    """The tuning(s) of loaded results, as one text: "name", or "name1 + name2" if mixed."""
    if devices is None or "tuning" not in devices:
        return tuning.UNKNOWN
    names = sorted(str(n) for n in devices["tuning"].dropna().unique())
    return " + ".join(names) if names else tuning.UNKNOWN


def load_all(wafers, positions, tunings=None, progress=None):
    """Results of several wafers ({name: results folder}) in one table, with a `wafer` and an
    `out_dir` column; devices keep their map position. progress(i, n, name) is called before each."""
    frames = []
    for i, (name, folder) in enumerate(wafers.items(), 1):
        if progress:
            progress(i, len(wafers), name)
        devices, _ = load_results(folder, positions, tunings)
        frames.append(devices.assign(wafer=name, out_dir=str(folder)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summary(df, param, metrics, order=None):
    """One row per wafer: n, mean, median, std, min, max of the parameter, and for parameters with
    tolerances the % of devices that pass / warn / fail / have no data."""
    rows = []
    status = param.statuses(df, metrics)
    values = param.values(df)
    for name in order if order is not None else df["wafer"].unique():
        sel = df["wafer"] == name
        v = values[sel].dropna()
        row = {"wafer": name, "devices": int(sel.sum()), "n": len(v), "tuning": tuning_of(df[sel])}
        if param.column is not None:
            row.update(mean=v.mean(), median=v.median(), std=v.std(), min=v.min(), max=v.max())
        if status is not None:
            s = status[sel]
            if param.column is None:   # overall: devices with any data
                row["n"] = int((s != "nodata").sum())
            for k in SEVERITY:
                row[f"{k}_pct"] = 100.0 * (s == k).mean() if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
