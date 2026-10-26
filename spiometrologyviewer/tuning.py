"""Tunings: every set of pipeline parameters that results were made with, kept as a timestamped file.

A tuning is identified by its parameter values: running or saving parameters that match an existing
tuning reuses its name, anything else creates tunings/<config name>/<config name>_<YYYYMMDD_HHMMSS>_<user>.yaml.
The batch writes the name into results.csv (column `tuning`), so displayed data always says which
tuning it came from. The `output` section (crop / thumbnail sizes) doesn't change any number, so it
is not part of the comparison.
"""
import copy
import re
import time
from pathlib import Path

import yaml

from . import io, workspace

ROOT = workspace.ROOT
TUNINGS = ROOT / "tunings"
UNKNOWN = "unknown"
_IGNORED = ("output",)


def tunings_dir(config_path):
    """Folder for the tunings of one parameter file, e.g. configs/P2_with_mirror.yaml ->
    tunings/P2_with_mirror/."""
    return TUNINGS / Path(config_path).stem


def _key(cfg):
    """The parts of a config that affect the results, normalised (60 == 60.0, tuples == lists)."""
    cfg = {k: v for k, v in cfg.items() if k not in _IGNORED}
    return yaml.safe_load(yaml.safe_dump(cfg, sort_keys=True))


_cache = {}   # tuning file -> (mtime, key)


def load_all(folder):
    """{name: key} of the tunings in `folder`, oldest first."""
    found = {}
    for p in sorted(Path(folder).glob("*.yaml")):
        try:
            mtime = p.stat().st_mtime
            if _cache.get(p, (None,))[0] != mtime:
                _cache[p] = (mtime, _key(io.load_config(p)))
            found[p.stem] = _cache[p][1]
        except (OSError, yaml.YAMLError, AttributeError):
            continue
    return found


def find(cfg, folder):
    """Name of the newest tuning in `folder` with the same parameters as `cfg`, or None."""
    key = _key(cfg)
    names = [n for n, k in load_all(folder).items() if k == key]
    return names[-1] if names else None


def register(cfg, folder, template=io.DEFAULT_CONFIG, when=None):
    """Name of the tuning for `cfg`: an existing one with the same parameters, or a new file
    named after the folder, the time (`when`, seconds since the epoch, default now) and the user."""
    folder = Path(folder)
    name = find(cfg, folder)
    if name is not None:
        return name
    folder.mkdir(parents=True, exist_ok=True)
    when = time.time() if when is None else when
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(when))
    who = re.sub(r"[^A-Za-z0-9]+", "", workspace.user()) or "unknown"
    name = f"{folder.name}_{stamp}_{who}"
    n = 2
    while (folder / f"{name}.yaml").exists():   # same second, different parameters
        name = f"{folder.name}_{stamp}_{who}_{n}"
        n += 1
    path = folder / f"{name}.yaml"
    io.save_config(path, copy.deepcopy(cfg), template=template)
    created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when))
    path.write_text(f"# Tuning {name}, created {created} by {workspace.user()}. "
                    f"Results made with it say tuning: {name}\n\n"
                    + path.read_text(encoding="utf-8"), encoding="utf-8")
    return name


def of_results(out_dir, results=None, folder=None):
    """The tuning a batch output folder was made with: the `tuning` column of its results.csv, or
    (older runs) its config_used.yaml matched against the tunings in `folder`. UNKNOWN otherwise.
    Several names joined with ' + ' if the column holds more than one."""
    out_dir = Path(out_dir)
    if results is not None and "tuning" in results:
        names = [str(n) for n in results["tuning"].dropna().unique()]
        if names:
            return " + ".join(sorted(names))
    used = out_dir / "config_used.yaml"
    if folder is not None and used.exists():
        try:
            return find(io.load_config(used), folder) or UNKNOWN
        except (OSError, yaml.YAMLError):
            pass
    return UNKNOWN
