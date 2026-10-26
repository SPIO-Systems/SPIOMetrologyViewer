"""The workspace: the shared folder (e.g. on Nextcloud) holding everything the program reads and
writes besides its code: measurements.yaml, configs, tolerances, tunings, position maps, images and
results. The code (this repo) can then be updated from GitHub without touching the data.

The workspace is, in this order:
  1. the environment variable SPIOMETROLOGYVIEWER_WORKSPACE
  2. `workspace:` in settings.yaml next to this package (per PC, not in git; the GUI writes it)
  3. the repo folder itself (how it worked before the split)

Paths written into results (image_path, folder) are relative to the workspace, so results made on
one PC open on another, wherever its Nextcloud folder is.
"""
import getpass
import os
import shutil
import socket
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SETTINGS = REPO / "settings.yaml"
TEMPLATE = REPO / "workspace_template"
ENV = "SPIOMETROLOGYVIEWER_WORKSPACE"
LOCK_NAME = "RUNNING.txt"
LOCK_HOURS = 12


def _configured():
    """The workspace folder from the environment or settings.yaml, or None."""
    if os.environ.get(ENV):
        return Path(os.environ[ENV]).expanduser()
    try:
        ws = (yaml.safe_load(SETTINGS.read_text(encoding="utf-8")) or {}).get("workspace")
    except (OSError, yaml.YAMLError, AttributeError):
        return None
    return Path(ws).expanduser() if ws else None


def is_workspace(folder):
    return (Path(folder) / "measurements.yaml").is_file()


ROOT = (_configured() or REPO).resolve()


def save_settings(folder):
    """Remember `folder` as this PC's workspace (settings.yaml); takes effect on the next start."""
    SETTINGS.write_text("# This PC's workspace folder (see spiometrologyviewer/workspace.py). Not in git.\n"
                        f"workspace: {Path(folder).resolve().as_posix()}\n", encoding="utf-8")


def create(folder):
    """Copy the default measurements.yaml, configs, tolerances and position maps into an empty
    `folder`, making it a workspace (images and results are added later)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE, folder, dirs_exist_ok=True)


def file(name):
    """A file of the workspace, or its default in workspace_template if the workspace has none."""
    p = ROOT / name
    return p if p.exists() or not (TEMPLATE / name).exists() else TEMPLATE / name


def relative(path):
    """`path` relative to the workspace (posix style) if it lies inside it, else absolute."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def resolve(path):
    """A path from results.csv: relative ones are taken from the workspace."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def user():
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ batch lock
# Nextcloud has no file locking: two people running the same measurement at once would overwrite each
# other's results and leave conflict copies. A batch leaves RUNNING.txt in its output folder meanwhile.

def lock_info(out):
    """Text of a fresh RUNNING.txt in `out` (someone is running a batch there), else None."""
    p = Path(out) / LOCK_NAME
    try:
        if time.time() - p.stat().st_mtime < LOCK_HOURS * 3600:
            return p.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return None


class Lock:
    """with Lock(out): ... writes out/RUNNING.txt (who, when) and removes it afterwards."""

    def __init__(self, out):
        self.path = Path(out) / LOCK_NAME

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"Batch run by {user()} on {socket.gethostname()}, "
                             f"started {time.strftime('%Y-%m-%d %H:%M')}\n", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass
