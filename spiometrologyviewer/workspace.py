"""The workspace and the current project.

The workspace is the shared program folder (e.g. 8 Software/24 SPIOMetrologyViewer on Nextcloud). It
holds projects.yaml, the list of projects. The code (this repo) can then be updated from GitHub
without touching any data.

A project (e.g. 17013_2) has its own folder, normally inside the project's metrology folder on
Nextcloud (05 Metrology/17013_2/SPIOMetrologyViewer): project.yaml (design: plates, units, lens),
measurements.yaml, configs, tolerances, tunings, position map, inventory and results. The measurement
files are read where the microscopes saved them, in the metrology folder (see project.py).

The workspace is, in this order:
  1. the environment variable SPIOMETROLOGYVIEWER_WORKSPACE
  2. `workspace:` in settings.yaml next to this package (per PC, not in git; the GUI writes it)
  3. the repo folder itself
The current project is SPIOMETROLOGYVIEWER_PROJECT, else `project:` in settings.yaml (per PC, so two
people can work on different projects), else the first one in projects.yaml. A workspace without
projects.yaml is its own project (how it worked before projects).

Paths in projects.yaml and in results (image_path, folder) are relative to the workspace, e.g.
../../1 Projects/.../17013_2/P2/..., so they work on every PC with the same Nextcloud folder tree.
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
TEMPLATE = REPO / "project_template"
ENV = "SPIOMETROLOGYVIEWER_WORKSPACE"
PROJECT_ENV = "SPIOMETROLOGYVIEWER_PROJECT"
PROJECTS_NAME = "projects.yaml"
PROJECT_FOLDER = "SPIOMetrologyViewer"   # a new project's folder inside its metrology folder
LOCK_NAME = "RUNNING.txt"
LOCK_HOURS = 12


def _settings():
    try:
        s = yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return s if isinstance(s, dict) else {}


def _configured():
    """The workspace folder from the environment or settings.yaml, or None."""
    if os.environ.get(ENV):
        return Path(os.environ[ENV]).expanduser()
    ws = _settings().get("workspace")
    return Path(ws).expanduser() if ws else None


def is_workspace(folder):
    return (Path(folder) / PROJECTS_NAME).is_file() or (Path(folder) / "measurements.yaml").is_file()


CONFIGURED = _configured()
ROOT = (CONFIGURED or REPO).resolve()


def reachable():
    """False if a workspace is configured but its folder isn't there (e.g. Nextcloud not connected).
    os.path.isdir, unlike Path.is_dir on Python < 3.14, also gives False on a network mount's I/O error."""
    return CONFIGURED is None or os.path.isdir(CONFIGURED)


def save_settings(folder=None, project=None):
    """Remember this PC's workspace and / or current project (settings.yaml); takes effect on the next
    start. What isn't given is kept."""
    s = _settings()
    if folder is not None:
        s["workspace"] = Path(folder).resolve().as_posix()
    if project is not None:
        s["project"] = project
    text = "# This PC's workspace folder and current project (see spiometrologyviewer/workspace.py). Not in git.\n"
    text += yaml.safe_dump(s, sort_keys=False, allow_unicode=True)   # quotes 17013_2 (YAML would read 170132)
    SETTINGS.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ projects

def projects(root=ROOT):
    """{name: project folder} from projects.yaml, in file order."""
    try:
        data = yaml.safe_load((Path(root) / PROJECTS_NAME).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return {str(name): resolve(folder, root) for name, folder in (data.get("projects") or {}).items()}


def register_project(name, folder, root=ROOT):
    """Add (or move) project `name` in projects.yaml, its folder stored relative to the workspace."""
    path = Path(root) / PROJECTS_NAME
    found = {n: relative(f, root) for n, f in projects(root).items()}
    found[name] = relative(folder, root)
    path.write_text("# Projects of this workspace: name: project folder (relative to this folder). The GUI adds\n"
                    "# them with 'New project...'; each PC picks its current one (settings.yaml).\n"
                    + yaml.safe_dump({"projects": found}, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _current():
    found = projects()
    if not found:
        return None, ROOT
    name = os.environ.get(PROJECT_ENV) or _settings().get("project")
    if name not in found:
        name = next(iter(found))
    return name, found[name]



def create(folder):
    """An empty workspace in `folder`: a projects.yaml without projects (the GUI then asks for the first)."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / PROJECTS_NAME).write_text("projects: {}\n", encoding="utf-8")


def create_project(folder, name, metrology):
    """A new project in `folder` (made if needed): the template's measurements.yaml, configs,
    tolerances and position map, and its project.yaml with `name` and the metrology folder (relative
    to `folder`). Fill in the design values in project.yaml afterwards (project.save_design)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE, folder, dirs_exist_ok=True)
    p = folder / "project.yaml"
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    text = text.replace(f"name: {data.get('name')}", f"name: {_quoted(name)}", 1)
    text = text.replace(f"measurements: {data.get('measurements')}",
                        f"measurements: {relative(metrology, folder)}", 1)
    p.write_text(text, encoding="utf-8")
    return folder


def _quoted(text):
    return yaml.safe_dump(str(text), default_style='"', width=10000).splitlines()[0]


def file(name):
    """A file of the current project, or its default in project_template if the project has none."""
    p = PROJECT / name
    return p if p.exists() or not (TEMPLATE / name).exists() else TEMPLATE / name


def relative(path, root=None):
    """`path` relative to `root` (default the workspace), posix style, with ".." where needed; absolute
    only if it can't be relative (another drive on Windows)."""
    p = Path(path).resolve()
    try:
        return Path(os.path.relpath(p, Path(root or ROOT).resolve())).as_posix()
    except ValueError:
        return str(p)


def resolve(path, root=None):
    """A stored path: relative ones are taken from `root` (default the workspace). Backslashes (a path
    written on Windows) are read as separators, so it also resolves on Linux. ".." parts are kept, so
    a path through a symlink isn't changed."""
    p = Path(str(path).replace("\\", "/"))
    return p if p.is_absolute() else Path(os.path.normpath(Path(root or ROOT) / p))


PROJECT_NAME, PROJECT = _current()


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
