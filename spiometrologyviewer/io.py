import fnmatch
import json
import re
from pathlib import Path

import cv2
import numpy as np
import tifffile
import yaml

from . import workspace

IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
DEFAULT_CONFIG = workspace.file("config.yaml")


def list_images(folder, pattern=None):
    """Image files directly in `folder`; with `pattern` (e.g. "*COLOR2D.tif") only matching names."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                  and (pattern is None or fnmatch.fnmatch(p.name.lower(), pattern.lower())))


def find_wafers(root, pattern=None, min_images=10):
    """{name: folder} for every folder under `root` (including itself) that directly holds at least
    `min_images` images matching `pattern`. The name is the folder name."""
    wafers = {}
    for folder in [Path(root), *sorted(p for p in Path(root).rglob("*") if p.is_dir())]:
        if len(list_images(folder, pattern)) >= min_images:
            wafers[folder.name] = folder
    return wafers


def load_image(path):
    """Load an image (TIF, JPG, PNG) as uint8 RGB (H, W, 3)."""
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        im = tifffile.imread(path)
    else:
        bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise OSError(f"Can't read image {path}")
        im = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB)
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    im = im[..., :3]
    if im.dtype != np.uint8:
        im = (255.0 * im / max(im.max(), 1)).astype(np.uint8)
    return im


load_tif = load_image   # older name


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _yaml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_yaml_value(x) for x in v) + "]"
    if isinstance(v, float):
        return repr(round(v, 6))
    if isinstance(v, str):
        return json.dumps(v)  # a double-quoted string is valid YAML, and safe for "*", ":" etc.
    return str(v)


_LINE = re.compile(r"^(\s*)([A-Za-z_]\w*):(\s*)([^#]*?)(\s*#.*)?$")


def save_config(path, cfg, template=DEFAULT_CONFIG):
    """Write `cfg` as YAML, keeping the layout and comments of `template` (same keys).

    Falls back to a plain dump if the template is missing or doesn't round-trip."""
    try:
        lines = Path(template).read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = None
    if lines:
        out, stack = [], []  # stack of (indent, key) for the current nesting
        for line in lines:
            m = _LINE.match(line)
            if not m:
                out.append(line)
                continue
            indent, key, gap, old, comment = len(m[1]), m[2], m[3], m[4], m[5] or ""
            stack = [(i, k) for i, k in stack if i < indent] + [(indent, key)]
            if not old:  # section header
                out.append(line)
                continue
            v = cfg
            try:
                for _, k in stack:
                    v = v[k]
            except (KeyError, TypeError, IndexError):   # the template has keys cfg doesn't
                lines = None
                break
            new = _yaml_value(v)
            out.append(f"{m[1]}{key}:{gap or ' '}{new.ljust(len(old)) if comment else new}{comment}")
        text = "\n".join(out) + "\n"
        if lines is not None and yaml.safe_load(text) == cfg:
            Path(path).write_text(text, encoding="utf-8")
            return
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
