import fnmatch
import functools
import json
import re
from pathlib import Path

import cv2
import numpy as np
import tifffile
import yaml

from . import workspace

IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".lext"}
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
    """Load an image (TIF, JPG, PNG; for a LEXT scan its laser intensity image) as uint8 RGB (H, W, 3)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".lext":
        im = load_lext(path)["intensity"]
    elif suffix in (".tif", ".tiff"):
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


# ---------------------------------------------------------------- LEXT 3D scans (Evident / Olympus OLS)
# A .lext file is a TIFF: page 0 laser intensity (uint16; its ImageDescription is an XML with the
# scale), 1 thumbnail, 2 "HEIGHT" (uint16 counts), 3 "INVALID" (1 bit per pixel). The scale is
# HeightDataPerPixelX/Y in pm per pixel and HeightDataPerPixelZ in pm per height count. It is the
# instrument's own scale: our calibration factors are applied on top (read_lext_calibration).

def _lext_scale(xml, key):
    m = re.search(rf"<{key}>([^<]+)</{key}>", xml)
    if not m:
        raise OSError(f"No {key} in the LEXT file's description")
    return float(m.group(1))


@functools.lru_cache(maxsize=1)
def _load_lext(path, mtime):
    with tifffile.TiffFile(path) as t:
        pages = {str(p.description).strip(): p for p in t.pages[1:]}
        if "HEIGHT" not in pages:
            raise OSError(f"{path}: no HEIGHT image, not a LEXT 3D scan")
        xml = str(t.pages[0].description)
        intensity = t.pages[0].asarray()
        height = pages["HEIGHT"].asarray()
        invalid = np.zeros(height.shape, bool)
        p = pages.get("INVALID")
        if p is not None:
            # 1 bit per pixel without the usual padding at the end of each row, which tifffile can't
            # unpack, so the bits are read directly
            t.filehandle.seek(p.dataoffsets[0])
            raw = np.frombuffer(t.filehandle.read(sum(p.databytecounts)), np.uint8)
            n = p.shape[0] * p.shape[1]
            if raw.size * 8 >= n:
                invalid = np.unpackbits(raw)[:n].reshape(p.shape[:2]).astype(bool)
    return {
        "intensity": intensity,
        "height_um": height.astype(np.float32) * np.float32(_lext_scale(xml, "HeightDataPerPixelZ") * 1e-6),
        "invalid": invalid,
        "um_per_px": (_lext_scale(xml, "HeightDataPerPixelX") * 1e-6, _lext_scale(xml, "HeightDataPerPixelY") * 1e-6),
        "nm_per_count": _lext_scale(xml, "HeightDataPerPixelZ") * 1e-3,
    }


def load_lext(path):
    """A LEXT 3D scan: {intensity (uint16), height_um (float32, the instrument's scale), invalid (bool),
    um_per_px (x, y), nm_per_count}. The last file read is kept in memory (the tuning GUI re-runs the
    analysis on the same file when a parameter changes)."""
    path = Path(path)
    return _load_lext(str(path), path.stat().st_mtime)


CALIBRATION_KEYS = {"X Calibration": "x", "Y Calibration": "y", "Z Calibration": "z",
                    "X user calibration": "x_user", "Y user calibration": "y_user", "Z user calibration": "z_user",
                    "X resolution (Xcv)": "x_res", "Y resolution (Ycv)": "y_res", "Z resolution (Zcv)": "z_res"}


def read_lext_calibration(path):
    """Calibration factors from the header of a LEXT CSV export (e.g. "X Calibration  = ,100.697519,").
    Only the header is read: a full height export is often over 1 GB. Returns {"x_pct", "y_pct",
    "z_pct" (calibration x user calibration), "x_res", "y_res", "z_res" (the export's own resolution,
    um), "source"}, or None if the file has no calibration lines."""
    vals = {}
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for _ in range(200):
            line = f.readline()
            if not line or line.startswith("DataLine"):
                break
            key, _, rest = line.partition("=")
            name = CALIBRATION_KEYS.get(" ".join(key.split()))
            if name:
                try:
                    vals[name] = float(rest.strip().strip(",").split(",")[0])
                except ValueError:
                    pass
    if not all(k in vals for k in ("x", "y", "z")):
        return None
    out = {f"{a}_pct": vals[a] * vals.get(f"{a}_user", 100.0) / 100.0 for a in "xyz"}
    out.update({f"{a}_res": vals.get(f"{a}_res", np.nan) for a in "xyz"})
    out["source"] = Path(path).name
    return out


def find_lext_calibration(folder):
    """The calibration CSV of a wafer folder (one LEXT CSV export with "X Calibration" in its header,
    see read_lext_calibration), or None. With several, the newest is used and the others are listed
    under "ignored"."""
    folder = Path(folder)
    csvs = [(p.stat().st_mtime, p.name) for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    return _find_lext_calibration(str(folder), tuple(sorted(csvs)))


@functools.lru_cache(maxsize=32)
def _find_lext_calibration(folder, csvs):
    found = []
    for mtime, name in csvs:
        try:
            cal = read_lext_calibration(Path(folder) / name)
        except OSError:
            continue
        if cal is not None:
            found.append((mtime, cal))
    if not found:
        return None
    found.sort(key=lambda t: t[0])
    cal = dict(found[-1][1])
    cal["ignored"] = [c["source"] for _, c in found[:-1]]
    return cal


@functools.lru_cache(maxsize=8)
def lext_stitching(folder):
    """Stitching layout of a wafer folder's scans from the microscope's matl.omp2info (in the
    timestamped subfolders next to the .poir files): {"tile_um": (w, h), "overlap_pct", "tiles":
    (nx, ny), "source"}, or None. All scans of a wafer use the same recipe, so the first file found
    is read (reading all of them over a network drive takes long)."""
    for sub in sorted(Path(folder).iterdir()):
        info = sub / "matl.omp2info"
        if not sub.is_dir() or not info.is_file():
            continue
        try:
            xml = info.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        get = lambda key: re.search(rf"<matl:{key}>([^<]+)</matl:{key}>", xml)
        w, h, ov, nx, ny = (get(k) for k in ("areaWidth", "areaHeight", "overlap", "numOfXAreas", "numOfYAreas"))
        if w and h:
            return {"tile_um": (float(w.group(1)) * 1e-3, float(h.group(1)) * 1e-3),   # nm -> um
                    "overlap_pct": float(ov.group(1)) if ov else 0.0,
                    "tiles": (int(nx.group(1)) if nx else 0, int(ny.group(1)) if ny else 0),
                    "source": f"{sub.name}/{info.name}"}
    return None


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
