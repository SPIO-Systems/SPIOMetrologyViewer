"""Plates and their measurement types, from measurements.yaml.

A measurement type (e.g. P2 / With mirror) has its own images, pipeline parameters, tolerances
and results; the GUI shows one tab per plate and, inside it, one per type.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import io, workspace

ROOT = workspace.ROOT
DEFAULT_REGISTRY = ROOT / "measurements.yaml"


@dataclass
class Measurement:
    plate: str
    name: str
    images: str        # folder of wafer folders, or a pattern with "*" for the wafer folder
    config: Path
    tolerances: Path
    output: Path
    positions: Path
    base: Path = ROOT  # folder `images` is relative to

    @property
    def key(self):
        return f"{self.plate}/{self.name}"

    def load_config(self):
        return io.load_config(self.config)

    def wafers(self, cfg=None):
        """{wafer name: image folder}, using the config's image pattern and minimum count."""
        cfg = cfg or self.load_config()
        inp = cfg.get("input", {})
        pattern, min_images = inp.get("image_glob"), inp.get("min_images", 10)
        if "*" not in self.images:
            root = self.base / self.images
            return io.find_wafers(root, pattern, min_images) if root.is_dir() else {}
        # e.g. Images/P3/*/Mirror: every match is one wafer, named after the "*" part
        parts = Path(self.images).parts
        i = next(k for k, p in enumerate(parts) if "*" in p)
        found = {}
        for folder in sorted(self.base.glob(self.images)):
            if folder.is_dir() and len(io.list_images(folder, pattern)) >= min_images:
                found[folder.relative_to(self.base).parts[i]] = folder
        return found

    def first_image(self, cfg=None):
        cfg = cfg or self.load_config()
        for folder in self.wafers(cfg).values():
            images = io.list_images(folder, cfg.get("input", {}).get("image_glob"))
            if images:
                return images[0]
        return None


def load(path=DEFAULT_REGISTRY):
    """{plate: [Measurement, ...]} in file order."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent
    plates = {}
    for plate, types in data.items():
        plates[plate] = [
            Measurement(str(plate), str(name), str(m["images"]), base / m["config"], base / m["tolerances"],
                        base / m["output"], base / m.get("positions", "DevicePositionMap.xlsx"), base)
            for name, m in (types or {}).items()
        ]
    return plates


def find(key, path=DEFAULT_REGISTRY):
    """The measurement called "plate/type", e.g. "P2/With mirror"."""
    for ms in load(path).values():
        for m in ms:
            if m.key.lower() == key.lower():
                return m
    raise KeyError(f"No measurement {key!r} in {path}")
