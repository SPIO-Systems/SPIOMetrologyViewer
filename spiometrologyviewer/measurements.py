"""Plates and their measurement types, from the project's measurements.yaml.

A measurement type (e.g. P2 / With mirror) has its own images, pipeline parameters, tolerances
and results; the GUI shows one tab per plate and, inside it, one per type. Image folders are
relative to the project's metrology folder, the rest to the project folder (see project.py).

Wafers the project lists as legacy (scanned before the current folder layout) are found as usual
and can be browsed, but a type only analyses them with `legacy: only` (and then nothing else).
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import io, project, workspace

ROOT = workspace.PROJECT
DEFAULT_REGISTRY = ROOT / "measurements.yaml"
BROWSE = "Browse"   # the type a plate without measurement types gets
LEGACY_LABEL = " (legacy, not analysed)"


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
    recursive: bool = False   # images anywhere below the wafer folder (P5: one per timestamp subfolder)
    legacy: str = "skip"      # legacy wafers: "skip" (browse only) or "only" (analyse only them)
    project: object = field(default=None, repr=False)

    @property
    def key(self):
        return f"{self.plate}/{self.name}"

    def load_config(self):
        """The config with the project's design values filled in (see project.Project.overlay)."""
        cfg = io.load_config(self.config)
        if self.project is None:
            return cfg
        from . import analyses
        return self.project.overlay(cfg, analyses.name_of(cfg))

    def is_browse(self, cfg=None):
        return (cfg or self.load_config()).get("analysis") == "none"

    def is_legacy(self, wafer):
        return self.project is not None and self.project.is_legacy(self.plate, wafer)

    def images_of(self, folder, cfg=None):
        """The images of one wafer folder."""
        cfg = cfg or self.load_config()
        pattern = cfg.get("input", {}).get("image_glob")
        return io.list_images_below(folder, pattern) if self.recursive else io.list_images(folder, pattern)

    def found(self, cfg=None):
        """{wafer name: image folder} of every wafer with images, legacy or not."""
        cfg = cfg or self.load_config()
        inp = cfg.get("input", {})
        pattern, min_images = inp.get("image_glob"), inp.get("min_images", 10)
        # Folder names are matched ignoring case, as on Windows (P3/*/Grading finds .../grading)
        if "*" not in self.images:
            root = self.base / self.images
            if not root.is_dir() and not Path(self.images).is_absolute():
                root = next(iter(sorted(self.base.glob(self.images, case_sensitive=False))), root)
            return io.find_wafers(root, pattern, min_images) if root.is_dir() else {}
        # e.g. P3/*/Mirror: every match is one wafer, named after the "*" part
        parts = Path(self.images).parts
        i = next(k for k, p in enumerate(parts) if "*" in p)
        found = {}
        for folder in sorted(self.base.glob(self.images, case_sensitive=False)):
            # recursive: every matching folder is a wafer, without counting its images (P5: one subfolder
            # per image, listing them all takes minutes on Nextcloud)
            if folder.is_dir() and (self.recursive or len(self.images_of(folder, cfg)) >= min_images):
                found.setdefault(folder.relative_to(self.base).parts[i], folder)
        return found

    def wafers(self, cfg=None):
        """{wafer name: image folder} of the wafers this type analyses."""
        only = self.legacy == "only"
        return {w: f for w, f in self.found(cfg).items() if self.is_legacy(w) == only}

    def browsable(self, cfg=None, found=None):
        """{label: image folder} of every wafer found (or of `found`), the analysed ones first, the
        others (legacy wafers in a "skip" type, the rest in an "only" type) labelled LEGACY_LABEL."""
        only = self.legacy == "only"
        found = self.found(cfg) if found is None else found
        return {**{w: f for w, f in found.items() if self.is_legacy(w) == only},
                **{w + LEGACY_LABEL: f for w, f in found.items() if self.is_legacy(w) != only}}

    def first_image(self, cfg=None, folders=None):
        """The first image of the first wafer (of `folders`, default browsable()) that has one."""
        cfg = cfg or self.load_config()
        for folder in (folders if folders is not None else self.browsable(cfg)).values():
            images = self.images_of(folder, cfg)
            if images:
                return images[0]
        return None


def load(path=DEFAULT_REGISTRY):
    """{plate: [Measurement, ...]}: the plates of project.yaml in its order, a plate without types
    getting a Browse type (without a plates list in project.yaml: those of measurements.yaml). Types
    with `legacy: only` are left out if the project lists no legacy wafers for their plate."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    folder = path.parent
    proj = project.load(folder)
    base = proj.metrology
    plates = {}
    for plate, types in data.items():
        plate = str(plate)
        if proj.plates and plate not in proj.plates:
            continue
        plates[plate] = [
            Measurement(plate, str(name), str(m["images"]), folder / m["config"], folder / m["tolerances"],
                        folder / m["output"], folder / m.get("positions", proj.positions.name), base,
                        bool(m.get("recursive", False)), str(m.get("legacy", "skip")), proj)
            for name, m in (types or {}).items()
            if str(m.get("legacy", "skip")) != "only" or proj.legacy.get(plate)
        ]
    for plate in proj.plates:
        if not plates.get(plate):
            plates[plate] = [Measurement(plate, BROWSE, f"{plate}/*", workspace.TEMPLATE / "configs" / "browse.yaml",
                                         folder / "tolerances" / f"{plate}_browse.yaml",
                                         folder / "output" / plate / BROWSE, proj.positions, base, True, "skip", proj)]
    order = {p: i for i, p in enumerate(proj.plates)}
    return dict(sorted(plates.items(), key=lambda kv: order.get(kv[0], 0)))


def find(key, path=DEFAULT_REGISTRY):
    """The measurement called "plate/type", e.g. "P2/With mirror"."""
    for ms in load(path).values():
        for m in ms:
            if m.key.lower() == key.lower():
                return m
    raise KeyError(f"No measurement {key!r} in {path}")
