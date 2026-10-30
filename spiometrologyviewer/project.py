"""A project's design, from project.yaml in the project folder (see workspace.py):

    name, description
    measurements   the metrology folder (one subfolder per plate), relative to the project folder
    plates         plate tabs in order; a plate without measurement types gets a Browse tab
    wafer          units per wafer and the device position map
    lens, mirror, aperture, slit, p5_mirror   design values, filled into the configs of the analyses that use them (DESIGN)
    legacy         per plate, wafers scanned before the current folder layout (see measurements.py)

A project folder without project.yaml (a workspace from before projects) is read with defaults: the
metrology folder is the project folder, and the plates are those in measurements.yaml.
"""
import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import workspace

FILE = "project.yaml"

# Design value in project.yaml -> {analysis: config path}. These win over the config, and are shown
# locked in the step-by-step tuning.
DESIGN = {
    ("lens", "nominal_roc_um"): {"lens3d": ("nominal_roc_um",)},
    ("mirror", "size_um"): {"mirror": ("mirror", "size_um")},
    ("aperture", "size_um"): {"p1_aperture": ("opening", "size_um")},
    ("slit", "size_um"): {"p1_slit": ("opening", "size_um")},
    ("p5_mirror", "size_um"): {"p5_mirror": ("mirror", "size_um")},
}


@dataclass
class Project:
    folder: Path
    name: str
    description: str = ""
    metrology: Path = None
    plates: list = field(default_factory=list)
    units: int = None
    positions: Path = None
    design: dict = field(default_factory=dict)   # {"lens": {...}, "mirror": {...}}
    legacy: dict = field(default_factory=dict)   # {plate: {wafer, ...}}

    def is_legacy(self, plate, wafer):
        return wafer in self.legacy.get(plate, ())

    def design_paths(self, analysis):
        """{config path: value} that project.yaml sets for `analysis`."""
        found = {}
        for (section, key), targets in DESIGN.items():
            value = self.design.get(section, {}).get(key)
            if value is not None and analysis in targets:
                found[targets[analysis]] = value
        return found

    def overlay(self, cfg, analysis):
        """`cfg` with the project's design values filled in (a copy)."""
        cfg = copy.deepcopy(cfg)
        for path, value in self.design_paths(analysis).items():
            d = cfg
            for k in path[:-1]:
                d = d.setdefault(k, {})
            d[path[-1]] = copy.deepcopy(value)
        return cfg


def load(folder=None):
    folder = Path(folder or workspace.PROJECT)
    try:
        data = yaml.safe_load((folder / FILE).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        data = {}
    w = data.get("wafer") or {}
    return Project(
        folder=folder,
        name=str(data.get("name") or workspace.PROJECT_NAME or folder.name),
        description=str(data.get("description") or ""),
        metrology=workspace.resolve(data.get("measurements") or ".", folder),
        plates=[str(p) for p in data.get("plates") or []],
        units=w.get("units"),
        positions=folder / w.get("positions", "DevicePositionMap.xlsx"),
        design={k: dict(data.get(k) or {}) for k in {s for s, _ in DESIGN}},
        legacy={str(p): {str(n) for n in names or []} for p, names in (data.get("legacy") or {}).items()},
    )


def _inline(value, **kw):
    """`value` as YAML on one line (without the "..." end marker PyYAML adds after a lone scalar)."""
    return yaml.safe_dump(value, width=10000, **kw).splitlines()[0]


def save_design(folder, description="", plates=None, units=None, design=None):
    """Write the design values of a new project into its project.yaml (made by workspace.create_project),
    keeping the template's comments where a line is replaced."""
    p = Path(folder) / FILE
    lines = p.read_text(encoding="utf-8").splitlines()

    def put(prefix, value, section=None):
        start = lines.index(f"{section}:") if section else 0   # keys like size_um are in several sections
        for i, line in enumerate(lines):
            if i >= start and line.startswith(prefix):
                comment = line[line.index("#"):] if "#" in line else ""
                lines[i] = f"{prefix} {value}".ljust(33) + comment if comment else f"{prefix} {value}"
                return

    put("description:", _inline(description, default_style='"') if description else '""')
    if plates is not None:
        put("plates:", "[" + ", ".join(plates) + "]")
    if units is not None:
        put("  units:", units)
    for (section, key), _ in DESIGN.items():
        value = (design or {}).get(section, {}).get(key)
        if value is not None and f"{section}:" in lines:
            put(f"  {key}:", _inline(value, default_flow_style=True), section)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
