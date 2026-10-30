"""Analysis types: which pipeline steps, tuning-GUI steps and default tolerances a measurement type uses.

The config key `analysis` picks one; P2 configs without it fall back to `region` ("mirror" / "lens").

    mirror      P2 with mirror: mirror rectangle, dirt, edge chips, offset to the lens
    lens        P2 without mirror: dirt on the lens surface
    p3_mirror   P3 mirror: small particles and large defects on the cream mirror (gold strip ignored)
    grating     P3 grating: particles, streaks and discoloured spots
    alignment   P3 alignment marks: dark-line pattern offset vs the bright-line pattern, and dirt
    lens3d      P4 3D profile: lens shape from LEXT 3D scans (radius of curvature, residual, defects)
    p1_aperture P1 aperture: size and edges of the opening, dirt on the coating around it
    p1_slit     P1 slit: the same at higher magnification
    p5_mirror   P5 mirror: size of the mirror, dirt and stains on it
    none        no analysis: the images can only be browsed

Each of the P1 / P3 / P4 / P5 modules defines STEPS (pipeline), step_ui() (tuning GUI) and METRICS (tolerances; per analysis
type if a module serves several).
"""
import importlib

MODULES = {"p3_mirror": "p3mirror", "grating": "grating", "alignment": "alignment", "lens3d": "lens3d",
           "p1_aperture": "p1opening", "p1_slit": "p1opening", "p5_mirror": "p5mirror"}
NAMES = ["mirror", "lens", *MODULES, "none"]


def name_of(cfg):
    a = cfg.get("analysis")
    if a:
        if a not in NAMES:
            raise ValueError(f"Unknown analysis {a!r}; one of {', '.join(NAMES)}")
        return a
    return "lens" if cfg.get("region") == "lens" else "mirror"


def _module(name):
    return importlib.import_module(f".{MODULES[name]}", __package__)


def steps(cfg):
    from . import pipeline
    name = name_of(cfg)
    if name == "mirror":
        return pipeline.STEPS
    if name == "lens":
        return pipeline.LENS_STEPS
    if name == "none":
        return []
    return _module(name).STEPS


def step_ui(cfg):
    from . import gui_steps
    name = name_of(cfg)
    if name == "mirror":
        return gui_steps.STEP_UI
    if name == "lens":
        return gui_steps.lens_steps()
    return _module(name).step_ui()


def default_metrics(name):
    if name == "none":
        return []
    metrics = _module(name).METRICS
    return metrics[name] if isinstance(metrics, dict) else metrics
