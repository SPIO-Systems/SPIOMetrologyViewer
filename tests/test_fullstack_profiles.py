"""Full stack layer thicknesses (ported from ThicknessImageProcessor). Synthetic stacks: vertically streaked
glass, smooth glue / polymer bands, air above and below."""
import copy

import cv2
import numpy as np
import pytest
import yaml

from spiometrologyviewer import fullstack as steps, fullstack_profiles as pr, pipeline

CFG = yaml.safe_load(steps.TEMPLATE_CONFIG.read_text(encoding="utf-8"))
NAMES = steps.glass_names(CFG)

# Six glass plates (top, bottom) and the bands between them; the P3|P4 band is speckled like glass
# in its middle, as in some of the real images.
GLASS = [(200, 1000), (1230, 2100), (2140, 3000), (3530, 4900), (4950, 6300), (6360, 7600)]
SPECKLED = (3150, 3400)
TOL = 3 * CFG["texture"]["smooth_sigma"]   # interfaces widen by up to ~3 sigma (smoothing + extension)

# Layer boundaries, each drawn as a thin dark line: P1|P2 glue 1000-1060 and polymer 1060-1230,
# P2P3 glue 2100-2140, P4_Polymer 3150-3530 (its middle is the speckled part), P4P5 glue 4900-4950,
# P5P6 glue 6300-6360.
LINES = [1000, 1060, 1230, 2100, 2140, 3000, 3150, 3530, 4900, 4950, 6300, 6360]
EXPECT = {"P2_Polymer": 170, "P4_Polymer": 380, "P1P2_Glue": 60, "P2P3_Glue": 40, "P4P5_Glue": 50,
          "P5P6_Glue": 60}


def make_stack(h=8000, w=400, tilt_deg=0.0, seed=0, lines=LINES, single_line=None):
    """`single_line` = (row, width): a wider dark line instead of a glue band's two lines."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    streaks = cv2.GaussianBlur(noise, (0, 0), sigmaX=0.7, sigmaY=6)     # vertical streaks: glass
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.5)                        # isotropic and soft: glue / polymer
    streaks /= streaks.std()
    smooth /= smooth.std()
    y = np.arange(h)
    glass = np.zeros(h, bool)
    for a, b in GLASS:
        glass |= (y >= a) & (y < b)
    glass |= (y >= SPECKLED[0]) & (y < SPECKLED[1])
    img = np.where(glass[:, None], 90 + 25 * streaks, 80 + 6 * smooth)
    for r in lines:
        img[np.abs(y - r) < 3] *= 0.4
    if single_line:
        img[np.abs(y - single_line[0]) < single_line[1] / 2] *= 0.4
    img[(y < GLASS[0][0]) | (y >= GLASS[-1][1])] = 5
    img = np.clip(img, 0, 255).astype(np.float32)
    if tilt_deg:
        img = pr.deskew(img, -tilt_deg)   # tilt it; the pipeline should take it out again
    return img


def plates_of(img, **kw):
    texture = pr.texture_profile(img)
    top, bottom = pr.air_edges(pr.project(img), 25)
    return pr.find_glass_plates(texture, top, bottom, NAMES, **kw)


def test_six_plates_found():
    """Edge accuracy of the threshold alone (the extension widens interfaces on purpose)."""
    p = plates_of(make_stack(), extend=CFG["texture"]["threshold"])
    assert [n for n, _, _ in p.plates] == NAMES
    for (name, a, b), (ta, tb) in zip(p.plates, GLASS):
        assert a == pytest.approx(ta, abs=TOL), name
        assert b == pytest.approx(tb, abs=TOL), name


def test_extension_only_widens_interfaces():
    core = plates_of(make_stack(), extend=CFG["texture"]["threshold"])
    p = plates_of(make_stack())
    assert [n for n, _, _ in p.plates] == NAMES
    for (a, b, _, _), (ca, cb, _, _) in zip(p.interfaces, core.interfaces):
        assert a <= ca and b >= cb


def test_speckled_band_stays_one_interface():
    p = plates_of(make_stack(), extend=CFG["texture"]["threshold"])
    a, b, above, below = p.interfaces[2]
    assert (above, below) == ("P3_Glass", "P4_Glass")
    assert a == pytest.approx(3000, abs=TOL) and b == pytest.approx(3530, abs=TOL)


def test_too_small_min_glass_splits_the_band():
    p = plates_of(make_stack(), min_glass=100)   # the speckled middle then counts as a glass plate
    assert any("deepest" in n for n in p.notes)


def test_crops_have_the_margin():
    img = make_stack()
    p = plates_of(img)
    crops = pr.interface_crops(img, p, margin=30)
    assert len(crops) == 5
    for (name, r0, r1, crop), (a, b, _, _) in zip(crops, p.interfaces):
        assert r0 == int(np.floor(a - 30)) and r1 == int(np.ceil(b + 30))
        assert crop.shape == (r1 - r0, img.shape[1])


def test_tilt_estimated_and_removed():
    img = make_stack(tilt_deg=0.4)
    angle, _, _ = pr.estimate_tilt(img, 1.0)
    assert abs(angle) == pytest.approx(0.4, abs=0.05)
    p = plates_of(pr.deskew(img, angle))
    assert len(p.plates) == 6


def test_pipeline_and_every_view_render():
    cfg = copy.deepcopy(CFG)
    rgb = np.dstack([make_stack().astype(np.uint8)] * 3)
    state = pipeline.run_steps({"name": "synthetic", "raw": rgb}, cfg, steps.STEPS)
    assert len(state["crops"]) == 5
    for ui in steps.step_ui():
        for name, view in ui.views.items():
            img = view(state, cfg)
            assert img.dtype == np.uint8 and img.ndim == 3, name
        assert isinstance(ui.info(state, cfg), str)


LINE_W = 6   # the dark lines drawn by make_stack are 6 px wide: the light glue interior is LINE_W narrower
GLUES = ("P1P2_Glue", "P2P3_Glue", "P4P5_Glue", "P5P6_Glue")


def layers_of(img, cfg=CFG):
    p = plates_of(img)
    _, _, depth = pr.relative_depth(pr.project(img))
    w = pr.interface_windows(depth, p, cfg["layers"])
    polymers = pr.polymer_layers(w)
    rel, grad = pr.gradient_profile(img)
    return polymers, pr.glue_layers(w, rel, grad, polymers)


def test_polymers_between_line_centres():
    polymers, _ = layers_of(make_stack())
    for name in ("P2_Polymer", "P4_Polymer"):
        assert polymers[name].thickness == pytest.approx(EXPECT[name], abs=2), name
        assert polymers[name].method == "lines"


def test_glues_between_rising_and_falling_edge():
    _, glues = layers_of(make_stack())
    assert set(glues) == set(GLUES)
    for name in GLUES:
        assert glues[name].method == "edges", name
        assert glues[name].thickness == pytest.approx(EXPECT[name] - LINE_W, abs=2), name


def test_glue_edges_robust_to_noise_and_a_nearby_scratch():
    img = make_stack(seed=3)
    y = np.arange(img.shape[0])
    img[np.abs(y - 4870) < 2] *= 0.5                 # a dark scratch in P4_Glass, 30 px above P4P5_Glue
    img += np.random.default_rng(1).normal(0, 6, img.shape).astype(np.float32)
    _, glues = layers_of(img)
    g = glues["P4P5_Glue"]
    assert g.method == "edges"
    assert g.top == pytest.approx(4900 + LINE_W / 2, abs=2) and g.bottom == pytest.approx(4950 - LINE_W / 2, abs=2)


def test_thin_glue_without_light_interior_is_the_dark_band():
    lines = [r for r in LINES if r not in (2100, 2140)]
    _, glues = layers_of(make_stack(lines=lines, single_line=(2120, 12)))
    g = glues["P2P3_Glue"]
    assert g.method == "dark band"
    assert g.thickness == pytest.approx(12, abs=2)


def test_p1p2_glue_searched_above_p2_polymer():
    polymers, glues = layers_of(make_stack())
    assert glues["P1P2_Glue"].bottom < polymers["P2_Polymer"].top + 10
    assert glues["P1P2_Glue"].top == pytest.approx(1000 + LINE_W / 2, abs=2)


def stack_of(img, cfg=CFG):
    p = plates_of(img)
    _, _, depth = pr.relative_depth(pr.project(img))
    w = pr.interface_windows(depth, p, cfg["layers"])
    polymers = pr.polymer_layers(w)
    rel, grad = pr.gradient_profile(img)
    found = {**polymers, **pr.glue_layers(w, rel, grad, polymers), **pr.mixed_layers(w)}
    return found, pr.stack_layers(cfg["layers"], p, found)


def test_p3_polymer_and_p3p4_glue_measured_together():
    found, _ = stack_of(make_stack())
    m = found["P3_Polymer+P3P4_Glue"]
    assert m.top == pytest.approx(3000, abs=1.5) and m.bottom == pytest.approx(3150, abs=1.5)
    assert m.bottom == pytest.approx(found["P4_Polymer"].top, abs=0.01)


def test_full_stack_in_order_with_glass_between_measured_layers():
    found, stack = stack_of(make_stack())
    assert [l.name for l in stack] == [lay["name"] for lay in CFG["layers"]]
    for a, b in zip(stack, stack[1:]):
        assert a.bottom <= b.top + 1e-6, (a.name, b.name)   # no overlaps
    glass = {l.name: l for l in stack if l.method == "glass"}
    assert glass["P1_Glass"].top == pytest.approx(200, abs=2)
    assert glass["P2_Glass"].top == pytest.approx(found["P2_Polymer"].bottom)
    assert glass["P4_Glass"].top == pytest.approx(3530, abs=2)
    assert glass["P4_Glass"].bottom == pytest.approx(found["P4P5_Glue"].top)
    assert glass["P6_Glass"].bottom == pytest.approx(7600, abs=2)


def test_interface_plan_from_the_layer_list():
    between = pr.interfaces(CFG["layers"])
    assert [n for _, n in map(pr.interface_plan, between)] == [3, 2, 3, 2, 2]
    bad = [{"name": "A", "type": "glue"}, {"name": "B", "type": "polymer", "extract": False}]
    with pytest.raises(ValueError):
        pr.interface_plan(bad)
