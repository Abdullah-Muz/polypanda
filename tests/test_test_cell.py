"""v4.1: the electronics test cell (simulation/test_cell.py, perception/inspection.py,
control/test_cell.py, evaluation/test_cell.py): layout, the modules' look and hidden truth, the
image rules, scoring, and a short shift run end to end."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from polypanda.control.ik import PoseIK, closing_yaw, top_down_quat
from polypanda.control.test_cell import (
    EnvDoor,
    LineDoor,
    ModuleRecord,
    NestInsertion,
    StraightDown,
    camera_belief,
    make_door,
    make_insertion,
)
from polypanda.evaluation.test_cell import first_pass, run_shift, utilization
from polypanda.perception.inspection import NOISE, ImagingNoise, inspect
from polypanda.simulation.line import build_line_spec
from polypanda.simulation.machine import RUNNING, MachineConfig
from polypanda.simulation.test_cell import (
    ELECTRICAL_DEFECTS,
    MIN_SEVERITY,
    ModuleTruth,
    Station,
    TestCell,
    TestCellConfig,
    TesterSpot,
    build_test_cell_spec,
    built_kinds,
    cell_config,
    draw_truths,
    planned_testers,
)
from polypanda.simulation.test_cell import tester as make_tester
from polypanda.simulation.workcell import _yaw_quat


@pytest.fixture(scope="module")
def cfg():
    return TestCellConfig()


@pytest.fixture(scope="module")
def cell(cfg):
    c = TestCell.build(cfg)
    yield c
    c.close()


def _box_xy(model, data, geoms):
    """The plan-view (x, y) bounds of a set of geoms (boxes), turned with their bodies."""
    pts = []
    for g in geoms:
        R, p, s = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g], model.geom_size[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                pts.append((p + R @ (np.array([sx, sy, 0.0]) * s))[:2])
    pts = np.array(pts)
    return pts.min(0), pts.max(0)


# --- layout -------------------------------------------------------------------------------------
def test_the_testers_and_stations_stand_where_the_layout_says(cell, cfg):
    for s in cfg.testers:
        np.testing.assert_allclose(cell.machines[s.name].pocket_xyz()[:2],
                                   cfg.xy(Station(s.stop, s.angle)), atol=1e-9)
    np.testing.assert_allclose(cell.reject_xy, [-0.9 + 0.5 * np.cos(np.radians(-150)),
                                                0.5 * np.sin(np.radians(-150))])
    np.testing.assert_allclose(cell.data.xpos[cell.model.body("reject_bin").id][:2],
                               cell.reject_xy, atol=1e-9)
    np.testing.assert_allclose(cell.pick_xy, [0.0, 0.5], atol=1e-9)
    lo, hi = cfg.rail.low, cfg.rail.high
    stops = [s.stop for s in cfg.testers] + [cfg.vision.stop, cfg.tray.stop, cfg.fail_bin.stop,
                                             cfg.infeed.stop]
    assert all(lo <= s <= hi for s in stops)
    assert len(cell.pockets) == cfg.tray_rows * cfg.tray_cols == 8
    for station in (cfg.vision, cfg.tray, cfg.fail_bin, cfg.infeed):
        assert abs(station.angle) <= 160    # inside joint 1's +-166 deg, with a margin


def test_every_station_is_reachable_from_its_stop(cell, cfg):
    """Top-down at a module's grasp height (its centre, 3 cm) and at the carry height, from the
    carry posture turned to the station, at the station's rail stop."""
    ik = PoseIK()
    home = ik.data.qpos[ik.idx.arm_qpos].copy()
    _, R = ik.forward(home)
    carry, ok = ik.solve(home, [0.5, 0.0, 0.32], top_down_quat(closing_yaw(R)))
    assert ok
    targets = [(s.stop, cell.machines[s.name].pocket_xyz()[:2]) for s in cfg.testers]
    targets += [(cfg.vision.stop, cell.vision_xy), (cfg.fail_bin.stop, cell.reject_xy),
                (cfg.infeed.stop, cell.pick_xy)]
    targets += [(cfg.tray.stop, p) for p in cell.pockets]
    for stop, xy in targets:
        rel = np.asarray(xy) - [stop, 0.0]
        a = float(np.arctan2(rel[1], rel[0]))
        seed = carry.copy()
        seed[0] += a
        for z in (0.32, 0.03):
            q, ok = ik.solve(seed, [*rel, z], top_down_quat(closing_yaw(R) + a))
            assert ok, (stop, xy, z)
            seed = q


def test_no_station_overlaps_a_tester_the_bin_or_the_conveyor(cell, cfg):
    m, d = cell.model, cell.data
    boxes = []
    for s in cfg.testers:                     # every colliding geom of each tester, lid open
        root = m.body(s.name).id
        gs = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root
              and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        boxes.append((s.name, *_box_xy(m, d, gs)))
    bin_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("reject_bin").id
                 and m.geom_contype[g]]
    boxes.append(("bin", *_box_xy(m, d, bin_geoms)))
    # the conveyor's frame (drawn) runs along y = 0.5 from x = -0.88 to +0.98, 0.12 m wide plus
    # its side frames; its far side carries a motor near the downstream end
    boxes.append(("conveyor", np.array([-0.9, 0.38]), np.array([1.0, 0.82])))
    # module footprints (with the 8 mm rib and the open fingers' 5 cm either way radially)
    spots = [("vision", cell.vision_xy)] + [(f"pocket{i}", p) for i, p in enumerate(cell.pockets)]
    for name, xy in spots:
        for other, lo, hi in boxes:
            gap = np.maximum(lo - xy, xy - hi).max()
            assert gap > 0.05, (name, other, gap)
    for i, (a, lo_a, hi_a) in enumerate(boxes):        # testers and bin apart from each other
        for b, lo_b, hi_b in boxes[i + 1:]:
            assert np.maximum(lo_a - hi_b, lo_b - hi_a).max() > 0.05, (a, b)


def test_testers_b_and_c_are_lid_stand_ins_until_the_tester_kinds_merge():
    cfg = cell_config(cycle_s=12.0)
    assert [(t.name, t.kind, t.planned_kind, t.stand_in) for t in cfg.testers] == [
        ("TA", "lid", "lid", False), ("TB", "lid", "drawer", True),
        ("TC", "lid", "pneumatic", True)]
    assert {t.config.cycle_s for t in cfg.testers} == {12.0}
    assert all(t.config.keyed and t.config.contact == "stiff" for t in cfg.testers)
    assert all(t.config.clearance >= 0.00025 for t in cfg.testers)
    with pytest.raises(ValueError):
        TesterSpot("TX", 0.0, kind="spaceship")
    with pytest.raises(ValueError):          # the spot's kind must be its machine's
        TesterSpot("TX", 0.0, kind="drawer")
    if "drawer" in built_kinds():
        assert [t.kind for t in planned_testers()] == ["lid", "drawer", "pneumatic"]
    else:
        assert built_kinds() == ("lid",)
        with pytest.raises(NotImplementedError):
            make_tester("TX", 0.0, "drawer")
        with pytest.raises(NotImplementedError):
            planned_testers()


# --- the module's look and truth ----------------------------------------------------------------
def test_the_module_look_adds_no_physics(cfg):
    """The look is visual geoms on the part bodies: masses, inertias and colliding geoms are the
    plain line's (same layout, no look); and the cell's own additions collide nowhere."""
    plain = build_line_spec(cfg.line_config()).compile()
    cell = build_test_cell_spec(cfg).compile()
    for k in range(cfg.n_bodies):
        a, b = plain.body(f"cube{k}").id, cell.body(f"cube{k}").id
        np.testing.assert_allclose(cell.body_mass[b], plain.body_mass[a])
        np.testing.assert_allclose(cell.body_inertia[b], plain.body_inertia[a])
        np.testing.assert_allclose(cell.body_ipos[b], plain.body_ipos[a])

    def colliding(model):
        return sorted(model.geom(g).name for g in range(model.ngeom)
                      if model.geom_contype[g] or model.geom_conaffinity[g])

    assert colliding(cell) == colliding(plain)
    for attr in ("nq", "nv", "nu", "njnt", "neq"):
        assert getattr(cell, attr) == getattr(plain, attr), attr
    looks = [g for g in range(cell.ngeom) if cell.geom(g).name.startswith("cube")
             and "_" in cell.geom(g).name and not cell.geom(g).name.endswith("_rib")]
    assert len(looks) >= 19 * cfg.n_bodies
    assert all(cell.geom_contype[g] == 0 and cell.geom_conaffinity[g] == 0 for g in looks)


def test_the_v40_line_is_unchanged_by_the_cell():
    """Building the test cell leaves LineConfig's defaults alone (v4.0's line: one machine, its
    rail, the factory look)."""
    from polypanda.simulation.line import LineConfig
    TestCellConfig()
    line = LineConfig()
    assert [s.name for s in line.machines] == ["M1"] and line.theme == "factory"
    assert (line.rail.low, line.rail.high) == (-0.65, 0.15) and not line.keyed_parts


def test_truths_are_seeded_and_have_the_planned_rates():
    big = draw_truths(4000, seed=3)
    fail = np.mean([not t.electrical_pass for t in big])
    defect = np.mean([t.defect is not None for t in big])
    residue = np.mean([t.residue > 0 for t in big])
    assert 0.04 < fail < 0.06 and 0.022 < defect < 0.038 and 0.03 < residue < 0.05
    assert 0.9 < np.mean([t.good for t in big]) < 0.94
    assert {t.defect for t in big} == {None, "missing_label", "red_mark"}
    assert {t.electrical_defect for t in big if not t.electrical_pass} == set(ELECTRICAL_DEFECTS)
    assert all(t.electrical_defect is None for t in big if t.electrical_pass)
    marks = [t.severity for t in big if t.defect == "red_mark"]
    assert MIN_SEVERITY <= min(marks) and max(marks) <= 1.0 and np.std(marks) > 0.1
    assert all(0.3 <= t.residue <= 1.0 for t in big if t.residue)
    assert all(t.good == (t.electrical_pass and t.defect is None) for t in big)   # residue: OK
    assert draw_truths(4000, seed=3) == big and draw_truths(10, seed=3) == big[:10]
    assert draw_truths(10, seed=4) != big[:10]
    # one module's truth does not depend on the rates of the other properties
    other = draw_truths(200, seed=3, defect_rate=0.5, residue_rate=0.0)
    assert [t.electrical_pass for t in other] == [t.electrical_pass for t in big[:200]]


def test_bodies_are_recycled_through_a_long_shift(cell):
    truths = draw_truths(30, seed=1)
    cell.reset_shift(truths)
    assert cell.loaded == 3 and cell.modules_due == 27        # the first batch, on the belt
    on_belt = [k for k, m in enumerate(cell.body_module) if m is not None]
    assert [cell.body_module[k] for k in on_belt] == [0, 1, 2]
    while cell.take_body() is not None:
        pass
    assert cell.loaded == len(cell.cubes) and not cell.free_bodies()
    cell.release(on_belt[1])
    assert cell.free_bodies() == [on_belt[1]] and cell.take_body() == on_belt[1]
    assert cell.body_module[on_belt[1]] == len(cell.cubes)
    assert cell.tester_result("TA") is None
    cell.reset_shift([])


# --- the image rules ------------------------------------------------------------------------------
@pytest.mark.parametrize("dressing,theme", [(True, "electronics"), (True, "factory"),
                                            (False, "electronics")])
def test_the_scan_and_the_vision_rule_see_what_is_drawn(dressing, theme):
    from polypanda.simulation.test_cell import ModuleTruth
    cell = TestCell.build(cell_config(dressing=dressing, theme=theme))
    looks = [ModuleTruth("a", True), ModuleTruth("b", True, "missing_label"),
             ModuleTruth("c", True, "red_mark")]
    cell.reset_shift(looks)
    m, d = cell.model, cell.data
    try:
        cell.camera_image("vision")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no offscreen renderer: {exc}")
    rng = np.random.default_rng(0)
    try:
        for k, truth in enumerate(looks):
            cube = cell.cubes[cell.body_module.index(k)]
            for camera, xy, lift in (("vision", cell.vision_xy, 0.0),
                                     ("scanner", cell.pick_xy, 0.002)):
                for yaw in (0.3, 2.2):
                    d.qpos[cube.qpos:cube.qpos + 7] = [*xy, 0.03 + lift, *_yaw_quat(yaw)]
                    mujoco.mj_forward(m, d)
                    image = cell.camera_image(camera)
                    for ins in (inspect(image), inspect(NOISE.apply(image, rng))):
                        assert ins.readable == (truth.defect != "missing_label"), (camera, ins)
                        if camera == "vision":
                            assert ins.defect == (truth.defect is not None), (truth, ins)
            d.qpos[cube.qpos:cube.qpos + 3] = [3.0, 3.0, 0.03]
        mujoco.mj_forward(m, d)
        assert not inspect(cell.camera_image("scanner")).readable      # nothing there
    finally:
        cell.close()


def test_the_vision_rule_misses_faint_marks_and_takes_big_residue_for_a_mark():
    """The rule's built-in imperfection (perception/inspection.py), on clean images: the
    faintest red mark is not seen (an escape, if the module passes its test), the largest
    residue patch is (a false reject), a small one is not; a full mark always is."""
    cell = TestCell.build(cell_config(dressing=False))
    cases = [(ModuleTruth("faint", True, "red_mark", severity=MIN_SEVERITY), False),
             (ModuleTruth("full", True, "red_mark", severity=1.0), True),
             (ModuleTruth("res_small", True, residue=0.3), False),
             (ModuleTruth("res_big", True, residue=1.0), True)]
    cell.reset_shift([])
    cube = cell.cubes[0]
    m, d = cell.model, cell.data
    try:
        for truth, seen in cases:
            cell.show(0, truth)
            for yaw in (0.3, 1.4, 2.2, 4.0):
                d.qpos[cube.qpos:cube.qpos + 7] = [*cell.vision_xy, 0.03, *_yaw_quat(yaw)]
                mujoco.mj_forward(m, d)
                try:
                    ins = inspect(cell.camera_image("vision"))
                except Exception as exc:  # noqa: BLE001
                    pytest.skip(f"no offscreen renderer: {exc}")
                assert ins.mark == seen and ins.label, (truth, yaw, ins)
    finally:
        cell.close()


def test_imaging_noise_is_seeded_and_can_be_off():
    image = np.full((8, 8, 3), 128, np.uint8)
    assert np.array_equal(ImagingNoise.none().apply(image, np.random.default_rng(0)), image)
    a = NOISE.apply(image, np.random.default_rng(1))
    assert np.array_equal(a, NOISE.apply(image, np.random.default_rng(1)))
    assert not np.array_equal(a, image) and a.dtype == np.uint8


def test_the_camera_refers_an_open_doors_handle_back_to_shut(cell):
    """Belief.handle is the handle with the door shut (move_lid adds the opening itself)."""
    m = cell.machines["TA"]
    shut = m.handle_xyz()
    cell.data.qpos[m.lid_qpos] = 0.08
    mujoco.mj_forward(cell.model, cell.data)
    try:
        b = camera_belief(cell, "TA", np.random.default_rng(0))
        assert np.linalg.norm(b.handle - shut) < 0.005
        assert np.linalg.norm(b.handle_at(0.08) - m.handle_xyz()) < 0.005
    finally:
        cell.data.qpos[m.lid_qpos] = 0.0
        mujoco.mj_forward(cell.model, cell.data)


def test_door_actors_by_name():
    assert make_door("line_camera").name == "line_camera"
    assert isinstance(make_door("feel_v2"), EnvDoor) and make_door("run:x").actor == "run:x"
    custom = LineDoor()
    assert make_door(custom) is custom
    with pytest.raises(ValueError):
        make_door("bogus")
    with pytest.raises(ValueError):
        EnvDoor("feel_v2", handover="bogus")


# --- scoring ------------------------------------------------------------------------------------
def test_a_sort_is_right_when_it_matches_the_truth():
    good = ModuleRecord.of(0, ModuleTruth("a", True, residue=0.5))
    bad = ModuleRecord.of(1, ModuleTruth("b", False, electrical_defect="short"))
    assert good.good and not bad.good and bad.electrical_defect == "short"
    for rec, where, right in ((good, "pass_tray", True), (good, "fail_bin", False),
                              (bad, "fail_bin", True), (bad, "pass_tray", False),
                              (good, None, False), (bad, None, False)):
        rec.sorted_to = where
        assert rec.correct == right


def test_first_pass_means_once_through_cleanly():
    rec = ModuleRecord.of(0, ModuleTruth("a", True))
    rec.scan, rec.sorted_to = {"read": True, "tries": 1}, "pass_tray"
    rec.inserts = [{"attempt": 0, "seated": True}]
    assert first_pass(rec)
    rec.scan["tries"] = 2
    assert not first_pass(rec)
    rec.scan["tries"] = 1
    rec.inserts.append({"attempt": 1, "seated": True})
    assert not first_pass(rec)
    rec.inserts.pop()
    rec.failures.append("vision_pick")
    assert not first_pass(rec)


def test_utilization_counts_the_running_time():
    timeline = [[0.0, "TA", "empty", False], [10.0, "TA", "loaded", False],
                [12.0, "TA", RUNNING, False], [42.0, "TA", "done", False],
                [50.0, "TA", "loaded", False], [60.0, "TA", RUNNING, False],
                [0.0, "TB", "empty", False]]
    u = utilization(timeline, ["TA", "TB"], 0.0, 100.0)
    assert u == {"TA": pytest.approx(0.7), "TB": 0.0}       # 30 s + 40 s (still running at 100)


# --- a shift, end to end ------------------------------------------------------------------------
class _Counting:
    """An actor that records its calls and hands them on (checks the slots are used)."""

    def __init__(self, inner):
        self.inner, self.name, self.calls = inner, f"counting_{inner.name}", []

    def insert(self, ctl, part, tester):
        self.calls.append(("insert", tester, ctl.cell.attached(part)))
        return {"by": "counting"} | self.inner.insert(ctl, part, tester)

    def move(self, ctl, tester, command):
        self.calls.append((command, tester))
        return self.inner.move(ctl, tester, command)


def test_a_short_shift_tests_and_sorts_every_module():
    """Four modules, one of each kind: good, failing the electrical test, a red mark (seen by
    the vision station), no label (no read at the scanner). Short tests (5 s), the nest's true
    pose (an oracle insertion: what this test checks is the cell's flow), the door and the
    insertion through their actor slots."""
    truths = [ModuleTruth("good", True), ModuleTruth("elec", False),
              ModuleTruth("mark", True, "red_mark"), ModuleTruth("nolabel", True,
                                                                 "missing_label")]
    door, insertion = _Counting(EnvDoor("feel_v2")), _Counting(StraightDown())
    try:
        out = run_shift(4, seed=0, truths=truths, cycle_s=5.0, nest_belief="true",
                        dressing=False, door=door, insertion=insertion)
    except Exception as exc:  # noqa: BLE001
        if "renderer" in repr(exc).lower() or "gl" in repr(exc).lower():
            pytest.skip(f"no offscreen renderer: {exc}")
        raise
    k = out["kpis"]
    rows = {r["serial"]: r for r in out["modules"]}
    assert k["sorted"] == 4 and k["sort_correct"] == 4, out["modules"]
    assert rows["good"]["sorted_to"] == "pass_tray" and rows["good"]["test_result"] == "pass"
    assert rows["elec"]["test_result"] == "fail" and rows["elec"]["sorted_to"] == "fail_bin"
    assert rows["elec"]["electrical_defect"] == "open_circuit"
    assert rows["mark"]["vision"]["defect"] and rows["mark"]["sorted_to"] == "fail_bin"
    assert not rows["nolabel"]["scan"]["read"] and rows["nolabel"]["tester"] is None
    assert rows["nolabel"]["scan"]["tries"] == 2
    assert [rows[s]["reject_reason"] for s in ("good", "elec", "mark", "nolabel")] == [
        None, "test_fail", "vision", "no_read"]
    assert k["caught"] == {"test_fail": 1, "vision": 1, "no_read": 1}
    assert k["escapes"] == k["false_rejects"] == 0
    assert k["tested"] == 3 and not k["timed_out"]
    assert k["test_yield"] == pytest.approx(2 / 3, abs=1e-3)
    assert 0.0 <= k["first_pass_yield"] <= 1 / 4      # only "good" can pass, if seated at once
    assert 0.0 < k["utilization_mean"] < 1.0 and k["modules_per_hour"] > 0
    assert set(k["activity_total_s"]) >= {"open", "close", "insert", "fetch", "vision", "sort"}
    testers = {row[1] for row in out["timeline"]}
    assert testers == {"TA", "TB", "TC"}
    assert any(row[2] == RUNNING for row in out["timeline"])
    # the slots: every insertion with the module held, every door move through the door actor
    inserts = [c for c in insertion.calls]
    assert len(inserts) == sum(len(r["inserts"]) for r in out["modules"]) >= 3
    assert all(held for _, _, held in inserts)
    assert all(i.get("by") == "counting" for r in out["modules"] for i in r["inserts"])
    assert {c[0] for c in door.calls} == {"open", "close"}
    assert out["config"]["door"] == "counting_feel_v2"
    assert out["config"]["insertion"] == "counting_straight_down"
    assert out["config"]["testers"]["TB"].items() >= {"planned": "drawer",
                                                      "stand_in": True}.items()


class _Descend:
    """A PoseInsertion: straight down to `xy` at `yaw` (ScriptedTending.load's second half),
    recording what it was told."""
    name = "descend"

    def __init__(self):
        self.calls = []

    def insert(self, ctl, part, xy, yaw):
        from polypanda.control.scripted_cell import CUBE_HEIGHT
        from polypanda.environments.grasp import GRIPPER_OPEN
        self.calls.append((np.array(xy), float(yaw), ctl.cell.attached(part)))
        tcp = ctl.d.site_xpos[ctl.arm.tcp_site]
        offset = tcp[2] - ctl.d.xpos[part.body][2]
        ctl.vertical(xy, CUBE_HEIGHT / 2 + ctl.config.release_clearance + offset, yaw,
                     track=part.body)
        ctl.cell.detach(part)
        ctl.grip = GRIPPER_OPEN
        ctl.step(ctl.config.grip_steps)
        ctl.vertical(ctl.d.site_xpos[ctl.arm.tcp_site][:2].copy(), ctl.config.carry_height, yaw)
        return {"told": [round(float(v), 4) for v in xy]}


def test_a_pose_insertion_goes_into_the_slot_through_nest_insertion():
    """An insertion with the call insert(ctl, part, xy, yaw) (as SearchInsertion on main) runs
    in the slot through NestInsertion, aimed at the believed nest (here the truth), and the
    module is seated and tested; the line's own door move fills the door slot."""
    inner = _Descend()
    out = run_shift(1, seed=0, truths=[ModuleTruth("one", True)], cycle_s=2.0,
                    nest_belief="true", dressing=False, door="line_camera",
                    insertion=NestInsertion(inner))
    rec = out["modules"][0]
    assert out["config"]["insertion"] == "descend" and out["config"]["door"] == "line_camera"
    assert rec["sorted_to"] == "pass_tray" and rec["test_result"] == "pass", rec
    assert rec["inserts"][0]["seated"] and rec["inserts"][0]["told"]
    assert len(inner.calls) == len(rec["inserts"]) and all(held for *_, held in inner.calls)
    nest = TestCell.build(cell_config(dressing=False))
    try:
        np.testing.assert_allclose(inner.calls[0][0], nest.machines[rec["tester"]]
                                   .pocket_xyz()[:2], atol=1e-6)
    finally:
        nest.close()


def test_insertion_actors_by_name():
    assert isinstance(make_insertion(None), StraightDown)
    assert isinstance(make_insertion("straight_down"), StraightDown)
    custom = NestInsertion(_Descend())
    assert make_insertion(custom) is custom and custom.name == "descend"
    with pytest.raises(ValueError):
        make_insertion("bogus")
    try:
        import polypanda.control.scripted_insertion  # noqa: F401
    except ImportError:
        with pytest.raises(NotImplementedError):
            make_insertion("search")
    else:
        assert make_insertion("search").name == "search"


def test_machine_config_default_is_not_the_testers():
    """The testers' tight nests are theirs only: MachineConfig's defaults are v4.0's."""
    assert MachineConfig().clearance == 0.002 and not MachineConfig().keyed
    assert replace(MachineConfig(), cycle_s=45.0).contact == "default"
