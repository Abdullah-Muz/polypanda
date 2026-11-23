"""v4.0: the keyed part and the keyed pocket (docs/keyed_part.md): a rib on the line's parts, a
slot in one pocket wall, the machine's seated check and the scripted load that turns the rib to
the slot."""

import mujoco
import numpy as np
import pytest

from polypanda.control.scripted_tending import ScriptedTending
from polypanda.simulation.line import Line, LineConfig, MachineSpot
from polypanda.simulation.machine import MachineConfig, Rib, rib_geom
from polypanda.simulation.workcell import WorkcellConfig, _yaw_quat


def make_line(keyed_parts: bool = True, keyed: bool = True, turn: float = 0.0) -> Line:
    cfg = MachineConfig(keyed=keyed, cycle_s=1e6)
    return Line.build_line(LineConfig(
        cell=WorkcellConfig(tower_angles=(-40.0,), dressing=False),
        machines=(MachineSpot("M1", -0.6, config=cfg, turn=turn),), keyed_parts=keyed_parts))


def geom_names(model: mujoco.MjModel) -> set[str]:
    return {model.geom(g).name for g in range(model.ngeom)}


def drop_into_pocket(line: Line, wrong_deg: float, seconds: float = 2.0):
    """Stand part 0 5 mm above the pocket's rim, centred, turned `wrong_deg` from the right way
    round, and let it fall in (arm held at home)."""
    m1, d = line.machines["M1"], line.data
    part = line.cubes[0]
    p = m1.pocket_xyz()
    R = d.xmat[m1.body].reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0]) + np.radians(wrong_deg)
    d.qpos[part.qpos:part.qpos + 7] = [p[0], p[1], part.half_size[2] + 0.02, *_yaw_quat(yaw)]
    d.qvel[:] = 0.0
    mujoco.mj_forward(line.model, d)
    ctl = ScriptedTending(line)
    ctl.step(int(seconds / ctl.dt))
    rel = R.T @ (d.xpos[part.body] - p)
    return part, float(rel[2] - part.half_size[2])


def test_the_default_line_has_no_rib_and_no_slot():
    line = Line.build_line(LineConfig(cell=WorkcellConfig(tower_angles=(-40.0,), dressing=False)))
    names = geom_names(line.model)
    assert not any(n.endswith("_rib") for n in names)
    assert not any("slot" in n for n in names)
    assert {f"M1_pocket_wall{k}" for k in range(4)} <= names
    assert all(rib_geom(line.model, c) < 0 for c in line.cubes)
    assert line.machines["M1"].key_error(line.cubes[0]) is None
    assert not LineConfig().keyed_parts and not MachineConfig().keyed


def test_the_keyed_options_build_a_rib_on_every_part_and_a_slot_in_one_wall():
    line = make_line()
    names = geom_names(line.model)
    assert all(rib_geom(line.model, c) >= 0 for c in line.cubes)
    assert {"M1_pocket_slot0", "M1_pocket_wall0a", "M1_pocket_wall0b"} <= names
    assert "M1_pocket_wall0" not in names
    assert {f"M1_pocket_wall{k}" for k in (1, 2, 3)} <= names     # the other walls are plain


def test_an_impossible_slot_is_refused():
    with pytest.raises(ValueError):
        MachineConfig(keyed=True, rib=Rib(width=0.04))


def test_the_rib_is_a_separate_geom_and_the_part_box_stays_its_main_geom():
    line = make_line()
    m = line.model
    for c in line.cubes:
        assert m.geom(c.geom).name == m.body(c.body).name               # cube.geom is the box
        np.testing.assert_allclose(m.geom_size[c.geom], [0.02, 0.02, 0.03])
        np.testing.assert_allclose(c.half_size, [0.02, 0.02, 0.03])
        r = rib_geom(m, c)
        assert m.geom_bodyid[r] == c.body
        np.testing.assert_allclose(m.geom_pos[r], [0.024, 0.0, -0.024])  # +x face, on the bottom
        assert m.geom_contype[r] == m.geom_contype[c.geom]
        assert m.geom_conaffinity[r] == m.geom_conaffinity[c.geom]
    plain = make_line(keyed_parts=False)
    c, p = line.cubes[0], plain.cubes[0]
    assert m.body_mass[c.body] == pytest.approx(plain.model.body_mass[p.body] + 0.00096, rel=1e-3)
    assert np.abs(m.body_ipos[c.body]).max() < 0.0003      # the centre of mass moves < 0.3 mm


def test_a_part_dropped_the_right_way_round_is_seated():
    line = make_line()
    part, gap = drop_into_pocket(line, 0.0)
    m1 = line.machines["M1"]
    assert m1.in_pocket(part) == (True, True)
    assert abs(m1.key_error(part)) < 1.0 and gap < 0.001


@pytest.mark.parametrize("wrong", [90.0, 180.0, 270.0])
def test_a_part_dropped_the_wrong_way_round_rests_on_its_rib_and_is_not_seated(wrong):
    line = make_line()
    part, gap = drop_into_pocket(line, wrong, seconds=4.0)
    m1 = line.machines["M1"]
    assert m1.in_pocket(part) == (True, False)
    assert abs(abs(m1.key_error(part)) - min(wrong, 360 - wrong)) < 5.0
    assert gap > 0.004                      # held up by the rib, not only refused by the check


def test_a_plain_part_fits_the_keyed_pocket_any_way_round():
    line = make_line(keyed_parts=False)
    part, _ = drop_into_pocket(line, 90.0)
    assert line.machines["M1"].key_error(part) is None
    assert line.machines["M1"].in_pocket(part) == (True, True)


def _scripted_load(line: Line, part_yaw_deg: float, keyed_pick: bool):
    m, d = line.model, line.data
    m1 = line.machines["M1"]
    part = line.cubes[0]
    d.qpos[m1.lid_qpos] = m1.cfg.travel - ScriptedTending.open_margin
    spot = line.line.station_xy(-0.6, -30.0)
    d.qpos[part.qpos:part.qpos + 7] = [*spot, part.half_size[2],
                                       *_yaw_quat(np.radians(part_yaw_deg))]
    mujoco.mj_forward(m, d)
    ctl = ScriptedTending(line)
    fingers = {m.body("left_finger").id, m.body("right_finger").id}
    rib = rib_geom(m, part)
    touched = []

    def watch():
        line.update_machines()
        for i in range(d.ncon):
            c = d.contact[i]
            if rib in (c.geom1, c.geom2) and {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} \
                    & fingers:
                touched.append(1)

    ctl.on_step = watch
    ctl.go_to("M1")
    ctl.swing_to(np.radians(-30.0))
    (ctl.pick_for(part, "M1") if keyed_pick else ctl.pick(part))
    ctl.swing_to(np.radians(-90.0))
    ctl.load(part, "M1")
    ctl.step(20)
    return ctl, part, touched


@pytest.mark.parametrize("part_yaw", [60.0, 150.0])
def test_a_scripted_load_turns_the_rib_to_the_slot_and_seats_the_part(part_yaw):
    line = make_line()
    ctl, part, touched = _scripted_load(line, part_yaw, keyed_pick=True)
    m1 = line.machines["M1"]
    assert m1.in_pocket(part) == (True, True)
    assert abs(m1.key_error(part)) < 3.0
    assert ctl.ik_failures == 0 and ctl.regrasps == 0
    assert not touched, "the fingers touched the rib"


def test_a_part_gripped_across_the_wrong_faces_is_regrasped_then_seated():
    # the cell's plain pick grips this part with its rib across the fingers' closing direction;
    # turning it to the slot would turn the hand across the lid's opening, so load regrasps
    line = make_line()
    ctl, part, _ = _scripted_load(line, 60.0, keyed_pick=False)
    m1 = line.machines["M1"]
    assert ctl.regrasps == 1
    assert m1.in_pocket(part) == (True, True) and ctl.ik_failures == 0
