"""Record the v4.1 short video (about 25 s): one insertion three ways, the learned door on a drawer,
and the one-shot numbers.

Usage: PYTHONPATH=scripts uv run --group media python scripts/record_v41_short.py \\
           --out media/v41_short [--width 1920 --height 1080] [--gif]
Segments (30 fps):
  1. 0-3 s: the title over a close-up of the nest.
  2. 3-13 s: the SAME insertion_dev_v1 episode three ways, side by side on one clock: the frozen
     straight-down and force-guided search (evaluation/insertion.run_script, the default env) and
     the learned insertion (runs/insertion_ppo_r2_seed1, final checkpoint, deterministic, in its
     own env: evaluation/insertion.policy_env_config). Each pane: a close camera on the nest with a
     cutaway, the nest force (the env's own, checked every 2 ms physics step; the window's peak
     per frame), a clock in simulated time, a top view of the module's centre relative to the
     true nest, and the outcome.
  3. 13-20 s: the learned door (runs/door_v3_ppo_r3_seed1, final, task door_v2, as
     scripts/evaluate_door.py runs it) opening a door_dev_v2 drawer, in slow motion.
  4. 20-25 s: the end card (docs/v4_1_targets.md, the one-shots as they came).

Episodes: development sets only (the final sets are refused), chosen by a rule fixed in this
script before any frame was looked at, from the committed development results:
- insertion: among the episodes where straight-down fails with damage, the search succeeds and
  the learned policy succeeds, the most typical one, by the smallest score
      |F - F~| / F~ + |t_search - t_search~| / t_search~ + |t_learned - t_learned~| / t_learned~
  (F: straight-down's peak nest force, F~ its median over straight-down's damage episodes, so
  its failure is typical of its failures, not a near miss; t~: the medians of the search's and
  the learned policy's times to success); ties to the lower index;
- door: among the drawer-open episodes the learned door opens without a slam, the one whose
  steps are closest to its median steps on them, then whose peak force is closest to its median
  peak force; ties to the lower index.
Every actor runs exactly as its evaluator runs it (the same env, config and reset seed); each
outcome is printed and checked against its committed row.

The look (render only): every episode is simulated in its evaluator's own (unthemed) scene; the
recorded states are drawn in a separately compiled TWIN of that scene with v4.1's electronics
theme (simulation/electronics.py: the ESD hall, lighting, the testers as RF shield boxes) and the
test cell's sensor-module look on the held part (test_cell.add_module_look: dark housing, green
board edge, label and barcode), all visual geoms. The twin gets the episode's layout (body
placements, joint axes and ranges, the re-cut nest or the door's handle, copied by name) and the
states joint by joint, by name. The door scene's twin is the door env's scene with the theme's
tester restyle and lights, and a light ESD-grey floor. Also render only: the tester box's walls
and roof nearest the insertion camera hidden (geom group 5), the nest's near walls drawn
semi-transparent, no shadows or reflections. The physics never sees any of it.
Deterministic given its arguments (MuJoCo, the deterministic policies, fixed seeds).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from record_demo import font

from polypanda.environments.insertion import InsertionEnv
from polypanda.evaluation.insertion import (
    NOISE_SEED,
    policy_env_config,
    run_policy,
    run_script,
)
from polypanda.evaluation.policies import load_run_policy, run_env_cls

DOOR_NOISE_SEED = 7_000_000          # scripts/evaluate_door.py
HIDDEN_GROUP = 5                     # render-only: geoms moved here are not drawn
FORCE_LIMIT = 50.0                   # N, the damage limit (docs/v4_1_targets.md)
BAR_MAX = 80.0                       # N, the force bar's full scale
INSET_MM = 3.5                       # the top view's half-width (mm)
PATH_MM = 5.0                        # the top view draws the path from this height over the rim
IN_NEST_MM = -14.5                   # the module's bottom this far over the rim: down in the
#                                      15 mm nest (within 0.5 mm of its floor)

WHITE, GREY = (255, 255, 255), (150, 150, 150)
GREEN, RED, AMBER = (30, 140, 60), (190, 35, 35), (230, 160, 20)
BG = (18, 20, 24)
ACTOR_COLOUR = {"straight_down": (150, 150, 150), "search": (230, 160, 20),
                "learned": (70, 170, 255)}
ACTOR_LABEL = {"straight_down": "Straight down (scripted)",
               "search": "Force-guided search (scripted)",
               "learned": "Learned (pure RL)"}

TITLE = "0.25 mm clearance. Camera off by 1 mm. Find the hole by feel."
CARD = [
    ("Insertion (pure RL)",
     "99.7 % on 1,800 unseen insertions vs 80.5 % for the industrial force-guided search; "
     "28 % faster; never over the 50 N damage limit."),
    ("Door",
     "98.3 % on unseen lids and drawers, as reliable as the best script, 2.5x faster."),
]
SMALL_PRINT = ("Simulation (MuJoCo). Learned insertion runs on its own compliant controller; "
               "the search on 20 Hz force regulation.")


# --- the episodes, by the rule fixed in advance -------------------------------------------------
def _rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def choose_insertion(results: Path, learned_name: str, set_name: str) -> tuple[int, dict]:
    """The rule (module docstring), on the committed dev results of the three actors."""
    sd = _rows(results / f"straight_down_{set_name}.json")
    se = _rows(results / f"search_{set_name}.json")
    le = _rows(results / f"{learned_name}_{set_name}.json")
    damaged = [r["peak_force"] for r in sd if r["failure"] == "damage"]
    med_f = float(np.median(damaged))
    med_s = float(np.median([r["time_s"] for r in se if r["success"]]))
    med_l = float(np.median([r["time_s"] for r in le if r["success"]]))
    cands = []
    for a, b, c in zip(sd, se, le, strict=True):
        if not a["index"] == b["index"] == c["index"]:
            raise ValueError("the result files' rows are not aligned")
        if a["failure"] == "damage" and b["success"] and c["success"]:
            score = round(abs(a["peak_force"] - med_f) / med_f + abs(b["time_s"] - med_s) / med_s
                          + abs(c["time_s"] - med_l) / med_l, 4)
            cands.append((score, a["index"]))
    if not cands:
        raise ValueError("no episode meets the rule")
    score, index = min(cands)
    rows = {"straight_down": sd[index], "search": se[index], "learned": le[index]}
    print(f"insertion: rule over {len(cands)} candidates (straight-down damaged, search and "
          f"learned seated). Dev medians: straight-down's peak force over its {len(damaged)} "
          f"damage episodes F~ = {med_f:.3f} N; search time {med_s:.3f} s; learned time "
          f"{med_l:.3f} s. Score = |F - F~|/F~ + |t_search - {med_s:.3f}|/{med_s:.3f} + "
          f"|t_learned - {med_l:.3f}|/{med_l:.3f}. Chosen #{index}: straight-down "
          f"{rows['straight_down']['peak_force']} N, search {rows['search']['time_s']} s, "
          f"learned {rows['learned']['time_s']} s, score {score:.4f} (next: "
          f"{', '.join(f'#{i} {s:.4f}' for s, i in sorted(cands)[1:4])})", flush=True)
    return index, rows


def choose_door(results: Path, learned_name: str, set_name: str) -> tuple[int, dict]:
    rows = _rows(results / f"{learned_name}_{set_name}.json")
    ok = [r for r in rows if r["kind"] == "drawer" and r["command"] == "open" and r["success"]]
    med_n = float(np.median([r["steps"] for r in ok]))
    med_f = float(np.median([r["peak_force"] for r in ok]))
    cands = sorted((abs(r["steps"] - med_n), round(abs(r["peak_force"] - med_f), 3), r["index"])
                   for r in ok if r["slams"] == 0)
    _, _, index = cands[0]
    row = next(r for r in rows if r["index"] == index)
    print(f"door: rule over {len(cands)} drawer opens without a slam (of {len(ok)} learned "
          f"successes); median {med_n:g} steps, {med_f:.2f} N; chosen #{index}: "
          f"{row['steps']} steps, {row['peak_force']} N", flush=True)
    return index, row


# --- recording the simulations ------------------------------------------------------------------
@dataclass
class Track:
    """One episode, sampled after every physics step (or chunk): simulated time from the start,
    the state to render, and what the overlays show."""
    t: list = field(default_factory=list)
    qpos: list = field(default_factory=list)
    mocap: list = field(default_factory=list)
    force: list = field(default_factory=list)       # N (insertion: the nest force)
    xy: list = field(default_factory=list)          # mm, the module's centre in the true nest
    z: list = field(default_factory=list)           # mm, the module's bottom over the rim
    extra: list = field(default_factory=list)       # door: the drawer's opening (m)
    outcome: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def done(self) -> None:
        for k in ("t", "qpos", "mocap", "force", "xy", "z", "extra"):
            setattr(self, k, np.asarray(getattr(self, k)))

    @property
    def end(self) -> float:
        return float(self.t[-1])

    def at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))

    def peak(self, t0: float, t1: float) -> float:
        i, j = self.at(t0) + 1, self.at(t1) + 1
        return float(self.force[min(i, j - 1):j].max()) if j > 0 else 0.0


def _sample_insertion(env, track: Track) -> None:
    d = env.data
    origin, R = env.nest_frame()
    rel = R.T @ (d.xpos[env.part.body] - origin)
    track.t.append(float(d.time - env.t0))
    track.qpos.append(d.qpos.copy())
    track.mocap.append(d.mocap_pos.copy())
    track.force.append(float(env._last_force))
    track.xy.append(rel[:2] * 1000.0)
    track.z.append(env.part_height_over_rim() * 1000.0)


def run_insertion(actor: str, ep: dict, index: int, learned: str) -> tuple[InsertionEnv, Track]:
    """`actor` on the episode exactly as scripts/evaluate_insertion.py runs it, sampled."""
    if actor == "learned":
        act = load_run_policy(f"{learned}@final")
        env = InsertionEnv(policy_env_config(learned))
    else:
        act, env = None, InsertionEnv()
    env.reset(seed=NOISE_SEED + index, options={"episode": ep})
    track = Track()
    env._last_force = 0.0
    _sample_insertion(env, track)
    hook = env.ctl.physics_hook          # the env's watch (and the learned layer's reflex)

    def physics():
        hook()
        _sample_insertion(env, track)

    env.ctl.physics_hook = physics
    try:
        out = run_script(env, actor) if act is None else run_policy(env, act)
    finally:
        env.ctl.physics_hook = hook
    track.done()
    track.outcome = out
    origin, R = env.nest_frame()
    belief = R.T @ (np.asarray(env.belief.origin) - origin)
    down = np.flatnonzero(track.z < IN_NEST_MM)
    track.info = {"belief_mm": belief[:2] * 1000.0,
                  "t_in": float(track.t[down[0]]) if len(down) else None,
                  "clearance_mm": ep["clearance"] * 1000.0,
                  "camera_mm": float(np.hypot(*ep["camera_xy"])) * 1000.0,
                  "tester": ep["tester"], "kind": ep["kind"]}
    return env, track


def run_door(ep: dict, index: int, episodes: list[dict], run: str) -> tuple[object, Track]:
    """The learned door on the episode exactly as scripts/evaluate_door.py runs it, sampled after
    every physics chunk."""
    from polypanda.environments.door import DOOR_KINDS, DoorConfig
    kinds = tuple(k for k in DOOR_KINDS if any(e.get("kind", "lid") == k for e in episodes))
    config = replace(DoorConfig(), variation=replace(DoorConfig().variation, kinds=kinds))
    spec = f"{run}@final"
    env = run_env_cls(spec)(config)
    act = load_run_policy(spec)
    obs, _ = env.reset(seed=DOOR_NOISE_SEED + index, options={"episode": ep})
    d, track, t0 = env.data, Track(), float(env.data.time)
    adr = env.machine.door_qpos

    def sample():
        track.t.append(float(d.time - t0))
        track.qpos.append(d.qpos.copy())
        track.mocap.append(d.mocap_pos.copy())
        track.extra.append(float(d.qpos[adr]))

    sample()
    real_step = mujoco.mj_step

    def step(m, data, nstep=1):
        real_step(m, data, nstep=nstep)
        if data is d:
            sample()

    mujoco.mj_step = step
    try:
        done, steps, info = False, 0, {}
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            steps += 1
            done = term or trunc
    finally:
        mujoco.mj_step = real_step
    track.done()
    st = info["stats"]
    track.outcome = {"success": bool(info["is_success"]), "failure": info["failure"],
                     "steps": steps, "peak_force": round(st["peak_force"], 2),
                     "slams": int(st["slams"])}
    track.info = {"dt": env.adapter.control_dt, "handle": env.handle_xyz().copy()}
    return env, track


def run_feel(ep: dict, index: int, episodes: list[dict]) -> dict:
    """feel_v2 on the same drawer, as scripts/evaluate_door.py runs it (not rendered: its time
    is quoted on screen)."""
    from polypanda.control.door_baselines import BaselineDoor
    from polypanda.environments.door import DOOR_KINDS, DoorConfig, DoorEnv
    kinds = tuple(k for k in DOOR_KINDS if any(e.get("kind", "lid") == k for e in episodes))
    env = DoorEnv(replace(DoorConfig(), variation=replace(DoorConfig().variation, kinds=kinds)))
    script = BaselineDoor(env, "feel_v2")
    env.reset(seed=DOOR_NOISE_SEED + index, options={"episode": ep})
    script.reset()
    done, steps, info = False, 0, {}
    while not done:
        _, _, term, trunc, info = env.step(script.act())
        steps += 1
        done = term or trunc
    return {"success": bool(info["is_success"]), "steps": steps,
            "time_s": steps * env.adapter.control_dt}


def check(label: str, out: dict, row: dict, keys) -> None:
    diff = {k: (out.get(k), row.get(k)) for k in keys
            if (round(out[k], 2) if isinstance(out.get(k), float) else out.get(k))
            != (round(row[k], 2) if isinstance(row.get(k), float) else row.get(k))}
    same = "  (as the committed row)" if not diff else f"  DIFFERS from the committed row: {diff}"
    print(f"  {label}: " + ", ".join(f"{k} {out.get(k)}" for k in keys) + same, flush=True)


# --- rendering ----------------------------------------------------------------------------------
def prepare_render(model: mujoco.MjModel, width: int, height: int,
                   fovy: float | None = None) -> None:
    """Render-only: no floor reflections or shadows (both streak off-screen), the buffer size,
    the free camera's field of view (narrow: a long lens, the hand not looming)."""
    model.mat_reflectance[:] = 0.0
    model.light_castshadow[:] = 0
    if fovy is not None:
        model.vis.global_.fovy = fovy
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)


def _id(model: mujoco.MjModel, kind, name: str) -> int:
    return mujoco.mj_name2id(model, kind, name) if name else -1


def insertion_twin(env: InsertionEnv) -> mujoco.MjModel:
    """The insertion scene as InsertionEnv builds it (environments/insertion.scene_config), with
    the hall dressing and the electronics theme on and the held part drawn as a sensor module:
    visual geoms only, compiled apart from the simulated model (render only)."""
    from polypanda.environments.insertion import scene_config
    from polypanda.simulation.line import build_line_spec
    from polypanda.simulation.test_cell import HOUSING_RGBA, add_module_look
    cfg = scene_config()
    cfg = replace(cfg, cell=replace(cfg.cell, dressing=True), theme="electronics")
    spec = build_line_spec(cfg)
    part = env.model.body(env.part.body).name
    add_module_look(spec, part)
    model = spec.compile()
    model.geom_rgba[model.geom(part).id] = HOUSING_RGBA      # (the test cell sets it at reset)
    return model


def door_twin(env) -> mujoco.MjModel:
    """The door env's scene (environments/door.build_door_scene) with the electronics theme's
    tester restyle and lights (simulation/electronics.py) and a light ESD-grey floor and sky
    (render only)."""
    from polypanda.environments import door as dm
    from polypanda.simulation import electronics
    cfg, kinds = env.config.scene, env.config.variation.kinds
    spec = dm.panda_spec()
    dm.set_finger_stiffness(spec, cfg.finger_stiffness)
    a = np.radians(cfg.station_angle)
    names = []
    for kind in dm.DOOR_KINDS:
        if kind in kinds:
            xy = (cfg.radius * np.cos(a), cfg.radius * np.sin(a))
            dm.add_machine(spec, dm.MACHINES[kind], xy, cfg.station_angle,
                           dm.MachineConfig(kind=kind))
            names.append(dm.MACHINES[kind])
    electronics._testers(spec, names)
    electronics._lights(spec)
    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.rgb1, tex.rgb2 = [0.9, 0.92, 0.95], [0.7, 0.73, 0.78]
        elif tex.builtin == mujoco.mjtBuiltin.mjBUILTIN_CHECKER:
            tex.rgb1, tex.rgb2 = [0.8, 0.81, 0.82], [0.75, 0.76, 0.78]
            tex.markrgb = [0.62, 0.64, 0.67]
    return spec.compile()


def mirror_layout(src: mujoco.MjModel, dst: mujoco.MjModel, geoms) -> None:
    """Give the twin `dst` the simulated model's episode layout, by name: every body's placement,
    every joint's axis and range, and the geoms `geoms` (a re-cut nest, a door's handle)."""
    for b in range(1, src.nbody):
        k = _id(dst, mujoco.mjtObj.mjOBJ_BODY, src.body(b).name)
        if k >= 0:
            dst.body_pos[k], dst.body_quat[k] = src.body_pos[b], src.body_quat[b]
    for j in range(src.njnt):
        k = _id(dst, mujoco.mjtObj.mjOBJ_JOINT, src.joint(j).name)
        if k >= 0:
            dst.jnt_axis[k], dst.jnt_range[k] = src.jnt_axis[j], src.jnt_range[j]
    for name in geoms:
        g, k = src.geom(name).id, dst.geom(name).id
        for f in ("geom_pos", "geom_quat", "geom_size", "geom_rbound", "geom_aabb"):
            getattr(dst, f)[k] = getattr(src, f)[g]


_QSIZE = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4,
          mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}


def state_map(src: mujoco.MjModel, dst: mujoco.MjModel) -> tuple:
    """Index arrays carrying a simulated qpos (and mocap positions) onto the twin, joint by joint
    and body by body, by name; every simulated joint must be in the twin."""
    si, di = [], []
    for j in range(src.njnt):
        k = _id(dst, mujoco.mjtObj.mjOBJ_JOINT, src.joint(j).name)
        if k < 0:
            raise ValueError(f"joint {j} ({src.joint(j).name!r}) is not in the twin")
        n = _QSIZE[mujoco.mjtJoint(src.jnt_type[j])]
        si += range(src.jnt_qposadr[j], src.jnt_qposadr[j] + n)
        di += range(dst.jnt_qposadr[k], dst.jnt_qposadr[k] + n)
    if dst.njnt != src.njnt:
        raise ValueError(f"the twin has {dst.njnt} joints, the simulation {src.njnt}")
    ms, md = [], []
    for b in range(src.nbody):
        if src.body_mocapid[b] >= 0:
            k = _id(dst, mujoco.mjtObj.mjOBJ_BODY, src.body(b).name)
            if k >= 0 and dst.body_mocapid[k] >= 0:
                ms.append(src.body_mocapid[b])
                md.append(dst.body_mocapid[k])
    return np.array(si), np.array(di), np.array(ms, int), np.array(md, int)


class View:
    """A renderer on a (twin) model, drawing recorded states of `src` from its own MjData."""

    def __init__(self, model: mujoco.MjModel, src: mujoco.MjModel, width: int,
                 height: int) -> None:
        self.model, self.data = model, mujoco.MjData(model)
        self.map = state_map(src, model)
        self.renderer = mujoco.Renderer(model, height, width)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[HIDDEN_GROUP] = 0

    def set(self, track: Track, i: int) -> None:
        m, d = self.model, self.data
        si, di, ms, md = self.map
        d.qpos[di] = track.qpos[i][si]
        if len(ms):
            d.mocap_pos[md] = track.mocap[i][ms]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)

    def frame(self, track: Track, i: int, cam: mujoco.MjvCamera) -> np.ndarray:
        self.set(track, i)
        self.renderer.update_scene(self.data, cam, scene_option=self.option)
        return self.renderer.render().copy()


def cutaway(model: mujoco.MjModel, data: mujoco.MjData, tester: str, origin: np.ndarray,
            toward: np.ndarray) -> None:
    """Render only: the tester's box walls, roof and dressing on the camera's side of the nest
    are not drawn (group 5), and the nest's walls on that side are drawn semi-transparent.
    `toward`: the horizontal unit vector from the nest to the camera; `data` posed."""
    m, d = model, data
    root = m.body(tester).id
    pockets = {m.body(f"{tester}_pocket{k}").id for k in range(4)}
    door = _id(m, mujoco.mjtObj.mjOBJ_BODY, f"{tester}_lid")
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if m.body_rootid[b] != root:
            continue
        s = float((d.geom_xpos[g][:2] - origin[:2]) @ toward)
        if b in pockets:
            if s > 0.006:
                m.geom_matid[g] = -1
                m.geom_rgba[g, 3] = 0.3
        elif b != door and s > 0.03:
            m.geom_group[g] = HIDDEN_GROUP


def camera(lookat, azimuth: float, elevation: float, distance: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    return cam


# --- overlays -----------------------------------------------------------------------------------
def text_box(draw, xy, text, f, fill=WHITE, bg=(0, 0, 0, 150), anchor="la", pad=5) -> tuple:
    box = draw.textbbox(xy, text, font=f, anchor=anchor)
    draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=bg)
    draw.text(xy, text, font=f, fill=fill, anchor=anchor)
    return box


def force_bar(draw, x0, y0, w, h, f, now: float, peak: float, s: float) -> None:
    """The nest force: a bar to BAR_MAX N, the 50 N damage limit as a red line, the peak so far."""
    draw.rectangle((x0 - 8 * s, y0 - 30 * s, x0 + w + 8 * s, y0 + h + 8 * s), fill=(0, 0, 0, 160))
    draw.text((x0, y0 - 26 * s), "nest force", font=f, fill=(220, 220, 220))
    over = peak > FORCE_LIMIT
    draw.text((x0 + w, y0 - 26 * s), f"{now:4.0f} N", font=f,
              fill=(255, 90, 90) if now > FORCE_LIMIT else WHITE, anchor="ra")
    draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(60, 60, 60, 220))
    fill = (220, 50, 50) if now > FORCE_LIMIT else (90, 200, 120) if now < 20 else AMBER
    draw.rectangle((x0, y0, x0 + w * min(now, BAR_MAX) / BAR_MAX, y0 + h), fill=fill)
    px = x0 + w * min(peak, BAR_MAX) / BAR_MAX
    draw.line((px, y0 - 3 * s, px, y0 + h + 3 * s), fill=(255, 120, 120) if over else WHITE,
              width=max(1, round(2 * s)))
    lx = x0 + w * FORCE_LIMIT / BAR_MAX
    draw.line((lx, y0 - 6 * s, lx, y0 + h + 6 * s), fill=(255, 40, 40), width=max(2, round(3 * s)))


def top_view(draw, x0, y0, size, f, track: Track, i: int, s: float) -> None:
    """The module's centre seen from above, relative to the true nest (mm): the square it fits
    in (the clearance), the camera's estimate of the nest's centre, the path so far."""
    half = INSET_MM
    draw.rectangle((x0, y0, x0 + size, y0 + size), fill=(0, 0, 0, 170))

    def px(p):
        return (x0 + size / 2 + p[0] / half * size / 2, y0 + size / 2 - p[1] / half * size / 2)

    c = track.info["clearance_mm"]
    a, b = px((-c, -c)), px((c, c))
    draw.rectangle((a[0], b[1], b[0], a[1]), outline=(90, 200, 120), width=max(1, round(2 * s)))
    e = px(track.info["belief_mm"])
    r = 6 * s
    draw.line((e[0] - r, e[1], e[0] + r, e[1]), fill=(255, 210, 60), width=max(1, round(2 * s)))
    draw.line((e[0], e[1] - r, e[0], e[1] + r), fill=(255, 210, 60), width=max(1, round(2 * s)))
    # the path from PATH_MM over the rim (above that the module is only on its way down)
    near = np.flatnonzero(track.z[:i + 1] < PATH_MM)
    if len(near) > 1:
        pts = [px(np.clip(p, -half, half)) for p in track.xy[near[0]:i + 1:4]]
        if len(pts) > 1:
            draw.line(pts, fill=(120, 190, 255), width=max(1, round(1.5 * s)))
    p = px(np.clip(track.xy[i], -half, half))
    draw.ellipse((p[0] - 4 * s, p[1] - 4 * s, p[0] + 4 * s, p[1] + 4 * s), fill=WHITE)
    draw.text((x0 + 4 * s, y0 + 3 * s), f"top view, ±{half:g} mm", font=f, fill=(200, 200, 200))
    draw.text((x0 + 4 * s, y0 + size - 3 * s), "+ camera's nest", font=f, fill=(255, 210, 60),
              anchor="ld")
    draw.text((x0 + size - 4 * s, y0 + size - 3 * s), "□ fits", font=f, fill=(90, 200, 120),
              anchor="rd")


def verdict(actor: str, track: Track) -> tuple[str, str, tuple]:
    o = track.outcome
    peak = max(o.get("peak_force", 0.0), float(track.force.max()))
    if o["success"]:
        return (f"SEATED {o['time_s']:.2f} s", f"released, hand back up; peak {peak:.0f} N",
                GREEN)
    if o["failure"] == "damage":
        return (f"DAMAGED {peak:.0f} N", ("seated, but " if o.get("seated") else "")
                + f"over the {FORCE_LIMIT:.0f} N limit", RED)
    return f"FAILED: {o['failure']}", f"peak {peak:.0f} N", (90, 90, 90)


def pane(img: np.ndarray, actor: str, track: Track, t: float, t_prev: float, s: float
         ) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    i = track.at(t)
    f_label, f_small = font(round(22 * s)), font(round(15 * s))
    # the label, coloured by actor
    draw.rectangle((0, 0, w, 40 * s), fill=(0, 0, 0, 170))
    draw.rectangle((0, 40 * s - 4 * s, w, 40 * s), fill=ACTOR_COLOUR[actor])
    draw.text((w / 2, 18 * s), ACTOR_LABEL[actor], font=f_label, fill=WHITE, anchor="mm")
    # the clock (simulated time), held at the episode's end
    tt = min(t, track.end)
    text_box(draw, (12 * s, 52 * s), f"t = {tt:5.2f} s", font(round(20 * s)))
    # the top view
    size = round(min(w * 0.46, 200 * s))
    top_view(draw, w - size - 10 * s, 50 * s, size, font(round(11 * s)), track, i, s)
    # the force bar
    now = track.peak(t_prev, t) if t <= track.end else 0.0
    peak = track.peak(-1.0, min(t, track.end))
    force_bar(draw, 16 * s, h - 34 * s, w - 32 * s, 16 * s, f_small, now, peak, s)
    # the outcome; before it, the moments that lead to it
    y = h - 125 * s
    if t >= track.end:
        big, small, colour = verdict(actor, track)
        fb = font(round(28 * s))
        bb = draw.textbbox((w / 2, y), big, font=fb, anchor="mm")
        draw.rectangle((bb[0] - 14 * s, bb[1] - 10 * s, bb[2] + 14 * s, bb[3] + 30 * s),
                       fill=(*colour, 225))
        draw.text((w / 2, y), big, font=fb, fill=WHITE, anchor="mm")
        draw.text((w / 2, bb[3] + 14 * s), small, font=f_small, fill=WHITE, anchor="mm")
        return np.asarray(im)
    t_in = track.info.get("t_in")
    if peak > FORCE_LIMIT:
        text_box(draw, (w / 2, y), f"OVER {FORCE_LIMIT:.0f} N: {peak:.0f} N",
                 font(round(24 * s)), bg=(*RED, 220), anchor="mm", pad=round(8 * s))
        y -= 46 * s
    if t_in is not None and t >= t_in:
        text_box(draw, (w / 2, y), f"down in the nest at {t_in:.2f} s", font(round(18 * s)),
                 bg=(20, 90, 45, 210), anchor="mm", pad=round(6 * s))
    return np.asarray(im)


def strip(width: int, height: int, left: str, right: str, s: float, size: int = 20
          ) -> np.ndarray:
    im = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(im)
    f = font(round(size * s))
    draw.text((16 * s, height / 2), left, font=f, fill=(235, 235, 235), anchor="lm")
    if right:
        draw.text((width - 16 * s, height / 2), right, font=f, fill=AMBER, anchor="rm")
    return np.asarray(im)


def wrap(draw, text: str, f, width: float) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        test = f"{line} {word}".strip()
        if draw.textlength(test, font=f) <= width:
            line = test
        else:
            lines.append(line)
            line = word
    return lines + [line]


def title_overlay(img: np.ndarray, alpha: float, s: float) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    draw.rectangle((0, h * 0.80, w, h * 0.98), fill=(0, 0, 0, round(170 * alpha)))
    a = round(255 * alpha)
    draw.text((w / 2, h * 0.86), TITLE, font=font(round(40 * s)), fill=(255, 255, 255, a),
              anchor="mm")
    draw.text((w / 2, h * 0.935), "PolyPanda v4.1 · a Franka Panda in simulation · the nest, "
              "cut away", font=font(round(20 * s)), fill=(200, 200, 200, a), anchor="mm")
    return np.asarray(im)


def door_overlay(img: np.ndarray, track: Track, t: float, feel: dict, s: float) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    i = track.at(t)
    text_box(draw, (16 * s, 16 * s), f"t = {min(t, track.end):4.2f} s", font(round(24 * s)))
    text_box(draw, (16 * s, 58 * s), f"drawer open {track.extra[i] * 1000:5.0f} mm",
             font(round(20 * s)))
    if t >= track.end:
        o = track.outcome
        big = f"DRAWER OPEN  {track.end:.2f} s"
        fb = font(round(34 * s))
        bb = draw.textbbox((w / 2, h * 0.80), big, font=fb, anchor="mm")
        draw.rectangle((bb[0] - 16 * s, bb[1] - 10 * s, bb[2] + 16 * s, bb[3] + 36 * s),
                       fill=(*GREEN, 225))
        draw.text((w / 2, h * 0.80), big, font=fb, fill=WHITE, anchor="mm")
        draw.text((w / 2, bb[3] + 18 * s),
                  f"{o['steps']} steps, peak {o['peak_force']:.0f} N; the feel_v2 script on "
                  f"this drawer: {feel['time_s']:.2f} s ({feel['steps']} steps)",
                  font=font(round(17 * s)), fill=WHITE, anchor="mm")
    return np.asarray(im)


def end_card(width: int, height: int, s: float) -> np.ndarray:
    """The one-shot results (CARD): per skill a heading, its headline rate large, the sentence."""
    im = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(im)
    fh, fn, fb = font(round(40 * s)), font(round(120 * s)), font(round(36 * s))
    fs = font(round(22 * s))
    x, y = 150 * s, 200 * s
    xb = x + 470 * s                      # the sentence's column
    colw = width - xb - 130 * s
    for (head, body), colour in zip(CARD, ((70, 170, 255), (90, 200, 120)), strict=True):
        draw.rectangle((x - 30 * s, y + 6 * s, x - 18 * s, y + 46 * s), fill=colour)
        draw.text((x, y), head, font=fh, fill=colour)
        y += 70 * s
        draw.text((x - 6 * s, y - 14 * s), body.split(" on ")[0], font=fn, fill=colour)
        yy = y + 6 * s
        for line in wrap(draw, body, fb, colw):
            draw.text((xb, yy), line, font=fb, fill=(235, 235, 235))
            yy += 50 * s
        y = max(y + 150 * s, yy) + 60 * s
    for k, line in enumerate(wrap(draw, SMALL_PRINT, fs, width - 2 * x)):
        draw.text((x, height - 90 * s + 30 * s * k), line, font=fs, fill=(150, 150, 150))
    return np.asarray(im)


def fade(frame: np.ndarray, k: int, n: int, edge: int) -> np.ndarray:
    """Fade in over the first `edge` and out over the last `edge` of n frames."""
    a = min(1.0, (k + 1) / edge, (n - k) / edge)
    return frame if a >= 1.0 else (frame.astype(np.float32) * a).astype(np.uint8)


# --- main ---------------------------------------------------------------------------------------
def parse_speeds(text: str) -> list[tuple[float, float]]:
    """"0:1,5.5:2": from simulated time 0 s at 1x, from 5.5 s at 2x."""
    out = sorted((float(a), float(b)) for a, b in (p.split(":") for p in text.split(",")))
    if not out or out[0][0] > 0.0:
        raise ValueError("the speed profile must start at 0")
    return out


def speed_at(profile, t: float) -> float:
    return [v for t0, v in profile if t0 <= t + 1e-9][-1]


def speed_label(speed: float) -> str:
    if speed == 1:
        return "real time (1x)"
    return f"{speed:g}x slow motion" if speed < 1 else f"{speed:g}x speed"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.split("\n")[1:]))
    p.add_argument("--insertion-set", type=Path, default=Path("eval_sets/insertion_dev_v1.json"))
    p.add_argument("--door-set", type=Path, default=Path("eval_sets/door_dev_v2.json"))
    p.add_argument("--learned", default="runs/insertion_ppo_r2_seed1")
    p.add_argument("--door-run", default="runs/door_v3_ppo_r3_seed1")
    p.add_argument("--insertion-results", type=Path, default=Path("results/v4/insertion"))
    p.add_argument("--door-results", type=Path, default=Path("results/v4/door"))
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--title-s", type=float, default=3.0)
    p.add_argument("--split-s", type=float, default=10.0)
    p.add_argument("--door-s", type=float, default=7.0)
    p.add_argument("--card-s", type=float, default=5.0)
    p.add_argument("--split-speed", default="0:2,1.3:0.5,3.5:1,5.5:3",
                   help="playback speed from each simulated time (s): 'time:speed,...'; the "
                        "default plays the descent at 2x, the contacts at 0.5x, the search's "
                        "push at 1x and the retracts at 3x, and the header states each")
    p.add_argument("--door-speed", type=float, default=0.5)
    p.add_argument("--pane-camera", type=float, nargs=4, default=[40.0, -22.0, 0.55, 0.05],
                   metavar=("AZ_REL", "ELEVATION", "DISTANCE", "LOOKAT_Z"),
                   help="azimuth relative to the tester's side, degrees; lookat height over "
                        "the nest's floor, m")
    p.add_argument("--pane-fovy", type=float, default=25.0,
                   help="the insertion views' vertical field of view (deg)")
    p.add_argument("--door-camera", type=float, nargs=5,
                   default=[-65.0, -20.0, 0.5, 0.75, 0.13],
                   metavar=("AZIMUTH", "ELEVATION", "DIST_START", "DIST_END", "LOOKAT_Z"))
    p.add_argument("--out", type=Path, default=Path("media/v41_short"))
    p.add_argument("--gif", action="store_true", help="also write <out>.gif")
    p.add_argument("--gif-width", type=int, default=480)
    p.add_argument("--gif-fps", type=int, default=10)
    p.add_argument("--stills", type=float, nargs="*", default=None,
                   help="only write split-screen stills at these simulated times (s)")
    args = p.parse_args()
    for path in (args.insertion_set, args.door_set):
        if "_final_" in path.name or "stretch" in path.name:
            p.error("one-shot sets are not for videos; use a development set")
    torch.set_num_threads(1)
    wall = time.perf_counter()
    W, H, fps = args.width, args.height, args.fps
    s = H / 1080.0                       # overlay scale

    # --- the episodes and the simulations ---
    ins = json.loads(args.insertion_set.read_text())
    learned_name = args.learned.rstrip("/").replace("/", "_")
    index, rows = choose_insertion(args.insertion_results, learned_name, ins["name"])
    ep = ins["episodes"][index]
    actors = ("straight_down", "search", "learned")
    envs, tracks = {}, {}
    for actor in actors:
        envs[actor], tracks[actor] = run_insertion(actor, ep, index, args.learned)
        check(actor, tracks[actor].outcome, rows[actor],
              ("success", "failure", "time_s", "peak_force"))
    door_set = json.loads(args.door_set.read_text())
    door_name = f"{args.door_run.rstrip('/').replace('/', '_')}_final"
    d_index, d_row = choose_door(args.door_results, door_name, door_set["name"])
    d_ep = door_set["episodes"][d_index]
    door_env, door_track = run_door(d_ep, d_index, door_set["episodes"], args.door_run)
    check("learned door", door_track.outcome, d_row, ("success", "steps", "peak_force", "slams"))
    feel = run_feel(d_ep, d_index, door_set["episodes"])
    feel_rows = _rows(args.door_results / f"feel_v2_{door_set['name']}.json")
    check("feel_v2 door", feel, feel_rows[d_index], ("success", "steps"))
    print(f"simulated in {time.perf_counter() - wall:.0f} s", flush=True)

    # --- the views (render-only changes from here on) ---
    header_h = round(56 * s)
    footer_h = round(34 * s)
    pw, ph = W // 3, H - header_h - footer_h
    env0 = envs["learned"]
    origin, R = env0.nest_frame()
    side = R[:, 1]                       # the tester's side (its +y), horizontal
    az_rel, elev, dist, look_z = args.pane_camera
    a = np.radians(az_rel)
    toward = np.cos(a) * side[:2] + np.sin(a) * R[:2, 0]
    toward /= np.linalg.norm(toward)
    azimuth = float(np.degrees(np.arctan2(-toward[1], -toward[0])))
    look = origin + [0.0, 0.0, look_z]
    pane_cam = camera(look, azimuth, elev, dist)
    # one themed twin for the three panes and the title (the same episode's layout), posed at
    # the episode's start to cut away
    twin = insertion_twin(env0)
    nest = [env0.model.geom(g).name for g in range(env0.model.ngeom)
            if f"{env0.tester}_pocket" in env0.model.body(env0.model.geom_bodyid[g]).name]
    mirror_layout(env0.model, twin, nest)
    prepare_render(twin, W, H, args.pane_fovy)
    title_view = View(twin, env0.model, W, H)
    title_view.set(tracks["learned"], 0)
    cutaway(twin, title_view.data, env0.tester, origin, toward)
    views = {a: View(twin, envs[a].model, pw, ph) for a in actors}
    dtwin = door_twin(door_env)
    handles = [door_env.model.geom(g).name for g in range(door_env.model.ngeom)
               if "handle" in door_env.model.geom(g).name]
    mirror_layout(door_env.model, dtwin, handles)
    prepare_render(dtwin, W, H)
    door_view = View(dtwin, door_env.model, W, H)
    print(f"render twins: insertion {twin.ngeom} geoms (simulated {env0.model.ngeom}), door "
          f"{dtwin.ngeom} ({door_env.model.ngeom})", flush=True)
    info = tracks["learned"].info
    split_left = (f"The same insertion three ways: insertion_dev_v1 #{index}, "
                  f"{info['kind']} tester, {info['clearance_mm']:.2f} mm clearance, camera off "
                  f"by {info['camera_mm']:.2f} mm")
    footer = strip(W, footer_h, "Nest force checked every 2 ms (damage limit 50 N). Learned: "
                   "its own compliant controller; search: 20 Hz force regulation.", "", s, 15)
    t_last = max(tr.end for tr in tracks.values())

    def split_frame(t: float, t_prev: float, speed: float) -> np.ndarray:
        row = [pane(views[a].frame(tracks[a], tracks[a].at(t), pane_cam), a, tracks[a], t,
                    t_prev, s) for a in actors]
        body = np.hstack(row)
        if body.shape[1] < W:
            body = np.pad(body, ((0, 0), (0, W - body.shape[1]), (0, 0)))
        top = strip(W, header_h, split_left, speed_label(speed) if t <= t_last else "", s)
        return np.vstack([top, body, footer])

    def title_frame(k: int, n: int) -> np.ndarray:
        """The start of the episode, the camera slowly turning and closing in."""
        u = k / max(1, n - 1)
        cam = camera(origin + [0.0, 0.0, 0.10], azimuth - 10.0 + 10.0 * u, -20.0,
                     1.15 - 0.15 * u)
        img = title_view.frame(tracks["learned"], 0, cam)
        return title_overlay(img, min(1.0, k / (0.4 * fps)), s)

    az_d, el_d, dist0, dist1, look_dz = args.door_camera
    handle = door_track.info["handle"]
    door_left = (f"And the door: the learned door (pure RL) opens a drawer from the "
                 f"development set (door_dev_v2 #{d_index})")

    def door_frame(k: int, n: int) -> np.ndarray:
        """The drawer opening in slow motion, the camera pulling back."""
        t = k * args.door_speed / fps
        u = min(1.0, k / max(1, n - 1))
        cam = camera([handle[0] - 0.02, handle[1], look_dz], az_d, el_d,
                     dist0 + (dist1 - dist0) * u)
        img = door_view.frame(door_track, door_track.at(t), cam)
        img = door_overlay(img[:H - header_h], door_track, t, feel, s)
        top = strip(W, header_h, door_left, speed_label(args.door_speed)
                    if t <= door_track.end else "", s)
        return np.vstack([top, img])

    if args.stills is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        stem = args.out.parent / args.out.name
        for t in args.stills:
            Image.fromarray(split_frame(t, max(0.0, t - 1 / fps), 1.0)).save(
                f"{stem}_split_{t:05.2f}.png")
        n = round(args.door_s * fps)
        for k in (0, n // 3, n - 1):
            Image.fromarray(door_frame(k, n)).save(f"{stem}_door_{k:03d}.png")
        n = round(args.title_s * fps)
        Image.fromarray(title_frame(n - 1, n)).save(f"{stem}_title.png")
        Image.fromarray(end_card(W, H, s)).save(f"{stem}_card.png")
        print(f"wrote stills {stem}_*.png")
        return

    # --- the video ---
    out = args.out.with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    gif_frames: list[np.ndarray] = []
    gif_every = max(1, round(fps / args.gif_fps))
    count = {"n": 0}
    edge = max(1, round(0.25 * fps))

    def emit(frame: np.ndarray) -> None:
        writer.append_data(frame)
        if args.gif and count["n"] % gif_every == 0:
            gh = round(H * args.gif_width / W)
            gif_frames.append(np.asarray(Image.fromarray(frame).resize((args.gif_width, gh),
                                                                       Image.LANCZOS)))
        count["n"] += 1

    # 1. the title over the nest
    n = round(args.title_s * fps)
    for k in range(n):
        emit(fade(title_frame(k, n), k, n, edge))
    # 2. the split
    profile = parse_speeds(args.split_speed)
    n = round(args.split_s * fps)
    t, t_prev = 0.0, 0.0
    for k in range(n):
        speed = speed_at(profile, t)
        emit(fade(split_frame(t, t_prev, speed), k, n, edge))
        t_prev, t = t, t + speed / fps
    print(f"split: played to {t_prev:.2f} s of simulated time (the longest episode "
          f"{max(tr.end for tr in tracks.values()):.2f} s)", flush=True)
    # 3. the door, pulling back
    n = round(args.door_s * fps)
    for k in range(n):
        emit(fade(door_frame(k, n), k, n, edge))
    # 4. the end card
    n = round(args.card_s * fps)
    card = end_card(W, H, s)
    for k in range(n):
        emit(fade(card, k, n, edge))
    writer.close()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {count['n']} frames) in "
          f"{time.perf_counter() - wall:.0f} s", flush=True)
    if args.gif:
        gif = args.out.with_suffix(".gif")
        imageio.mimwrite(gif, gif_frames, duration=1000 / args.gif_fps, loop=0)
        print(f"wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames)")


if __name__ == "__main__":
    main()
