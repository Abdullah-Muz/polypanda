"""v4 sensing: how accurately does a simple simulated depth camera locate the cubes?

See docs/pose_error.md.

A measurement, not a skill evaluation and not a target. Test scenes are drawn from their own
seed: upright cubes on the towers (1-4 levels), the buffer slots, the belt's pick point and the
open bench, sometimes a second cube 5-8 cm from another, every cube scaled 0.85-1.15; the arm at
home or held over a station (partly or fully hiding it). Each scene is rendered (depth, plus a
segmentation render used only to say which cluster is which cube) by three cameras:

- overhead: fixed, 1.5 m above the bench, sees the whole work area;
- station_b: fixed near the machine at tower B, 0.6 m away;
- hover_b: 0.55 m straight above tower B, standing in for a wrist camera paused above the
  station (arm-at-home scenes only).

The depth image gets noise at several levels (perception/camera.py: NOISE) and the classical
estimator (perception/estimator.py) returns each visible stack's top cube: xy, yaw mod 90 deg,
top height. "medium+calib" adds an extrinsic calibration error (the estimator believes a camera
pose off by 1 mm and 0.05 deg, a new random error per scene) to the medium noise.

Usage:
  uv run python scripts/measure_pose_error.py                       # 400 scenes, 4 processes
  uv run python scripts/measure_pose_error.py --n 10 --out-dir /tmp/pe  # smoke run
Writes <out-dir>/pose_error.json (per-condition statistics, provenance) and pose_error_rows.csv
(one row per cube, camera and noise condition).
"""

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from polypanda.control.ik import PoseIK, top_down_quat
from polypanda.perception.camera import NOISE, SETUPS, add_camera, camera_pose
from polypanda.perception.estimator import EstimatorConfig, estimate_cubes, identify, wrap_quarter
from polypanda.simulation.assets import menagerie_revision
from polypanda.simulation.panda import PandaIndices, reset_to_home
from polypanda.simulation.randomization import CubePhysics
from polypanda.simulation.scenes import CubeIndices, cube_yaw
from polypanda.simulation.workcell import (
    CUBE_NAME,
    WorkcellConfig,
    _yaw_quat,
    belt_direction,
    build_workcell_spec,
    hopper_xy,
    station_xy,
)

SEED = 20261123
N_CUBES = 24                   # cube bodies in the model: 3 towers x 4, 3 buffers, belt, 4 loose
CONDITIONS = ("none", "low", "medium", "high", "medium+calib")
CALIB = {"translation_m": 0.001, "rotation_deg": 0.05}
SIZE_RANGE = (0.85, 1.15)
LOOSE_RADIUS = (0.30, 0.75)    # m from the base axis (disturbances.REACH, clear of the signs)
BASE_RANGE = 162.0             # deg either way (disturbances.BASE_RANGE)
PAIR_GAP = (0.05, 0.08)        # m between the centres of a close pair
P_ARM_OVER = 0.4
MIN_VISIBLE_PX = 25            # a cube is visible to a camera if it shows at least this many px
OCCLUDED = 0.05                # arm hides more than this share of the cube's pixels: occluded
NEIGHBOUR = 0.09               # m: another stack's centre closer than this is a close neighbour
THRESHOLDS_MM = (1, 2, 3, 4)
CFG = WorkcellConfig(dressing=False, n_cubes=N_CUBES)


# --- scenes -----------------------------------------------------------------------------------
def _corners(xy, half, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(xy) + (np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * half) @ np.array(
        [[c, -s], [s, c]]).T


def squares_overlap(a, b, margin=0.003) -> bool:
    """Separating-axis test for two squares (xy, half side, yaw), with a clearance margin."""
    pa, pb = _corners(*a), _corners(*b)
    for yaw in (a[2], b[2]):
        for ax in (np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])):
            qa, qb = pa @ ax, pb @ ax
            if qa.max() + margin < qb.min() or qb.max() + margin < qa.min():
                return False
    return True


def _clear_of_stations(xy) -> bool:
    if any(np.linalg.norm(xy - station_xy(a, CFG.radius)) < 0.09 for a in CFG.tower_angles):
        return False
    if any(np.linalg.norm(xy - station_xy(a, CFG.radius)) < 0.08 for a in CFG.buffer_angles):
        return False
    if np.linalg.norm(xy - station_xy(CFG.reject_angle, CFG.radius)) < 0.17:
        return False
    return not (xy[1] > 0.36 and xy[0] < 0.22)   # the belt, its tunnel and the hopper


def draw_scene(rng: np.random.Generator) -> dict:
    """Stacks of upright cubes (bottom first) and the arm's pose, from one scene's stream."""
    half = 0.02
    stacks = []   # each: kind, cubes [(xy, yaw, size)], footprint for overlap tests

    def add(kind, cubes):
        stacks.append({"kind": kind, "cubes": cubes})

    def fits(xy, yaw, size):
        return all(not squares_overlap((xy, half * size, yaw), (s["cubes"][0][0],
                   half * s["cubes"][0][2], s["cubes"][0][1])) for s in stacks)

    for t, angle in enumerate(CFG.tower_angles):
        h = int(rng.integers(0, 5))
        base = station_xy(angle, CFG.radius)
        cubes = [(base + rng.uniform(-0.002, 0.002, 2),
                  np.radians(angle + rng.uniform(-5, 5)), rng.uniform(*SIZE_RANGE))
                 for _ in range(h)]
        if cubes:
            add(f"tower{'ABC'[t]}", cubes)
    for angle in CFG.buffer_angles:
        if rng.random() < 0.5:
            add("buffer", [(station_xy(angle, CFG.radius) + rng.uniform(-0.01, 0.01, 2),
                            rng.uniform(-np.pi, np.pi), rng.uniform(*SIZE_RANGE))])
    if rng.random() < 0.5:
        travel = belt_direction(CFG)
        normal = np.array([-travel[1], travel[0]])
        xy = station_xy(CFG.infeed_angle, CFG.radius) + rng.uniform(-0.015, 0.015) * normal
        add("belt", [(xy, np.arctan2(travel[1], travel[0]) + np.radians(rng.uniform(-15, 15)),
                      rng.uniform(*SIZE_RANGE))])
    for _ in range(int(rng.integers(1, 4))):
        for _try in range(50):
            r, a = rng.uniform(*LOOSE_RADIUS), np.radians(rng.uniform(-BASE_RANGE, BASE_RANGE))
            xy, yaw, size = r * np.array([np.cos(a), np.sin(a)]), rng.uniform(-np.pi, np.pi), \
                rng.uniform(*SIZE_RANGE)
            if _clear_of_stations(xy) and fits(xy, yaw, size):
                add("loose", [(xy, yaw, size)])
                break
    if rng.random() < 0.5:   # a close pair: a cube 5-8 cm from a cube on the bench
        anchors = [s for s in stacks if s["kind"] != "belt"]
        for _try in range(50):
            anchor = anchors[int(rng.integers(len(anchors)))]["cubes"][0][0]
            a, d = rng.uniform(-np.pi, np.pi), rng.uniform(*PAIR_GAP)
            xy = anchor + d * np.array([np.cos(a), np.sin(a)])
            yaw, size = rng.uniform(-np.pi, np.pi), rng.uniform(*SIZE_RANGE)
            radius, angle = np.linalg.norm(xy), np.degrees(np.arctan2(xy[1], xy[0]))
            if (LOOSE_RADIUS[0] <= radius <= 0.78 and abs(angle) <= BASE_RANGE
                    and not (xy[1] > 0.36 and xy[0] < 0.22) and fits(xy, yaw, size)):
                add("pair", [(xy, yaw, size)])
                break
    arm = None
    if rng.random() < P_ARM_OVER:   # the hand over one of the stacks, at some offset and height
        s = stacks[int(rng.integers(len(stacks)))]
        offset = rng.uniform(0, 0.08) * np.array([np.cos(a := rng.uniform(-np.pi, np.pi)),
                                                  np.sin(a)])
        top = sum(0.06 * c[2] for c in s["cubes"])   # nominal cube height 6 cm x size
        arm = {"xy": s["cubes"][0][0] + offset, "z": top + rng.uniform(0.04, 0.20),
               "yaw": rng.uniform(-np.pi / 2, np.pi / 2)}
    return {"stacks": stacks, "arm": arm}


# --- worker -----------------------------------------------------------------------------------
_W: dict = {}


def _init_worker() -> None:
    spec = build_workcell_spec(CFG)
    for setup in SETUPS.values():
        add_camera(spec, setup)
    model = spec.compile()
    data = mujoco.MjData(model)
    cubes = [CubeIndices.from_model(model, CUBE_NAME.format(k)) for k in range(N_CUBES)]
    _W.update(model=model, data=data, cubes=cubes,
              physics=[CubePhysics(model, c) for c in cubes], ik=PoseIK(),
              panda=PandaIndices.from_model(model),
              renderers={n: mujoco.Renderer(model, s.height, s.width) for n, s in SETUPS.items()},
              rays={}, estimator=EstimatorConfig())
    noarm = mujoco.MjvOption()
    noarm.geomgroup[2] = 0          # the Panda's visual meshes (everything else is group 0)
    _W["noarm"] = noarm


def _render(name: str, mode: str, option=None) -> np.ndarray:
    r = _W["renderers"][name]
    r.disable_depth_rendering()
    r.disable_segmentation_rendering()
    if mode == "depth":
        r.enable_depth_rendering()
    else:
        r.enable_segmentation_rendering()
    r.update_scene(_W["data"], camera=name, scene_option=option)
    out = r.render().copy()
    if mode == "seg":
        geom = np.where(out[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), out[..., 0], -1)
        return geom
    return out


def _stage(scene: dict) -> tuple[list[dict], bool]:
    """Write the scene into the model and data; returns the stacks' ground truth."""
    m, d, cubes = _W["model"], _W["data"], _W["cubes"]
    reset_to_home(m, d, _W["panda"])
    for c, spot, phys in zip(cubes, hopper_xy(CFG), _W["physics"], strict=True):
        phys.apply(size_scale=1.0)
        d.qpos[c.qpos:c.qpos + 7] = [*spot, c.half_size[2], 1, 0, 0, 0]
    used, truth = iter(range(N_CUBES)), []
    for s_id, s in enumerate(scene["stacks"]):
        z = CFG.belt_lift if s["kind"] == "belt" else 0.0
        geoms = []
        for xy, yaw, size in s["cubes"]:
            k = next(used)
            c = cubes[k]
            _W["physics"][k].apply(size_scale=size)
            d.qpos[c.qpos:c.qpos + 7] = [*xy, z + c.half_size[2], *_yaw_quat(yaw)]
            z += 2 * c.half_size[2]
            geoms.append(c.geom)
        truth.append({"stack": s_id, "kind": s["kind"], "level": len(s["cubes"]), "size": size,
                      "cube": k, "geoms": geoms})
    arm_over = False
    if scene["arm"] is not None:
        a = scene["arm"]
        ik = _W["ik"]
        q0 = m.key("home").qpos[_W["panda"].arm_qpos].copy()
        q0[0] = np.arctan2(a["xy"][1], a["xy"][0])
        q, ok = ik.solve(q0, [*a["xy"], a["z"]], top_down_quat(a["yaw"]))
        if ok:
            d.qpos[_W["panda"].arm_qpos] = q
            arm_over = True
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    for t in truth:
        c = cubes[t["cube"]]
        p = d.xpos[c.body]
        t.update(xy=p[:2].copy(), yaw=cube_yaw(d, c), top=float(p[2] + c.half_size[2]))
    return truth, arm_over


def _expected_depth(name: str) -> np.ndarray:
    """The known scene from this camera: cell and robot, every cube moved out of sight."""
    m, d = _W["model"], _W["data"]
    saved = d.qpos.copy()
    for c in _W["cubes"]:
        d.qpos[c.qpos + 2] = -1.0
    mujoco.mj_kinematics(m, d)
    depth = _render(name, "depth")
    d.qpos[:] = saved
    mujoco.mj_kinematics(m, d)
    return depth


def _perturbed_pose(R, t, rng):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(CALIB["rotation_deg"])
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    shift = rng.standard_normal(3)
    return dR @ R, t + CALIB["translation_m"] * shift / np.linalg.norm(shift)


def run_scene(args) -> list[dict]:
    index, seed = args
    rng = np.random.default_rng([seed, index])
    scene = draw_scene(rng)
    truth, arm_over = _stage(scene)
    m, d = _W["model"], _W["data"]
    rows = []
    centres = np.array([t["xy"] for t in truth])
    for name, setup in SETUPS.items():
        if name == "hover_b" and arm_over:
            continue          # the hand holding a wrist camera would be where the arm is
        K = setup.intrinsics
        t_cam, R_cam = camera_pose(m, d, name)
        if name not in _W["rays"]:
            _W["rays"][name] = K.rays() @ R_cam.T
        rays = _W["rays"][name]
        seg = _render(name, "seg")
        seg_noarm = _render(name, "seg", _W["noarm"])   # (the arm at home hides cubes too)
        clean = _render(name, "depth")
        expected = _expected_depth(name)
        stack_of_geom = {g: t["stack"] for t in truth for g in t["geoms"]}
        h, w = seg.shape
        border = np.zeros((h, w), dtype=bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        info = {}
        for t in truth:
            g = t["geoms"][-1]
            px, px_free = int((seg == g).sum()), int((seg_noarm == g).sum())
            info[t["stack"]] = {
                "px": px, "px_noarm": px_free,
                "occ": 1.0 - px / px_free if px_free else 0.0,
                "at_border": bool(((seg_noarm == g) & border).any()),
            }
        nrng = np.random.default_rng([seed, index, list(SETUPS).index(name)])
        noisy = {lvl: (clean if NOISE[lvl].is_clean else NOISE[lvl].apply(clean, nrng))
                 for lvl in ("none", "low", "medium", "high")}
        R_cal, t_cal = _perturbed_pose(R_cam, t_cam, nrng)
        for cond in CONDITIONS:
            lvl = cond.split("+")[0]
            noise = NOISE[lvl]
            if cond.endswith("calib"):
                R_est, t_est, rays_est = R_cal, t_cal, K.rays() @ R_cal.T
            else:
                R_est, t_est, rays_est = R_cam, t_cam, rays
            est = estimate_cubes(noisy[lvl], expected, K, R_est, t_est, _W["estimator"],
                                 None if noise.is_clean else noise.sigma, rays_est)
            ids = [identify(e, seg, stack_of_geom) for e in est]
            for t in truth:
                s = t["stack"]
                inf = info[s]
                if inf["px_noarm"] < MIN_VISIBLE_PX:
                    continue      # out of this camera's view, or hidden behind other cubes
                mine = [(e, i) for e, i in zip(est, ids, strict=True) if i.target == s]
                others = np.delete(centres, s, axis=0)
                nd = float(np.linalg.norm(others - t["xy"], axis=1).min()) if len(others) else 9.9
                horiz = float(np.linalg.norm(t["xy"] - t_cam[:2]))
                row = {
                    "scene": index, "camera": name, "condition": cond, "stack": s,
                    "kind": t["kind"], "level": t["level"], "size": round(t["size"], 4),
                    "arm": "over" if arm_over else "home", "px": inf["px"],
                    "occ": round(inf["occ"], 4), "at_border": inf["at_border"],
                    "visible": inf["px"] >= MIN_VISIBLE_PX, "neighbour_m": round(nd, 4),
                    "horiz_m": round(horiz, 4),
                    "range_m": round(float(np.linalg.norm([*(t["xy"] - t_cam[:2]),
                                                           t["top"] - t_cam[2]])), 4),
                    "true_x": t["xy"][0], "true_y": t["xy"][1],
                    "true_yaw_deg": np.degrees(wrap_quarter(t["yaw"])), "true_top": t["top"],
                    "detected": bool(mine), "merged": False, "withheld": False,
                    "split": len(mine) > 1,
                }
                if mine:
                    e, i = max(mine, key=lambda ei: len(ei[0].top_pixels))
                    row.update(
                        merged=i.merged, withheld=e.occluded, n_top=len(e.top_pixels),
                        est_x=e.xy[0], est_y=e.xy[1],
                        est_yaw_deg=np.degrees(e.yaw), est_top=e.top_z,
                        xy_err_mm=1000 * float(np.linalg.norm(e.xy - t["xy"])),
                        xy_err_centroid_mm=1000 * float(np.linalg.norm(e.xy_centroid - t["xy"])),
                        yaw_err_deg=abs(np.degrees(wrap_quarter(e.yaw - t["yaw"]))),
                        yaw_rect_err_deg=abs(np.degrees(wrap_quarter(e.yaw_rect - t["yaw"]))),
                        z_err_mm=1000 * (e.top_z - t["top"]))
                rows.append(row)
    return rows


# --- statistics -------------------------------------------------------------------------------
def _dist(values) -> dict:
    v = np.asarray(values, dtype=float)
    if not len(v):
        return {}
    return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max()), "mean": float(v.mean())}


def _errors(rs: list[dict]) -> dict:
    if not rs:
        return {}
    xy = np.array([r["xy_err_mm"] for r in rs])
    return {"n": len(rs), "xy_mm": _dist(xy),
            "xy_over_mm": {str(k): float((xy > k).mean()) for k in THRESHOLDS_MM},
            "xy_centroid_mm": _dist([r["xy_err_centroid_mm"] for r in rs]),
            "yaw_deg": _dist([r["yaw_err_deg"] for r in rs]),
            "yaw_rect_deg": _dist([r["yaw_rect_err_deg"] for r in rs]),
            "z_abs_mm": _dist([abs(r["z_err_mm"]) for r in rs]),
            "z_bias_mm": float(np.mean([r["z_err_mm"] for r in rs]))}


def block(rows: list[dict]) -> dict:
    """Detection counts over visible cubes; error statistics over the accepted estimates
    (detected, not flagged occluded) and, separately, over every detected cube."""
    vis = [r for r in rows if r["visible"] and not r["at_border"]]
    det = [r for r in vis if r["detected"]]
    accepted = [r for r in det if not r["withheld"]]
    return {
        "in_view": len([r for r in rows if not r["at_border"]]),
        "hidden_by_arm": len([r for r in rows if not r["visible"] and not r["at_border"]]),
        "at_border": len([r for r in rows if r["at_border"]]),
        "visible": len(vis), "detected": len(det), "missed": len(vis) - len(det),
        "withheld": len(det) - len(accepted), "accepted": len(accepted),
        "merged": sum(r["merged"] for r in det),
        "merged_accepted": sum(r["merged"] for r in accepted),
        "split": sum(r["split"] for r in det),
        "errors": _errors(accepted),
        "errors_all_detected": _errors(det),
    }


def _horiz_bin(r) -> str:
    if r["camera"] != "overhead":
        return "station"
    h = r["horiz_m"]
    return "near (<0.45 m)" if h < 0.45 else "mid (0.45-0.60 m)" if h < 0.60 else "far (>0.60 m)"


GROUPS = {
    "distance": _horiz_bin,
    "arm": lambda r: r["arm"],
    "occlusion": lambda r: ("partly occluded" if r["occ"] > OCCLUDED else "clear"),
    "location": lambda r: (f"tower level {r['level']}" if r["kind"].startswith("tower")
                           else "bench (buffer, belt, open)"),
    "kind": lambda r: "tower" if r["kind"].startswith("tower") else r["kind"],
    "neighbour": lambda r: "neighbour < 9 cm" if r["neighbour_m"] < NEIGHBOUR else "no neighbour",
    "size": lambda r: ("small (<0.95)" if r["size"] < 0.95 else
                       "large (>1.05)" if r["size"] > 1.05 else "nominal"),
}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for cam in SETUPS:
        for cond in CONDITIONS:
            rs = [r for r in rows if r["camera"] == cam and r["condition"] == cond]
            if not rs:
                continue
            entry = {"all": block(rs)}
            for g, key in GROUPS.items():
                labels = sorted({key(r) for r in rs})
                entry[g] = {lab: block([r for r in rs if key(r) == lab]) for lab in labels}
            out.setdefault(cam, {})[cond] = entry
    return out


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("--n", type=int, default=400, help="test scenes (default 400)")
    p.add_argument("--seed", type=int, default=SEED, help=f"scene seed (default {SEED})")
    p.add_argument("--jobs", type=int, default=4, help="worker processes (default 4)")
    p.add_argument("--out-dir", type=Path, default=Path("results/v4"))
    args = p.parse_args()
    git = git_state()   # recorded before running, so later commits cannot mislabel the result
    started = time.time()
    with Pool(args.jobs, initializer=_init_worker) as pool:
        rows = [r for rs in pool.imap(run_scene, [(i, args.seed) for i in range(args.n)],
                                      chunksize=4) for r in rs]
    elapsed = time.time() - started
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cams = {n: {"pos": list(s.pos), "target": list(s.target), "fovy_deg": s.fovy,
                "resolution": [s.width, s.height], "note": s.note,
                "intrinsics": {"fx": s.intrinsics.fx, "fy": s.intrinsics.fy,
                               "cx": s.intrinsics.cx, "cy": s.intrinsics.cy},
                "mm_per_px_at_target": 1000 * s.intrinsics.pixel_size(
                    float(np.linalg.norm(np.subtract(s.target, s.pos))))}
            for n, s in SETUPS.items()}
    payload = {
        "what": "pose error of a classical depth-camera cube estimator in the v3.2 cell "
                "(docs/pose_error.md)",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "git": git,
        "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                     "numpy": np.__version__, "menagerie": menagerie_revision()},
        "seed": args.seed, "n_scenes": args.n, "jobs": args.jobs, "seconds": round(elapsed, 1),
        "cameras": cams,
        "noise": {k: {"sigma0_m": v.sigma0, "k2_m_per_m2": v.k2, "z0_m": v.z0,
                      "dropout": v.dropout, "edge_dropout": v.edge_dropout, "note": v.note}
                  for k, v in NOISE.items()},
        "calibration_error": CALIB,
        "estimator": EstimatorConfig().__dict__,
        "scene_model": {"size_range": SIZE_RANGE, "loose_radius_m": LOOSE_RADIUS,
                        "pair_gap_m": PAIR_GAP, "p_arm_over": P_ARM_OVER,
                        "min_visible_px": MIN_VISIBLE_PX, "occluded_share": OCCLUDED,
                        "neighbour_m": NEIGHBOUR},
        "definitions": {
            "visible": f"the stack's top cube shows >= {MIN_VISIBLE_PX} px in the segmentation "
                       "render, and does not touch the image border",
            "detected": "some cluster's top face is mostly that cube (segmentation, identity "
                        "only)",
            "merged": "the cluster also holds >= 15 px of another stack",
            "xy_mm": "distance between estimated and true top-face centre, detected cubes",
            "yaw_deg": "|estimated - true| yaw, mod 90 deg",
            "z": "estimated - true top-face height",
        },
        "summary": summarize(rows),
    }
    out = args.out_dir / "pose_error.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out_dir / "pose_error_rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                             for k, v in r.items()})
    print(f"{len(rows)} rows from {args.n} scenes in {elapsed:.0f} s -> {out}")
    for cam, conds in payload["summary"].items():
        for cond, entry in conds.items():
            print(f"{cam:9s} {cond:12s} {_line(entry['all'])}")


def _line(a: dict) -> str:
    e = a["errors"]
    if not e:
        return f"vis {a['visible']} accepted 0"
    xy = e["xy_mm"]
    return (f"vis {a['visible']:4d} miss {a['missed']:3d} withheld {a['withheld']:3d} merged "
            f"{a['merged_accepted']:2d} | xy med {xy['median']:.2f} p95 {xy['p95']:.2f} max "
            f"{xy['max']:.1f} mm, >1/2/3/4 mm "
            + "/".join(f"{100 * e['xy_over_mm'][str(k)]:.0f}" for k in THRESHOLDS_MM)
            + f" % | yaw p95 {e['yaw_deg']['p95']:.2f} (rect {e['yaw_rect_deg']['p95']:.2f}) "
            f"| |z| p95 {e['z_abs_mm']['p95']:.2f} mm")


if __name__ == "__main__":
    main()
