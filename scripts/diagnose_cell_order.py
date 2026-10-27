"""Replay one evaluation order of the Milestone 9 cell and report every placement.

Usage:
  uv run python scripts/diagnose_cell_order.py --pick run:runs/<grasp> --place run:runs/<place> \
      --index 59 [--mixed] [--aim trained] [--watch A] [--footage out/dir]
The order is the one evaluate_cell.py ran as order INDEX (same seed and configuration). For each
learned placement it prints, in mm in the station frame (radial = away from the robot, sideways
= to its left), the aim relative to the fixture, and the cube relative to the aim when the skill
starts, at release, when the skill ends and after the hand has left; the cube's offset from the
cube below, and how far the cube below moved. --watch prints a tower's cube offsets over the
whole order (slow creep, a fall and when it happened). --footage (needs `uv sync --group media`)
saves close-up stills of the station after each placement, of the watched tower whenever it
moves, and of every tower at the end, and an MP4 of the placements.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from polypanda.evaluation.cell import CellEvalConfig, run_cell_order, skill_from_spec
from polypanda.simulation.workcell import Workcell, WorkcellConfig
from polypanda.supervisor.cell import LearnedCell


def station_frame(xy) -> np.ndarray:
    """Rows: the radial and sideways unit vectors at a station."""
    u = np.asarray(xy, dtype=float) / np.linalg.norm(xy)
    return np.array([u, [-u[1], u[0]]])


def mm(v, frame) -> np.ndarray:
    return frame @ v * 1000


class Diagnosis(LearnedCell):
    """LearnedCell that records where each placed cube is at each moment of its placement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows, self.filming, self.on_placed = [], None, None
        self._now = {}

    def station(self, xy) -> str:
        names = {f"tower {'ABC'[t]}": self.cell.tower_xy(t) for t in range(3)}
        names.update({f"slot {s}": self.cell.buffer_xy(s) for s in range(3)})
        return min(names, key=lambda k: np.linalg.norm(names[k] - xy))

    def _run_skill(self, skill, env, view, obs, after_step=None):
        if skill is not self.placer:
            return super()._run_skill(skill, env, view, obs, after_step)
        now = self._now
        now["start"] = self.d.xpos[env.cube.body][:2].copy()
        now["target"] = env.target[:2].copy()

        def watch():
            if after_step:
                after_step()
            if env.released and "release" not in now:
                now["release"] = self.d.xpos[env.cube.body][:2].copy()

        result = super()._run_skill(skill, env, view, obs, watch)
        now["end"] = self.d.xpos[env.cube.body][:2].copy()
        return result

    def place(self, cube, xy, level):
        if self.placer is None:
            return super().place(cube, xy, level)
        fixture = np.asarray(xy, dtype=float)
        below = self.top_cube(fixture, level) if level else None
        below0 = self.d.xpos[below.body][:2].copy() if below else None
        self._now = {}
        self.filming = fixture
        super().place(cube, xy, level)
        self.filming = None
        now, f = self._now, station_frame(fixture)
        aim = now["target"]
        settled = self.d.xpos[cube.body][:2].copy()
        self.rows.append({
            "station": self.station(fixture), "level": level + 1, "cube": cube,
            "fixture": fixture, "result": self.places[-1],
            "aim": mm(aim - fixture, f), "start": mm(now["start"] - aim, f),
            "release": mm(now.get("release", now["end"]) - aim, f),
            "end": mm(now["end"] - aim, f), "settled": mm(settled - aim, f),
            "on_below": mm(settled - self.d.xpos[below.body][:2], f) if below else None,
            "below_moved": mm(self.d.xpos[below.body][:2] - below0, f) if below else None})
        if self.on_placed:
            self.on_placed(self.rows[-1])


class Camera:
    """Close-up side views of a station (looking sideways and looking at the robot)."""

    def __init__(self, cell: Workcell, size: int = 360):
        from PIL import Image, ImageDraw
        self.Image, self.Draw = Image, ImageDraw
        self.d = cell.data
        self.renderer = mujoco.Renderer(cell.model, size, size)

    def views(self, xy, label: str = "") -> np.ndarray:
        deg = np.degrees(np.arctan2(xy[1], xy[0]))
        imgs = []
        for az in (deg + 90, deg + 180):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [xy[0], xy[1], 0.13]
            cam.distance, cam.azimuth, cam.elevation = 0.55, az, -8
            self.renderer.update_scene(self.d, cam)
            imgs.append(self.renderer.render())
        img = self.Image.fromarray(np.concatenate(imgs, axis=1))
        self.Draw.Draw(img).text((8, 8), label, fill=(255, 255, 0))
        return np.asarray(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", default="scripted")
    parser.add_argument("--place", required=True, help="run:<place run dir>")
    parser.add_argument("--index", type=int, required=True, help="evaluation order number")
    parser.add_argument("--seed", type=int, default=CellEvalConfig.seed)
    parser.add_argument("--heights", type=int, nargs=2, default=CellEvalConfig.heights)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--slots", type=int, default=CellEvalConfig.slots)
    parser.add_argument("--aim", choices=("fixture", "trained"), default=CellEvalConfig.aim)
    parser.add_argument("--watch", choices=("A", "B", "C"), help="follow a tower's cubes")
    parser.add_argument("--footage", type=Path, help="directory for stills and an MP4")
    args = parser.parse_args()

    config = CellEvalConfig(seed=args.seed, heights=tuple(args.heights), mixed=args.mixed,
                            slots=args.slots, aim=args.aim)
    pick, *_ = skill_from_spec(args.pick, "pick")
    place, *_ = skill_from_spec(args.place, "place")
    cell = Workcell.build(WorkcellConfig(n_cubes=config.n_cubes))
    ctl = Diagnosis(cell, grasp=pick, place=place, aim=args.aim)
    camera = Camera(cell) if args.footage else None
    frames, steps, watched, last = [], [0], [], {}
    if args.footage:
        args.footage.mkdir(parents=True, exist_ok=True)

    def on_step():
        steps[0] += 1
        if camera and ctl.filming is not None and steps[0] % 2 == 0:
            frames.append(camera.views(ctl.filming))
        if args.watch and steps[0] % 20 == 0:  # every second of simulated time
            xy = cell.tower_xy("ABC".index(args.watch))
            cubes = [r["cube"] for r in ctl.rows if r["station"] == f"tower {args.watch}"]
            offs = [mm(cell.data.xpos[c.body][:2] - xy, station_frame(xy)) for c in cubes]
            if offs and (len(offs) != len(last.get("offs", [])) or max(
                    np.abs(a - b).max() for a, b in zip(offs, last["offs"], strict=False)) > 0.5):
                t = steps[0] * ctl.dt
                watched.append((t, offs, np.degrees(cell.data.qpos[0])))
                last["offs"] = offs
                if camera:
                    camera.Image.fromarray(camera.views(xy, f"t={t:.0f} s")).save(
                        args.footage / f"watch_{args.watch}_{t:06.1f}s.png")

    def on_placed(row):
        if camera:
            label = f"after {row['station']} level {row['level']}"
            camera.Image.fromarray(camera.views(row["fixture"], label)).save(
                args.footage / f"{len(ctl.rows):02d}_{row['station'].replace(' ', '')}"
                               f"_L{row['level']}.png")

    ctl.on_step, ctl.on_placed = on_step, on_placed
    episode = run_cell_order(ctl, config, args.index)
    print(episode)

    def fmt(v):
        return "      -      " if v is None else f"{v[0]:+5.1f},{v[1]:+5.1f}"

    print("\nper placement, mm as (radial, sideways); start to settled are relative to the aim")
    print("station   lv  aim-fixture   start         release       end           settled     "
          "  on cube below  below moved   steps result")
    for r in ctl.rows:
        res = r["result"]
        print(f"{r['station']:8s}  {r['level']}  {fmt(r['aim'])}  {fmt(r['start'])}  "
              f"{fmt(r['release'])}  {fmt(r['end'])}  {fmt(r['settled'])}  "
              f"{fmt(r['on_below'])}  {fmt(r['below_moved'])}  {res['steps']:4d}  "
              f"{'ok' if res['success'] else res['failure'] or 'timeout'}")
    print("\nat the end, cube - fixture (mm, radial, sideways):")
    for t in range(3):
        f = station_frame(cell.tower_xy(t))
        rows = [r for r in ctl.rows if r["station"] == f"tower {'ABC'[t]}"]
        print(f"  tower {'ABC'[t]}: " + "  ".join(
            f"L{r['level']} {fmt(mm(cell.data.xpos[r['cube'].body][:2] - cell.tower_xy(t), f))}"
            for r in rows))
    if args.watch:
        print(f"\ntower {args.watch} over time (each cube's offset from the fixture, mm as "
              "(radial, sideways); printed when any moves by more than 0.5 mm; base = where the "
              "arm is):")
        for t, offs, base in watched:
            print(f"  t={t:6.1f} s  base {base:+5.0f} deg  " + " ".join(fmt(o) for o in offs))
    if camera:
        for t in range(3):
            camera.Image.fromarray(camera.views(cell.tower_xy(t), f"end: tower {'ABC'[t]}")).save(
                args.footage / f"end_tower{'ABC'[t]}.png")
        import imageio.v2 as imageio
        imageio.mimsave(args.footage / "placements.mp4", frames, fps=20)
        print(f"\nwrote stills and placements.mp4 ({len(frames)} frames) to {args.footage}")


if __name__ == "__main__":
    main()
