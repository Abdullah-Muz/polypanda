"""Record one work order of the Milestone 9 cell to video: an MP4 and a short GIF for the README.

Usage: uv run --group media python scripts/record_cell.py --pick run:runs/grasp_cart_seed2 \
           --place run:runs/place_stack_v3_seed2 --index 110 --mixed --out media/cell
The order is the one evaluate_cell.py ran as order INDEX. Renders off-screen (under WSL2 the
default GLFW backend works), over the robot's shoulder toward the towers, captioned with what is
running (a learned policy or a scripted move) and the planner's decisions. The MP4 keeps every
--speed-th control step at 20 fps (2x real time by default); the GIF keeps every --gif-speed-th
step at 20 fps (12x), smaller, with one palette for the whole clip (needs ffmpeg, which
imageio-ffmpeg provides).
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image
from record_demo import caption

from polypanda.evaluation.cell import CellEvalConfig, run_cell_order, skill_from_spec
from polypanda.simulation.workcell import TOWERS, Workcell, WorkcellConfig
from polypanda.supervisor.cell import LearnedCell


class CaptionedCell(LearnedCell):
    """LearnedCell that keeps a caption of what is running."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task, self.skill, self.active = "", "", False

    def station(self, angle: float) -> str:
        cfg = self.cell.config
        names = {"the infeed": cfg.infeed_angle, "the reject bin": cfg.reject_angle,
                 **{f"tower {TOWERS[t]}": a for t, a in enumerate(cfg.tower_angles)},
                 **{f"buffer slot {s + 1}": a for s, a in enumerate(cfg.buffer_angles)}}
        return min(names, key=lambda k: abs(np.radians(names[k]) - angle))

    def caption(self) -> str:
        return self.skill if self.active else self.task

    def swing_to(self, angle: float) -> None:
        self.task = f"scripted: swing to {self.station(angle)}"
        super().swing_to(angle)

    def _run_skill(self, skill, env, view, obs, after_step=None):
        self.active = True
        try:
            return super()._run_skill(skill, env, view, obs, after_step)
        finally:
            self.active = False
            self.task = ("scripted: lift to carry height" if skill is self.grasp
                         else "scripted: open and rise")

    def pick(self, cube):
        c = self.d.xpos[cube.body]
        what = (f"{self.cell.cube_colour(cube)} cube, "
                f"{self.station(float(np.arctan2(c[1], c[0]))).removeprefix('the ')}")
        self.task = f"scripted {'turn to' if self.grasp else 'grasp:'} {what}"
        self.skill = f"LEARNED GRASP (PPO): {what}"
        super().pick(cube)

    def place(self, cube, xy, level):
        where = self.station(float(np.arctan2(xy[1], xy[0])))
        spot = f"{where}, level {level + 1}" if where.startswith("tower") else where
        self.task = f"scripted {'descent:' if self.placer else 'place:'} {spot}"
        self.skill = f"LEARNED PLACE (PPO): {spot}"
        super().place(cube, xy, level)

    def reject(self, cube):
        self.task = "scripted: drop in the reject bin"
        super().reject(cube)


def write_gif(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    """A GIF small enough for a README: one palette for the whole clip, and each frame stores
    only the rectangle that changed (ffmpeg's palettegen/paletteuse)."""
    import imageio_ffmpeg
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        imageio.mimwrite(clip, frames, fps=fps, codec="libx264", quality=10, macro_block_size=2)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i",
                        str(clip), "-vf", "split[a][b];[a]palettegen=max_colors=128:"
                        "stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:"
                        "diff_mode=rectangle", "-loop", "0", str(path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", default="scripted")
    parser.add_argument("--place", default="scripted")
    parser.add_argument("--index", type=int, required=True, help="evaluation order number")
    parser.add_argument("--mixed", action="store_true", help="a Layer 2 order")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--speed", type=int, default=2, help="MP4: keep every n-th step")
    parser.add_argument("--gif-width", type=int, default=480)
    parser.add_argument("--gif-speed", type=int, default=12, help="GIF: keep every n-th step")
    parser.add_argument("--out", type=Path, default=Path("media/cell"))
    args = parser.parse_args()

    config = CellEvalConfig(mixed=args.mixed)
    pick, pick_desc, *_ = skill_from_spec(args.pick, "pick")
    place, place_desc, *_ = skill_from_spec(args.place, "place")
    cell = Workcell.build(WorkcellConfig(n_cubes=config.n_cubes))
    model = cell.model
    model.mat_reflectance[:] = 0.0  # floor reflections render with streak artifacts off-screen
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, args.height, args.width)
    camera = mujoco.MjvCamera()  # towers, signs, buffer, reject bin and conveyor all in view
    camera.azimuth, camera.elevation, camera.distance = 250, -35, 1.95
    camera.lookat[:] = [0.05, -0.22, 0.12]
    gif_size = (args.gif_width, round(args.height * args.gif_width / args.width))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mp4, gif = args.out.with_suffix(".mp4"), args.out.with_suffix(".gif")
    writer = imageio.get_writer(mp4, fps=20, codec="libx264", quality=8, macro_block_size=8)
    gif_frames: list[np.ndarray] = []
    learned = " and ".join(s for s, on in (("grasp", pick), ("place", place)) if on)
    title = ("PolyPanda work-order cell: " + (f"learned {learned} (PPO), " if learned else "")
             + "scripted planner and moves between stations")
    state = {"steps": 0, "plan": "", "order": "", "stats": ""}
    ctl = CaptionedCell(cell, grasp=pick, place=place)

    def frame(banner: str | None, scale: float, size=None, full: bool = True) -> np.ndarray:
        renderer.update_scene(cell.data, camera)
        raw = renderer.render()
        if size:
            raw = np.asarray(Image.fromarray(raw).resize(size, Image.LANCZOS))
        lines = [(ctl.caption(), round(26 * scale))]
        if state["plan"]:
            lines.append((f"planner: {state['plan']}", round(20 * scale)))
        if full:
            lines = [(title, round(22 * scale))] + lines
            lines.append((f"{state['order']}   t = {state['steps'] * ctl.dt:5.1f} s"
                          + state["stats"], round(18 * scale)))
        return caption(raw, lines, banner)

    def emit(banner: str | None = None, repeat: int = 1) -> None:
        n = state["steps"]
        if n % args.speed == 0 or repeat > 1:
            full = frame(banner, args.width / 1280)
            for _ in range(repeat):
                writer.append_data(full)
        if n % args.gif_speed == 0 or repeat > 1:
            short = banner and banner.split(":")[0]  # (the full banner is too wide here)
            small = frame(short, 1.5 * args.gif_width / 1280, gif_size, full=False)
            gif_frames.extend([small] * max(1, repeat // 2))

    def on_step():
        state["steps"] += 1
        if cell.order is not None and not state["order"]:
            kind = "Layer 2 order" if args.mixed else "Layer 1 order"
            state["order"] = f"{kind} {args.index}: " + "   ".join(
                f"{TOWERS[t]} " + "-".join(r) for t, r in enumerate(cell.order.recipes))
        if args.mixed and hasattr(ctl, "stats"):
            s = ctl.stats
            state["stats"] = f"   parked {s['buffered']}   rejected {s['rejected']}"
        emit()

    ctl.on_step = on_step
    original = ctl.run_mixed_order if args.mixed else ctl.run_order

    def run(order, rng, **kw):  # the planner's decisions, from the controller's log
        log = kw.get("log")

        def show(line: str) -> None:
            state["plan"] = line.strip()
            if log:
                log(line)

        return original(order, rng, **{**kw, "log": show})

    if args.mixed:
        ctl.run_mixed_order = run
    else:
        ctl.run_order = run
    episode = run_cell_order(ctl, config, args.index)
    ctl.task = ""
    banner = (f"ORDER COMPLETE: all {len(episode.towers_ok)} towers correct"
              if episode.success else f"ORDER FAILED: {episode.failure}")
    emit(banner, repeat=60)
    writer.close()
    write_gif(gif, gif_frames)
    print(f"{episode}\nwrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB) and {gif} "
          f"({gif.stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames); pick: {pick_desc}; "
          f"place: {place_desc}")


if __name__ == "__main__":
    main()
