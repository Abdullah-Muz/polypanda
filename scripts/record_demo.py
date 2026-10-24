"""Record pick-and-place episodes to video: an MP4 (with captions) and a short GIF for the README.

Usage: uv run --group media python scripts/record_demo.py --grasp run:runs/<grasp run> \
           --scenes 0 3 7 --out media/pickplace
Renders off-screen (needs an OpenGL context; under WSL2 the default GLFW backend works), one frame
per control step (20 fps, real time), with the active stage and its skill captioned. Writes
<out>.mp4 and, for the first scene, <out>.gif (every second frame, reduced size).
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from polypanda.control.scripted_place import ScriptedPlace
from polypanda.environments.handoff import capture_state, restore_state
from polypanda.evaluation.pickplace import grasp_driver_from_spec, grasp_env_for
from polypanda.evaluation.sets import PickPlaceEvalSet
from polypanda.evaluation.transport import ScriptedCarryDriver
from polypanda.simulation.scenes import load_pick_scene
from polypanda.supervisor.fsm import PickPlaceSupervisor, Stage

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def caption(frame: np.ndarray, lines: list[tuple[str, int]], banner: str | None = None
            ) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    y = 14
    for text, size in lines:
        f = font(size)
        box = draw.textbbox((16, y), text, font=f)
        draw.rectangle((box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5), fill=(0, 0, 0, 140))
        draw.text((16, y), text, font=f, fill=(255, 255, 255))
        y = box[3] + 14
    if banner:
        f = font(max(28, img.width // 30))
        box = draw.textbbox((0, 0), banner, font=f)
        x = (img.width - (box[2] - box[0])) // 2
        yb = img.height - (box[3] - box[1]) - 40
        draw.rectangle((x - 16, yb - 10, x + box[2] - box[0] + 16, yb + box[3] - box[1] + 16),
                       fill=(20, 110, 40, 210))
        draw.text((x, yb), banner, font=f, fill=(255, 255, 255))
    return np.asarray(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp", default="scripted", help="scripted | run:DIR[@best|final]")
    parser.add_argument("--scenes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--set", type=Path, default=Path("eval_sets/pickplace_nominal_v1.json"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gif-width", type=int, default=360)
    parser.add_argument("--gif-step", type=int, default=3, help="keep every n-th frame; the "
                        "GIF plays at 2x speed")
    parser.add_argument("--out", type=Path, default=Path("media/pickplace"))
    args = parser.parse_args()

    grasp, grasp_desc = grasp_driver_from_spec(args.grasp)[:2]
    skills = {Stage.GRASP: "learned PPO policy" if args.grasp != "scripted" else "scripted IK",
              Stage.TRANSPORT: "scripted closed-loop carry",
              Stage.PLACE: "scripted place (stand upright, lower, release)",
              Stage.HOME: "recovery: release, lift clear, return home"}
    supervisor = PickPlaceSupervisor(grasp, ScriptedCarryDriver(), ScriptedPlace(),
                                     grasp_env=grasp_env_for(args.grasp))
    eval_set = PickPlaceEvalSet.load(args.set)

    model, data = load_pick_scene()  # display copy mirroring the active skill's environment
    model.mat_reflectance[:] = 0.0  # floor reflections render with streak artifacts off-screen
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, args.height, args.width)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.40, 0.0, 0.24]
    camera.distance, camera.azimuth, camera.elevation = 1.5, 150.0, -20.0
    gif_size = (args.gif_width, round(args.height * args.gif_width / args.width))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mp4, gif = args.out.with_suffix(".mp4"), args.out.with_suffix(".gif")
    writer = imageio.get_writer(mp4, fps=round(1 / supervisor.dt), codec="libx264", quality=8,
                                macro_block_size=8)
    gif_frames: list[np.ndarray] = []
    state = {"target": None, "scene": 0, "t": 0, "frames": 0, "gif": True}

    def lines_for(stage: Stage | None, scale: float, full: bool) -> list[tuple[str, int]]:
        lines = []
        if full:
            lines.append(("PolyPanda: autonomous pick-and-place (simulated Franka Panda, MuJoCo)",
                          round(22 * scale)))
        if stage is not None:
            lines.append((f"{stage.value.upper()}: {skills[stage]}", round(26 * scale)))
        if full:
            lines.append((f"scene {state['scene']}   t = {state['t'] * supervisor.dt:4.1f} s   "
                          "green disc = target spot", round(18 * scale)))
        return lines

    def emit(stage: Stage | None, banner: str | None = None, repeat: int = 1) -> None:
        renderer.update_scene(data, camera)
        scn = renderer.scene
        if scn.ngeom < scn.maxgeom and state["target"] is not None:
            tx, ty = state["target"]
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                size=[0.02, 0.0006, 0], pos=[tx, ty, 0.0006],
                                mat=np.eye(3).flatten(), rgba=[0.1, 0.9, 0.2, 0.6])
            scn.ngeom += 1
        raw = renderer.render()
        frame = caption(raw, lines_for(stage, args.width / 1280, True), banner)
        small = None
        for _ in range(repeat):
            writer.append_data(frame)
            if state["gif"] and state["frames"] % args.gif_step == 0:
                if small is None:  # caption the GIF at its own size so it stays readable
                    resized = np.asarray(Image.fromarray(raw).resize(gif_size, Image.LANCZOS))
                    small = caption(resized, lines_for(stage, 2.0 * args.gif_width / 1280,
                                                       False), banner)
                gif_frames.append(small)
            state["frames"] += 1

    def on_step(stage, env):
        restore_state(model, data, capture_state(env.model, env.data))
        state["t"] += 1
        emit(stage)

    supervisor.on_step = on_step
    for i in args.scenes:
        pose, target = eval_set.placements[i], eval_set.targets[i]
        state.update(target=target, scene=i, t=0)
        grasp_env = supervisor.envs[Stage.GRASP]
        grasp_env.reset(options={"cube": tuple(pose)})
        restore_state(model, data, capture_state(grasp_env.model, grasp_env.data))
        emit(None, repeat=15)
        r = supervisor.run(pose, target)
        banner = (f"PLACED {r.final_xy_error_mm:.1f} mm from target in {r.total_s:.1f} s"
                  if r.success else f"FAILED at {r.failure_stage}: {r.failure_reason}")
        emit(None, banner, repeat=30)
        state["gif"] = False  # the GIF shows the first scene only
        print(f"scene {i}: {banner} ({state['frames']} frames so far)")
    writer.close()
    imageio.mimwrite(gif, gif_frames, duration=1000 * args.gif_step * supervisor.dt / 2, loop=0)
    print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB, {state['frames']} frames) and "
          f"{gif} ({gif.stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames); "
          f"grasp skill: {grasp_desc}")

if __name__ == "__main__":
    main()
