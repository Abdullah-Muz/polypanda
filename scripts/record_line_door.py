"""Record the v4.0 line cycle side by side: two door actors on the same machine (the learned door
against the feel script), the rest of the cycle the same scripted code.

Usage: uv run --group media python scripts/record_line_door.py --machines 117 120 109 \\
           --out media/line_door_v40
Each machine of the set is run exactly as scripts/evaluate_line_door.py runs it (evaluation/
line_door.run_cycle, hand-over "door_start", as in T5b), once per actor, and rendered off-screen
at 20 frames per simulated second (real time). The two cycles play beside each other with a
clock; the faster one holds on its last frame, showing its outcome and time, until the slower
finishes. The outcomes are printed, to check against results/v4/line/.
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from record_demo import font

from polypanda.evaluation import line_door
from polypanda.simulation.machine import RUNNING

GREEN, GREY = (20, 110, 40, 215), (90, 90, 90, 215)


def record(actor: str, index: int, ep: dict, args) -> tuple[list, dict]:
    """One cycle of `actor` on machine `index`: (simulated time, frame) every 0.05 s of
    simulated time. Physics advanced many steps in one call is split so no frame is skipped."""
    frames, held = [], {"next": 0.0}
    real_step, real_build = mujoco.mj_step, line_door.Line.build_line

    def build(cfg):
        line = real_build(cfg)
        model = line.model
        model.mat_reflectance[:] = 0.0   # floor reflections render with streaks off-screen
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        held["renderer"] = mujoco.Renderer(model, args.height, args.width)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = args.lookat
        cam.azimuth, cam.elevation, cam.distance = args.camera
        held["camera"], held["data"] = cam, line.data
        return line

    def step(m, d, nstep=1):
        left = nstep
        while left > 0:
            if "renderer" not in held:
                real_step(m, d, nstep=left)
                return
            k = min(left, max(1, int(np.ceil((held["next"] - d.time) / m.opt.timestep))))
            real_step(m, d, nstep=k)
            left -= k
            if d.time >= held["next"] - 1e-9:
                held["renderer"].update_scene(d, held["camera"])
                frames.append((float(d.time), held["renderer"].render().copy()))
                held["next"] += 0.05

    mujoco.mj_step = step
    line_door.Line.build_line = staticmethod(build)
    try:
        row = line_door.run_cycle(actor, index, ep, "door_start")
    finally:
        mujoco.mj_step = real_step
        line_door.Line.build_line = real_build
        held["renderer"].close()
    return frames, row


def panel(frame: np.ndarray, label: str, t: float, banner) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    f = font(max(14, img.width // 34))
    for (x, y), text in (((12, 10), label), ((12, 10 + f.size + 12), f"{t:5.1f} s")):
        box = draw.textbbox((x, y), text, font=f)
        draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(0, 0, 0, 150))
        draw.text((x, y), text, font=f, fill=(255, 255, 255))
    if banner:
        text, colour = banner
        fb = font(max(20, img.width // 20))
        w, h = (lambda b: (b[2] - b[0], b[3] - b[1]))(draw.textbbox((0, 0), text, font=fb))
        x, y = (img.width - w) // 2, img.height - h - 30
        draw.rectangle((x - 14, y - 8, x + w + 14, y + h + 14), fill=colour)
        draw.text((x, y), text, font=fb, fill=(255, 255, 255))
    return np.asarray(img)


def header(width: int, text: str) -> np.ndarray:
    img = Image.new("RGB", (width, max(28, width // 30)), (25, 25, 25))
    draw = ImageDraw.Draw(img)
    f = font(max(13, width // 70))
    b = draw.textbbox((0, 0), text, font=f)
    draw.text(((width - b[2] + b[0]) // 2, (img.height - b[3] + b[1]) // 2 - b[1]), text,
              font=f, fill=(235, 235, 235))
    return np.asarray(img)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.split("\n")[1:]))
    p.add_argument("--set", type=Path, default=Path("eval_sets/door_final_v1.json"))
    p.add_argument("--machines", type=int, nargs="+", required=True)
    p.add_argument("--left", default="run:runs/door_v2_bc_ppo_r5_seed0@final")
    p.add_argument("--right", default="feel_v2")
    p.add_argument("--left-label", default="Learned door (PPO, seed 0)")
    p.add_argument("--right-label", default="Feel script (feel_v2)")
    p.add_argument("--width", type=int, default=640, help="per panel")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--lookat", type=float, nargs=3, default=[-0.55, -0.4, 0.22])
    p.add_argument("--camera", type=float, nargs=3, default=[120.0, -30.0, 1.35],
                   metavar=("AZIMUTH", "ELEVATION", "DISTANCE"))
    p.add_argument("--hold", type=int, default=60, help="frames held after both finish")
    p.add_argument("--out", type=Path, default=Path("media/line_door"))
    args = p.parse_args()

    import json
    torch.set_num_threads(1)
    data = json.loads(args.set.read_text())
    out = args.out.with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=20, codec="libx264", quality=8, macro_block_size=8)
    for index in args.machines:
        ep = data["episodes"][index]
        runs = [record(a, index, ep, args) for a in (args.left, args.right)]
        for (_, r), label in zip(runs, (args.left_label, args.right_label), strict=True):
            print(f"{data['name']} machine {index} ({ep['slide']}, spring {ep['spring'] > 0}) "
                  f"{label}: running {r['success']}, failure {r['failure']}, "
                  f"cycle {r['cycle_s']:.1f} s", flush=True)
        kind = (f"lid slides {ep['slide']}" + (", with a closing spring" if ep["spring"] > 0
                                                else ""))
        top = header(2 * args.width, f"An unseen machine ({kind}): drive to it, open the lid, "
                                     f"load the part, close. Only the door actor differs.")
        n = max(len(f) for f, _ in runs)
        for i in range(n + args.hold):
            row = []
            for (f, r), label in zip(runs, (args.left_label, args.right_label), strict=True):
                done = i >= len(f)
                t, frame = f[min(i, len(f) - 1)]
                banner = None
                if done:
                    banner = ((f"MACHINE RUNNING  {r['cycle_s']:.0f} s", GREEN)
                              if r["success"] and r["state"] == RUNNING
                              else (f"NOT RUNNING: {r['failure']}", GREY))
                row.append(panel(frame, label, t, banner))
            writer.append_data(np.vstack([top, np.hstack(row)]))
    writer.close()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
