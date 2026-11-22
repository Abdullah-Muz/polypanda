"""Record pick-check cases side by side: two grasps on the same staged cube (the v3.2 before/after).

Usage: uv run --group media python scripts/record_pick_compare.py \\
           --set eval_sets/grasp_belt_final_v1.json --cases 58 190 \\
           --before run:runs/grasp_anypose6_seed0 --after run:runs/grasp_anypose9_seed0 \\
           --out media/pick_compare_v32
Each case is staged and picked exactly as scripts/evaluate_cell_picks.py does
(evaluation/cell_picks.run_case: a fresh cell, the cell's own pick), once per grasp, and rendered
off-screen (under WSL2 the default GLFW backend works) at one frame per control step, 20 fps
(real time; --slow N shows every frame N times). The two picks play beside each other, the
shorter one held on its last frame, and each case ends on its outcome. The outcomes are printed,
to check against the committed results/grasp/cellpicks_<set>_<run>_best_episodes.csv.
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from record_demo import font

from polypanda.evaluation.cell import skill_from_spec
from polypanda.evaluation.cell_picks import TEST_CUBE, PickSet, run_case
from polypanda.supervisor.cell import LearnedCell

GREEN, RED, GREY = (20, 110, 40, 215), (170, 30, 30, 215), (60, 60, 60, 215)


def record(case, grasp, args) -> tuple[list[np.ndarray], dict]:
    """Pick `case` with `grasp`, one rendered frame per control step."""
    frames, held = [], {}

    def make_controller(cell):
        model = cell.model
        model.mat_reflectance[:] = 0.0   # floor reflections render with streaks off-screen
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        held["renderer"] = renderer = mujoco.Renderer(model, args.height, args.width)
        camera = mujoco.MjvCamera()      # beside the pick point, the hand coming from the base
        camera.azimuth, camera.elevation, camera.distance = args.camera
        camera.lookat[:] = cell.data.xpos[cell.cubes[TEST_CUBE].body] + [0.0, 0.0, 0.08]
        ctl = LearnedCell(cell, grasp=grasp, aim="fixture")

        def on_step():
            renderer.update_scene(cell.data, camera)
            frames.append(renderer.render().copy())

        ctl.on_step = on_step
        return ctl

    result = run_case(case, make_controller)
    held["renderer"].close()
    return frames, result


def outcome(r: dict) -> tuple[str, tuple]:
    if r["success"]:
        return "PICKED", GREEN
    if r["knocked_over"]:
        return "KNOCKED OVER", RED
    return f"FAILED: {r['failure'] or 'timeout'}", GREY


def panel(frame: np.ndarray, label: str, banner: tuple[str, tuple] | None) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    f = font(max(14, img.width // 34))
    box = draw.textbbox((12, 10), label, font=f)
    draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(0, 0, 0, 150))
    draw.text((12, 10), label, font=f, fill=(255, 255, 255))
    if banner:
        text, colour = banner
        fb = font(max(20, img.width // 18))
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="\n".join(__doc__.split("\n")[1:]))
    parser.add_argument("--set", type=Path, default=Path("eval_sets/grasp_belt_final_v1.json"))
    parser.add_argument("--cases", type=int, nargs="+", required=True, help="case indices")
    parser.add_argument("--before", required=True, help="run:<grasp run dir>[@best|final]")
    parser.add_argument("--after", required=True, help="run:<grasp run dir>[@best|final]")
    parser.add_argument("--before-label", default="v3.1 grasp (grasp_anypose6, seed 0)")
    parser.add_argument("--after-label", default="v3.2 grasp (grasp_anypose9, seed 0)")
    parser.add_argument("--width", type=int, default=640, help="per panel")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", type=float, nargs=3, default=[140.0, -22.0, 0.85],
                        metavar=("AZIMUTH", "ELEVATION", "DISTANCE"))
    parser.add_argument("--slow", type=int, default=1, help="show every frame N times")
    parser.add_argument("--hold", type=int, default=50, help="frames the outcome is held")
    parser.add_argument("--out", type=Path, default=Path("media/pick_compare"))
    args = parser.parse_args()

    torch.set_num_threads(1)
    pick_set = PickSet.load(args.set)
    grasps = [skill_from_spec(spec, "pick")[0] for spec in (args.before, args.after)]
    labels = (args.before_label, args.after_label)
    out = args.out.with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=20, codec="libx264", quality=8, macro_block_size=8)
    for index in args.cases:
        case = pick_set.cases[index]
        runs = [record(case, g, args) for g in grasps]
        for (_, r), label in zip(runs, labels, strict=True):
            print(f"{pick_set.name} case {index} ({case.kind}, yaw_rel {case.yaw_rel:+.1f} deg) "
                  f"{label}: success {r['success']}, failure {r['failure']}, "
                  f"steps {r['steps']}", flush=True)
        top = header(2 * args.width, f"{pick_set.name}, case {index}: the same {case.kind} "
                                     f"cube, turned {case.yaw_rel:+.0f} deg, the cell's own pick")
        n = max(len(f) for f, _ in runs)
        for i in range(n + args.hold):
            done = i >= n
            row = [panel(f[min(i, len(f) - 1)], label, outcome(r) if done else None)
                   for (f, r), label in zip(runs, labels, strict=True)]
            frame = np.vstack([top, np.hstack(row)])
            for _ in range(1 if done else args.slow):
                writer.append_data(frame)
    writer.close()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
