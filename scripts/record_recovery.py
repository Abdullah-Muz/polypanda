"""Record the v3 demo: a work order whose tallest tower is knocked flat the moment it is complete,
and the cell rebuilding it with learned skills, as a short captioned MP4 and a GIF.

Usage: uv run --group media python scripts/record_recovery.py [--out media/recovery]
  ... --order 91 --tower B --angle 0 --build-speed 48 --out media/recovery_91   # 4-4-4 order
  ... --order 191 --seed 1 --suite eval_sets/disturbance_calib_v1.json   # the order's own knock
By default: calibration order 9 (`eval_sets/disturbance_calib_v1.json`'s orders, seed 20261103)
with the seed-2 learned skills (grasp, place, stand-up), and one chosen knock: a 1.2 m/s shove on
the middle cube of tower A the moment it reaches 4 cubes. The knock was chosen for the demo:
among 16 shoves, it brought the whole tower down and was rebuilt without a single failed skill
attempt. Order 91, tower B (0 deg): all three towers stand complete at 4 cubes when the middle one
is knocked flat; it is rebuilt with one grasp retried. With --suite, the order runs with that
suite's own knock (the calibration run's, chosen by nobody). Everything after the knock is the
cell's own, unscripted recovery. The video plays the build (x32 by default), the knock (its first
0.6 s in slow motion, x0.2, rendered between physics steps; then x2) and the rebuild (x12). On
screen: the work order (its squares light up as cubes stand on the towers), what is running (each
caption held 1.5 s at least), the robot's last decisions from its own log in plain words, and the
playback speed; it ends on the held-out results and the repository.
"""

import argparse
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from record_cell import write_gif
from record_demo import font

from polypanda.evaluation.cell import CellEvalConfig, run_cell_order, skill_from_spec
from polypanda.simulation.disturbances import Disturbance, Knock, load_suite, save_suite
from polypanda.simulation.workcell import COLOURS, TOWERS, Workcell, WorkcellConfig
from polypanda.supervisor.cell import LearnedCell
from polypanda.supervisor.observe import observe

ORDERS_SEED = 20261103
SPEED = {"build": 32, "knock": 1, "rebuild": 12}
AFTER_KNOCK = 40          # control steps shown (at x2) after the knock settles
FPS = 20                  # one frame per control step in real time
HOLD = 30                 # frames a caption stays up at least (1.5 s)
FINE, SUB = 12, 5         # the knock's first control steps, rendered 5 times per step (x0.2)


def plain(line: str, source: str) -> str | None:
    """The controller's log line (ScriptedCell.run_mixed_order) in a few plain words, for the
    robot's log on screen; None for lines not worth showing."""
    line = line.strip()
    if line.startswith("knock"):
        return None                                        # (the caption says it)
    if line.startswith("tower ") and "fell" in line:
        t, rest = line.split(":", 1)
        return f"Sees {t}: {rest.split(';')[0].strip()}"   # e.g. "Sees tower B: 4 cubes fell"
    if "written off (leaning" in line:
        return f"{line.split()[0].capitalize()} cube leaning: left alone"
    if "written off" in line:
        return f"{line.split()[0].capitalize()} cube out of reach: left"
    if line.startswith("clearing the"):
        return f"Moving the {line.split()[2]} cube clear"
    if line.startswith("recovering the"):
        return f"Picking up the {line.split()[2]} cube"
    if line.startswith("buffer slot") and "-> tower" in line:
        colour = line.split("(")[1].split(")")[0]
        return f"{colour.capitalize()} from the buffer -> {line.split('-> ')[1]}"
    if "cube -> tower" in line:
        colour, where = line.split()[0], line.split("-> ")[1]
        new = " new" if source == "belt" else ""
        return f"{colour.capitalize()}{new} cube -> {where}"
    if "cube -> buffer" in line:
        return f"{line.split()[0].capitalize()} cube -> buffer (needed later)"
    if "reject bin" in line:
        return f"{line.split()[0].capitalize()} cube not needed -> bin"
    return None


class RecordingCell(LearnedCell):
    """LearnedCell that tells the recorder what it is doing and what it decides."""

    def __init__(self, *args, knocked: str = "A", **kwargs):
        super().__init__(*args, **kwargs)
        self.phase, self.doing, self.knock_end, self.knocked = "build", "", None, knocked
        self.log_lines: list[str] = []
        self.fine_left, self.in_fine, self.sub_render = 0, False, None

    def step(self, n: int = 1) -> None:
        """As ScriptedCell.step; during the knock's first FINE control steps the 25 physics
        steps of each are taken in SUB pieces, with a frame after each (the same physics)."""
        for _ in range(n):
            if self.fine_left <= 0 or self.sub_render is None:
                super().step(1)
                continue
            self.fine_left -= 1
            self.d.ctrl[self.arm.arm_actuators] = self.q_cmd
            self.d.ctrl[self.arm.gripper_actuator] = self.grip
            self.cell.conveyor.update()
            for _ in range(SUB):
                mujoco.mj_step(self.m, self.d, nstep=25 // SUB)
                self.sub_render()
            self.in_fine = True
            if self.on_step:
                self.on_step()
            self.in_fine = False

    def run_mixed_order(self, order, rng, **kw):
        log = kw.get("log")

        def keep(line: str) -> None:
            text = plain(line, self.source)
            if text and self.phase != "build":
                self.log_lines.append(text)
            if log:
                log(line)
        return super().run_mixed_order(order, rng, **{**kw, "log": keep})

    def knock(self, tower, knock, log=None):
        self.phase, self.doing = "knock", f"Tower {self.knocked} knocked over"
        self.fine_left = FINE
        super().knock(tower, knock, log)
        self.knock_end = self.steps_seen + AFTER_KNOCK

    def stand_up(self, cube, spot):
        self.doing = "Learned stand-up (PPO): turning a fallen cube upright"
        super().stand_up(cube, spot)

    def pick(self, cube):
        if self.phase != "build" and self.source in ("loose", "stage"):
            self.doing = "Learned grasp (PPO): picking up a fallen cube"
        elif self.phase != "build":
            self.doing = "Learned grasp (PPO): the next cube from the belt"
        super().pick(cube)

    def place(self, cube, xy, level):
        if self.phase != "build":
            angle = float(np.degrees(np.arctan2(xy[1], xy[0])))
            towers = self.cell.config.tower_angles
            t = int(np.argmin([abs(angle - a) for a in towers]))
            self.doing = (f"Learned place (PPO): rebuilding tower {TOWERS[t]}, level {level + 1}"
                          if abs(angle - towers[t]) < 5 else
                          "Moving it clear of the towers")
        super().place(cube, xy, level)


def text_scale(img: Image.Image) -> float:
    """Text sized for 1280 x 720 (1.0); a square 1080 x 1080 frame keeps the same pixel sizes."""
    return min(img.width / 1080, img.height / 720)


def rgb(colour: str, scale: float = 1.0) -> tuple[int, int, int]:
    return tuple(int(255 * c * scale) for c in COLOURS[colour][:3])


def overlay(frame: np.ndarray, recipes, heights, doing: str, speed: str,
            banner: str | None = None, log_lines=()) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    s = text_scale(img)
    # the work order: one row per tower, its squares lit once their cubes stand on it
    sq, gap, x0, y0 = round(30 * s), round(8 * s), round(20 * s), round(20 * s)
    title = font(round(24 * s))
    width = round(60 * s) + 4 * (sq + gap)
    draw.rounded_rectangle((x0 - 12 * s, y0 - 12 * s, x0 + width, y0 + 40 * s + 3 * (sq + gap)),
                           radius=round(10 * s), fill=(0, 0, 0, 150))
    draw.text((x0, y0), "Work order", font=title, fill=(255, 255, 255))
    label = font(round(26 * s))
    for t, recipe in enumerate(recipes):
        y = y0 + round(40 * s) + t * (sq + gap)
        draw.text((x0, y), TOWERS[t], font=label, fill=(255, 255, 255))
        for k, colour in enumerate(recipe):
            x = x0 + round(40 * s) + k * (sq + gap)
            lit = k < heights[t]
            draw.rectangle((x, y, x + sq, y + sq), fill=rgb(colour, 1.0 if lit else 0.25),
                           outline=rgb(colour), width=max(2, round(3 * s)))
    # the playback speed, top right
    f = font(round(28 * s))
    box = draw.textbbox((0, 0), speed, font=f)
    xs = img.width - (box[2] - box[0]) - round(24 * s)
    draw.rectangle((xs - 10 * s, 16 * s, img.width - 14 * s, 26 * s + box[3] - box[1] + 6 * s),
                   fill=(0, 0, 0, 150))
    draw.text((xs, 18 * s), speed, font=f, fill=(255, 255, 255))
    # the robot's log, under it: its last decisions, newest first
    if log_lines:
        f, ft = font(round(24 * s)), font(round(20 * s))
        rows = [("Robot's log", ft, (190, 190, 190))] + [
            (ln, f, (255, 255, 255) if i == 0 else (160, 160, 160))
            for i, ln in enumerate(reversed(log_lines[-3:]))]
        widths = [draw.textbbox((0, 0), t, font=ff)[2] for t, ff, _ in rows]
        x_right, y = img.width - round(14 * s), round(70 * s)
        x_left = x_right - max(widths) - round(24 * s)
        heights_px = [draw.textbbox((0, 0), t, font=ff)[3] for t, ff, _ in rows]
        draw.rectangle((x_left, y, x_right, y + sum(heights_px) + round(12 * s) * len(rows)
                        + round(8 * s)), fill=(0, 0, 0, 150))
        y += round(8 * s)
        for (t, ff, colour), h in zip(rows, heights_px, strict=True):
            draw.text((x_left + round(12 * s), y), t, font=ff, fill=colour)
            y += h + round(12 * s)
    # what is running, bottom centre (a banner replaces it at the end)
    text, fill = (banner, (20, 110, 40, 220)) if banner else (doing, (0, 0, 0, 160))
    if text:
        knocked = text.endswith("knocked over")
        f = font(round(34 * s))
        box = draw.textbbox((0, 0), text, font=f)
        x = (img.width - (box[2] - box[0])) // 2
        y = img.height - (box[3] - box[1]) - round(36 * s)
        draw.rectangle((x - 18 * s, y - 12 * s, x + box[2] - box[0] + 18 * s,
                        y + box[3] - box[1] + 18 * s),
                       fill=(170, 30, 30, 220) if knocked else fill)
        draw.text((x, y), text, font=f, fill=(255, 255, 255))
    return np.asarray(img)


def end_card(width: int, height: int, lines: list[tuple[str, int, tuple]]) -> np.ndarray:
    img = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(img)
    s = text_scale(img)
    sizes = [draw.textbbox((0, 0), t, font=font(round(z * s))) for t, z, _ in lines]
    total = sum(b[3] - b[1] for b in sizes) + round(26 * s) * (len(lines) - 1)
    y = (height - total) // 2
    for (text, size, colour), b in zip(lines, sizes, strict=True):
        draw.text(((width - (b[2] - b[0])) // 2, y), text, font=font(round(size * s)), fill=colour)
        y += b[3] - b[1] + round(26 * s)
    return np.asarray(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2, help="which trained seed's skills")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gif-width", type=int, default=480)
    parser.add_argument("--order", type=int, default=9, help="calibration order number")
    parser.add_argument("--tower", choices=TOWERS, default="A", help="the tower knocked at 4")
    parser.add_argument("--angle", type=float, default=180.0, help="shove direction, deg")
    parser.add_argument("--strength", type=float, default=1.2, help="shove speed, m/s")
    parser.add_argument("--build-speed", type=int, default=SPEED["build"])
    parser.add_argument("--rebuild-speed", type=int, default=SPEED["rebuild"])
    parser.add_argument("--square", action="store_true",
                        help="1080 x 1080 for social feeds, framed closer on the towers")
    parser.add_argument("--suite", type=Path,
                        help="film the order with this suite's own knock instead of a chosen "
                             "one (e.g. eval_sets/disturbance_calib_v1.json)")
    parser.add_argument("--out", type=Path, default=Path("media/recovery"))
    args = parser.parse_args()
    SPEED.update(build=args.build_speed, rebuild=args.rebuild_speed)
    if args.square:
        args.width = args.height = 1080
    index, tower = args.order, TOWERS.index(args.tower)

    with tempfile.TemporaryDirectory() as tmp:
        if args.suite:
            suite, knocked = args.suite, TOWERS[load_suite(args.suite)[index].tower]
        else:
            suite, knocked = Path(tmp) / "demo_knock.json", args.tower
            knocks = [None] * (index + 1)
            knocks[index] = Disturbance(tower, 4, Knock("shove_middle",
                                                        float(np.radians(args.angle)),
                                                        args.strength))
            save_suite(suite, knocks, {"demo": f"tower {knocked} knocked at 4 cubes, "
                                               f"order {index}"})
        config = CellEvalConfig(seed=ORDERS_SEED, mixed=True, recover=True,
                                disturbances=str(suite))
        s = args.seed
        skills = {stage: skill_from_spec(f"run:runs/{run}_seed{s}", stage)[0]
                  for stage, run in (("pick", "grasp_anypose5"), ("place", "place_stack_v3"),
                                     ("standup", "standup_round11"))}
        cell = Workcell.build(WorkcellConfig(n_cubes=config.n_cubes))
        model = cell.model
        model.mat_reflectance[:] = 0.0  # floor reflections render with streak artifacts
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        renderer = mujoco.Renderer(model, args.height, args.width)
        camera = mujoco.MjvCamera()  # the towers, their recipe signs, the knock and the belt
        camera.azimuth, camera.elevation, camera.distance = 250, -35, 1.95
        camera.lookat[:] = [0.05, -0.22, 0.12]
        if args.square:  # a narrower view: back off and centre on the towers
            camera.distance, camera.lookat[:] = 2.3, [0.03, -0.27, 0.15]
        ctl = RecordingCell(cell, grasp=skills["pick"], place=skills["place"],
                            standup=skills["standup"], knocked=knocked)
        ctl.steps_seen = 0
        frames: list[np.ndarray] = []
        shown = {"caption": "", "since": 0}

        def caption() -> str:
            """What is running, each caption held for at least HOLD frames (the knock's at
            once)."""
            want = ("Building three towers to their recipes" if ctl.phase == "build"
                    else ctl.doing)
            urgent = ctl.phase == "knock" and shown["caption"] != want
            if want != shown["caption"] and (urgent or len(frames) - shown["since"] >= HOLD):
                shown.update(caption=want, since=len(frames))
            return shown["caption"]

        def render(banner=None, fine=False) -> np.ndarray:
            renderer.update_scene(cell.data, camera)
            heights = [len(t) for t in observe(cell).towers]
            speed = ("slow motion" if fine else "x2" if ctl.phase == "knock"
                     else f"x{SPEED[ctl.phase]}")
            return overlay(renderer.render(), cell.order.recipes, heights, caption(), speed,
                           banner, ctl.log_lines)

        def on_step():
            ctl.steps_seen += 1
            if ctl.phase == "knock" and ctl.knock_end and ctl.steps_seen >= ctl.knock_end:
                ctl.phase, ctl.doing = "rebuild", ""
            if ctl.in_fine:
                return                            # (its frames came from sub_render)
            speed = 2 if ctl.phase == "knock" else SPEED[ctl.phase]
            if ctl.steps_seen % speed == 0:
                frames.append(render())

        ctl.on_step = on_step
        ctl.sub_render = lambda: frames.append(render(fine=True))
        episode = run_cell_order(ctl, config, index)
        last = render(f"Order complete: tower {knocked} rebuilt")
        frames += [last] * round(1.5 * FPS)
        card = end_card(args.width, args.height, [
            ("Knocked over, then rebuilt:", 40, (255, 255, 255)),
            ("learned skills only", 40, (255, 255, 255)),
            ("PPO policies grasp, stand cubes upright and stack them", 26, (200, 200, 200)),
            ("Held-out test, 200 orders x 3 seeds:", 26, (120, 210, 140)),
            ("93 % / 88 % / 92 % of recoverable knocks rebuilt", 26, (120, 210, 140)),
            ("(one cube / some / the whole tower knocked off)", 22, (120, 210, 140)),
            ("github.com/Abdullah-Muz/polypanda", 28, (255, 210, 80)),
            ("PolyPanda v3 · MuJoCo · Stable-Baselines3 · simulation", 22, (150, 150, 150))])
        frames += [card] * round(2.5 * FPS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mp4, gif = args.out.with_suffix(".mp4"), args.out.with_suffix(".gif")
    imageio.mimwrite(mp4, frames, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    small = (args.gif_width, round(args.height * args.gif_width / args.width))
    gif_frames = [np.asarray(Image.fromarray(f).resize(small, Image.LANCZOS))
                  for f in frames[::2]]
    write_gif(gif, gif_frames, fps=FPS // 2)
    print(f"order {index}: {'complete' if episode.success else episode.failure}, knock "
          f"{episode.disturbance}, {episode.fallen} cubes fell, {episode.sim_s:.0f} s simulated; "
          f"{mp4}: {len(frames) / FPS:.1f} s ({mp4.stat().st_size / 1e6:.1f} MB); {gif}")


if __name__ == "__main__":
    main()
