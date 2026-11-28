"""v4.1's tester kinds leave v4.0's lid machine exactly as it was (docs/v4_1_plan.md, section 4.1).

With `MachineConfig.kind = "lid"` (the default) the machines, the door environment and its scripted
baselines must be bit-identical to v4.0's, so v4.0's trained door policies and results stand. These
fingerprints were taken from main at 2328db4, before the tester kinds were added, by running this
file against that commit's source:

    PYTHONPATH=<a checkout of 2328db4>/src uv run python tests/test_lid_identity.py

which prints the digests below. Each is a sha256 over exact bytes: a compiled model's binary
(mj_saveModel), or every observation, reward and end-of-episode record of a scripted door run on
`door_dev_v1` episodes (the development set). A change of a single bit changes it. The digests
are of this machine's floating point (the project's WSL with uv.lock's MuJoCo): run elsewhere,
regenerate them from 2328db4 first.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

# The digests (and the dev set's redraw) are this machine's floating point: on GitHub's runners
# (CI=true) some CPUs give other bits, so they are checked on the project's WSL only.
pytestmark = pytest.mark.skipif(os.environ.get("CI") == "true",
                                reason="exact floating point of the project's WSL machine")

ROOT = Path(__file__).resolve().parents[1]
DEV_SET = ROOT / "eval_sets" / "door_dev_v1.json"
# dev episodes covering every slide, both commands, springs: 0 open/right, 1 close/out,
# 4 open/out/spring, 9 close/right/spring, 10 open/left/spring, 23 close/left
EPISODES = (0, 1, 4, 9, 10, 23)
SCRIPTS = {"expert": EPISODES, "feel_v2": EPISODES, "feel": EPISODES[:3],
           "camera": EPISODES[:3], "nominal": EPISODES[:3]}
NOISE_SEED = 7_000_000          # scripts/evaluate_door.py's

EXPECTED = {
    "model:door_scene": "bd9663bb672a3f9d4402e7a2905efca638c4d8b5b57b0cc0cb1becb532ea74ac",
    "model:line": "5711786406814747da7c54f23679198f3495ddaa1b05c2a4442399bdd2e41648",
    "model:line_electronics": "4e3a3e8badffc9bee51f30292af217bf9593d199e0e9e2037a9be47890438193",
    "model:line_tight_nest": "379f023f34519c2b17f6abaa6067252f707c9a379d97f110d6604d7870cc8c9c",
    "model:machines": "bcb1233263b0adc5b2ef271da2c859764e9ce18c1c2a477e6e74476524ccb94c",
    "run:expert": "9df24820bdc0a1ee1937c49229b265c5cb3949e738eee02fa1357e0f7ceef765",
    "run:feel_v2": "857d7a65bed77216ee5b971a81715e8662fcef315fe1f928797ee3fd3b591f83",
    "run:feel": "5971a4de1415cc403cd0813d6e996e9b410ad5d846866d999a7e803cbfd9af55",
    "run:camera": "89fa4ba581d1826e06ade3f2ef5aa7d8fef105d79dbeb7a3ff416790ace31c17",
    "run:nominal": "72081984c0edd3cbd7508cd41d5a60b44b29195bdb695f2d6e674763d92354d7",
    "run:feel_v2_v2env": "ef5bdd33f173e3b788f39d3b2c919d02ec8af1697d9b21e66c0286550a69b71c",
}


def model_digest(model: mujoco.MjModel) -> str:
    buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


def machines_digest() -> str:
    """Bare specs with one machine each: every lid layout the door draws (slides, springs,
    handle placements), a keyed pocket and a tight nest."""
    from polypanda.simulation.machine import MachineConfig, add_machine, tight_nest
    h = hashlib.sha256()
    configs = [MachineConfig(), MachineConfig(slide="left", lid_spring=20.0),
               MachineConfig(slide="out", travel=0.27, handle_offset=0.02, handle_shift=-0.01,
                             handle_height=0.04),
               MachineConfig(keyed=True, key_wall=1), tight_nest()]
    for k, cfg in enumerate(configs):
        spec = mujoco.MjSpec()
        add_machine(spec, f"M{k}", (0.5, 0.1 * k), -30.0 * k, cfg)
        h.update(model_digest(spec.compile()).encode())
    return h.hexdigest()


def line_digest(**kw) -> str:
    from polypanda.simulation.line import Line, LineConfig
    return model_digest(Line.build_line(LineConfig(**kw)).model)


def run_digest(policy: str, indices, env_cls=None) -> str:
    """Every observation, reward and final info record of `policy` (a scripted door) on the
    given dev episodes, as scripts/evaluate_door.py runs them."""
    from polypanda.control.door_baselines import BaselineDoor
    from polypanda.control.scripted_door import DoorExpert
    from polypanda.environments.door import DoorEnv
    env = (env_cls or DoorEnv)()
    script = DoorExpert(env) if policy == "expert" else BaselineDoor(env, policy)
    episodes = json.loads(DEV_SET.read_text())["episodes"]
    h = hashlib.sha256()
    for i in indices:
        obs, _ = env.reset(seed=NOISE_SEED + i, options={"episode": episodes[i]})
        script.reset()
        h.update(np.asarray(obs, dtype=np.float32).tobytes())
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(script.act())
            h.update(np.asarray(obs, dtype=np.float32).tobytes())
            h.update(np.float64(r).tobytes())
            done = term or trunc
        st = info["stats"]
        h.update(json.dumps([info["is_success"], info["failure"], env.elapsed_steps,
                             float(st["peak_force"]), int(st["slams"]),
                             int(st["collision_steps"]),
                             env.data.qpos.tolist()]).encode())
    return h.hexdigest()


def compute(key: str) -> str:
    kind, name = key.split(":")
    if key == "model:door_scene":
        from polypanda.environments.door import DoorEnv
        return model_digest(DoorEnv().model)
    if key == "model:line":
        return line_digest()
    if key == "model:line_electronics":
        return line_digest(theme="electronics")
    if key == "model:line_tight_nest":
        from polypanda.simulation.line import MachineSpot
        from polypanda.simulation.machine import tight_nest
        return line_digest(machines=(MachineSpot("M1", -0.6, config=tight_nest()),),
                           keyed_parts=True, part_contact="stiff", theme="electronics")
    if key == "model:machines":
        return machines_digest()
    if key == "run:feel_v2_v2env":
        from polypanda.environments.door import DoorV2Env
        return run_digest("feel_v2", EPISODES[:2], DoorV2Env)
    return run_digest(name, SCRIPTS[name])


@pytest.mark.parametrize("key", list(EXPECTED))
def test_the_lid_is_bit_identical_to_v4_0(key):
    assert compute(key) == EXPECTED[key], f"{key} differs from main at 2328db4"


def test_the_door_dev_set_is_drawn_again_exactly():
    """scripts/make_door_sets.py, now with --kinds, still draws v4.0's development set to the
    byte (episodes and the recorded variation)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_door_sets",
                                                  ROOT / "scripts" / "make_door_sets.py")
    sets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sets)
    stored = json.loads(DEV_SET.read_text())
    episodes, variation = sets.make_set(stored["n"], stored["seed"], stored["stretch"])
    assert json.dumps(episodes, sort_keys=True) == json.dumps(stored["episodes"], sort_keys=True)
    assert json.loads(json.dumps(variation)) == stored["variation"]


if __name__ == "__main__":
    keys = sys.argv[1:] or list(EXPECTED)
    print(json.dumps({k: compute(k) for k in keys}, indent=1))
