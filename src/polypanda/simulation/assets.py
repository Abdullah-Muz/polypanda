"""Locate and load the pinned MuJoCo Menagerie Franka Panda model."""

import os
import subprocess
from pathlib import Path

import mujoco

MENAGERIE_COMMIT = "cda5236f98383f6fbaa4820c6b18dba892cd087f"

_REPO_ROOT = Path(__file__).resolve().parents[3]


def menagerie_dir() -> Path:
    """Menagerie checkout; override with POLYPANDA_MENAGERIE_DIR."""
    override = os.environ.get("POLYPANDA_MENAGERIE_DIR")
    return Path(override) if override else _REPO_ROOT / "external" / "mujoco_menagerie"


def panda_scene_path() -> Path:
    path = menagerie_dir() / "franka_emika_panda" / "scene.xml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run scripts/fetch_assets.sh first.")
    return path


def menagerie_revision() -> str | None:
    """Commit currently checked out in the Menagerie directory, if it is a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(menagerie_dir()), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def load_panda_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(panda_scene_path()))
    return model, mujoco.MjData(model)
