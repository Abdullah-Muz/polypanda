#!/usr/bin/env bash
# Fetch the pinned MuJoCo Menagerie Franka Panda model into external/mujoco_menagerie.
# Uses a blobless sparse clone so only the Panda model (plus Menagerie's root files) is downloaded.
set -euo pipefail

MENAGERIE_URL="https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT="cda5236f98383f6fbaa4820c6b18dba892cd087f"  # 2026-10-19

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/mujoco_menagerie"

if [ ! -d "$DEST/.git" ]; then
    git clone --quiet --filter=blob:none --sparse "$MENAGERIE_URL" "$DEST"
fi
git -C "$DEST" sparse-checkout set franka_emika_panda
git -C "$DEST" fetch --quiet origin "$MENAGERIE_COMMIT" 2>/dev/null || true
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$MENAGERIE_COMMIT"

echo "Menagerie pinned at $(git -C "$DEST" rev-parse HEAD)"
