#!/usr/bin/env bash
# Download the trained policies published with a GitHub Release into runs/, verifying the
# archive's checksum. Afterwards every "run:runs/<name>" command in the docs works without
# training, e.g.: uv run python scripts/demo_pickplace.py --grasp run:runs/grasp_ppo_curr_seed1
# Usage: scripts/fetch_models.sh
set -euo pipefail

TAG="v1.0"
ARCHIVE="polypanda-models-${TAG}.tar.gz"
SHA256="__FILLED_AT_RELEASE__"
URL="https://github.com/Abdullah-Muz/polypanda/releases/download/${TAG}/${ARCHIVE}"

cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "downloading ${URL}"
curl -fL --retry 3 -o "${tmp}/${ARCHIVE}" "${URL}"
echo "${SHA256}  ${tmp}/${ARCHIVE}" | sha256sum -c -
tar -xzf "${tmp}/${ARCHIVE}" -C .
echo "extracted into runs/ ($(ls runs | wc -l) entries); see runs/MANIFEST.json"
