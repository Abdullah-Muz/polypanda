#!/usr/bin/env bash
# Download the trained policies published with a GitHub Release into runs/, verifying the
# archive's checksum. Afterwards every "run:runs/<name>" command in the docs works without
# training, e.g.: uv run python scripts/view_workcell.py --pick run:runs/grasp_cart_seed2 \
#     --place run:runs/place_stack_v3_seed2
# Usage: scripts/fetch_models.sh   (v3.2 holds every run of v1.0, v2.0, v3.0 and v3.1 as well)
set -euo pipefail

TAG="v3.2"
ARCHIVE="polypanda-models-${TAG}.tar.gz"
SHA256="b1ec9f4f4a8835ea4744b51a3ae0446874a36782f744b65fab285aea5e013449"
URL="https://github.com/Abdullah-Muz/polypanda/releases/download/${TAG}/${ARCHIVE}"

cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "downloading ${URL}"
curl -fL --retry 3 -o "${tmp}/${ARCHIVE}" "${URL}"
echo "${SHA256}  ${tmp}/${ARCHIVE}" | sha256sum -c -
tar -xzf "${tmp}/${ARCHIVE}" -C .
echo "extracted into runs/ ($(ls runs | wc -l) entries); see runs/MANIFEST.json"
