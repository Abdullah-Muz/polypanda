#!/usr/bin/env bash
# Download the trained policies published with a GitHub Release into runs/, verifying the
# archive's checksum. Afterwards every "run:runs/<name>" command in the docs works without
# training, e.g.: uv run python scripts/view_workcell.py --pick run:runs/grasp_cart_seed2 \
#     --place run:runs/place_stack_v3_seed2
# Usage: scripts/fetch_models.sh   (v4.0 holds every run of v1.0 to v3.2 as well)
set -euo pipefail

TAG="v4.0"
ARCHIVE="polypanda-models-${TAG}.tar.gz"
SHA256="513eda0c052c75be4a3f9cffcbd5e919ebb55ab823a65705bb37cd81241c1bd6"
URL="https://github.com/Abdullah-Muz/polypanda/releases/download/${TAG}/${ARCHIVE}"

cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "downloading ${URL}"
curl -fL --retry 3 -o "${tmp}/${ARCHIVE}" "${URL}"
echo "${SHA256}  ${tmp}/${ARCHIVE}" | sha256sum -c -
tar -xzf "${tmp}/${ARCHIVE}" -C .
echo "extracted into runs/ ($(ls runs | wc -l) entries); see runs/MANIFEST.json"
