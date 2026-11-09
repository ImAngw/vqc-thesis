#!/usr/bin/env bash
# Downloads the checkpoints and raw data that are not in git, verifies the sha256 and unpacks them in place.
#
#   ARTIFACTS_URL=https://zenodo.org/records/<id>/files/vqc-thesis-artifacts.tar.gz?download=1 bash scripts/fetch_artifacts.sh
#
# The URL is that of the published archive (see artifacts/README.md); the checksum is artifacts/SHA256SUMS.
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${ARTIFACTS_URL:-}"
SUMS=artifacts/SHA256SUMS
ARCHIVE=artifacts/vqc-thesis-artifacts.tar.gz

if [ -z "$URL" ] || [ ! -s "$SUMS" ]; then
  echo "The artifact archive is not published yet: set ARTIFACTS_URL and provide artifacts/SHA256SUMS (artifacts/README.md)." >&2
  exit 2
fi
[ -f "$ARCHIVE" ] || curl -L --fail -o "$ARCHIVE" "$URL"
( cd artifacts && { shasum -a 256 -c SHA256SUMS 2>/dev/null || sha256sum -c SHA256SUMS; } )
tar -xzf "$ARCHIVE"
echo "unpacked: checkpoints/ and quantum_heads/analysis/{data,runs}"
