#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_URL="${HAND_LANDMARKER_URL:-https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task}"
OUT="${1:-$ROOT/models/hand_landmarker.task}"

usage() {
  cat >&2 <<EOF
usage: $0 [OUT] [--dry-run]

Fetches the MediaPipe Hand Landmarker task bundle.

Environment overrides:
  HAND_LANDMARKER_URL   default: $MODEL_URL
EOF
}

dry_run=0
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${2:-}" == "--dry-run" || "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  if [[ "${1:-}" == "--dry-run" ]]; then
    OUT="$ROOT/models/hand_landmarker.task"
  fi
fi

echo "url=$MODEL_URL"
echo "out=$OUT"
if [[ "$dry_run" == "1" ]]; then
  echo "dry_run=1"
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
tmp="$OUT.tmp.$$"
trap 'rm -f "$tmp"' EXIT
curl -L --fail --show-error --output "$tmp" "$MODEL_URL"

bytes=$(wc -c <"$tmp" | tr -d ' ')
if [[ "$bytes" -lt 1000000 ]]; then
  echo "error: downloaded model is unexpectedly small: $bytes bytes" >&2
  exit 1
fi

mv "$tmp" "$OUT"
trap - EXIT
echo "$OUT"
