#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/target/release/tortuise-apple-vision-helper}"

mkdir -p "$(dirname "$OUT")"
xcrun swiftc \
  "$ROOT/helpers/apple_vision_hands.swift" \
  -O \
  -framework AVFoundation \
  -framework Vision \
  -o "$OUT"

echo "$OUT"
