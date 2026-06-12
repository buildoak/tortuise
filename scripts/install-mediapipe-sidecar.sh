#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${TORTUISE_MEDIAPIPE_ENV_DIR:-$ROOT/.mediapipe-sidecar}"
PYTHON_VERSION="${TORTUISE_MEDIAPIPE_PYTHON:-3.12}"

usage() {
  cat >&2 <<EOF
usage: $0 [--dry-run]

Creates a project-local uv environment for helpers/mediapipe_hands_sidecar.py.

Environment overrides:
  TORTUISE_MEDIAPIPE_ENV_DIR   default: $ROOT/.mediapipe-sidecar
  TORTUISE_MEDIAPIPE_PYTHON    default: 3.12
EOF
}

dry_run=0
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi
if [[ $# -ne 0 ]]; then
  usage
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required; refusing to use global pip" >&2
  exit 127
fi

echo "env_dir=$ENV_DIR"
echo "python=$PYTHON_VERSION"
if [[ "$dry_run" == "1" ]]; then
  echo "dry_run=1"
  exit 0
fi

mkdir -p "$ENV_DIR"
cat >"$ENV_DIR/pyproject.toml" <<'EOF'
[project]
name = "tortuise-mediapipe-sidecar"
version = "0.0.0"
requires-python = ">=3.10,<3.13"
dependencies = [
  "mediapipe>=0.10.14",
  "numpy>=1.26",
  "opencv-python>=4.9",
]
EOF

uv sync --project "$ENV_DIR" --python "$PYTHON_VERSION"
"$ENV_DIR/.venv/bin/python" "$ROOT/helpers/mediapipe_hands_sidecar.py" --doctor || true

cat <<EOF
installed=$ENV_DIR/.venv
run=$ENV_DIR/.venv/bin/python $ROOT/helpers/mediapipe_hands_sidecar.py --doctor
EOF
