#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 OUT_DIR [-- cargo-run-extra-args...]" >&2
  exit 2
fi

out_dir=$1
shift || true
if [[ "${1:-}" == "--" ]]; then
  shift
fi
cargo_args=()
if [[ $# -gt 0 ]]; then
  cargo_args=("$@")
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$out_dir"

git_hash=$(git -C "$repo_root" rev-parse --short=12 HEAD 2>/dev/null || true)
git_status=$(git -C "$repo_root" status --short 2>/dev/null | wc -l | tr -d ' ')
uname_value=$(uname -a)
rustc_value=$(rustc --version 2>/dev/null || true)
cargo_value=$(cargo --version 2>/dev/null || true)

cat >"$out_dir/metadata.json" <<EOF
{
  "version": 1,
  "repo": "$repo_root",
  "git_hash": "$git_hash",
  "git_dirty_entries": $git_status,
  "uname": "$uname_value",
  "rustc": "$rustc_value",
  "cargo": "$cargo_value"
}
EOF

commands_log="$out_dir/commands.txt"
: >"$commands_log"
failures_log="$out_dir/failures.txt"
: >"$failures_log"
matrix_status=0

run_probe() {
  local name=$1
  local fail_on_mismatch=$2
  shift 2

  local case_dir="$out_dir/$name"
  mkdir -p "$case_dir"

  local cmd=(cargo run --features metal)
  if [[ ${#cargo_args[@]} -gt 0 ]]; then
    cmd+=("${cargo_args[@]}")
  fi
  cmd+=(
    --
    --render-probe "$case_dir"
    --probe-backends both
    --probe-size "${PROBE_MATRIX_SIZE:-256x192}"
    --probe-camera-pos "${PROBE_MATRIX_CAMERA:-0,0,4}"
    --probe-look-at "${PROBE_MATRIX_LOOK_AT:-0,0,0}"
    --probe-warmup "${PROBE_MATRIX_WARMUP:-1}"
    --probe-inspect-scale "${PROBE_MATRIX_INSPECT_SCALE:-2}"
    --probe-timing
    "$@"
  )

  if [[ "$fail_on_mismatch" == "yes" && "${PROBE_MATRIX_STRICT_SYNTHETICS:-1}" != "0" ]]; then
    cmd+=(--probe-fail-on-mismatch)
  fi

  printf '%s\n' "$name: ${cmd[*]}" | tee -a "$commands_log"
  set +e
  (cd "$repo_root" && "${cmd[@]}")
  local status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    printf '%s\n' "$name: exit_status=$status" | tee -a "$failures_log" >&2
    matrix_status=1
  fi
}

run_probe blank yes --probe-case blank
run_probe channels yes --probe-case channels
run_probe depth yes --probe-case depth
run_probe tile-boundary yes --probe-case tile-boundary

run_probe demo no --demo --probe-case loaded

if [[ -f "$repo_root/scenes/luigi.ply" ]]; then
  run_probe luigi no scenes/luigi.ply --probe-case loaded
fi

if [[ -f "$repo_root/scenes/bonsai.splat" && "${PROBE_MATRIX_SKIP_BONSAI:-0}" == "0" ]]; then
  run_probe bonsai no scenes/bonsai.splat --probe-case loaded
fi

if [[ $matrix_status -eq 0 ]]; then
  echo "{\"status\":\"ok\",\"out_dir\":\"$out_dir\",\"metadata\":\"$out_dir/metadata.json\",\"commands\":\"$commands_log\",\"failures\":\"$failures_log\"}"
else
  echo "{\"status\":\"failed\",\"out_dir\":\"$out_dir\",\"metadata\":\"$out_dir/metadata.json\",\"commands\":\"$commands_log\",\"failures\":\"$failures_log\"}"
fi
exit "$matrix_status"
