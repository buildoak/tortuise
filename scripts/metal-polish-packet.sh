#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/metal-polish-packet.sh OUT_DIR [options]

Options:
  --smoke                 Run the first available exact calibrated packet only.
  --release               Pass --release to cargo run.
  --camera-pack PATH      Camera pack JSON (default: docs/metal-camera-packs.json).
  --probe-size WxH        Probe size (default: 256x192).
  --warmup N              Warmup frames (default: 3).
  --frames N              Benchmark frames (default: 10).
  --quality ID|all        Quality filter (default: exact).

Generated artifacts stay under OUT_DIR.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

out_dir=$1
shift

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
camera_pack="$repo_root/docs/metal-camera-packs.json"
probe_size="${METAL_POLISH_PROBE_SIZE:-256x192}"
warmup="${METAL_POLISH_WARMUP:-3}"
frames="${METAL_POLISH_FRAMES:-10}"
quality_filter="${METAL_POLISH_QUALITY:-exact}"
smoke=0
cargo_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      smoke=1
      shift
      ;;
    --release)
      cargo_args+=(--release)
      shift
      ;;
    --camera-pack)
      camera_pack=$2
      shift 2
      ;;
    --probe-size)
      probe_size=$2
      shift 2
      ;;
    --warmup)
      warmup=$2
      shift 2
      ;;
    --frames)
      frames=$2
      shift 2
      ;;
    --quality)
      quality_filter=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "$out_dir"
commands_log="$out_dir/commands.txt"
failures_log="$out_dir/failures.txt"
: >"$commands_log"
: >"$failures_log"

git_hash=$(git -C "$repo_root" rev-parse --short=12 HEAD 2>/dev/null || true)
git_dirty_entries=$(git -C "$repo_root" status --short 2>/dev/null | wc -l | tr -d ' ')
machine=$(uname -a)
run_id=$(basename "$out_dir")

cat >"$out_dir/packet_run.json" <<EOF
{
  "schema_version": 1,
  "repo": "$repo_root",
  "camera_pack": "$camera_pack",
  "git_hash": "$git_hash",
  "git_dirty_entries": "$git_dirty_entries",
  "machine": "$machine",
  "probe_size": "$probe_size",
  "warmup_frames": $warmup,
  "benchmark_frames": $frames,
  "quality_filter": "$quality_filter",
  "smoke": $smoke
}
EOF

packet_rows=$(python3 - "$camera_pack" "$quality_filter" "$smoke" <<'PY'
import json
import sys

pack_path, quality_filter, smoke_raw = sys.argv[1], sys.argv[2], sys.argv[3]
smoke = smoke_raw == "1"
with open(pack_path, "r", encoding="utf-8") as f:
    pack = json.load(f)
qualities = pack.get("qualities", [])
if quality_filter != "all":
    qualities = [q for q in qualities if q.get("quality_id") == quality_filter]

rows = []
for scene in pack.get("scenes", []):
    for angle in scene.get("angles", []):
        if smoke and angle.get("tier") not in ("calibrated", "primary"):
            continue
        for quality in qualities:
            rows.append((
                scene.get("scene_id", ""),
                scene.get("path", ""),
                str(scene.get("required", False)).lower(),
                angle.get("angle_id", ""),
                ",".join(str(x) for x in angle.get("camera_pos", [0, 0, 4])),
                ",".join(str(x) for x in angle.get("look_at", [0, 0, 0])),
                str(angle.get("fov_deg", 60.0)),
                angle.get("tier", "unknown"),
                quality.get("quality_id", "exact"),
                quality.get("probe_backends", "both"),
            ))
            if smoke:
                break
        if smoke and rows:
            break
    if smoke and rows:
        break

for row in rows:
    print("\t".join(row))
PY
)

status=0
ran=0
while IFS=$'\t' read -r scene_id scene_path required angle_id camera_pos look_at fov_deg tier quality_id probe_backends; do
  [[ -n "${scene_id:-}" ]] || continue
  if [[ ! -f "$repo_root/$scene_path" && ! -f "$scene_path" ]]; then
    if [[ "$required" == "true" ]]; then
      echo "$scene_id/$angle_id/$quality_id: missing scene $scene_path" | tee -a "$failures_log" >&2
      status=1
    fi
    continue
  fi
  if [[ -f "$repo_root/$scene_path" ]]; then
    scene_file="$repo_root/$scene_path"
  else
    scene_file="$scene_path"
  fi

  packet_dir="$out_dir/$scene_id/$angle_id/$quality_id"
  mkdir -p "$packet_dir"
  packet_commands="$packet_dir/commands.txt"
  : >"$packet_commands"

  cmd=(cargo run --features metal)
  if [[ ${#cargo_args[@]} -gt 0 ]]; then
    cmd+=("${cargo_args[@]}")
  fi
  cmd+=(
    --
    "$scene_file"
    --render-probe "$packet_dir"
    --probe-case loaded
    --probe-backends "$probe_backends"
    --probe-size "$probe_size"
    --probe-camera-pos "$camera_pos"
    --probe-look-at "$look_at"
    --probe-fov-deg "$fov_deg"
    --probe-warmup "$warmup"
    --probe-benchmark-frames "$frames"
    --probe-inspect-scale 2
    --probe-timing
    --probe-kitty-payload
  )
  if [[ "$probe_backends" == "both" ]]; then
    cmd+=(--probe-fail-on-mismatch)
  else
    cmd+=(--probe-stage-telemetry)
  fi
  if [[ "$quality_id" == "fast-preview" || "$quality_id" == "turbo" ]]; then
    cmd+=(--metal-quality "$quality_id")
  fi

  printf '%s\n' "${cmd[*]}" | tee -a "$commands_log" "$packet_commands"
  set +e
  (
    cd "$repo_root"
    TORTUISE_PROBE_RUN_ID="$run_id" \
    TORTUISE_PROBE_SCENE_ID="$scene_id" \
    TORTUISE_PROBE_ANGLE_ID="$angle_id" \
    TORTUISE_PROBE_QUALITY_ID="$quality_id" \
    TORTUISE_PROBE_GIT_HASH="$git_hash" \
    TORTUISE_PROBE_GIT_DIRTY_ENTRIES="$git_dirty_entries" \
    TORTUISE_PROBE_MACHINE="$machine" \
      "${cmd[@]}"
  )
  rc=$?
  set -e

  python3 - "$packet_dir" "$scene_id" "$angle_id" "$quality_id" "$tier" "$scene_file" "$rc" <<'PY'
import json
import os
import sys

packet_dir, scene_id, angle_id, quality_id, tier, scene_file, rc = sys.argv[1:]
packet = {
    "schema_version": 1,
    "scene_id": scene_id,
    "angle_id": angle_id,
    "quality_id": quality_id,
    "tier": tier,
    "scene_path": scene_file,
    "exit_status": int(rc),
    "probe_manifest": os.path.join(packet_dir, "probe_manifest.json"),
    "timing": os.path.join(packet_dir, "probe_timing.json"),
    "diff_summary": os.path.join(packet_dir, "diff", "summary.json"),
    "commands": os.path.join(packet_dir, "commands.txt"),
}
review = {
    "schema_version": 1,
    "task": "metal_polish_visual_review",
    "scene_id": scene_id,
    "angle_id": angle_id,
    "quality_id": quality_id,
    "tier": tier,
    "packet_manifest": os.path.join(packet_dir, "packet_manifest.json"),
    "probe_manifest": packet["probe_manifest"],
    "diff_summary": packet["diff_summary"],
    "primary_images": [
        os.path.join(packet_dir, "inspect_contact_sheet.png"),
        os.path.join(packet_dir, "metal", "frame_000.inspect.png"),
        os.path.join(packet_dir, "cpu", "frame_000.inspect.png"),
    ],
    "rubric": [
        "visible scene is not blank",
        "scene is not severely cropped unless tier is stress",
        "exact CPU/Metal artifacts do not show obvious tearing, holes, or wrong orientation",
        "approximate packets must be compared against exact references and temporal arcs",
    ],
    "output_schema": {
        "verdict": "pass|fail|uncertain",
        "release_blocker": "boolean",
        "issue_type": "none|crop|blank|flicker|holes|tearing|color|depth|performance_claim",
        "evidence_path": "path",
        "telemetry_path": "path",
        "confidence": "low|medium|high",
        "notes": "string",
    },
}
with open(os.path.join(packet_dir, "packet_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(packet, f, indent=2)
    f.write("\n")
with open(os.path.join(packet_dir, "review_task.json"), "w", encoding="utf-8") as f:
    json.dump(review, f, indent=2)
    f.write("\n")
PY

  if [[ $rc -ne 0 ]]; then
    echo "$scene_id/$angle_id/$quality_id: exit_status=$rc" | tee -a "$failures_log" >&2
    status=1
  fi
  ran=$((ran + 1))
done <<<"$packet_rows"

python3 "$repo_root/scripts/metal-polish-analyze.py" "$out_dir" >"$out_dir/coverage_index.json"

if [[ $ran -eq 0 ]]; then
  echo "no packets ran" | tee -a "$failures_log" >&2
  status=1
fi

if [[ $status -eq 0 ]]; then
  final_status=ok
else
  final_status=failed
fi
echo "{\"status\":\"$final_status\",\"out_dir\":\"$out_dir\",\"packets\":$ran,\"coverage_index\":\"$out_dir/coverage_index.json\"}"
exit "$status"
