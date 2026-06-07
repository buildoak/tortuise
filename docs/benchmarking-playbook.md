# Benchmarking Playbook

This playbook defines how to measure tortuise rendering performance and
correctness without comparing incomparable views.

The benchmark unit is a deterministic `--render-probe` run. A speed number is
not valid unless the run also records correctness, occupancy, camera, timing,
and stage telemetry.

## Probe Artifact Contract

Keep each benchmark under a timestamped `target/` directory. When using
`scripts/probe-matrix.sh`, preserve:

- `metadata.json`
- `commands.txt`
- `failures.txt`

Each probe case should preserve:

- `probe_manifest.json`
- `probe_timing.json` when timing is enabled
- `contact_sheet.ppm`
- `inspect/contact_sheet.xN.png`
- `cpu/frame_000.ppm` and `cpu/frame_000.json` for CPU runs
- `metal/frame_000.ppm`, `metal/frame_000.json`, and
  `metal/frame_000.packed_u32le.bin` for Metal runs
- `diff/summary.json` and `inspect/diff/cpu_vs_metal_frame_000.xN.png` for
  CPU-vs-Metal runs
- `terminal/*_frame_000.ansi.txt` when `--probe-terminal` is used
- `kitty/*_frame_000.rgba` and `kitty/*_frame_000.json` when
  `--probe-kitty-payload` is used

`probe_manifest.json` is not enough provenance by itself. Keep the exact
command line in `commands.txt` or in the benchmark notes.

## Canonical Assets

Repo fixtures:

- `scenes/luigi.ply`
- `scenes/bonsai.splat`

Canonical external splats live under:

```text
/Users/otonashi/thinking/external/splats/
```

Use the `splat-ops` skill and `splat-assets find <name>` before introducing a
new benchmark asset. Do not move repo-owned sample scenes out of the repo.

Known useful external scenes:

- Bee: `imported/staged/supersplat/bee/bee_cf6ac78e.ply`
- Ramen: `imported/staged/supersplat/ramen/ramen_d281f99f.ply`
- Ignatius stable: `imported/staged/benchmarks/ignatius/ignatius_10000_stable.splat`
- Meeting room stable dec50:
  `imported/staged/benchmarks/meetingroom/meetingroom_30k_20000_stable_dec50.splat`

## Camera Packs

Never compare two runs unless these match:

- scene file
- probe size
- camera position
- look-at target or yaw/pitch
- FOV
- backend mode and Metal env flags
- release/debug mode
- machine
- git revision

Default synthetic camera:

```text
--probe-camera-pos 0,0,4 --probe-look-at 0,0,0 --probe-fov-deg 60
```

Loaded scenes must use named camera packs:

- `loaded-calibrated`: scene-specific view that passes the occupancy gate.
- `loaded-stress`: pathological or intentionally sparse/dense view. Report
  separately from calibrated speed.

The Bee failure mode is the reason this rule exists: one tested Bee view had
about `2.3M` source splats but only `48` nonblack pixels, so comparing its
speedup to a dense scene was misleading.

## Correctness Gate

Run strict synthetic correctness before any speed claim:

```bash
PROBE_MATRIX_SIZE=256x192 PROBE_MATRIX_WARMUP=3 \
  ./scripts/probe-matrix.sh target/probe-matrix-YYYYMMDD-HHMM -- --release
```

For individual strict cases:

```bash
cargo run --features metal --release -- \
  --render-probe target/probe/channels \
  --probe-backends both \
  --probe-size 256x192 \
  --probe-camera-pos 0,0,4 \
  --probe-look-at 0,0,0 \
  --probe-warmup 3 \
  --probe-frames 3 \
  --probe-case channels \
  --probe-timing \
  --probe-stage-telemetry \
  --probe-fail-on-mismatch
```

Synthetic cases `blank`, `channels`, `depth`, and `tile-boundary` must pass
with `--probe-fail-on-mismatch`.

For exact loaded CPU-vs-Metal checks, require `diff/summary.json` classification
to be `pass`. Treat `channel_swap`, `global_shift`, and `structured_mismatch` as
failures unless the run is explicitly filed as triage evidence.

Approximate modes do not need exact CPU parity, but they must stay opt-in and
pass visual review.

## Occupancy Gate

For loaded scenes, record occupancy from `cpu/frame_000.json` or
`metal/frame_000.json`:

```bash
jq '{nonblack_pixels,bbox,luma_mean,luma_p95,checksum}' \
  "$OUT/$LABEL/metal/frame_000.json"
```

Compute:

```text
visible_pct = nonblack_pixels / (width * height) * 100
```

Occupancy tiers:

- `0%`: invalid unless the case is `blank`.
- `<1%`: pathology/stress only. Do not use for dense-scene speed claims.
- `1-10%`: sparse tier. Compare only to other sparse-tier runs.
- `10-95%`: primary benchmark tier.
- `>95%`: saturated/dense tier. Requires visual review for clipping.

Also report bbox area ratio when the object is tiny or cropped.

## Tile Pressure Gate

Tile pressure requires `--probe-stage-telemetry`.

Check:

```bash
jq '{sort_path,actual_total_overlaps,valid_count,attempt_sort_count,overflow_flag,tile_density}' \
  "$OUT/$LABEL/metal/stage_telemetry_frame_000.json"
```

Rules:

- `overflow_flag` must be `0`.
- `retry_count` must be `0` unless overflow recovery is the test.
- `actual_total_overlaps` must match `tile_density.total_tile_entries`.
- Always report `total_tile_entries`, `max_tile_range`, `p99_tile_range`, and
  `tile_ranges_ge_8192`.

Tile pressure tiers, using entries per pixel:

- `<1`: light
- `1-10`: moderate
- `10-30`: heavy
- `>30`: extreme

Any `tile_ranges_ge_8192 > 0` is a hotspot/stress run and must be labeled.

## Speed Gate

Speed claims require release builds:

```bash
cargo run --features metal --release -- "$SCENE" \
  --render-probe "$OUT/$LABEL-metal" \
  --probe-case loaded \
  --probe-backends metal \
  --probe-size 256x192 \
  --probe-camera-pos "$CAM" \
  --probe-look-at "$LOOK" \
  --probe-warmup 3 \
  --probe-benchmark-frames 30 \
  --probe-stage-telemetry \
  --probe-inspect-scale 2
```

Use CPU-only and Metal-only runs for speed comparison:

```bash
cargo run --features metal --release -- "$SCENE" \
  --render-probe "$OUT/$LABEL-cpu" \
  --probe-case loaded \
  --probe-backends cpu \
  --probe-size 256x192 \
  --probe-camera-pos "$CAM" \
  --probe-look-at "$LOOK" \
  --probe-warmup 3 \
  --probe-benchmark-frames 30
```

Compare `render_avg_ms`, not wall time and not artifact write time.

Benchmark decision rules:

- Run at least three benchmark runs and compare the median.
- A win requires `>=10%` faster median with the same correctness and occupancy
  gates.
- `+5%` to `+15%` slower is a watch result.
- `>15%` slower is a regression unless the change intentionally moves work to a
  different quality tier.

## Visual-Subagent Gate

Visual review is required for:

- new camera packs
- public benchmark tables
- occupancy tier changes
- approximate renderer modes
- claimed speed wins over `20%`

Send visual reviewers:

- `inspect/contact_sheet.xN.png`
- `inspect/metal/frame_000.xN.png`
- `inspect/cpu/frame_000.xN.png` when available
- `inspect/diff/cpu_vs_metal_frame_000.xN.png` when available
- `diff/summary.json`
- `metal/stage_telemetry_frame_000.json`

Rubric:

- scene is not blank unless expected
- expected object/scene is visible
- no misleading tiny-object view for primary speed claims
- no channel swap
- no global shift
- no axis flip
- no severe crop or clipping
- no obvious depth-order artifacts for approximate modes

## Hard Rules

- Do not compare Bee-like tiny-visible frames to dense scenes.
- Do not publish speed without occupancy, correctness, timing, telemetry, and
  visual gates.
- Do not use debug builds for speed.
- Do not make tile-density claims without `--probe-stage-telemetry`.
- Do not compare across camera packs, resolutions, scene files, backend modes,
  sort paths, or machines.
- Do not treat `probe_timing.json` as sufficient provenance; keep exact
  commands.
- Do not make Kitty throughput claims without `--probe-kitty-payload`; report
  RGBA bytes, base64 bytes, and chunk count.

## Checklist

- Relevant tests pass.
- Strict synthetic matrix passes.
- Loaded run uses a named camera pack.
- Exact command is preserved.
- Occupancy tier is assigned.
- Tile pressure tier is assigned.
- Stage telemetry is checked.
- Three-run median speed is compared to a matching baseline.
- Visual packet is reviewed when required.
- Result is labeled: `correctness`, `primary-speed`, `sparse`, `dense`,
  `stress`, or `pathology`.
