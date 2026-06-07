# Metal Optimization Plan

This plan follows the CLI-first probe work. Metal changes must be measured with
the benchmarking playbook before they are treated as real wins.

The current goal is exact Metal becoming first-class: correct by default,
measurably faster than CPU, and instrumented enough that future approximate
paths can be judged honestly.

## Current State

The exact Metal path now has synthetic correctness gates, loaded-scene
CPU-vs-Metal diff artifacts, timing summaries, stage telemetry, SnugBox tile
localization, and fused command-buffer execution when capacity is already
available.

Known telemetry fields to preserve:

- `sort_path`
- `valid_count`
- `actual_total_overlaps`
- `estimated_overlaps`
- `attempt_sort_count`
- `overflow_flag`
- `retry_count`
- `tile_density`

The most important current pathology is sparse high-source scenes. Bee can have
about `2.3M` source splats but only tens of thousands of visible overlaps in a
small probe view. If the exact fused path sorts millions of padded entries for
that view, the benchmark is measuring allocator/sort waste, not useful splat
work.

## Baseline Gate

Before each optimization wave, capture a release benchmark matrix:

```bash
cargo run --features metal --release -- "$SCENE" \
  --render-probe "$OUT/$LABEL" \
  --probe-case loaded \
  --probe-backends metal \
  --probe-size "$SIZE" \
  --probe-camera-pos "$CAM" \
  --probe-look-at "$LOOK" \
  --probe-warmup 5 \
  --probe-benchmark-frames 30 \
  --probe-stage-telemetry \
  --probe-timing \
  --probe-inspect-scale 2
```

Minimum matrix:

- scenes: `scenes/luigi.ply`, `scenes/bonsai.splat`, Bee, Ramen, Ignatius
- sizes: `128x96`, `256x192`, `512x384`
- modes: exact fused default, `TORTUISE_METAL_SORT_PATH=hybrid`, and
  `TORTUISE_METAL_SORT_PATH=coarse-depth TORTUISE_METAL_FAST_DEPTH_BITS=14`
  as approximate reference only

Use `--probe-warmup 5`, `--probe-benchmark-frames 30`, stage telemetry, and a
three-run median for speed claims.

Correctness smoke must still run strict synthetic cases:

```bash
PROBE_MATRIX_SIZE=256x192 PROBE_MATRIX_WARMUP=3 \
  ./scripts/probe-matrix.sh target/probe-matrix-YYYYMMDD-HHMM -- --release
```

## Phase 1: Exact Fused Estimate And Retry

Immediate ticket:

```text
metal: fix warmed fused sort estimate and add bounded overflow retry
```

Scope:

- `src/render/metal/render.rs`
- focused tests if the estimator is extracted
- no shader rewrite
- no hybrid auto-policy
- no approximate defaults

Change the warmed high-source estimate so it is based on the previous overlap
count and tile-count scaling, not floored to `splat_count`. Keep cold start
conservative. If the estimate overflows, retry the same frame once or twice with
capacity based on observed overlaps, then report `OverflowDeferred` only after
the retry budget is exhausted.

Done means:

- warmed sparse scenes no longer use `splat_count` as the fused attempt floor
- overflow retries are reported through `retry_count`
- Bee `256x192` warm frame has
  `attempt_sort_count / actual_total_overlaps <= 2.0`
- `cargo test` and `cargo test --features metal` pass
- release matrix shows at least `25%` Bee exact improvement and no more than
  `5%` regression on Bonsai, Ramen, or Ignatius

Expected impact:

- largest exact win for sparse high-source scenes
- lower padded radix work without changing output semantics

## Phase 2: Sparse Policy Decision

The `hybrid` path already sorts actual overlaps, but it pays a CPU wait/readback
and a second command-buffer split. Do not make it default by theory.

Gate:

- compare fixed fused default vs `TORTUISE_METAL_SORT_PATH=hybrid`
- require at least `10%` median win and no `>5%` loss across the benchmark
  matrix before auto-selecting hybrid for any class of scene
- keep `sort_path` in telemetry so benchmark tables can separate policy modes

Outcome:

- either keep hybrid as a manual diagnostic mode
- or add a narrow automatic sparse policy with explicit telemetry

## Phase 3: GPU Valid-Count Dispatch

Projection currently must inspect source splats, but later stages should not
continue dispatching over source count when only a small valid subset survives.

Implement GPU-side indirect dispatch so count/emit work is driven by
`valid_count` where the pipeline permits it.

Gate:

- Bee-like sparse views show lower stage time after projection
- dense scenes do not regress by more than `5%`
- telemetry still reports source count, valid count, and overlap count
- overflow behavior remains exact

## Phase 4: GPU Actual-Overlap Radix

If Phase 1 leaves padded sort as a dominant cost, remove the hybrid path's CPU
split by deriving the actual radix dispatch count on GPU.

Gate:

- exact output still matches CPU/diff gates
- no same-frame CPU wait for overlap count
- `attempt_sort_count` tracks actual overlaps with bounded headroom
- release matrix beats fixed fused default by at least `10%` where selected

## Phase 5: Exact Cluster Culling

Do not start with broad LOD. First add exact cluster-level rejection only if
telemetry shows projection remains the dominant cost after sort waste is fixed.

Requirements:

- cluster bounds must never drop visible splats
- loaded-scene exact diff must still pass
- counters must report clusters tested, clusters rejected, splats projected,
  valid splats, and overlaps

This phase is worthwhile only for large scenes with small occupied screen area.

## Phase 6: Approximate Renderer Track

Approximate renderers are separate products, not correctness shortcuts.

Near-term candidate:

- stabilize `coarse-depth` with deterministic tie bits before any default use

Implemented first fast tier:

- `--metal-quality exact` keeps the exact fused renderer.
- `--metal-quality fast-preview` keeps exact depth ordering but enables
  opacity-aware tile radius tightening and faster raster accumulation.
- `--metal-quality turbo` is the aggressive unsorted diagnostic tier.
- `TORTUISE_METAL_FAST_ALPHA_CUTOFF` tunes the fast-preview peak alpha cutoff;
  the default is `0.01`.
- `TORTUISE_METAL_FAST_TILE_BUDGET` caps front-to-back raster work per tile in
  fast-preview; the default is `16384`. This is the first budgeted-compositor
  primitive and must stay approximate-only.
- `--splat-budget N` deterministically keeps at most `N` evenly spaced source
  splats before upload/render. This is active-set/LoD groundwork rather than a
  finished quality LoD, but it lets benchmarks test fixed-budget behavior on
  the same camera packs.

Gate:

- visual review packet passes
- no obvious depth flicker across adjacent camera frames
- exact mode remains default
- benchmark tables label approximate speed separately from exact speed

Later candidates:

- clustered LOD
- depth-bin/front streaming
- splat budget modes for terminal-only previews

## Phase 7: Terminal Throughput

Only optimize terminal transport after Metal frame cost is under control.

Track:

- ANSI bytes per frame
- dirty rows/cells
- terminal write time
- render time vs terminal flush time

Potential work:

- dirty-row output
- halfblock damage tracking
- lower-frequency preview overlay for hands/camera

## Stop Rules

- Stop exact sort work once warmed exact
  `attempt_sort_count <= 1.5 * actual_total_overlaps` and sort is no longer a
  dominant stage.
- Reject optimizations that hide overflow by dropping overlaps.
- Do not claim a win without the benchmarking playbook gates.
- Do not make approximate paths default until exact mode is correct and the
  approximate path has its own visual/flicker gate.
