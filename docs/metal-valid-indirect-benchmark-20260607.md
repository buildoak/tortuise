# Metal Valid-Indirect Benchmark 2026-06-07

This note records the GPU valid-count indirect-dispatch optimization.

## Change

Projection already compacts visible splats into `projected[0..valid_count)`.
Before this change, fused Metal still dispatched `count_tile_overlaps` and
`emit_tile_keys` over full `splat_count`, leaving the kernels to early-return
for inactive indexes.

The new path adds a small GPU kernel that writes
`MTLDispatchThreadgroupsIndirectArguments` from `valid_count`, then dispatches
count and emit through `dispatchThreadgroupsWithIndirectBuffer`. The fused path
keeps one command buffer and avoids a CPU readback.

## Artifacts

- baseline: `target/metal-estimate-bench-20260607-after/`
- after: `target/metal-valid-indirect-bench-20260607-after/`
- correctness: `target/probe-matrix-20260607-valid-indirect/`

All benchmark runs used `--release`, `--probe-warmup 5`,
`--probe-benchmark-frames 30`, `--probe-stage-telemetry`, and
`--probe-timing`.

## Result

| Scene | Baseline avg ms | After avg ms | Delta | Checksum |
| --- | ---: | ---: | ---: | --- |
| Bee sparse median | 27.087 | 24.287 | +10.3% | unchanged |
| Bonsai sentinel | 39.935 | 33.324 | +16.6% | unchanged |
| Ignatius sentinel | 28.404 | 30.952 | -9.0% | unchanged |

Bee sparse telemetry stayed semantically identical:

| Metric | Value |
| --- | ---: |
| nonblack pixels | 48 |
| valid splats | 37,523 |
| actual overlaps | 59,901 |
| attempted sort entries | 74,877 |
| retry count | 0 |
| overflow flag | 0 |

## Correctness

Passed:

```bash
cargo test
cargo test --features metal
PROBE_MATRIX_SIZE=256x192 PROBE_MATRIX_WARMUP=3 \
  ./scripts/probe-matrix.sh target/probe-matrix-20260607-valid-indirect -- --release
```

The loaded-frame checksum was unchanged for Bee, Bonsai, and Ignatius.

## Read

This is a real exact-path win, but not universally dominant yet. Bee and Bonsai
improved, while the single Ignatius sentinel regressed. The next gate should use
a wider three-run matrix for Ignatius/Ramen/Meetingroom and then tune dispatch
policy or hotspot handling from that evidence.
