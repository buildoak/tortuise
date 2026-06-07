# Metal Estimate Benchmark 2026-06-07

This note records the measurement for the warmed fused overlap-estimate change.

## Command Shape

All runs used:

```bash
cargo run --features metal --release -- "$SCENE" \
  --render-probe "$OUT/$LABEL" \
  --probe-case loaded \
  --probe-backends metal \
  --probe-size 256x192 \
  --probe-camera-pos 0,0,4 \
  --probe-look-at 0,0,0 \
  --probe-warmup 5 \
  --probe-benchmark-frames 30 \
  --probe-stage-telemetry \
  --probe-timing \
  --probe-inspect-scale 2
```

Artifacts:

- before: `target/metal-estimate-bench-20260607-before/`
- after: `target/metal-estimate-bench-20260607-after/`
- correctness: `target/probe-matrix-20260607-metal-estimate/`

## Result

| Scene | Before avg ms | After avg ms | Speed delta | Before attempt/actual | After attempt/actual | Checksum |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bee sparse median | 28.277 | 27.087 | +4.2% | 38.66x | 1.25x | unchanged |
| Bonsai sentinel | 36.729 | 39.935 | -8.7% | 1.95x | 1.25x | unchanged |
| Ignatius sentinel | 31.940 | 28.404 | +11.1% | 1.32x | 1.25x | unchanged |

Bee sparse run details:

| Metric | Before | After |
| --- | ---: | ---: |
| nonblack pixels | 48 | 48 |
| valid splats | 37,523 | 37,523 |
| actual overlaps | 59,901 | 59,901 |
| estimated overlaps | 2,315,943 | 74,877 |
| attempted sort entries | 2,315,943 | 74,877 |
| retry count | 0 | 0 |
| overflow flag | 0 | 0 |

The change removes `96.8%` of padded sort work in the Bee sparse view. The
render-time win is much smaller than the work-reduction win, which indicates the
next bottleneck is not padded radix sorting for this view. Projection, overlap
counting, key emission over `splat_count`, and raster hotspots are the next
places to measure.

## Correctness

Passed:

```bash
cargo test
cargo test --features metal
PROBE_MATRIX_SIZE=256x192 PROBE_MATRIX_WARMUP=3 \
  ./scripts/probe-matrix.sh target/probe-matrix-20260607-metal-estimate -- --release
```

The before/after loaded checksums were unchanged for Bee, Bonsai, and Ignatius.

## Next Optimization Gate

Run stage-timing probes with `TORTUISE_METAL_STAGE_TIMING=1`, then optimize the
largest remaining exact stage:

1. GPU valid-count dispatch for count/emit work.
2. GPU actual-overlap radix dispatch without the hybrid CPU readback split.
3. Exact cluster culling only if projection remains dominant after the first two
   gates.
