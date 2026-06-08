# Metal Polish Artifact Contract

This contract keeps Metal optimization claims tied to reproducible evidence.

## Packet Layout

`scripts/metal-polish-packet.sh` writes packets under:

```text
target/metal-polish-YYYYMMDD/{scene_id}/{angle_id}/{quality_id}/
```

Each packet must preserve:

- `commands.txt` - exact command line executed for the packet.
- `packet_manifest.json` - runner metadata and links to probe outputs.
- `review_task.json` - compact task for vision/agent review.
- `probe_manifest.json` - renderer artifact manifest.
- `probe_timing.json` when timing is enabled.
- `diff/summary.json` for exact CPU/Metal comparison packets.
- `metal/stage_telemetry_frame_*.json` when stage telemetry is enabled.

The root `coverage_index.json` is derived from packet manifests. Do not edit it
by hand and do not treat it as an authority over the packet files.

## Quality Tiers

- `exact`: default path and the only release-claim path.
- `fast-preview`: approximate visual/interactive packet.
- `turbo`: diagnostic approximate packet.

Approximate packets must be reviewed against exact references and temporal arcs.
Still images alone are not sufficient for approximate-mode acceptance.

## Acceptance Gates

- `cargo test` and `cargo test --features metal` pass.
- Synthetic probes pass with `--probe-fail-on-mismatch`.
- Loaded exact calibrated packets pass CPU/Metal diff.
- Primary speed claims use only 10-95% occupancy views.
- Warmed Metal packets have `overflow_flag == 0` and `retry_count == 0`.
- `actual_total_overlaps == tile_density.total_tile_entries` for accepted Metal
  telemetry packets.
- Three release runs are required before accepting a policy change.
- Policy changes need a median `render_avg_ms` win of at least 10% where
  selected and no calibrated regression above 5%.

## Review Task Shape

Reviewers consume `review_task.json` plus the referenced PNG/JSON artifacts and
return:

```json
{
  "verdict": "pass|fail|uncertain",
  "release_blocker": false,
  "issue_type": "none|crop|blank|flicker|holes|tearing|color|depth|performance_claim",
  "evidence_path": "path/to/artifact",
  "telemetry_path": "path/to/telemetry",
  "confidence": "low|medium|high",
  "notes": "short grounded observation"
}
```

Start with raw PNGs and JSON. Add rendered label overlays only if reviewers fail
to localize issues from the artifact packet.
