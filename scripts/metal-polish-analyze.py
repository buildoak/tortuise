#!/usr/bin/env python3
"""Derive a coverage index from Metal polish packet artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        return {"_error": f"json_decode_error: {exc}"}


def first_frame_stats(probe: dict[str, Any] | None, backend: str) -> dict[str, Any] | None:
    if not probe:
        return None
    frames = probe.get(f"{backend}_frames") or []
    if not frames:
        return None
    return frames[0].get("metrics") or frames[0].get("stats")


def occupancy(stats: dict[str, Any] | None) -> float | None:
    if not stats:
        return None
    width = stats.get("width") or 0
    height = stats.get("height") or 0
    nonblack = stats.get("nonblack_pixels") or 0
    pixels = width * height
    if pixels <= 0:
        return None
    return nonblack / pixels


def occupancy_tier(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.10:
        return "sparse"
    if value > 0.95:
        return "saturated"
    return "primary"


def diff_classification(diff: dict[str, Any] | None) -> str | None:
    if not diff:
        return None
    overall = diff.get("overall") or {}
    return overall.get("classification")


def metal_invariants(probe: dict[str, Any] | None) -> dict[str, Any]:
    frames = (probe or {}).get("metal_frames") or []
    telemetry = None
    if frames:
        telemetry = frames[0].get("telemetry")
    if not telemetry:
        return {"available": False}
    tile_density = telemetry.get("tile_density") or {}
    return {
        "available": True,
        "overflow_flag": telemetry.get("overflow_flag"),
        "retry_count": telemetry.get("retry_count"),
        "actual_total_overlaps": telemetry.get("actual_total_overlaps"),
        "tile_entries": tile_density.get("total_tile_entries"),
        "overlap_count_matches_tile_entries": telemetry.get("actual_total_overlaps")
        == tile_density.get("total_tile_entries"),
        "sort_path": telemetry.get("sort_path"),
        "lod_mode": telemetry.get("lod_mode"),
        "active_splat_count": telemetry.get("active_splat_count"),
        "valid_count": telemetry.get("valid_count"),
        "max_tile_range": tile_density.get("max_tile_range"),
        "p95_tile_range": tile_density.get("p95_tile_range"),
        "p99_tile_range": tile_density.get("p99_tile_range"),
    }


def packet_record(packet_manifest_path: Path, root: Path) -> dict[str, Any]:
    packet = load_json(packet_manifest_path) or {}
    probe = load_json(Path(packet.get("probe_manifest", "")))
    diff = load_json(Path(packet.get("diff_summary", "")))
    metal_stats = first_frame_stats(probe, "metal")
    cpu_stats = first_frame_stats(probe, "cpu")
    occ = occupancy(metal_stats or cpu_stats)
    return {
        "packet": str(packet_manifest_path.relative_to(root)),
        "scene_id": packet.get("scene_id"),
        "angle_id": packet.get("angle_id"),
        "quality_id": packet.get("quality_id"),
        "tier": packet.get("tier"),
        "exit_status": packet.get("exit_status"),
        "probe_manifest": packet.get("probe_manifest"),
        "review_task": str(packet_manifest_path.with_name("review_task.json")),
        "diff_classification": diff_classification(diff),
        "occupancy": occ,
        "occupancy_tier": occupancy_tier(occ),
        "metal": metal_invariants(probe),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: metal-polish-analyze.py OUT_DIR", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    packets = [
        packet_record(path, root)
        for path in sorted(root.glob("*/*/*/packet_manifest.json"))
    ]
    summary = {
        "schema_version": 1,
        "root": str(root),
        "packet_count": len(packets),
        "primary_claim_packet_count": sum(
            1
            for packet in packets
            if packet.get("quality_id") == "exact"
            and packet.get("occupancy_tier") == "primary"
            and packet.get("diff_classification") in (None, "pass")
        ),
        "packets": packets,
    }
    json.dump(summary, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
