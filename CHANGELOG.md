# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- WIP Metal + Kitty graphics preview for macOS/Apple Silicon behind explicit `--kitty` and `--live-preset macbook-neo` launch flags.
- Dramatically faster heavy-splat live rendering path using the Metal backend, fixed LoD mode, Kitty RGB output, and a MacBook Neo tuned preset.
- Live interaction telemetry via `--live-telemetry-jsonl`, covering frame timing, render timing, terminal write timing, and input age.
- Probe and benchmark tooling for rendered-image validation, visual review packets, and performance comparisons.

### Changed
- `M` now keeps Kitty flag-gated: normal launches cycle through terminal render modes only; Kitty is included in the mode cycle only after `--kitty` or a live preset enables it.
- README now documents the Metal/Kitty path as a useful WIP preview, not a default polished release.

### Notes
- This is a WIP version with a major Metal performance jump, but the path still needs final polish.
- Full hands-control integration is planned within this week.

## [0.1.1] - 2026-02-24

### Added
- `scripts/supersplat-dl.sh` — download any SuperSplat scene as a tortuise-compatible PLY in one command
- GitHub Actions CI (build, test, clippy, fmt on macOS + Linux)
- CONTRIBUTING.md and CHANGELOG.md

### Fixed
- Camera init restored to proven `(0,0,5)` at origin; all AABB/scene-center heuristics removed
- Orbit mode always targets origin; WASD no longer shifts the orbit center
- Crate package excludes large scene files and binary assets

### Changed
- Demo image switched from GIF (9.3 MB) to animated WebP (4.4 MB)
- README tagline, intro, and "Why this exists" section rewritten for clarity
- Hardware table updated; ratatui attribution corrected

## [0.1.0] - 2026-02-14

### Added
- Metal GPU pipeline: single command buffer, tile-based radix sort, zero-copy readback
- CPU fallback renderer with packed framebuffer and color diffing
- Modal camera system: Free (WASD + R/F vertical) and Orbit modes; reset on Z
- Input thread with delta-time movement
- True-color detection for Ghostty, Kitty, WezTerm, iTerm2; 256-color fallback for Terminal.app
- Perceptual grayscale matching for 256-color environments
- `.ply` scene parser with checked arithmetic, file-size validation, and error context
- Clap CLI with `--flip-y` flag and graceful GPU-error degradation
- Procedural torus-knot scene as built-in fallback when no `.ply` is provided
- MSRV set to Rust 1.80
- MIT LICENSE, crates.io metadata, cross-platform dependencies
