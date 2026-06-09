# tortuise

Gaussian splats viewer that works in your terminal. Yes, it's made of symbols!

![tortuise demo](https://raw.githubusercontent.com/buildoak/tortuise/main/assets/demo.webp)

[![crates.io](https://img.shields.io/crates/v/tortuise.svg)](https://crates.io/crates/tortuise)
[![CI](https://github.com/buildoak/tortuise/actions/workflows/ci.yml/badge.svg)](https://github.com/buildoak/tortuise/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux-lightgrey)

[![Terminal Trove - Tool of the Week](https://cdn.terminaltrove.com/media/badges/tool_of_the_week/svg/terminal_trove_tool_of_the_week_gold_transparent.svg)](https://terminaltrove.com/tortuise/)

A CPU-first 3D Gaussian Splatting viewer inspired by [ratatui](https://github.com/ratatui/ratatui), built on [crossterm](https://github.com/crossterm-rs/crossterm). Fully parallelized rendering pipeline via [rayon](https://github.com/rayon-rs/rayon), perceptual color mapping, six terminal render modes, and an optional WIP Metal + Kitty graphics preview on macOS. Real scenes with 1.1M splats hold 10–25 FPS on CPU, and the new Metal path can be dramatically faster on Apple Silicon. No GPU required for the default experience. Scenes [download](#where-to-get-scenes) included!

<!-- Demo recorded in Ghostty, halfblock mode, no Kitty graphics protocol — pure Unicode characters -->

## Why this exists

Gaussian Splats are cool. Beautiful tech. Especially now with the rise of image-to-splat pipelines like [SHARP](https://github.com/apple/ml-sharp). Btw, it will be soon available [here](#roadmap).

Inspiration by [ratatui](https://github.com/ratatui/ratatui) merged with binge watching of *Common Side Effects* which resulted in building this "The Tortuise" TUI 3GS viewer.

## Features

| Feature | Details |
|---------|---------|
| **6 terminal render modes** | Halfblock (default), point cloud, matrix, block density, braille, ASCII. Cycle with `M` |
| **WIP Metal + Kitty preview** | macOS/Apple Silicon acceleration behind explicit `--kitty` / `--live-preset macbook-neo` launch flags. Fast, useful, and still being polished |
| **Full 3D navigation** | WASD movement, R/F vertical, arrow keys for yaw/pitch. Smooth held-key input |
| **Two camera modes** | Free (fly anywhere) and Orbit (auto-rotate around origin). Switch with `Space` |
| **.ply and .splat files** | Standard 3DGS formats. Binary little-endian PLY with SH coefficients, 32-byte .splat records |
| **Built-in scenes** | Bundled `bonsai.splat` (1.1M splats) and `luigi.ply` (14K). `--demo` runs a built-in demo scene |
| **Terminal detection** | Truecolor for modern terminals, perceptual 256-color fallback for Terminal.app. Zero config |
| **Supersampling** | 1x/2x/3x factor for higher fidelity at the cost of compute |
| **Cross-platform** | macOS and Linux |

## Quick start

**Requires Rust 1.80+** (`rustup update` to upgrade)

```bash
# Install from crates.io (recommended)
cargo install tortuise

# Or build from source
git clone https://github.com/buildoak/tortuise.git
cd tortuise
cargo install --path .

# Built-in demo (no scene file needed)
tortuise --demo

# Load a bundled scene
tortuise --flip-y scenes/bonsai.splat

# Load any .ply or .splat file
tortuise your-scene.splat
tortuise your-scene.ply

# Some scenes need axis flips depending on capture coordinate system
tortuise --flip-y scene.ply
tortuise --flip-z scene.splat

# Don't have a scene file? Download one from SuperSplat
pip install Pillow numpy  # needed once, for SOG scene format
./scripts/supersplat-dl.sh "https://superspl.at/scene/d281f99f" ramen.ply
tortuise ramen.ply
```

### WIP Metal + Kitty preview

The current branch includes a WIP macOS Metal renderer with Kitty graphics protocol output. It is intentionally flag-gated while polish continues: normal launches stay on the terminal/CPU path, and `M` will not cycle into Kitty unless the session was launched with `--kitty` or a live preset.

```bash
# Build this preview from source with the Metal feature
cargo install --path . --features metal

# Recommended Apple Silicon live preset for local Kitty-capable terminals
tortuise bee.ply --live-preset macbook-neo

# Manual preview flags
tortuise bee.ply \
  --kitty \
  --flip-y \
  --camera-pos 0,0,0.5 \
  --look-at 0,0,0 \
  --metal-quality turbo \
  --metal-lod fixed \
  --metal-lod-splat-count 2100000 \
  --metal-lod-order voxel \
  --kitty-format rgb
```

This preview is already a large performance jump versus the original CPU terminal renderer on heavy splats, but it is not the final polished release. Hands control and the full interaction layer are planned for this week.

### CLI options

```
tortuise [OPTIONS] [INPUT]

Arguments:
  [INPUT]    Path to a .ply or .splat scene file (use --demo for built-in scene)

Options:
  --demo              Run built-in demo scene
  --flip-y            Flip Y axis (some capture tools use Y-down)
  --flip-z            Flip Z axis
  --supersample <N>   Supersampling factor [default: 1]
  --cpu               Force CPU rendering
  --camera-pos <XYZ>  Start camera position, for example 0,0,0.5
  --look-at <XYZ>     Initial look-at target, for example 0,0,0
  --live-telemetry-jsonl <PATH>
                      Write live frame/render/terminal/input telemetry as JSONL
  -h, --help          Print help
  -V, --version       Print version

Metal builds also include:
  --kitty             Enable WIP Kitty graphics protocol renderer
  --live-preset <ID>  Apply a live preset, currently `macbook-neo`
  --kitty-format <F>  Kitty payload format: rgba or rgb
  --metal-quality <Q> Metal quality: exact, fast-preview, or turbo
  --metal-lod <MODE>  Metal LoD mode: off or fixed
  --metal-lod-splat-count <N>
                      Active splat count for fixed Metal LoD
```

## Controls

### Free mode

| Key | Action |
|-----|--------|
| `W` / `A` / `S` / `D` | Move forward / left / back / right |
| `R` / `F` | Move up / down |
| Arrow keys | Yaw and pitch (look around) |
| `Space` | Switch to Orbit mode |
| `M` | Cycle render mode. Kitty appears only when launched with `--kitty` or a live preset |
| `+` / `-` | Adjust movement speed |
| `Tab` | Cycle HUD: minimal / debug / hidden |
| `Z` | Reset camera |
| `Q` / `Esc` | Quit |

### Orbit mode

| Key | Action |
|-----|--------|
| Arrow Up / Down | Adjust elevation |
| Arrow Left / Right | Nudge orbit angle |
| `Space` | Switch to Free mode |
| `+` / `-` | Adjust orbit speed |

## Supported terminals

**Truecolor (best experience):** Ghostty, iTerm2, Kitty, WezTerm, Alacritty

**256-color fallback:** Apple Terminal.app -- works, but reduced color fidelity. The perceptual color mapping does its best.

Auto-detected via `COLORTERM`, `TERM_PROGRAM`, and `TERM` environment variables. No configuration needed.

## Tested hardware

| Device | CPU | Scene | Reference FPS |
|--------|-----|-------|---------------|
| Mac Mini M4 | Apple M4 | luigi.ply (14K) | 120+ |
| Mac Mini M4 | Apple M4 | bonsai.splat (1.1M) | 80+ |
| MacBook Air M2 | Apple M2 | luigi.ply (14K) | ✓ |
| MacBook Air M2 | Apple M2 | bonsai.splat (1.1M) | 20–30 |
| Jetson Orin Nano* | ARM Cortex-A78AE | luigi.ply (14K) | ~30 |
| Jetson Orin Nano* | ARM Cortex-A78AE | bonsai.splat (1.1M) | 10–15 |

Numbers in parentheses are splat counts. FPS is approximate and depends heavily on terminal window size — a smaller window (⌘-) renders fewer cells and runs faster. *Jetson tested over SSH, which may be a bottleneck.

## Where to get scenes

### SuperSplat (recommended)

The fastest way to get a real scene is from [SuperSplat](https://superspl.at/) -- thousands of community-uploaded Gaussian splat scenes. The included download script handles everything:

```bash
# Download any SuperSplat scene — paste a share URL, scene page URL, or bare ID
./scripts/supersplat-dl.sh "https://superspl.at/scene/d281f99f" ramen.ply
./scripts/supersplat-dl.sh "https://superspl.at/s?id=cf6ac78e" bee.ply
./scripts/supersplat-dl.sh cf6ac78e bee.ply

# View it
tortuise ramen.ply
```

The script auto-detects the scene format and handles both legacy (compressed PLY) and modern (SOG) SuperSplat scenes. No Node.js required — conversion is pure Python.

> **Note:** `supersplat-dl.sh` is provided for personal, educational, and interoperability purposes. It converts SuperSplat's proprietary formats to standard PLY — the same data access pattern used by SuperSplat's own [MIT-licensed viewer](https://github.com/playcanvas/supersplat-viewer). Downloaded content remains the intellectual property of its creator. Please respect content creators' rights and provide attribution.

**Requirements:** Python 3.8+, curl. For SOG scenes (most scenes uploaded after mid-2025), you also need Pillow and numpy:

```bash
pip install -r scripts/requirements.txt
```

### Other sources

- `tortuise --demo` -- instant built-in demo scene, no downloads required
- [Polycam](https://poly.cam/explore) -- photogrammetry captures, some with Gaussian splat export
- [nerfstudio](https://docs.nerf.studio/) -- train your own splats from video, exports to .ply
- Any standard 3DGS pipeline output in .ply or .splat format

Both formats are well-supported: PLY files with spherical harmonic coefficients (`f_dc_0/1/2`) or direct RGB, and the compact 32-byte .splat format used by most web viewers.

## How it works

The pipeline is straightforward: load splats, project them into screen space, depth-sort, splat onto a framebuffer, then convert to terminal characters. Each frame:

1. **Project** -- every Gaussian is transformed from world space through the camera view matrix. Frustum culling drops anything behind the near plane or outside the viewport. This step is parallelized with rayon.
2. **Sort** -- projected splats are depth-sorted back-to-front for correct alpha compositing.
3. **Rasterize** -- each splat is splatted onto an RGB framebuffer using its 2D covariance (scale + rotation). Front-to-back compositing with early alpha termination -- once a pixel is fully opaque, all remaining splats behind it are skipped. Per-splat saturation probes skip entire Gaussians when they land on already-saturated regions. At 1M+ splats, the back 80% are often invisible behind the front 20%.
4. **Encode** -- the framebuffer is converted to terminal output. In halfblock mode, each cell packs two vertical pixels using the `▄` character with separate foreground/background colors. Other modes use braille patterns, ASCII density ramps, or single characters.

The frame target is 8ms (~120fps). On truecolor terminals, colors are passed as 24-bit RGB. On 256-color terminals, a perceptual distance function maps each pixel to the closest ANSI color -- weighted toward green sensitivity, which is where human vision is sharpest.

## Roadmap

Things I want to improve next -- and contribution opportunities:

- **Hands control** -- camera-based gesture control for splat orbit/fly navigation. Targeting the full WIP version this week.
- **Kitty graphics protocol polish** -- pixel-perfect rendering via the terminal image protocol. The Metal + Kitty path is working behind flags, but still needs final polish before it becomes a default UX.
- **SHARP integration** -- image-to-splat-to-view pipeline. Single photo to 3D in your terminal.
- **Sample scene bundle** -- curated downloadable scenes so people can skip the "where do I find a .splat file" step.
- **GPU acceleration** -- the Metal compute backend is now useful on Apple Silicon and still improving.
- **Performance** -- radix sort for depth ordering, SIMD-accelerated projection via glam, tighter memory layout.

## Built with

- [Rust](https://www.rust-lang.org/)
- [crossterm](https://github.com/crossterm-rs/crossterm) -- terminal control and input
- [rayon](https://github.com/rayon-rs/rayon) -- data parallelism for projection and rasterization
- [clap](https://github.com/clap-rs/clap) -- CLI argument parsing

[ratatui](https://github.com/ratatui/ratatui) + tortoise = tortuise.

## License

MIT -- Nick Oak, 2026
