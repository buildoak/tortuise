# Kitty Protocol Support

This repo now has a live, Metal-backed Kitty graphics protocol mode plus the
probe artifacts needed to measure terminal payload cost separately from renderer
cost.

## Live Mode

Run a scene in Kitty graphics protocol mode:

```bash
cargo run --features metal --release -- "$SCENE" --kitty
```

Transport/quality knobs:

```bash
cargo run --features metal --release -- "$SCENE" --kitty \
  --kitty-format rgb \
  --kitty-scale-divisor 2
```

- `--camera-pos x,y,z` and `--look-at x,y,z` set the live viewer's initial
  camera. This matters for compact scenes such as Bee, where the default
  `0,0,5` camera can make the first frame appear effectively black.
- `--kitty-format rgb` sends `f=24` RGB payloads instead of `f=32` RGBA,
  cutting raw/base64 transport by 25% when terminal alpha blending is not
  needed.
- `--kitty-scale-divisor N` renders the Kitty image at `1/N` resolution and
  asks Kitty to place it over the same terminal cell area. This is a deliberate
  bandwidth/quality tradeoff for interactive preview. For compact scenes such
  as Bee, prefer divisor `1` or `2`; divisor `4` can make the splat effectively
  disappear at normal terminal dimensions.

The normal `M` render-mode cycle includes:

```text
Halfblock -> Kitty -> PointCloud -> Matrix -> BlockDensity -> Braille -> AsciiClassic
```

Runtime behavior:

- Kitty mode is available only with the `metal` feature.
- The renderer uses the Metal packed framebuffer as the source of truth.
- Each frame is converted to RGBA8, base64 encoded, and emitted as Kitty direct
  image data chunks.
- HUD telemetry reports raw RGBA bytes, base64 bytes, and chunk count as
  `Kitty:<payload>B/<base64>B <chunks>ch`.
- If Metal is unavailable or fails, the app falls back to halfblock rendering.

The current live path sends a full RGBA frame. Dirty-frame updates, placement
reuse, and transport timing breakdowns are still separate optimization gates.

## Probe Payload Gate

Use:

```bash
cargo run --features metal --release -- "$SCENE" \
  --render-probe "$OUT/$LABEL" \
  --probe-case loaded \
  --probe-backends metal \
  --probe-size 256x192 \
  --probe-camera-pos "$CAM" \
  --probe-look-at "$LOOK" \
  --probe-warmup 5 \
  --probe-benchmark-frames 30 \
  --probe-stage-telemetry \
  --probe-timing \
  --probe-kitty-payload
```

Artifacts:

- `kitty/metal_frame_000.rgba`
- `kitty/metal_frame_000.json`
- manifest fields `kitty_rgba` and `kitty_metadata`

Metadata fields:

- `format`: currently `rgba8`
- `width`
- `height`
- `payload_bytes`
- `base64_bytes`
- `chunks_4096`

The raw RGBA payload is the renderer-to-terminal payload candidate. The
`base64_bytes` and `chunks_4096` numbers estimate Kitty escape transport cost.

## Replay / Transport Measurement

Existing `.rgba` probe payloads can be measured without rendering or opening a
terminal:

```bash
cargo run --release -- \
  --kitty-replay target/kitty-bench-20260607-1524/calibration/bee_256_192/kitty/metal_frame_000.rgba
```

If the sidecar JSON is missing, pass dimensions explicitly:

```bash
cargo run --release -- \
  --kitty-replay path/to/frame.rgba \
  --kitty-replay-size 256x192
```

The command prints JSON with:

- `variants`: measured full-frame Kitty payload variants:
  - `format: "rgba"`, `kitty_f: 32`
  - `format: "rgb"`, `kitty_f: 24`
- `payload_bytes`, `base64_bytes`, `chunks`, and `encode_ms` per variant.
- `downscale_estimates` for 1x, 1/2, and 1/4 byte budgets. These are
  deterministic transport estimates only; they do not resample pixels.

Use `--kitty-replay-chunk-size N` to estimate a chunk size other than the live
4096-byte base64 chunks.

## Current 64x48 Smoke

Command:

```bash
cargo run --features metal --release -- \
  --render-probe target/probe-kitty-payload-20260607/channels \
  --probe-backends both \
  --probe-size 64x48 \
  --probe-camera-pos 0,0,4 \
  --probe-look-at 0,0,0 \
  --probe-warmup 1 \
  --probe-case channels \
  --probe-kitty-payload \
  --probe-timing \
  --probe-fail-on-mismatch
```

Result:

- payload bytes: `12,288`
- base64 bytes: `16,384`
- 4096-byte chunks: `4`
- CPU-vs-Metal mismatches: `0`

## Remaining Integration Gates

1. Add byte-budget telemetry beside render timing: render ms, readback ms,
   encode/base64 ms, write bytes, terminal flush ms.
2. Add dirty-frame policy so static frames do not resend full payloads.
3. Detect terminal capability for auto-selection; keep `--kitty` as the explicit
   override.
4. Benchmark live transport separately from renderer speed. A fast Metal frame
   can still be bottlenecked by base64 and terminal I/O.
