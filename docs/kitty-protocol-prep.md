# Kitty Protocol Prep

This repo is not wired to live Kitty graphics protocol output yet, but the
rendering side now has the probe artifacts needed to measure that path before
integrating terminal escape transport.

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

## Live Integration Plan

1. Add a terminal transport module that can write Kitty graphics protocol image
   frames from RGBA bytes.
2. Add byte-budget telemetry beside render timing: render ms, readback ms,
   encode/base64 ms, write bytes, terminal flush ms.
3. Add dirty-frame policy so static frames do not resend full payloads.
4. Keep halfblock as fallback; select Kitty only when terminal capability is
   detected or explicitly requested.
5. Benchmark live transport separately from renderer speed. A fast Metal frame
   can still be bottlenecked by base64 and terminal I/O.
