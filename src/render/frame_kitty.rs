use std::io::{self, Write};
use std::time::Instant;

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
#[cfg(feature = "metal")]
use crossterm::{cursor, queue};

#[cfg(feature = "metal")]
use super::{AppState, Backend};

pub const KITTY_DIRECT_CHUNK_SIZE: usize = 4096;

pub fn kitty_base64_len(payload_bytes: usize) -> usize {
    payload_bytes.div_ceil(3) * 4
}

pub fn kitty_chunk_count(base64_bytes: usize, chunk_size: usize) -> usize {
    if base64_bytes == 0 {
        0
    } else {
        base64_bytes.div_ceil(chunk_size.max(1))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KittyPayloadFormat {
    Rgba32,
    Rgb24,
}

impl KittyPayloadFormat {
    fn name(self) -> &'static str {
        match self {
            Self::Rgba32 => "rgba",
            Self::Rgb24 => "rgb",
        }
    }

    fn kitty_format(self) -> u8 {
        match self {
            Self::Rgba32 => 32,
            Self::Rgb24 => 24,
        }
    }

    fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba32 => 4,
            Self::Rgb24 => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KittyPayloadMeasurement {
    pub format: KittyPayloadFormat,
    pub width: usize,
    pub height: usize,
    pub payload_bytes: usize,
    pub base64_bytes: usize,
    pub chunks: usize,
    pub encode_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KittyDownscaleBudget {
    pub divisor: usize,
    pub width: usize,
    pub height: usize,
    pub rgba_payload_bytes: usize,
    pub rgba_base64_bytes: usize,
    pub rgba_chunks: usize,
    pub rgb_payload_bytes: usize,
    pub rgb_base64_bytes: usize,
    pub rgb_chunks: usize,
}

#[cfg_attr(not(any(test, feature = "metal")), allow(dead_code))]
fn packed_framebuffer_to_rgba(packed: &[u32], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(packed.len().saturating_mul(4));
    for pixel in packed {
        out.push((pixel & 0xFF) as u8);
        out.push(((pixel >> 8) & 0xFF) as u8);
        out.push(((pixel >> 16) & 0xFF) as u8);
        out.push(((pixel >> 24) & 0xFF) as u8);
    }
}

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
    for pixel in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&pixel[..3]);
    }
    rgb
}

fn validate_rgba_dimensions(width: usize, height: usize, rgba: &[u8]) -> io::Result<usize> {
    if width == 0 || height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Kitty replay dimensions must be non-zero",
        ));
    }
    let pixels = width.checked_mul(height).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Kitty replay dimensions {width}x{height} overflow usize"),
        )
    })?;
    let expected = pixels.checked_mul(4).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Kitty replay RGBA byte count for {width}x{height} overflows usize"),
        )
    })?;
    if rgba.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Kitty replay payload has {} bytes, expected {expected} for {width}x{height} RGBA",
                rgba.len()
            ),
        ));
    }
    Ok(pixels)
}

fn measure_payload(
    width: usize,
    height: usize,
    payload: &[u8],
    format: KittyPayloadFormat,
    chunk_size: usize,
) -> KittyPayloadMeasurement {
    let start = Instant::now();
    let encoded = BASE64_STANDARD.encode(payload);
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
    let base64_bytes = encoded.len();
    debug_assert_eq!(base64_bytes, kitty_base64_len(payload.len()));
    KittyPayloadMeasurement {
        format,
        width,
        height,
        payload_bytes: payload.len(),
        base64_bytes,
        chunks: kitty_chunk_count(base64_bytes, chunk_size),
        encode_ms,
    }
}

pub fn measure_kitty_replay_variants(
    width: usize,
    height: usize,
    rgba: &[u8],
    chunk_size: usize,
) -> io::Result<Vec<KittyPayloadMeasurement>> {
    validate_rgba_dimensions(width, height, rgba)?;
    let chunk_size = chunk_size.max(1);
    let rgb = rgba_to_rgb(rgba);
    Ok(vec![
        measure_payload(width, height, rgba, KittyPayloadFormat::Rgba32, chunk_size),
        measure_payload(width, height, &rgb, KittyPayloadFormat::Rgb24, chunk_size),
    ])
}

pub fn kitty_downscale_budgets(
    width: usize,
    height: usize,
    chunk_size: usize,
    divisors: &[usize],
) -> io::Result<Vec<KittyDownscaleBudget>> {
    if width == 0 || height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Kitty replay dimensions must be non-zero",
        ));
    }
    let chunk_size = chunk_size.max(1);
    divisors
        .iter()
        .copied()
        .filter(|divisor| *divisor > 0)
        .map(|divisor| {
            let scaled_width = width.div_ceil(divisor).max(1);
            let scaled_height = height.div_ceil(divisor).max(1);
            let pixels = scaled_width.checked_mul(scaled_height).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "Kitty downscale estimate dimensions {scaled_width}x{scaled_height} overflow usize"
                    ),
                )
            })?;
            let rgba_payload_bytes =
                pixels
                    .checked_mul(KittyPayloadFormat::Rgba32.bytes_per_pixel())
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "Kitty RGBA downscale byte estimate overflows usize",
                        )
                    })?;
            let rgb_payload_bytes =
                pixels
                    .checked_mul(KittyPayloadFormat::Rgb24.bytes_per_pixel())
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "Kitty RGB downscale byte estimate overflows usize",
                        )
                    })?;
            let rgba_base64_bytes = kitty_base64_len(rgba_payload_bytes);
            let rgb_base64_bytes = kitty_base64_len(rgb_payload_bytes);
            Ok(KittyDownscaleBudget {
                divisor,
                width: scaled_width,
                height: scaled_height,
                rgba_payload_bytes,
                rgba_base64_bytes,
                rgba_chunks: kitty_chunk_count(rgba_base64_bytes, chunk_size),
                rgb_payload_bytes,
                rgb_base64_bytes,
                rgb_chunks: kitty_chunk_count(rgb_base64_bytes, chunk_size),
            })
        })
        .collect()
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

pub fn kitty_replay_report_json(
    source: &str,
    width: usize,
    height: usize,
    rgba: &[u8],
    chunk_size: usize,
) -> io::Result<String> {
    let variants = measure_kitty_replay_variants(width, height, rgba, chunk_size)?;
    let budgets = kitty_downscale_budgets(width, height, chunk_size, &[1, 2, 4])?;

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"source\": {},\n", json_string(source)));
    json.push_str(&format!("  \"width\": {width},\n"));
    json.push_str(&format!("  \"height\": {height},\n"));
    json.push_str(&format!("  \"pixels\": {},\n", width * height));
    json.push_str(&format!("  \"source_bytes\": {},\n", rgba.len()));
    json.push_str(&format!("  \"chunk_size\": {},\n", chunk_size.max(1)));
    json.push_str("  \"variants\": [\n");
    for (idx, variant) in variants.iter().enumerate() {
        let comma = if idx + 1 == variants.len() { "" } else { "," };
        json.push_str(&format!(
            concat!(
                "    {{",
                "\"format\":{},",
                "\"kitty_f\":{},",
                "\"payload_bytes\":{},",
                "\"base64_bytes\":{},",
                "\"chunks\":{},",
                "\"encode_ms\":{:.3}",
                "}}{}\n"
            ),
            json_string(variant.format.name()),
            variant.format.kitty_format(),
            variant.payload_bytes,
            variant.base64_bytes,
            variant.chunks,
            variant.encode_ms,
            comma
        ));
    }
    json.push_str("  ],\n");
    json.push_str("  \"downscale_estimates\": [\n");
    for (idx, budget) in budgets.iter().enumerate() {
        let comma = if idx + 1 == budgets.len() { "" } else { "," };
        json.push_str(&format!(
            concat!(
                "    {{",
                "\"divisor\":{},",
                "\"width\":{},",
                "\"height\":{},",
                "\"rgba_payload_bytes\":{},",
                "\"rgba_base64_bytes\":{},",
                "\"rgba_chunks\":{},",
                "\"rgb_payload_bytes\":{},",
                "\"rgb_base64_bytes\":{},",
                "\"rgb_chunks\":{}",
                "}}{}\n"
            ),
            budget.divisor,
            budget.width,
            budget.height,
            budget.rgba_payload_bytes,
            budget.rgba_base64_bytes,
            budget.rgba_chunks,
            budget.rgb_payload_bytes,
            budget.rgb_base64_bytes,
            budget.rgb_chunks,
            comma
        ));
    }
    json.push_str("  ]\n");
    json.push('}');
    Ok(json)
}

#[cfg_attr(not(any(test, feature = "metal")), allow(dead_code))]
fn write_kitty_rgba_direct(
    stdout: &mut impl Write,
    image_id: u32,
    width: usize,
    height: usize,
    term_cols: usize,
    term_rows: usize,
    rgba: &[u8],
) -> io::Result<(usize, usize)> {
    if rgba.is_empty() {
        return Ok((0, 0));
    }

    let encoded = BASE64_STANDARD.encode(rgba);
    debug_assert_eq!(encoded.len(), kitty_base64_len(rgba.len()));
    let mut chunks = 0usize;
    let mut offset = 0usize;
    while offset < encoded.len() {
        let end = (offset + KITTY_DIRECT_CHUNK_SIZE).min(encoded.len());
        let chunk = &encoded[offset..end];
        let more = if end < encoded.len() { 1 } else { 0 };
        if offset == 0 {
            write!(
                stdout,
                "\x1b_Ga=T,f=32,t=d,s={width},v={height},c={term_cols},r={term_rows},i={image_id},q=2,C=1,m={more};{chunk}\x1b\\"
            )?;
        } else {
            write!(stdout, "\x1b_Gq=2,m={more};{chunk}\x1b\\")?;
        }
        chunks += 1;
        offset = end;
    }
    Ok((encoded.len(), chunks))
}

#[cfg(feature = "metal")]
fn render_metal_framebuffer(
    app_state: &mut AppState,
    width: usize,
    height: usize,
) -> Result<(), crate::render::metal::MetalRenderError> {
    match app_state.metal_backend.as_mut() {
        Some(mb) => mb.render(&app_state.camera, width, height, app_state.splats.len()),
        None => Err(crate::render::metal::MetalRenderError::GpuDisabled),
    }
}

#[cfg(feature = "metal")]
fn record_kitty_gpu_error(app_state: &mut AppState, err: &crate::render::metal::MetalRenderError) {
    app_state.last_gpu_error = Some(err.to_string());
    if err.should_disable_gpu() {
        app_state.backend = Backend::Cpu;
        app_state.metal_backend = None;
        app_state.gpu_fallback_active = true;
    }
}

#[cfg(feature = "metal")]
pub fn render_kitty_frame(
    app_state: &mut AppState,
    term_cols: usize,
    term_rows: usize,
    stdout: &mut impl Write,
) -> io::Result<()> {
    if app_state.backend != Backend::Metal || app_state.metal_backend.is_none() {
        return super::frame_halfblock::render_halfblock_frame(
            app_state, term_cols, term_rows, stdout,
        );
    }

    let ss = app_state.supersample_factor as usize;
    let width = term_cols.saturating_mul(ss).max(1);
    let height = term_rows.saturating_mul(2).saturating_mul(ss).max(1);
    super::pipeline::resize_render_state(&mut app_state.render_state, width, height);

    if let Err(err) = render_metal_framebuffer(app_state, width, height) {
        record_kitty_gpu_error(app_state, &err);
        return super::frame_halfblock::render_halfblock_frame(
            app_state, term_cols, term_rows, stdout,
        );
    }

    let mut rgba = Vec::with_capacity(width.saturating_mul(height).saturating_mul(4));
    if let Some(mb) = app_state.metal_backend.as_ref() {
        packed_framebuffer_to_rgba(mb.framebuffer_slice(), &mut rgba);
    }

    queue!(stdout, cursor::MoveTo(0, 0))?;
    let (base64_bytes, chunks) = write_kitty_rgba_direct(
        stdout,
        app_state.kitty_image_id,
        width,
        height,
        term_cols,
        term_rows,
        &rgba,
    )?;
    app_state.kitty_payload_bytes = rgba.len();
    app_state.kitty_base64_bytes = base64_bytes;
    app_state.kitty_chunks = chunks;
    app_state.visible_splat_count = app_state.splats.len();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_framebuffer_to_rgba_uses_low_byte_red_layout() {
        let mut out = Vec::new();
        packed_framebuffer_to_rgba(&[0x44332211], &mut out);
        assert_eq!(out, vec![0x11, 0x22, 0x33, 0x44]);
    }

    #[test]
    fn kitty_direct_writer_chunks_payload() {
        let rgba = vec![7u8; 4096];
        let mut out = Vec::new();

        let (base64_bytes, chunks) =
            write_kitty_rgba_direct(&mut out, 1, 32, 32, 16, 16, &rgba).unwrap();

        assert_eq!(base64_bytes, kitty_base64_len(rgba.len()));
        assert_eq!(chunks, 2);
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("\x1b_Ga=T,f=32,t=d,s=32,v=32,c=16,r=16,i=1,q=2,C=1,m=1;"));
        assert!(text.contains("\x1b_Gq=2,m=0;"));
    }

    #[test]
    fn kitty_replay_measures_rgba_and_rgb_variants() {
        let rgba = vec![1, 2, 3, 255, 4, 5, 6, 255];

        let measurements = measure_kitty_replay_variants(2, 1, &rgba, 8).unwrap();

        assert_eq!(measurements.len(), 2);
        assert_eq!(measurements[0].format, KittyPayloadFormat::Rgba32);
        assert_eq!(measurements[0].payload_bytes, 8);
        assert_eq!(measurements[0].base64_bytes, 12);
        assert_eq!(measurements[0].chunks, 2);
        assert_eq!(measurements[1].format, KittyPayloadFormat::Rgb24);
        assert_eq!(measurements[1].payload_bytes, 6);
        assert_eq!(measurements[1].base64_bytes, 8);
        assert_eq!(measurements[1].chunks, 1);
    }

    #[test]
    fn kitty_replay_report_rejects_dimension_mismatch() {
        let err = kitty_replay_report_json("bad.rgba", 2, 2, &[0, 0, 0, 255], 4096).unwrap_err();

        assert!(err.to_string().contains("expected 16"));
    }

    #[test]
    fn kitty_downscale_budgets_estimate_rgba_and_rgb_chunks() {
        let budgets = kitty_downscale_budgets(4, 2, 8, &[1, 2]).unwrap();

        assert_eq!(budgets[0].rgba_payload_bytes, 32);
        assert_eq!(budgets[0].rgb_payload_bytes, 24);
        assert_eq!(budgets[1].width, 2);
        assert_eq!(budgets[1].height, 1);
        assert_eq!(budgets[1].rgba_base64_bytes, 12);
        assert_eq!(budgets[1].rgb_base64_bytes, 8);
    }
}
