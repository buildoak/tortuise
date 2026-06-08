use std::io::{self, Write};
use std::time::Instant;

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
#[cfg(feature = "metal")]
use crossterm::{cursor, queue};

#[cfg(feature = "metal")]
use super::{AppState, Backend, MetalLodMode};

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
    pub fn name(self) -> &'static str {
        match self {
            Self::Rgba32 => "rgba",
            Self::Rgb24 => "rgb",
        }
    }

    pub fn kitty_format(self) -> u8 {
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

#[derive(Debug, Clone, Copy)]
struct KittyImageSpec {
    image_id: u32,
    width: usize,
    height: usize,
    term_cols: usize,
    term_rows: usize,
    format: KittyPayloadFormat,
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

#[cfg_attr(not(any(test, feature = "metal")), allow(dead_code))]
fn packed_framebuffer_to_rgb(packed: &[u32], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(packed.len().saturating_mul(3));
    for pixel in packed {
        out.push((pixel & 0xFF) as u8);
        out.push(((pixel >> 8) & 0xFF) as u8);
        out.push(((pixel >> 16) & 0xFF) as u8);
    }
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

#[cfg(test)]
fn write_kitty_rgba_direct(
    stdout: &mut impl Write,
    spec: KittyImageSpec,
    payload: &[u8],
) -> io::Result<(usize, usize)> {
    if payload.is_empty() {
        return Ok((0, 0));
    }

    let encoded = BASE64_STANDARD.encode(payload);
    debug_assert_eq!(encoded.len(), kitty_base64_len(payload.len()));
    write_kitty_encoded_direct(stdout, spec, &encoded)
}

#[cfg_attr(not(any(test, feature = "metal")), allow(dead_code))]
fn write_kitty_encoded_direct(
    stdout: &mut impl Write,
    spec: KittyImageSpec,
    encoded: &str,
) -> io::Result<(usize, usize)> {
    if encoded.is_empty() {
        return Ok((0, 0));
    }

    let kitty_format = spec.format.kitty_format();
    let mut chunks = 0usize;
    let mut offset = 0usize;
    while offset < encoded.len() {
        let end = (offset + KITTY_DIRECT_CHUNK_SIZE).min(encoded.len());
        let chunk = &encoded[offset..end];
        let more = if end < encoded.len() { 1 } else { 0 };
        if offset == 0 {
            write!(
                stdout,
                "\x1b_Ga=T,f={kitty_format},t=d,s={},v={},c={},r={},i={},q=2,C=1,m={more};{chunk}\x1b\\",
                spec.width,
                spec.height,
                spec.term_cols,
                spec.term_rows,
                spec.image_id
            )?;
        } else {
            write!(stdout, "\x1b_Gq=2,m={more};{chunk}\x1b\\")?;
        }
        chunks += 1;
        offset = end;
    }
    Ok((encoded.len(), chunks))
}

#[cfg_attr(not(any(test, feature = "metal")), allow(dead_code))]
pub fn delete_kitty_image(stdout: &mut impl Write, image_id: u32) -> io::Result<()> {
    write!(stdout, "\x1b_Ga=d,d=i,i={image_id},q=2\x1b\\")
}

#[cfg_attr(not(any(test, feature = "metal")), allow(dead_code))]
fn next_kitty_image_id(image_id: u32) -> u32 {
    if image_id == 1 {
        2
    } else {
        1
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
fn kitty_payload_format_from_env() -> KittyPayloadFormat {
    std::env::var("TORTUISE_KITTY_FORMAT")
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "rgb" | "24" | "f24" => KittyPayloadFormat::Rgb24,
            _ => KittyPayloadFormat::Rgba32,
        })
        .unwrap_or(KittyPayloadFormat::Rgba32)
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
fn kitty_scale_divisor_from_env() -> usize {
    std::env::var("TORTUISE_KITTY_SCALE_DIVISOR")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
        .min(16)
}

#[cfg(feature = "metal")]
fn render_metal_framebuffer(
    app_state: &mut AppState,
    width: usize,
    height: usize,
) -> Result<(), crate::render::metal::MetalRenderError> {
    match app_state.metal_backend.as_mut() {
        Some(mb) => mb.render(
            &app_state.camera,
            width,
            height,
            app_state.metal_active_splat_count,
            app_state.splats.len(),
            app_state.metal_lod_mode,
            app_state.metal_lod_order,
            app_state.metal_lod_requested_splat_count,
        ),
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
fn clear_kitty_transport_stats(app_state: &mut AppState) {
    app_state.kitty_payload_bytes = 0;
    app_state.kitty_base64_bytes = 0;
    app_state.kitty_chunks = 0;
    app_state.kitty_convert_ms = 0.0;
    app_state.kitty_encode_ms = 0.0;
    app_state.kitty_write_ms = 0.0;
}

#[cfg_attr(not(any(test, feature = "metal")), allow(dead_code))]
fn kitty_image_placement_rows(show_hud: bool, term_rows: usize) -> (usize, usize) {
    let reserved_hud_rows = if show_hud && term_rows > 2 { 2 } else { 0 };
    let image_term_rows = term_rows.saturating_sub(reserved_hud_rows).max(1);
    let image_start_row = if reserved_hud_rows > 0 { 1 } else { 0 };
    (image_start_row, image_term_rows)
}

#[cfg(feature = "metal")]
pub fn render_kitty_frame(
    app_state: &mut AppState,
    term_cols: usize,
    term_rows: usize,
    stdout: &mut impl Write,
) -> io::Result<()> {
    if app_state.backend != Backend::Metal || app_state.metal_backend.is_none() {
        clear_kitty_transport_stats(app_state);
        return super::frame_halfblock::render_halfblock_frame(
            app_state, term_cols, term_rows, stdout,
        );
    }

    let ss = app_state.supersample_factor as usize;
    let scale_divisor = kitty_scale_divisor_from_env();
    let (image_start_row, image_term_rows) =
        kitty_image_placement_rows(app_state.show_hud, term_rows);
    let width = term_cols.saturating_mul(ss).div_ceil(scale_divisor).max(1);
    let height = image_term_rows
        .saturating_mul(2)
        .saturating_mul(ss)
        .div_ceil(scale_divisor)
        .max(1);
    super::pipeline::resize_render_state(&mut app_state.render_state, width, height);

    if let Err(err) = render_metal_framebuffer(app_state, width, height) {
        record_kitty_gpu_error(app_state, &err);
        if app_state.metal_lod_mode == MetalLodMode::Fixed && err.should_disable_gpu() {
            return Err(io::Error::other(format!(
                "Metal fixed LoD render failed before CPU fallback: {err}"
            )));
        }
        clear_kitty_transport_stats(app_state);
        return super::frame_halfblock::render_halfblock_frame(
            app_state, term_cols, term_rows, stdout,
        );
    }

    let format = kitty_payload_format_from_env();
    let mut payload = Vec::with_capacity(
        width
            .saturating_mul(height)
            .saturating_mul(format.bytes_per_pixel()),
    );
    let convert_start = Instant::now();
    if let Some(mb) = app_state.metal_backend.as_ref() {
        match format {
            KittyPayloadFormat::Rgba32 => {
                packed_framebuffer_to_rgba(mb.framebuffer_slice(), &mut payload)
            }
            KittyPayloadFormat::Rgb24 => {
                packed_framebuffer_to_rgb(mb.framebuffer_slice(), &mut payload)
            }
        }
    }
    app_state.kitty_convert_ms = convert_start.elapsed().as_secs_f32() * 1000.0;

    let encode_start = Instant::now();
    let encoded = BASE64_STANDARD.encode(&payload);
    app_state.kitty_encode_ms = encode_start.elapsed().as_secs_f32() * 1000.0;
    debug_assert_eq!(encoded.len(), kitty_base64_len(payload.len()));

    let write_start = Instant::now();
    let image_id = app_state.kitty_image_id;
    let previous_image_id = app_state.kitty_visible_image_id;
    queue!(stdout, cursor::MoveTo(0, image_start_row as u16))?;
    let spec = KittyImageSpec {
        image_id,
        width,
        height,
        term_cols,
        term_rows: image_term_rows,
        format,
    };
    let (base64_bytes, chunks) = write_kitty_encoded_direct(stdout, spec, &encoded)?;
    if previous_image_id != 0 && previous_image_id != image_id {
        delete_kitty_image(stdout, previous_image_id)?;
    }
    app_state.kitty_visible_image_id = image_id;
    app_state.kitty_image_id = next_kitty_image_id(image_id);
    app_state.kitty_payload_bytes = payload.len();
    app_state.kitty_base64_bytes = base64_bytes;
    app_state.kitty_chunks = chunks;
    app_state.kitty_write_ms = write_start.elapsed().as_secs_f32() * 1000.0;
    if let Some(mb) = app_state.metal_backend.as_ref() {
        let telemetry = mb.probe_telemetry();
        app_state.visible_splat_count = telemetry.valid_count as usize;
    }
    app_state.effective_render_path = "metal_kitty";
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

        let (base64_bytes, chunks) = write_kitty_rgba_direct(
            &mut out,
            KittyImageSpec {
                image_id: 1,
                width: 32,
                height: 32,
                term_cols: 16,
                term_rows: 16,
                format: KittyPayloadFormat::Rgba32,
            },
            &rgba,
        )
        .unwrap();

        assert_eq!(base64_bytes, kitty_base64_len(rgba.len()));
        assert_eq!(chunks, 2);
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("\x1b_Ga=T,f=32,t=d,s=32,v=32,c=16,r=16,i=1,q=2,C=1,m=1;"));
        assert!(text.contains("\x1b_Gq=2,m=0;"));
    }

    #[test]
    fn packed_framebuffer_to_rgb_drops_alpha() {
        let mut out = Vec::new();
        packed_framebuffer_to_rgb(&[0x44332211], &mut out);
        assert_eq!(out, vec![0x11, 0x22, 0x33]);
    }

    #[test]
    fn kitty_delete_image_uses_id_specific_quiet_delete() {
        let mut out = Vec::new();
        delete_kitty_image(&mut out, 7).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\x1b_Ga=d,d=i,i=7,q=2\x1b\\"
        );
    }

    #[test]
    fn kitty_image_ids_alternate_between_two_slots() {
        assert_eq!(next_kitty_image_id(1), 2);
        assert_eq!(next_kitty_image_id(2), 1);
        assert_eq!(next_kitty_image_id(99), 1);
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
    fn kitty_image_placement_reserves_hud_rows_when_possible() {
        assert_eq!(kitty_image_placement_rows(true, 40), (1, 38));
        assert_eq!(kitty_image_placement_rows(false, 40), (0, 40));
        assert_eq!(kitty_image_placement_rows(true, 2), (0, 2));
        assert_eq!(kitty_image_placement_rows(true, 0), (0, 1));
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
