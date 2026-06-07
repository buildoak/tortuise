use super::{types, MetalBackend};
use std::{
    mem,
    sync::{Mutex, MutexGuard, Once, OnceLock},
    time::Duration,
};

use rand::{Rng, SeedableRng};

use crate::{
    camera::{look_at_origin, Camera},
    demo::generate_demo_splats,
    math::Vec3,
    render::{
        modes::halfblock::downsample_packed_to_terminal_into, pipeline, rasterizer, HalfblockCell,
        RenderState,
    },
    sort::sort_by_depth,
    splat::Splat,
};

static ENV_INIT: Once = Once::new();
static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn test_guard() -> MutexGuard<'static, ()> {
    let mutex = TEST_MUTEX.get_or_init(|| Mutex::new(()));
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn init_metal_validation_env() {
    ENV_INIT.call_once(|| {
        std::env::set_var("MTL_DEBUG_LAYER", "1");
        std::env::set_var("MTL_SHADER_VALIDATION", "1");
    });
}

fn setup_metal_test() -> Option<MutexGuard<'static, ()>> {
    let guard = test_guard();
    init_metal_validation_env();

    if metal::Device::system_default().is_none() {
        eprintln!("Skipping Metal test: no system-default Metal device.");
        return None;
    }

    Some(guard)
}

fn make_test_camera() -> Camera {
    let mut camera = Camera::new(Vec3::new(0.0, 0.0, 5.0), -std::f32::consts::FRAC_PI_2, 0.0);
    look_at_origin(&mut camera);
    camera
}

fn make_render_state(width: usize, height: usize) -> RenderState {
    let len = width.saturating_mul(height);
    RenderState {
        framebuffer: vec![[0, 0, 0]; len],
        alpha_buffer: vec![0.0; len],
        depth_buffer: vec![f32::INFINITY; len],
        width,
        height,
    }
}

fn unpack_rgb(framebuffer: &[u32]) -> Vec<[u8; 3]> {
    framebuffer
        .iter()
        .map(|&p| {
            [
                (p & 0xFF) as u8,
                ((p >> 8) & 0xFF) as u8,
                ((p >> 16) & 0xFF) as u8,
            ]
        })
        .collect()
}

fn unpack_halfblock_cells(cells: &[types::GpuHalfblockCell]) -> Vec<HalfblockCell> {
    cells
        .iter()
        .map(|cell| {
            let top = cell.top_rgb;
            let bottom = cell.bottom_rgb;
            (
                [
                    (top & 0xFF) as u8,
                    ((top >> 8) & 0xFF) as u8,
                    ((top >> 16) & 0xFF) as u8,
                ],
                [
                    (bottom & 0xFF) as u8,
                    ((bottom >> 8) & 0xFF) as u8,
                    ((bottom >> 16) & 0xFF) as u8,
                ],
            )
        })
        .collect()
}

fn cpu_reference_framebuffer(splats: &[Splat], width: usize, height: usize) -> Vec<[u8; 3]> {
    let camera = make_test_camera();
    let mut projected = Vec::with_capacity(splats.len());
    let mut visible_count = 0usize;

    pipeline::project_and_cull_splats(
        splats,
        &mut projected,
        &camera,
        width,
        height,
        &mut visible_count,
    );
    sort_by_depth(&mut projected);

    let mut render_state = make_render_state(width, height);
    rasterizer::rasterize_splats(&projected, &mut render_state, width, height);
    render_state.framebuffer
}

#[test]
fn test_halfblock_downsample_matches_cpu_oracle_for_synthetic_framebuffer() {
    let _guard = match setup_metal_test() {
        Some(g) => g,
        None => return,
    };

    let width = 4usize;
    let height = 4usize;
    let term_cols = 2usize;
    let term_rows = 1usize;
    let ss = 2usize;
    let framebuffer = vec![
        0x000000ff, 0x0000ff00, 0x00ff0000, 0x00000000, 0x000000ff, 0x0000ff00, 0x00ff0000,
        0x00000000, 0x00000000, 0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000, 0x00ff0000,
        0x0000ff00, 0x000000ff,
    ];

    let mut backend = MetalBackend::new(0).expect("MetalBackend::new should succeed");
    backend
        .ensure_framebuffer_capacity(width, height)
        .expect("framebuffer capacity should grow");
    unsafe {
        let dst = backend.framebuffer.contents() as *mut u32;
        std::ptr::copy_nonoverlapping(framebuffer.as_ptr(), dst, framebuffer.len());
    }

    backend
        .downsample_halfblock_cells(width, height, term_cols, term_rows, ss)
        .expect("GPU halfblock downsample should succeed");

    let gpu_cells = unpack_halfblock_cells(backend.halfblock_cells_slice());
    let mut expected = Vec::new();
    downsample_packed_to_terminal_into(
        &framebuffer,
        width,
        height,
        term_cols,
        term_rows,
        ss,
        &mut expected,
    );
    assert_eq!(gpu_cells, expected);
}

#[test]
fn test_halfblock_downsample_matches_cpu_oracle_after_render() {
    let _guard = match setup_metal_test() {
        Some(g) => g,
        None => return,
    };

    let width = 32usize;
    let height = 32usize;
    let term_cols = 16usize;
    let term_rows = 8usize;
    let ss = 2usize;
    let camera = make_test_camera();
    let splats = generate_seeded_splats(20, 0xABCDEF_u64);

    let mut backend = MetalBackend::new(splats.len()).expect("MetalBackend::new should succeed");
    backend
        .upload_splats(&splats)
        .expect("upload_splats should succeed");
    backend
        .render(&camera, width, height, splats.len())
        .expect("render should succeed");
    backend
        .downsample_halfblock_cells(width, height, term_cols, term_rows, ss)
        .expect("GPU halfblock downsample should succeed");

    let gpu_cells = unpack_halfblock_cells(backend.halfblock_cells_slice());
    let mut expected = Vec::new();
    downsample_packed_to_terminal_into(
        backend.framebuffer_slice(),
        width,
        height,
        term_cols,
        term_rows,
        ss,
        &mut expected,
    );
    assert_eq!(gpu_cells, expected);
}

fn make_center_red_splat() -> Splat {
    Splat {
        position: Vec3::new(0.0, 0.0, 0.0),
        color: [255, 0, 0],
        opacity: 1.0,
        scale: Vec3::new(0.5, 0.5, 0.5),
        rotation: [1.0, 0.0, 0.0, 0.0],
    }
}

fn generate_seeded_splats(count: usize, seed: u64) -> Vec<Splat> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut splats = Vec::with_capacity(count);

    for _ in 0..count {
        splats.push(Splat {
            position: Vec3::new(
                rng.random_range(-1.6_f32..1.6_f32),
                rng.random_range(-1.6_f32..1.6_f32),
                rng.random_range(-1.5_f32..1.5_f32),
            ),
            color: [
                rng.random_range(24_u8..=255_u8),
                rng.random_range(24_u8..=255_u8),
                rng.random_range(24_u8..=255_u8),
            ],
            opacity: rng.random_range(0.35_f32..0.95_f32),
            scale: Vec3::new(
                rng.random_range(0.03_f32..0.12_f32),
                rng.random_range(0.03_f32..0.12_f32),
                rng.random_range(0.03_f32..0.12_f32),
            ),
            rotation: [1.0, 0.0, 0.0, 0.0],
        });
    }

    splats
}

fn make_radix_sort_stability_input(count: usize) -> Vec<(u64, u32)> {
    let low_digits = [0x00_u64, 0x7f, 0x7f, 0x80, 0xff, 0x2a, 0x2a, 0x00];
    let high_bytes = [0x11_u64, 0x11, 0x8e, 0x8e, 0xe4, 0xff];

    (0..count)
        .map(|i| {
            let class = (i % 181) as u64;
            let low = low_digits[class as usize % low_digits.len()];
            let byte_1 = ((class / 8) % 4) * 0x11;
            let byte_2 = (class.wrapping_mul(7).wrapping_add(class / 3)) & 0xff;
            let high = high_bytes[class as usize % high_bytes.len()];
            let key = (high << 56) | (class << 24) | (byte_2 << 16) | (byte_1 << 8) | low;
            (key, i as u32)
        })
        .collect()
}

fn write_shared_buffer_slice<T: Copy>(buffer: &metal::Buffer, values: &[T]) {
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), buffer.contents() as *mut T, values.len());
    }
}

fn read_shared_buffer_slice<T: Copy>(buffer: &metal::Buffer, count: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(buffer.contents() as *const T, count).to_vec() }
}

#[test]
fn test_warmed_overlap_estimate_does_not_floor_large_sparse_scene_to_splat_count() {
    let estimate = super::render::estimate_overlaps_for_attempt(2_315_943, 59_901, 192, 192);
    assert_eq!(estimate, 74_877);
}

#[test]
fn test_warmed_overlap_estimate_scales_with_tile_count() {
    let estimate = super::render::estimate_overlaps_for_attempt(2_315_943, 59_901, 96, 192);
    assert_eq!(estimate, 149_753);
}

#[test]
fn test_warmed_overlap_estimate_keeps_small_scene_floor() {
    let estimate = super::render::estimate_overlaps_for_attempt(64, 20, 16, 16);
    assert_eq!(estimate, 1024);
}

#[test]
fn test_cold_overlap_estimate_stays_conservative() {
    assert_eq!(
        super::render::estimate_overlaps_for_attempt(2_315_943, 0, 0, 192),
        18_527_544
    );
    assert_eq!(
        super::render::estimate_overlaps_for_attempt(64, 0, 0, 16),
        1024
    );
}

#[test]
fn test_struct_sizes() {
    assert_eq!(std::mem::size_of::<types::GpuSplatData>(), 48);
    assert_eq!(std::mem::size_of::<types::GpuCameraData>(), 72);
    assert_eq!(std::mem::size_of::<types::GpuProjectedSplat>(), 52);
    assert_eq!(std::mem::size_of::<types::TileConfig>(), 16);
}

#[test]
fn test_radix_sort_direct_matches_cpu_stable_sort_with_repeated_digits() {
    let _guard = match setup_metal_test() {
        Some(g) => g,
        None => return,
    };

    let input = make_radix_sort_stability_input(769);
    let mut expected = input.clone();
    expected.sort_by_key(|&(key, _value)| key);

    let keys: Vec<u64> = input.iter().map(|&(key, _value)| key).collect();
    let values: Vec<u32> = input.iter().map(|&(_key, value)| value).collect();
    let key_bytes = mem::size_of_val(keys.as_slice());
    let value_bytes = mem::size_of_val(values.as_slice());
    let count_u32 = u32::try_from(input.len()).expect("test input should fit u32");

    let mut backend = MetalBackend::new(0).expect("MetalBackend::new should succeed");
    backend
        .ensure_sort_capacity(input.len())
        .expect("sort buffers should grow for test input");

    let num_blocks = super::sort::div_ceil_u32(count_u32, types::THREADS_PER_GROUP_1D);
    let histogram_count = num_blocks
        .checked_mul(types::RADIX_BUCKETS)
        .expect("test histogram count should fit u32");
    backend
        .ensure_histogram_capacity(histogram_count as usize)
        .expect("histogram buffer should grow for test input");
    backend
        .ensure_block_sums_capacity_for_count(histogram_count)
        .expect("block-sum scratch should grow for histogram scan");

    unsafe {
        *(backend.total_overlaps_buffer.contents() as *mut u32) = count_u32;
    }

    let key_upload = super::pipeline::new_shared_buffer(&backend.device, key_bytes);
    let value_upload = super::pipeline::new_shared_buffer(&backend.device, value_bytes);
    let key_readback = super::pipeline::new_shared_buffer(&backend.device, key_bytes);
    let value_readback = super::pipeline::new_shared_buffer(&backend.device, value_bytes);
    write_shared_buffer_slice(&key_upload, &keys);
    write_shared_buffer_slice(&value_upload, &values);

    let command_buffer = backend.command_queue.new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_buffer(&key_upload, 0, &backend.sort_keys_a, 0, key_bytes as u64);
    blit.copy_from_buffer(
        &value_upload,
        0,
        &backend.sort_values_a,
        0,
        value_bytes as u64,
    );
    blit.end_encoding();

    let mut keys_in_a = true;
    backend
        .run_radix_sort_passes(command_buffer, count_u32, &mut keys_in_a)
        .expect("radix sort kernels should encode");

    let (sorted_keys, sorted_values) = if keys_in_a {
        (&backend.sort_keys_a, &backend.sort_values_a)
    } else {
        (&backend.sort_keys_b, &backend.sort_values_b)
    };
    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_buffer(sorted_keys, 0, &key_readback, 0, key_bytes as u64);
    blit.copy_from_buffer(sorted_values, 0, &value_readback, 0, value_bytes as u64);
    blit.end_encoding();

    super::sync::commit_and_wait_with_timeout(
        command_buffer,
        "radix_sort_direct_stability_test",
        Duration::from_secs(5),
    )
    .expect("radix sort command buffer should complete");

    let actual_keys = read_shared_buffer_slice::<u64>(&key_readback, input.len());
    let actual_values = read_shared_buffer_slice::<u32>(&value_readback, input.len());
    let actual: Vec<(u64, u32)> = actual_keys.into_iter().zip(actual_values).collect();

    assert_eq!(actual, expected);
}

#[test]
fn test_metal_backend_creation() {
    let _guard = match setup_metal_test() {
        Some(g) => g,
        None => return,
    };

    let backend = MetalBackend::new(16).expect("MetalBackend::new should succeed");
    assert!(
        !backend.is_ready(),
        "backend should not be ready before upload"
    );
}

#[test]
fn test_upload_splats() {
    let _guard = match setup_metal_test() {
        Some(g) => g,
        None => return,
    };

    let splats = generate_demo_splats();
    let mut backend = MetalBackend::new(splats.len()).expect("MetalBackend::new should succeed");
    backend
        .upload_splats(&splats)
        .expect("upload_splats should succeed for demo data");
    assert!(
        backend.is_ready(),
        "backend should report ready after upload"
    );
}

#[test]
fn test_render_empty_scene() {
    let _guard = match setup_metal_test() {
        Some(g) => g,
        None => return,
    };

    let camera = make_test_camera();
    let mut backend = MetalBackend::new(0).expect("MetalBackend::new should succeed");
    backend
        .upload_splats(&[])
        .expect("upload_splats should accept empty slice");

    backend
        .render(&camera, 64, 64, 0)
        .expect("render should succeed for empty scene");
    let framebuffer = backend.framebuffer_slice().to_vec();
    assert!(framebuffer.is_empty() || framebuffer.iter().all(|&p| p == 0));
}

#[test]
fn test_render_matches_cpu() {
    let _guard = match setup_metal_test() {
        Some(g) => g,
        None => return,
    };

    let width = 128usize;
    let height = 128usize;
    let camera = make_test_camera();
    let splats = generate_seeded_splats(50, 0xC0FFEE_u64);

    let mut backend = MetalBackend::new(splats.len()).expect("MetalBackend::new should work");
    backend
        .upload_splats(&splats)
        .expect("upload_splats should succeed");

    backend
        .render(&camera, width, height, splats.len())
        .expect("GPU render should succeed");
    let gpu_packed = backend.framebuffer_slice().to_vec();
    let gpu_rgb = unpack_rgb(&gpu_packed);
    let cpu_rgb = cpu_reference_framebuffer(&splats, width, height);

    let tolerance = 8u8;
    let mut out_of_tolerance = 0usize;
    for (gpu_px, cpu_px) in gpu_rgb.iter().zip(cpu_rgb.iter()) {
        let within = gpu_px[0].abs_diff(cpu_px[0]) <= tolerance
            && gpu_px[1].abs_diff(cpu_px[1]) <= tolerance
            && gpu_px[2].abs_diff(cpu_px[2]) <= tolerance;
        if !within {
            out_of_tolerance += 1;
        }
    }

    let pixel_count = width * height;
    let allowed = (pixel_count as f32 * 0.20).ceil() as usize;
    assert!(out_of_tolerance <= allowed);
}

#[test]
fn test_resize_handling() {
    let _guard = match setup_metal_test() {
        Some(g) => g,
        None => return,
    };

    let camera = make_test_camera();
    let splats = vec![make_center_red_splat()];

    let mut backend = MetalBackend::new(splats.len()).expect("MetalBackend::new should work");
    backend
        .upload_splats(&splats)
        .expect("upload_splats should succeed");

    backend
        .render(&camera, 64, 64, splats.len())
        .expect("64x64 render should succeed");
    let fb_64_a = backend.framebuffer_slice().to_vec();
    backend
        .render(&camera, 256, 256, splats.len())
        .expect("256x256 render should succeed");
    let fb_256 = backend.framebuffer_slice().to_vec();
    backend
        .render(&camera, 64, 64, splats.len())
        .expect("second 64x64 render should succeed");
    let fb_64_b = backend.framebuffer_slice().to_vec();

    assert_eq!(fb_64_a.len(), 64 * 64);
    assert_eq!(fb_256.len(), 256 * 256);
    assert_eq!(fb_64_b.len(), 64 * 64);
}
