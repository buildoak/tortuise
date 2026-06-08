use crate::camera;
use crate::render::AppState;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HeldMovementKeys {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
}

#[derive(Debug, Default)]
pub struct InputState {
    pub held: HeldMovementKeys,
    pub quit_requested: bool,
}

pub fn apply_movement_from_held_keys(app_state: &mut AppState, delta_time: f32) {
    let step = app_state.move_speed * delta_time.max(0.0);
    if step <= 0.0 {
        return;
    }

    let held = app_state.input_state.held;
    let forward = (held.forward as i8 - held.back as i8) as f32;
    let right = (held.right as i8 - held.left as i8) as f32;
    let up = (held.up as i8 - held.down as i8) as f32;

    if forward != 0.0 {
        camera::move_forward(&mut app_state.camera, forward * step);
    }
    if right != 0.0 {
        camera::move_right(&mut app_state.camera, right * step);
    }
    if up != 0.0 && matches!(app_state.camera_mode, crate::render::CameraMode::Free) {
        camera::move_up(&mut app_state.camera, up * step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::math::Vec3;
    use crate::render::{AppState, Backend, CameraMode, RenderMode, RenderState};
    use std::time::Instant;

    fn make_state() -> AppState {
        AppState {
            camera: Camera::new(Vec3::new(0.0, 0.0, 5.0), -std::f32::consts::FRAC_PI_2, 0.0),
            splats: Vec::new(),
            projected_splats: Vec::new(),
            render_state: RenderState {
                framebuffer: vec![[0, 0, 0]; 4],
                alpha_buffer: vec![0.0; 4],
                depth_buffer: vec![f32::INFINITY; 4],
                width: 2,
                height: 2,
            },
            halfblock_cells: Vec::new(),
            hud_string_buf: String::new(),
            input_state: InputState::default(),
            show_hud: true,
            camera_mode: CameraMode::Free,
            move_speed: 2.0,
            frame_count: 0,
            last_frame_time: Instant::now(),
            fps: 0.0,
            visible_splat_count: 0,
            orbit_angle: 0.0,
            orbit_radius: 5.0,
            orbit_height: 0.0,
            orbit_target: Vec3::ZERO,
            supersample_factor: 1,
            render_mode: RenderMode::Halfblock,
            backend: Backend::Cpu,
            use_truecolor: false,
            #[cfg(feature = "metal")]
            kitty_image_id: 1,
            #[cfg(feature = "metal")]
            kitty_visible_image_id: 0,
            #[cfg(feature = "metal")]
            kitty_payload_bytes: 0,
            #[cfg(feature = "metal")]
            kitty_base64_bytes: 0,
            #[cfg(feature = "metal")]
            kitty_chunks: 0,
            #[cfg(feature = "metal")]
            kitty_write_ms: 0.0,
            #[cfg(feature = "metal")]
            metal_backend: None,
            #[cfg(feature = "metal")]
            last_gpu_error: None,
            #[cfg(feature = "metal")]
            gpu_fallback_active: false,
        }
    }

    #[test]
    fn movement_scales_with_delta_time() {
        let mut app = make_state();
        app.input_state.held.forward = true;
        let z0 = app.camera.position.z;
        apply_movement_from_held_keys(&mut app, 0.016);
        let d1 = (app.camera.position.z - z0).abs();

        let mut app2 = make_state();
        app2.input_state.held.forward = true;
        let z1 = app2.camera.position.z;
        apply_movement_from_held_keys(&mut app2, 0.032);
        let d2 = (app2.camera.position.z - z1).abs();

        assert!((d2 - d1 * 2.0).abs() < 1e-4);
    }
}
