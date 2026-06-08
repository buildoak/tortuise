pub mod state;
pub mod thread;

use crate::camera;
use crate::math::Vec3;
use crate::render::{AppState, CameraMode};
use crossterm::event::{Event, KeyCode, KeyEventKind};
use std::sync::mpsc::{Receiver, TryRecvError};

use crate::AppResult;

pub fn drain_input_events(
    app_state: &mut AppState,
    input_rx: &Receiver<crate::input::thread::InputMessage>,
) -> AppResult<bool> {
    loop {
        match input_rx.try_recv() {
            Ok(crate::input::thread::InputMessage::Event(event)) => {
                handle_input_event(app_state, event)?;
                if app_state.input_state.quit_requested {
                    return Ok(true);
                }
            }
            Ok(crate::input::thread::InputMessage::ReadError(err)) => {
                return Err(format!("Input thread read failed: {err}").into());
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                return Err("Input channel disconnected".into());
            }
        }
    }

    Ok(app_state.input_state.quit_requested)
}

/// Transition from Free camera to Orbit mode.
///
/// Always orbits around the origin — WASD navigation does not shift the
/// orbit center. Matches the pre-modal-camera behavior.
fn transition_to_orbit(app_state: &mut AppState) {
    let target = Vec3::ZERO;
    app_state.orbit_target = target;

    let dx = app_state.camera.position.x - target.x;
    let dz = app_state.camera.position.z - target.z;
    app_state.orbit_radius = (dx * dx + dz * dz).sqrt().max(0.5);
    app_state.orbit_angle = dz.atan2(dx);
    app_state.orbit_height = app_state.camera.position.y - target.y;

    // Clear held movement keys so WASD state doesn't leak
    app_state.input_state.held = crate::input::state::HeldMovementKeys::default();
    app_state.camera_mode = CameraMode::Orbit;
}

/// Transition from Orbit mode back to Free camera.
///
/// Keeps the camera at its current position and sets yaw/pitch to face
/// the orbit target, so the view is seamless.
fn transition_to_free(app_state: &mut AppState) {
    camera::look_at_target(&mut app_state.camera, app_state.orbit_target);
    app_state.camera_mode = CameraMode::Free;
}

pub fn handle_input_event(app_state: &mut AppState, event: Event) -> AppResult<()> {
    match event {
        Event::Key(key_event) => {
            // Track held WASD keys (press/repeat/release) — only meaningful in Free mode,
            // but we track state always and just ignore it in Orbit's movement path.
            if let KeyCode::Char(c) = key_event.code {
                let lc = c.to_ascii_lowercase();
                if matches!(
                    key_event.kind,
                    KeyEventKind::Press | KeyEventKind::Repeat | KeyEventKind::Release
                ) {
                    let pressed = key_event.kind != KeyEventKind::Release;
                    match lc {
                        'w' => app_state.input_state.held.forward = pressed,
                        's' => app_state.input_state.held.back = pressed,
                        'a' => app_state.input_state.held.left = pressed,
                        'd' => app_state.input_state.held.right = pressed,
                        'r' => app_state.input_state.held.up = pressed,
                        'f' => app_state.input_state.held.down = pressed,
                        _ => {}
                    }
                }
            }

            // Only process press/repeat for discrete actions below
            if !matches!(key_event.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                return Ok(());
            }

            match key_event.code {
                KeyCode::Esc => app_state.input_state.quit_requested = true,
                KeyCode::Tab => app_state.show_hud = !app_state.show_hud,
                KeyCode::Char('+') | KeyCode::Char('=') => {
                    app_state.move_speed = (app_state.move_speed * 1.2).min(10.0);
                }
                KeyCode::Char('-') | KeyCode::Char('_') => {
                    app_state.move_speed = (app_state.move_speed / 1.2).max(0.01);
                }
                KeyCode::Char(' ') => match app_state.camera_mode {
                    CameraMode::Free => transition_to_orbit(app_state),
                    CameraMode::Orbit => transition_to_free(app_state),
                },

                // Arrow keys: modal behavior
                KeyCode::Up => match app_state.camera_mode {
                    CameraMode::Free => {
                        camera::adjust_pitch(&mut app_state.camera, 0.08 * app_state.move_speed);
                    }
                    CameraMode::Orbit => {
                        app_state.orbit_height += 0.15 * app_state.move_speed;
                    }
                },
                KeyCode::Down => match app_state.camera_mode {
                    CameraMode::Free => {
                        camera::adjust_pitch(&mut app_state.camera, -0.08 * app_state.move_speed);
                    }
                    CameraMode::Orbit => {
                        app_state.orbit_height -= 0.15 * app_state.move_speed;
                    }
                },
                KeyCode::Left => match app_state.camera_mode {
                    CameraMode::Free => {
                        camera::adjust_yaw(&mut app_state.camera, -0.08 * app_state.move_speed);
                    }
                    CameraMode::Orbit => {
                        // Manual orbit nudge (in addition to auto-orbit)
                        app_state.orbit_angle -= 0.1 * app_state.move_speed;
                    }
                },
                KeyCode::Right => match app_state.camera_mode {
                    CameraMode::Free => {
                        camera::adjust_yaw(&mut app_state.camera, 0.08 * app_state.move_speed);
                    }
                    CameraMode::Orbit => {
                        app_state.orbit_angle += 0.1 * app_state.move_speed;
                    }
                },

                KeyCode::Char(c) => match c.to_ascii_lowercase() {
                    'q' => app_state.input_state.quit_requested = true,
                    'm' => {
                        app_state.render_mode = app_state.render_mode.next();
                    }
                    'z' => {
                        camera::reset(&mut app_state.camera, Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO);
                        app_state.camera_mode = CameraMode::Free;
                        app_state.orbit_target = Vec3::ZERO;
                        app_state.orbit_angle = 0.0;
                        app_state.orbit_radius = 5.0;
                        app_state.orbit_height = 0.0;
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        Event::FocusLost => {
            app_state.input_state.held = crate::input::state::HeldMovementKeys::default();
        }
        Event::Resize(_, _) => {}
        _ => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::render::{AppState, Backend, CameraMode, RenderMode, RenderState};
    use std::sync::mpsc;
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
            input_state: crate::input::state::InputState::default(),
            show_hud: true,
            camera_mode: CameraMode::Free,
            move_speed: 0.3,
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
    fn held_keys_toggle_on_press_and_release() {
        let mut app = make_state();
        handle_input_event(
            &mut app,
            Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('w'),
                crossterm::event::KeyModifiers::NONE,
            )),
        )
        .expect("press should succeed");
        assert!(app.input_state.held.forward);

        let release = crossterm::event::KeyEvent {
            code: KeyCode::Char('w'),
            modifiers: crossterm::event::KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: crossterm::event::KeyEventState::NONE,
        };
        handle_input_event(&mut app, Event::Key(release)).expect("release should succeed");
        assert!(!app.input_state.held.forward);
    }

    #[test]
    fn drain_consumes_all_queued_events() {
        let (tx, rx) = mpsc::channel();
        tx.send(crate::input::thread::InputMessage::Event(Event::Key(
            crossterm::event::KeyEvent::new(
                KeyCode::Char('w'),
                crossterm::event::KeyModifiers::NONE,
            ),
        )))
        .expect("send w");
        tx.send(crate::input::thread::InputMessage::Event(Event::Key(
            crossterm::event::KeyEvent::new(
                KeyCode::Char('a'),
                crossterm::event::KeyModifiers::NONE,
            ),
        )))
        .expect("send a");

        let mut app = make_state();
        let quit = drain_input_events(&mut app, &rx).expect("drain should succeed");
        assert!(!quit);
        assert!(app.input_state.held.forward);
        assert!(app.input_state.held.left);
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn speed_keys_adjust_move_speed() {
        let mut app = make_state();
        let base = app.move_speed;
        handle_input_event(
            &mut app,
            Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('='),
                crossterm::event::KeyModifiers::NONE,
            )),
        )
        .expect("increase speed");
        assert!(app.move_speed > base);

        let increased = app.move_speed;
        handle_input_event(
            &mut app,
            Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('_'),
                crossterm::event::KeyModifiers::SHIFT,
            )),
        )
        .expect("decrease speed");
        assert!(app.move_speed < increased);
    }

    #[test]
    fn focus_lost_clears_held_movement() {
        let mut app = make_state();
        app.input_state.held.forward = true;
        app.input_state.held.left = true;

        handle_input_event(&mut app, Event::FocusLost).expect("focus lost should succeed");
        assert!(!app.input_state.held.forward);
        assert!(!app.input_state.held.left);
    }
}
