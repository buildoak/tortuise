use crate::math::Vec3;

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(position: Vec3, yaw: f32, pitch: f32) -> Self {
        let mut camera = Self {
            position,
            forward: Vec3::new(0.0, 0.0, -1.0),
            right: Vec3::new(1.0, 0.0, 0.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            yaw,
            pitch,
            fov: std::f32::consts::PI / 3.0,
            near: 0.1,
            far: 1000.0,
        };
        camera.update_vectors();
        camera
    }

    pub fn update_vectors(&mut self) {
        let forward = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize();

        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = forward.cross(world_up).normalize();
        let up = right.cross(forward).normalize();

        self.forward = forward;
        self.right = if right.length_squared() < 1e-6 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            right
        };
        self.up = up;
    }

    pub fn world_to_view(&self, point: Vec3) -> Vec3 {
        let rel = point - self.position;
        Vec3::new(rel.dot(self.right), rel.dot(self.up), rel.dot(self.forward))
    }

    pub fn view_rotation(&self) -> [[f32; 3]; 3] {
        [
            [self.right.x, self.right.y, self.right.z],
            [self.up.x, self.up.y, self.up.z],
            [self.forward.x, self.forward.y, self.forward.z],
        ]
    }

    pub fn focal_lengths(&self, width: usize, height: usize) -> (f32, f32) {
        let h = height.max(1) as f32;
        let w = width.max(1) as f32;
        let tan_half = (self.fov * 0.5).tan().max(1e-6);
        let fy = h / (2.0 * tan_half);
        let fx = fy * (w / h);
        (fx, fy)
    }
}

pub fn reset(camera: &mut Camera, start: Vec3, target: Vec3) {
    *camera = Camera::new(start, -std::f32::consts::FRAC_PI_2, 0.0);
    look_at_target(camera, target);
}

pub fn move_forward(camera: &mut Camera, distance: f32) {
    camera.position += camera.forward * distance;
}

pub fn move_right(camera: &mut Camera, distance: f32) {
    camera.position += camera.right * distance;
}

pub fn move_up(camera: &mut Camera, distance: f32) {
    let world_up = crate::math::Vec3::new(0.0, 1.0, 0.0);
    camera.position += world_up * distance;
}

pub fn adjust_pitch(camera: &mut Camera, delta: f32) {
    camera.pitch = (camera.pitch + delta).clamp(-1.5, 1.5);
    camera.update_vectors();
}

pub fn adjust_yaw(camera: &mut Camera, delta: f32) {
    camera.yaw += delta;
    camera.update_vectors();
}

pub fn look_at_target(camera: &mut Camera, target: Vec3) {
    let to_target = (target - camera.position).normalize();
    if to_target.length_squared() < 1e-8 {
        return;
    }
    camera.yaw = to_target.z.atan2(to_target.x);
    camera.pitch = to_target.y.clamp(-1.0, 1.0).asin();
    camera.update_vectors();
}

#[allow(dead_code)]
pub fn look_at_origin(camera: &mut Camera) {
    look_at_target(camera, Vec3::ZERO);
}
