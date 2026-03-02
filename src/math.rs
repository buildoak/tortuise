use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len <= 1e-8 {
            Self::ZERO
        } else {
            self * (1.0 / len)
        }
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Vec3) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub fn clamp_u8(v: f32) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

pub fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len <= 1e-8 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    }
}

pub fn quat_to_rotation_matrix(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = quat_normalize(q);
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]
}

pub fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

pub fn mat3_transpose(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

pub fn hsv_to_rgb(h_deg: f32, s: f32, v: f32) -> [u8; 3] {
    let h = (h_deg.rem_euclid(360.0)) / 60.0;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r1, g1, b1) = if h < 1.0 {
        (c, x, 0.0)
    } else if h < 2.0 {
        (x, c, 0.0)
    } else if h < 3.0 {
        (0.0, c, x)
    } else if h < 4.0 {
        (0.0, x, c)
    } else if h < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let m = v - c;
    [
        clamp_u8((r1 + m) * 255.0),
        clamp_u8((g1 + m) * 255.0),
        clamp_u8((b1 + m) * 255.0),
    ]
}

pub type Mat4 = [[f32; 4]; 4];

pub const MAT4_IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = a[r][0] * b[0][c]
                + a[r][1] * b[1][c]
                + a[r][2] * b[2][c]
                + a[r][3] * b[3][c];
        }
    }
    out
}

pub fn mat4_inverse(m: Mat4) -> Option<Mat4> {
    let mut aug = [[0.0f32; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            aug[r][c] = m[r][c];
        }
        aug[r][4 + r] = 1.0;
    }

    for col in 0..4 {
        let mut pivot_row = col;
        let mut pivot_abs = aug[col][col].abs();
        for (r, row) in aug.iter().enumerate().skip(col + 1) {
            let candidate = row[col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = r;
            }
        }
        if pivot_abs <= 1e-8 {
            return None;
        }

        if pivot_row != col {
            aug.swap(col, pivot_row);
        }

        let pivot = aug[col][col];
        for value in &mut aug[col] {
            *value /= pivot;
        }

        let pivot_values = aug[col];
        for (r, row) in aug.iter_mut().enumerate() {
            if r == col {
                continue;
            }
            let factor = row[col];
            if factor.abs() <= 1e-12 {
                continue;
            }
            for (target, pivot_value) in row.iter_mut().zip(pivot_values.iter()) {
                *target -= factor * *pivot_value;
            }
        }
    }

    let mut inv = [[0.0f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            inv[r][c] = aug[r][4 + c];
        }
    }
    Some(inv)
}

pub fn symmetric_eigen3(m: [[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3]) {
    let mut a = m;
    let mut v = [[0.0f32; 3]; 3];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for _ in 0..10 {
        jacobi_rotate_3x3(&mut a, &mut v, 0, 1);
        jacobi_rotate_3x3(&mut a, &mut v, 0, 2);
        jacobi_rotate_3x3(&mut a, &mut v, 1, 2);

        let off_diag =
            a[0][1].abs() + a[0][2].abs() + a[1][0].abs() + a[1][2].abs() + a[2][0].abs() + a[2][1].abs();
        if off_diag <= 1e-6 {
            break;
        }
    }

    let eigenvalues = [a[0][0], a[1][1], a[2][2]];
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| {
        eigenvalues[j]
            .partial_cmp(&eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_values = [0.0f32; 3];
    let mut sorted_vectors = [[0.0f32; 3]; 3];
    for col in 0..3 {
        sorted_values[col] = eigenvalues[order[col]];
        for row in 0..3 {
            sorted_vectors[row][col] = v[row][order[col]];
        }
    }

    let det = mat3_determinant(sorted_vectors);
    if det < 0.0 {
        for row in &mut sorted_vectors {
            row[2] = -row[2];
        }
    }

    (sorted_vectors, sorted_values)
}

pub fn quaternion_from_rotation_matrix(m: [[f32; 3]; 3]) -> [f32; 4] {
    let w_sq = 1.0 + m[0][0] + m[1][1] + m[2][2];
    let x_sq = 1.0 + m[0][0] - m[1][1] - m[2][2];
    let y_sq = 1.0 - m[0][0] + m[1][1] - m[2][2];
    let z_sq = 1.0 - m[0][0] - m[1][1] + m[2][2];

    let mut q = [0.0f32; 4];
    let mut max_index = 0usize;
    let mut max_value = w_sq;
    if x_sq > max_value {
        max_value = x_sq;
        max_index = 1;
    }
    if y_sq > max_value {
        max_value = y_sq;
        max_index = 2;
    }
    if z_sq > max_value {
        max_index = 3;
    }

    match max_index {
        0 => {
            let w = 0.5 * w_sq.max(0.0).sqrt();
            let denom = 4.0 * w.max(1e-8);
            q[0] = w;
            q[1] = (m[2][1] - m[1][2]) / denom;
            q[2] = (m[0][2] - m[2][0]) / denom;
            q[3] = (m[1][0] - m[0][1]) / denom;
        }
        1 => {
            let x = 0.5 * x_sq.max(0.0).sqrt();
            let denom = 4.0 * x.max(1e-8);
            q[0] = (m[2][1] - m[1][2]) / denom;
            q[1] = x;
            q[2] = (m[0][1] + m[1][0]) / denom;
            q[3] = (m[0][2] + m[2][0]) / denom;
        }
        2 => {
            let y = 0.5 * y_sq.max(0.0).sqrt();
            let denom = 4.0 * y.max(1e-8);
            q[0] = (m[0][2] - m[2][0]) / denom;
            q[1] = (m[0][1] + m[1][0]) / denom;
            q[2] = y;
            q[3] = (m[1][2] + m[2][1]) / denom;
        }
        _ => {
            let z = 0.5 * z_sq.max(0.0).sqrt();
            let denom = 4.0 * z.max(1e-8);
            q[0] = (m[1][0] - m[0][1]) / denom;
            q[1] = (m[0][2] + m[2][0]) / denom;
            q[2] = (m[1][2] + m[2][1]) / denom;
            q[3] = z;
        }
    }

    quat_normalize(q)
}

fn jacobi_rotate_3x3(a: &mut [[f32; 3]; 3], v: &mut [[f32; 3]; 3], p: usize, q: usize) {
    let apq = a[p][q];
    if apq.abs() <= 1e-10 {
        return;
    }

    let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;

    let app = a[p][p];
    let aqq = a[q][q];

    a[p][p] = app - t * apq;
    a[q][q] = aqq + t * apq;
    a[p][q] = 0.0;
    a[q][p] = 0.0;

    for r in [0usize, 1, 2] {
        if r == p || r == q {
            continue;
        }
        let arp = a[r][p];
        let arq = a[r][q];
        let new_rp = c * arp - s * arq;
        let new_rq = s * arp + c * arq;
        a[r][p] = new_rp;
        a[p][r] = new_rp;
        a[r][q] = new_rq;
        a[q][r] = new_rq;
    }

    for row in v.iter_mut() {
        let vrp = row[p];
        let vrq = row[q];
        row[p] = c * vrp - s * vrq;
        row[q] = s * vrp + c * vrq;
    }
}

fn mat3_determinant(m: [[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}
