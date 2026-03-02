use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::splat::Splat;

pub fn save_ply(splats: &[Splat], path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    write!(
        writer,
        "ply\n\
format binary_little_endian 1.0\n\
element vertex {}\n\
property float x\n\
property float y\n\
property float z\n\
property float f_dc_0\n\
property float f_dc_1\n\
property float f_dc_2\n\
property float opacity\n\
property float scale_0\n\
property float scale_1\n\
property float scale_2\n\
property float rot_0\n\
property float rot_1\n\
property float rot_2\n\
property float rot_3\n\
end_header\n",
        splats.len()
    )?;

    for splat in splats {
        let f_dc_0 = inverse_sigmoid((splat.color[0] as f32 / 255.0).clamp(1e-6, 1.0 - 1e-6));
        let f_dc_1 = inverse_sigmoid((splat.color[1] as f32 / 255.0).clamp(1e-6, 1.0 - 1e-6));
        let f_dc_2 = inverse_sigmoid((splat.color[2] as f32 / 255.0).clamp(1e-6, 1.0 - 1e-6));
        let opacity = inverse_sigmoid(splat.opacity.clamp(1e-6, 1.0 - 1e-6));
        let scale_0 = splat.scale.x.max(1e-7).ln();
        let scale_1 = splat.scale.y.max(1e-7).ln();
        let scale_2 = splat.scale.z.max(1e-7).ln();

        write_f32(&mut writer, splat.position.x)?;
        write_f32(&mut writer, splat.position.y)?;
        write_f32(&mut writer, splat.position.z)?;
        write_f32(&mut writer, f_dc_0)?;
        write_f32(&mut writer, f_dc_1)?;
        write_f32(&mut writer, f_dc_2)?;
        write_f32(&mut writer, opacity)?;
        write_f32(&mut writer, scale_0)?;
        write_f32(&mut writer, scale_1)?;
        write_f32(&mut writer, scale_2)?;
        write_f32(&mut writer, splat.rotation[0])?;
        write_f32(&mut writer, splat.rotation[1])?;
        write_f32(&mut writer, splat.rotation[2])?;
        write_f32(&mut writer, splat.rotation[3])?;
    }

    writer.flush()?;
    Ok(())
}

fn inverse_sigmoid(v: f32) -> f32 {
    (v / (1.0 - v)).ln()
}

fn write_f32(writer: &mut BufWriter<File>, value: f32) -> Result<(), std::io::Error> {
    writer.write_all(&value.to_le_bytes())
}
