use std::fs;
fn main() {
    let mut content = fs::read_to_string("crates/cellm-model/src/qwen.rs").unwrap();
    
    // Insert parity check after projection and RoPE
    content = content.replace(
        "self.linear_f16_out_in(&x_norm, &q_name, q_raw.len(), hidden, &mut q_raw)?;",
        "self.linear_f16_out_in(&x_norm, &q_name, q_raw.len(), hidden, &mut q_raw)?;\n            let q_metal = q_raw.clone();\n            // run CPU fallback to compare\n            let mut q_cpu = vec![0.0f32; q_raw.len()];\n            self.linear_f16_out_in_cpu(&x_norm, &q_name, q_raw.len(), hidden, &mut q_cpu).unwrap();\n            let mut diff = 0.0f32;\n            for i in 0..q_metal.len() { diff += (q_metal[i] - q_cpu[i]).abs(); }\n            eprintln!(\"[parity-debug] pos={} layer={} q_diff={:.6}\", pos, layer, diff);"
    );

    // We need to add linear_f16_out_in_cpu for comparison
    content = content.replace(
        "    fn linear_f16_out_in(",
        "    fn linear_f16_out_in_cpu(&self, x: &[f32], weight_name: &str, out_dim: usize, in_dim: usize, out: &mut [f32]) -> Result<(), CoreError> {\n        let dtype = self.tensor_dtype(weight_name)?;\n        match dtype.as_str() {\n            \"i8\" => {\n                let w = self.tensor_i8(weight_name)?;\n                let s = self.tensor_f16(&format!(\"{weight_name}.qscale\"))?;\n                for j in 0..out_dim {\n                    let row = &w[j * in_dim..(j + 1) * in_dim];\n                    let scale = f16::from_bits(s[j]).to_f32();\n                    let mut acc = 0.0f32;\n                    for i in 0..in_dim { acc += x[i] * (row[i] as f32) * scale; }\n                    out[j] = acc;\n                }\n                Ok(())\n            }\n            _ => self.linear_f16_out_in(x, weight_name, out_dim, in_dim, out)\n        }\n    }\n    fn linear_f16_out_in("
    );
    
    fs::write("crates/cellm-model/src/qwen.rs", content).unwrap();
}
