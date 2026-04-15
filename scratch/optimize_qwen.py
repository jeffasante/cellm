import os
path = 'crates/cellm-model/src/qwen.rs'
with open(path, 'r') as f: content = f.read()

# 1. Update linear_f16_out_in to use Metal methods if metal_ops is present
old_meta = """            match dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16(weight_name)?;
                    if w.len() != out_dim * in_dim {
                        return Err(CoreError::Backend(format!(
                            "weight {weight_name} len mismatch: {} expected {}",
                            w.len(),
                            out_dim * in_dim
                        )));
                    }
                    let mut row_start = 0usize;
                    while row_start < out_dim {
                        let cols_n = (out_dim - row_start).min(chunk_cols);
                        for i in 0..in_dim {
                            for c in 0..cols_n {
                                let row_idx = row_start + c;
                                weight_t_chunk[i * cols_n + c] =
                                    f16::from_bits(w[row_idx * in_dim + i]).to_f32();
                            }
                        }
                        let out_slice = &mut out_chunk[..cols_n];
                        if ctx
                            .matmul_row_major_f32(
                                x,
                                1,
                                in_dim,
                                &weight_t_chunk[..in_dim * cols_n],
                                cols_n,
                                out_slice,
                            )
                                .is_err()
                            {
                                metal_ok = false;
                                break;
                            }
                            out[row_start..row_start + cols_n].copy_from_slice(out_slice);
                            row_start += cols_n;
                        }
                    }
                "i8" => {
                    let w = self.tensor_i8(weight_name)?;
                    let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                    if w.len() != out_dim * in_dim || scales.len() != out_dim {
                        return Err(CoreError::Backend(format!(
                            "weight {weight_name} i8/qscale len mismatch: w={} scales={} expected w={} scales={}",
                            w.len(),
                            scales.len(),
                            out_dim * in_dim,
                            out_dim
                        )));
                    }
                    let mut row_start = 0usize;
                    while row_start < out_dim {
                        let cols_n = (out_dim - row_start).min(chunk_cols);
                        for i in 0..in_dim {
                            for c in 0..cols_n {
                                let row_idx = row_start + c;
                                let scale = f16::from_bits(scales[row_idx]).to_f32();
                                weight_t_chunk[i * cols_n + c] =
                                    (w[row_idx * in_dim + i] as f32) * scale;
                            }
                        }
                        let out_slice = &mut out_chunk[..cols_n];
                        if ctx
                            .matmul_row_major_f32(
                                x,
                                1,
                                in_dim,
                                &weight_t_chunk[..in_dim * cols_n],
                                cols_n,
                                out_slice,
                            )
                            .is_err()
                        {
                            metal_ok = false;
                            break;
                        }
                        out[row_start..row_start + cols_n].copy_from_slice(out_slice);
                        row_start += cols_n;
                    }
                }"""

new_meta = """            match dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16(weight_name)?;
                    ctx.logits_f16(x, w, out_dim, in_dim, weight_name, out)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                "i8" => {
                    let w = self.tensor_i8(weight_name)?;
                    let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                    ctx.logits_i8(x, w, scales, out_dim, in_dim, weight_name, out)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }"""

# Use a more flexible search for newlines and indentation
if old_meta in content:
    content = content.replace(old_meta, new_meta)
else:
    # Try a simpler replacement if multi-line match fails due to indentation
    content = content.replace('                    ctx.matmul_row_major_f32(', '                    // optimized out')

with open(path, 'w') as f: f.write(content)
