use std::fs;
fn main() {
    let mut s = fs::read_to_string("crates/cellm-model/src/qwen.rs").unwrap();
    
    // Add rmsnorm_weight_f16 helper
    if !s.contains("fn rmsnorm_weight_f16") {
        s = s.replace(
            "fn tensor_f16(&self, name: &str) -> Result<&[u16], CoreError> {",
            "fn rmsnorm_weight_f16(&self, name: &str) -> Result<Vec<u16>, CoreError> {
                let shape = self.tensor_shape(name)?;
                let mut f32_out = vec![0.0f32; shape.iter().product()];
                self.rmsnorm_weight(name, &mut f32_out)?;
                Ok(f32_out.into_iter().map(|v| half::f16::from_f32(v).to_bits()).collect())
            }

            fn tensor_f16(&self, name: &str) -> Result<&[u16], CoreError> {"
        );
    }

    // Fix rms_norm_f16w calls with unique cache keys and correct argument count
    // Use regex-like manual replace for robustness
    s = s.replace(
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_norm)",
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &format!(\"qwen.input_layernorm.{layer}\"), &mut x_norm)"
    );
     s = s.replace(
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut mlp_in)",
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &format!(\"qwen.post_attention_layernorm.{layer}\"), &mut mlp_in)"
    );
     s = s.replace(
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_final)",
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, \"qwen.final_norm\", &mut x_final)"
    );

    // Fix cr.attention... call
    s = s.replace(
        "head_dim,\n                    &mut attn_out,",
        "head_dim,\n                    None,\n                    &mut attn_out,"
    );

    // Add back relevant debug prints
    if !s.contains("[qwen-debug] layer={} x_norm") {
        s = s.replace(
            "let layer_post_mixer_norm = l2norm_value(&x);",
            "let layer_post_mixer_norm = l2norm_value(&x);\n            if self.debug_pos == Some(pos) { eprintln!(\"[qwen-debug] layer={} x_norm={:.6}\", layer, l2norm_value(&x)); }"
        );
    }

    fs::write("crates/cellm-model/src/qwen.rs", s).unwrap();
}
