use std::fs;
fn main() {
    let mut s = fs::read_to_string("crates/cellm-model/src/qwen.rs").unwrap();
    s = s.replace(
        "let layer_post_mixer_norm = l2norm_value(&x);",
        "let layer_post_mixer_norm = l2norm_value(&x); if self.debug_pos == Some(pos) { eprintln!(\"[qwen-debug] layer={} x_norm={:.6}\", layer, l2norm_value(&x)); }"
    );
    s = s.replace(
        "self.linear_f16_out_in(&gate, &format!(\"language_model.model.layers.{layer}.mlp.down_proj.weight\"), hidden, cfg.intermediate_size, &mut down)?;",
        "self.linear_f16_out_in(&gate, &format!(\"language_model.model.layers.{layer}.mlp.down_proj.weight\"), hidden, cfg.intermediate_size, &mut down)?; if self.debug_pos == Some(pos) { eprintln!(\"[qwen-debug] layer={} gate_norm={:.6} up_norm={:.6} down_norm={:.6}\", layer, l2norm_value(&gate), l2norm_value(&up), l2norm_value(&down)); }"
    );
    fs::write("crates/cellm-model/src/qwen.rs", s).unwrap();
}
