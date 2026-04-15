use std::fs;
fn main() {
    let mut s = fs::read_to_string("crates/cellm-model/src/qwen.rs").unwrap();
    s = s.replace("\"qwen.input_layernorm\"", "&format!(\"qwen.input_layernorm.{layer}\")");
    s = s.replace("\"qwen.post_attention_layernorm\"", "&format!(\"qwen.post_attention_layernorm.{layer}\")");
    s = s.replace("ops.rms_norm_f16w(&q[start..end], &qw,", "ops.rms_norm_f16w(&q[start..end], &qw[start..end],");
    s = s.replace("&format!(\"qwen.q_norm.h{h}\")", "&format!(\"qwen.q_norm.l{layer}.h{h}\")");
    s = s.replace("ops.rms_norm_f16w(&k[start..end], &kw,", "ops.rms_norm_f16w(&k[start..end], &kw[start..end],");
    s = s.replace("&format!(\"qwen.k_norm.h{h}\")", "&format!(\"qwen.k_norm.l{layer}.h{h}\")");
    fs::write("crates/cellm-model/src/qwen.rs", s).unwrap();
}
