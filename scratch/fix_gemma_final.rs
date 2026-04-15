use std::fs;
fn main() {
    let mut content = fs::read_to_string("crates/cellm-model/src/gemma.rs").unwrap();
    
    // Fix rms_norm_f16w to 6 args in gemma.rs
    content = content.replace(
        ".rms_norm_f16w(&attn_proj, &w, cfg.rms_norm_eps, add_one, &mut x_out)",
        ".rms_norm_f16w(&attn_proj, &w, cfg.rms_norm_eps, add_one, \"attn_norm\", &mut x_out)"
    );
    content = content.replace(
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut mlp_in)",
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, \"mlp_norm\", &mut mlp_in)"
    );
    content = content.replace(
        ".rms_norm_f16w(&down, &wffn, cfg.rms_norm_eps, add_one, &mut x_out)",
        ".rms_norm_f16w(&down, &wffn, cfg.rms_norm_eps, add_one, \"down_norm\", &mut x_out)"
    );
    content = content.replace(
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_final)",
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, \"final_norm\", &mut x_final)"
    );

    // Fix attention_single_token_gqa_from_bases
    content = content.replace(
        "cr.attention_single_token_gqa_from_bases(\n                &gather_bases,\n                &q_slice,\n                n_heads,\n                n_kv_heads,\n                head_dim,\n                attn_out_slice,",
        "cr.attention_single_token_gqa_from_bases(\n                &gather_bases,\n                &q_slice,\n                n_heads,\n                n_kv_heads,\n                head_dim,\n                None,\n                attn_out_slice,"
    );

    fs::write("crates/cellm-model/src/gemma.rs", content).unwrap();
}
