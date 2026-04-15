use std::fs;
fn main() {
    for path in &["crates/cellm-model/src/qwen.rs", "crates/cellm-model/src/gemma.rs"] {
        let mut content = fs::read_to_string(path).unwrap();
        
        // Fix rms_norm_f16w
        // Match: .rms_norm_f16w(arg1, arg2, arg3, arg4, arg5) where arg5 is &mut...
        // This is hard with regex in simple rust, I'll use simple search/replace for known patterns
        
        content = content.replace(
            ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_norm)",
            ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, \"pre_norm\", &mut x_norm)"
        );
        content = content.replace(
            ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut mlp_in)",
            ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, \"post_norm\", &mut mlp_in)"
        );
        content = content.replace(
            ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_final)",
            ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, \"final_norm\", &mut x_final)"
        );

        // q_head_norm
        content = content.replace(
            "ops.rms_norm_f16w(&inp, &qw, cfg.rms_norm_eps, add_one, seg)",
            "ops.rms_norm_f16w(&inp, &qw, cfg.rms_norm_eps, add_one, \"q_head_norm\", seg)"
        );
        content = content.replace(
            "ops.rms_norm_f16w(&inp, &kw, cfg.rms_norm_eps, add_one, seg)",
            "ops.rms_norm_f16w(&inp, &kw, cfg.rms_norm_eps, add_one, \"k_head_norm\", seg)"
        );

        // Fix attention calls
        content = content.replace(
            "cr.attention_single_token_gqa_from_bases(\n                    &gather_bases,\n                    &q,\n                    n_heads,\n                    n_kv_heads,\n                    head_dim,\n                    &mut attn_out,",
            "cr.attention_single_token_gqa_from_bases(\n                    &gather_bases,\n                    &q,\n                    n_heads,\n                    n_kv_heads,\n                    head_dim,\n                    None,\n                    &mut attn_out,"
        );
        content = content.replace(
            "cr.attention_single_token_gqa_from_bases(\n                &gather_bases,\n                &q,\n                n_heads,\n                n_kv_heads,\n                head_dim,\n                &mut attn_out,",
            "cr.attention_single_token_gqa_from_bases(\n                &gather_bases,\n                &q,\n                n_heads,\n                n_kv_heads,\n                head_dim,\n                None,\n                &mut attn_out,"
        );
        
        // Gemma specific attention
        content = content.replace(
            "cr.attention_single_token_gqa_from_bases(\n                &gather_bases,\n                &q_slice,\n                n_heads,\n                n_kv_heads,\n                head_dim,\n                attn_out_slice,",
            "cr.attention_single_token_gqa_from_bases(\n                &gather_bases,\n                &q_slice,\n                n_heads,\n                n_kv_heads,\n                head_dim,\n                None,\n                attn_out_slice,"
        );

        fs::write(path, content).unwrap();
    }
}
