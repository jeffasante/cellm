use std::fs;
fn main() {
    let mut content = fs::read_to_string("crates/cellm-model/src/qwen.rs").unwrap();
    
    // Fix rms_norm_f16w calls in qwen.rs
    content = content.replace(
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_norm)",
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, \"pre_norm\", &mut x_norm)"
    );
    content = content.replace(
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut mlp_in)",
        ".rms_norm_f16w(&x, &mut w, cfg.rms_norm_eps, add_one, \"post_norm\", &mut mlp_in)"
    );
    content = content.replace(
        ".rms_norm_f16w(&x, &mut w, cfg.rms_norm_eps, add_one, \"post_norm\", &mut mlp_in)",
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, \"post_norm\", &mut mlp_in)"
    ); // Fix my double-fix
    content = content.replace(
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_final)",
        ".rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, \"final_norm\", &mut x_final)"
    );
    
    // Fix attention_single_token_gqa_from_bases call
    content = content.replace(
        "cr.attention_single_token_gqa_from_bases(\n                    &q,\n                    &k,\n                    &v,\n                    n_heads,\n                    head_dim,\n                    &mut attn_out,",
        "cr.attention_single_token_gqa_from_bases(\n                    &q,\n                    &k,\n                    &v,\n                    n_heads,\n                    head_dim,\n                    None,\n                    &mut attn_out,"
    );

    // Fix matmul_row_major_f32 - need more generic approach or direct replace of the blocks
    // I will use direct replace for the blocks I saw in the error
    content = content.replace(
        ".matmul_row_major_f32(\n                                x,\n                                1,\n                                in_dim,\n                                &weight_t_chunk[..in_dim * cols_n],",
        ".matmul_row_major_f32(\n                                x,\n                                &weight_t_chunk[..in_dim * cols_n],\n                                1,\n                                cols_n,\n                                in_dim,"
    );

    fs::write("crates/cellm-model/src/qwen.rs", content).unwrap();
}
