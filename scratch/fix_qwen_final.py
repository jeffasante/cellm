import re
with open('crates/cellm-model/src/qwen.rs', 'r') as f: content = f.read()

# 1. &mut self -> &self for metal_ops
content = content.replace('metal_ops.as_mut()', 'metal_ops.as_ref()')

# 2. Fix rms_norm_f16w (add key, results in map_err)
# We want to change:
# .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_norm)
# .map_err(|e| CoreError::Backend(e.to_string()))?;
# TO:
# .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, "qwen.norm", &mut x_norm)
# .map_err(|e| CoreError::Backend(e.to_string()))?;

content = content.replace(
    '.rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_norm)',
    '.rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, "qwen.norm", &mut x_norm)'
)
content = content.replace(
    '.rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut mlp_in)',
    '.rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, "qwen.mlp_norm", &mut mlp_in)'
)
content = content.replace(
    '.rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_final)',
    '.rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, "qwen.final_norm", &mut x_final)'
)

# Fix window attention calls where seg is used
content = content.replace(
    'ops.rms_norm_f16w(&inp, &qw, cfg.rms_norm_eps, add_one, seg)',
    'ops.rms_norm_f16w(&inp, &qw, cfg.rms_norm_eps, add_one, "qwen.win_q", seg)'
)
content = content.replace(
    'ops.rms_norm_f16w(&inp, &kw, cfg.rms_norm_eps, add_one, seg)',
    'ops.rms_norm_f16w(&inp, &kw, cfg.rms_norm_eps, add_one, "qwen.win_k", seg)'
)

# 3. Fix attention_single_token_gqa_from_bases
# Add None as soft_cap and Result handling
content = content.replace(
    'cr.attention_single_token_gqa_from_bases(&gather_bases, &q, n_heads, n_kv_heads, head_dim, &mut attn_out)',
    'cr.attention_single_token_gqa_from_bases(&gather_bases, &q, n_heads, n_kv_heads, head_dim, None, &mut attn_out)?'
)

with open('crates/cellm-model/src/qwen.rs', 'w') as f: f.write(content)
