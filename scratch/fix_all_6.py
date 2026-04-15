import os

def fix_llama():
    path = 'crates/cellm-model/src/llama.rs'
    with open(path, 'r') as f: content = f.read()
    # Fix import
    content = content.replace('rope_inplace_f32', 'rope_interleaved_inplace_f32')
    # Fix attention call
    content = content.replace(
        'cr.attention_single_token_gqa_from_bases(\n                &gather_bases,\n                &q,\n                n_heads,\n                n_kv_heads,\n                head_dim,\n                &mut attn_out,\n            )?',
        'cr.attention_single_token_gqa_from_bases(\n                &gather_bases,\n                &q,\n                n_heads,\n                n_kv_heads,\n                head_dim,\n                None,\n                &mut attn_out,\n            )?'
    )
    # Fix rope calls (interleaved)
    content = content.replace(
        'rope_interleaved_inplace_f32(&mut q, n_heads, head_dim, pos, cfg.rope_theta);',
        'rope_interleaved_inplace_f32(&mut q, n_heads, head_dim, pos, cfg.rope_theta);' # already ok
    )
    # Fix rope calls (non-interleaved)
    content = content.replace(
        'rope_non_interleaved_inplace_f32(&mut q, n_heads, head_dim, pos, cfg.rope_theta);',
        'rope_non_interleaved_inplace_f32(&mut q, n_heads, head_dim, head_dim, pos, cfg.rope_theta);'
    )
    content = content.replace(
        'rope_non_interleaved_inplace_f32(&mut k, n_kv_heads, head_dim, pos, cfg.rope_theta);',
        'rope_non_interleaved_inplace_f32(&mut k, n_kv_heads, head_dim, head_dim, pos, cfg.rope_theta);'
    )
    with open(path, 'w') as f: f.write(content)

def fix_gemma():
    path = 'crates/cellm-model/src/gemma.rs'
    with open(path, 'r') as f: content = f.read()
    content = content.replace(
        'cr.attention_single_token_gqa_from_bases(\n                &gather_bases,\n                q_for_attn,\n                n_heads,\n                n_kv_heads,\n                head_dim,\n                attn_out_slice,\n            )?',
        'cr.attention_single_token_gqa_from_bases(\n                &gather_bases,\n                q_for_attn,\n                n_heads,\n                n_kv_heads,\n                head_dim,\n                None,\n                attn_out_slice,\n            )?'
    )
    # Fix the Python script's broken output if it happened
    content = content.replace(', attn_out_slice, None, attn_out_slice)?', ', None, attn_out_slice)?')
    with open(path, 'w') as f: f.write(content)

def fix_qwen():
    path = 'crates/cellm-model/src/qwen.rs'
    with open(path, 'r') as f: content = f.read()
    # The last python script might have failed multi-line replace.
    # Manual precise find/replace for qwen.
    content = content.replace(
        'cr.attention_single_token_gqa_from_bases(\n                    &q,\n                    &k_bases,\n                    &v_bases,\n                    cfg.rope_theta,\n                    head_dim,\n                    None, // soft_cap\n                    &mut attn_out,\n                )?;',
        'cr.attention_single_token_gqa_from_bases(&gather_bases, &q, n_heads, n_kv_heads, head_dim, None, &mut attn_out)?;'
    )
    with open(path, 'w') as f: f.write(content)

fix_llama()
fix_gemma()
fix_qwen()
