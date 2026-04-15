import re
import os

def fix_file(path, model_name):
    with open(path, 'r') as f: content = f.read()
    
    # 1. &mut self -> &self
    content = content.replace('metal_ops.as_mut()', 'metal_ops.as_ref()')
    content = content.replace('ops.as_mut()', 'ops.as_ref()')
    
    # 2. Fix attention_single_token_gqa_from_bases (add None, ?)
    # Find calls and inject None before last arg
    parts = re.split(r'(cr\.attention_single_token_gqa_from_bases\(.*?,.*?,.*?,.*?,.*?,)(.*?\))', content, flags=re.DOTALL)
    # Wait, simple replace is safer if we know the blocks
    content = content.replace(
        'cr.attention_single_token_gqa_from_bases(&gather_bases, &q, n_heads, n_kv_heads, head_dim, &mut attn_out)',
        'cr.attention_single_token_gqa_from_bases(&gather_bases, &q, n_heads, n_kv_heads, head_dim, None, &mut attn_out)?'
    )
    content = content.replace(
        'cr.attention_single_token_gqa_from_bases(&gather_bases, q_for_attn, n_heads, n_kv_heads, head_dim, attn_out_slice)',
        'cr.attention_single_token_gqa_from_bases(&gather_bases, q_for_attn, n_heads, n_kv_heads, head_dim, None, attn_out_slice)?'
    )

    # 3. Fix rms_norm_f16w (add key)
    # We want to insert a string before the last &mut arg.
    def repl_rms(m):
        args = m.group(1).split(',')
        if len(args) == 5:
            # Add key
            args.insert(4, f' "{model_name}.norm"')
            return f'rms_norm_f16w({",".join(args)})'
        return m.group(0)
    
    content = re.sub(r'rms_norm_f16w\((.*?)\)', repl_rms, content)
    
    with open(path, 'w') as f: f.write(content)

fix_file('crates/cellm-model/src/qwen.rs', 'qwen')
fix_file('crates/cellm-model/src/gemma.rs', 'gemma')
fix_file('crates/cellm-model/src/llama.rs', 'llama')

# Fix llama_graph.rs
with open('crates/cellm-model/src/llama_graph.rs', 'r') as f: c = f.read()
c = c.replace('head_dim as u32)', 'head_dim as u32, None)')
with open('crates/cellm-model/src/llama_graph.rs', 'w') as f: f.write(c)
