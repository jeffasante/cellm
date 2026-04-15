import re
with open('crates/cellm-model/src/qwen.rs', 'r') as f: content = f.read()

# Fix attention_single_token_gqa_from_bases
# The call looks like: cr.attention_single_token_gqa_from_bases(&gather_bases, &q, n_heads, n_kv_heads, head_dim, &mut attn_out)
# We need: cr.attention_single_token_gqa_from_bases(&gather_bases, &q, n_heads, n_kv_heads, head_dim, None, &mut attn_out)?
content = content.replace(
    'cr.attention_single_token_gqa_from_bases(&gather_bases, &q, n_heads, n_kv_heads, head_dim, &mut attn_out)',
    'cr.attention_single_token_gqa_from_bases(&gather_bases, &q, n_heads, n_kv_heads, head_dim, None, &mut attn_out)?'
)

# Fix rms_norm_f16w
# Find all calls to rms_norm_f16w and add a dummy key if missing
def fix_rms(m):
    args = m.group(1).split(',')
    if len(args) == 5:
        # Add a placeholder key before the last arg (out)
        args.insert(4, ' "qwen.norm"')
        return 'rms_norm_f16w(' + ','.join(args) + ')?'
    return m.group(0)

content = re.sub(r'rms_norm_f16w\((.*?)\)', fix_rms, content)

with open('crates/cellm-model/src/qwen.rs', 'w') as f: f.write(content)
