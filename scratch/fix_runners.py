import os

def fix_runner(path):
    with open(path, 'r') as f: lines = f.readlines()
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'cr.attention_single_token_gqa_from_bases(' in line:
            # Need to find the end of the call and replace it.
            # We want to change the signature to (q, k_bases, v_bases, rope_theta, head_dim, soft_cap, out)
            # Find the arguments being passed.
            j = i
            while ')' not in lines[j]: j += 1
            call_block = "".join(lines[i:j+1])
            
            # Extract arguments
            # This is fragile but let's try to detect based on typical pattern
            if '&gather_bases' in call_block:
                indent = line[:line.find('cr')]
                q_arg = 'q_for_attn' if 'q_for_attn' in call_block else '&q'
                if 'qwen.rs' in path or 'llama.rs' in path: q_arg = '&q'
                
                new_block = f"{indent}let k_bases: Vec<_> = gather_bases.iter().map(|b| b.k_base).collect();\n"
                new_block += f"{indent}let v_bases: Vec<_> = gather_bases.iter().map(|b| b.v_base).collect();\n"
                new_block += f"{indent}cr.attention_single_token_gqa_from_bases(\n"
                new_block += f"{indent}    {q_arg},\n"
                new_block += f"{indent}    &k_bases,\n"
                new_block += f"{indent}    &v_bases,\n"
                new_block += f"{indent}    cfg.rope_theta,\n"
                new_block += f"{indent}    head_dim,\n"
                new_block += f"{indent}    None, // soft_cap\n"
                new_block += f"{indent}    attn_out_slice if \"gemma\" in path else &mut attn_out,\n"
                new_block += f"{indent})?;\n" # Add ? because it returns Result now
                
                # Special fix for gemma's out arg
                if 'gemma.rs' in path:
                    new_block = new_block.replace('&mut attn_out', 'attn_out_slice')
                
                new_lines.append(new_block)
                i = j + 1
                continue
        
        # Replace .as_mut().unwrap() with .as_ref().unwrap()
        line = line.replace('metal_ops.as_mut().unwrap()', 'metal_ops.as_ref().unwrap()')
        new_lines.append(line)
        i += 1
        
    with open(path, 'w') as f: f.writelines(new_lines)

for p in ['crates/cellm-model/src/qwen.rs', 'crates/cellm-model/src/gemma.rs', 'crates/cellm-model/src/llama.rs']:
    fix_runner(p)
