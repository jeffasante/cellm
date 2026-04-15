import sys

def apply_patch(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if 'scores[t] = dot * scale;' in line:
            indent = line[:line.find('scores[t]')]
            # Avoid double patching
            if 'let mut s =' in "".join(new_lines[-2:]):
                 new_lines.append(line)
                 continue
            new_lines.append(f"{indent}let mut s = dot * scale;\n")
            new_lines.append(f"{indent}if let Some(cap) = soft_cap {{\n")
            new_lines.append(f"{indent}    s = (s / cap).tanh() * cap;\n")
            new_lines.append(f"{indent}}}\n")
            new_lines.append(f"{indent}scores[t] = s;\n")
        else:
            new_lines.append(line)
            
    with open(path, 'w') as f:
        f.writelines(new_lines)

apply_patch('crates/cellm-cache/src/kvcache.rs')
