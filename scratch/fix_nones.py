import os

def fix_file(path):
    with open(path, 'r') as f: content = f.read()
    # Replace the pattern that contains double None
    # We look for head_dim, followed by None, and another None
    content = content.replace('head_dim, None, None,', 'head_dim, None,')
    # Try with single line as well
    content = content.replace('head_dim, None,\n                None,', 'head_dim, None,')
    with open(path, 'w') as f: f.write(content)

fix_file('crates/cellm-model/src/llama.rs')
fix_file('crates/cellm-model/src/gemma.rs')
