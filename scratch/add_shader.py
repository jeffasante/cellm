import os
path = 'crates/cellm-kernels/src/metal.rs'
with open(path, 'r') as f: content = f.read()

shader_code = """
kernel void mv2_i8(
    device const char* w0 [[buffer(0)]],
    device const char* w1 [[buffer(1)]],
    device const half* s0 [[buffer(2)]],
    device const half* s1 [[buffer(3)]],
    device const float* x [[buffer(4)]],
    device float* o0 [[buffer(5)]],
    device float* o1 [[buffer(6)]],
    constant uint& r0 [[buffer(7)]],
    constant uint& r1 [[buffer(8)]],
    constant uint& c [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= r0 + r1) return;
    bool is_r1 = (gid >= r0);
    uint row = is_r1 ? (gid - r0) : gid;
    device const char* w = is_r1 ? w1 : w0;
    device const half* s = is_r1 ? s1 : s0;
    device float* o = is_r1 ? o1 : o0;
    
    float acc = 0.0f;
    float row_scale = float(s[row]);
    device const char* w_row = w + row * c;
    for (uint i = 0; i < c; i++) {
        acc += float(w_row[i]) * x[i];
    }
    o[row] = acc * row_scale;
}
"""

if 'kernel void mv2_i8' not in content:
    content = content.replace('// Shader source embedded\nconst ELEM_OPS_SHADER: &str = r#"', '// Shader source embedded\nconst ELEM_OPS_SHADER: &str = r#"' + shader_code)

with open(path, 'w') as f: f.write(content)
