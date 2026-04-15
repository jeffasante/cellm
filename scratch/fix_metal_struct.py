import os
path = 'crates/cellm-kernels/src/metal.rs'
with open(path, 'r') as f: lines = f.readlines()
new_lines = []
for line in lines:
    if 'pub pso_mv2_f16: ComputePipelineState,' in line:
        new_lines.append(line)
        new_lines.append('    pub pso_mv2_i8: ComputePipelineState,\n')
        continue
    if 'pso_mv2_f16: pso("mv2_f16")?,' in line:
        new_lines.append(line)
        new_lines.append('            pso_mv2_i8: pso("mv2_i8")?,\n')
        continue
    # Standard create call too
    if 'let pso_mv2_f16   = build_pso_ops(&device, &lib, "mv2_f16")?;' in line:
        new_lines.append(line)
        new_lines.append('        let pso_mv2_i8    = build_pso_ops(&device, &lib, "mv2_i8")?;\n')
        continue
    if 'pso_mv2_f16,' in line and 'Ok(Self {' in lines[new_lines.__len__()-1 if new_lines.__len__()>0 else 0]:
         new_lines.append(line.replace('pso_mv2_f16,', 'pso_mv2_f16, pso_mv2_i8,'))
         continue
    new_lines.append(line)
with open(path, 'w') as f: f.writelines(new_lines)
