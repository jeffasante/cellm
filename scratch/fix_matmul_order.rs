use std::fs;
fn main() {
    for path in &["crates/cellm-model/src/qwen.rs", "crates/cellm-model/src/gemma.rs"] {
        let mut content = fs::read_to_string(path).unwrap();
        
        // Find and fix the matmul calls
        // We are looking for:
        // .matmul_row_major_f32(
        //     x,
        //     [arg2],
        //     [arg3],
        //     &weight_t_chunk...,
        //     [arg5],
        //     out_slice,
        // )
        // And we want (x, 1, in_dim, weight, cols_n, out)
        
        // Gemma usually has: (x, 1, in_dim, weight, cols_n, out)
        // Qwen had something else.
        
        // I'll just use a very specific replace for the qwen blocks I saw
        content = content.replace(
            ".matmul_row_major_f32(\n                                x,\n                                &weight_t_chunk[..in_dim * cols_n],\n                                1,\n                                cols_n,\n                                in_dim,\n                                out_slice,\n                            )",
            ".matmul_row_major_f32(\n                                x,\n                                1,\n                                in_dim,\n                                &weight_t_chunk[..in_dim * cols_n],\n                                cols_n,\n                                out_slice,\n                            )"
        );

        fs::write(path, content).unwrap();
    }
}
