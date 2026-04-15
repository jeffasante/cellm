use std::fs;
fn main() {
    let mut s = fs::read_to_string("crates/cellm-model/src/qwen.rs").unwrap();
    s = s.replace(
        "l2norm_value(&x),\n                l2norm_value(&x_final)",
        "l2norm_value(&x), l2norm_value(&x_final)); eprintln!(\"[qwen-debug] x[0..5]={:?} x_final[0..5]={:?}\", &x[0..5], &x_final[0..5]"
    );
    fs::write("crates/cellm-model/src/qwen.rs", s).unwrap();
}
