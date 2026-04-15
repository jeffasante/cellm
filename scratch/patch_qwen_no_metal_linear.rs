use std::fs;
fn main() {
    let mut s = fs::read_to_string("crates/cellm-model/src/qwen.rs").unwrap();
    s = s.replace("if self.metal_ops.is_some() {", "if false { // self.metal_ops.is_some() {");
    fs::write("crates/cellm-model/src/qwen.rs", s).unwrap();
}
