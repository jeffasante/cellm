use std::fs;
fn main() {
    let mut content = fs::read_to_string("crates/cellm-model/src/qwen.rs").unwrap();
    content = content.replace("if self.metal_ops.is_some() {", "if false && self.metal_ops.is_some() {");
    fs::write("crates/cellm-model/src/qwen.rs", content).unwrap();
}
