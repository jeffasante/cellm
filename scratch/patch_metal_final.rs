use std::fs;
fn main() {
    let mut s = fs::read_to_string("crates/cellm-kernels/src/metal.rs").unwrap();
    // Signature change
    s = s.replace(
        "pub fn rms_norm_f16w(\n        &self,\n        x: &[f32],\n        w_f16: &[u16],\n        eps: f32,\n        w_add_one: bool,\n        out: &mut [f32],\n    ) -> anyhow::Result<()> {",
        "pub fn rms_norm_f16w(\n        &self,\n        x: &[f32],\n        w_f16: &[u16],\n        eps: f32,\n        w_add_one: bool,\n        cache_key: &str,\n        out: &mut [f32],\n    ) -> anyhow::Result<()> {"
    );
    // Cache key usage
    s = s.replace(
        "let wkey = \"rmsnorm.w\";",
        "let wkey = format!(\"rmsnorm.w.{}\", cache_key);"
    );
     s = s.replace(
        "let skey = \"rmsnorm.s\";", // wait, there is no skey for rmsnorm usually
        "// No skey for rmsnorm"
    );
    // Wait, let me check the actual code in metal.rs
    fs::write("crates/cellm-kernels/src/metal.rs", s).unwrap();
}
