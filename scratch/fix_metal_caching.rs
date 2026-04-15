use std::fs;
fn main() {
    let mut s = fs::read_to_string("crates/cellm-kernels/src/metal.rs").unwrap();
    
    // Update rms_norm_f16w signature to include cache_key
    s = s.replace(
        "pub fn rms_norm_f16w(\n        &mut self,\n        x: &[f32],\n        w_f16: &[u16],\n        eps: f32,\n        w_add_one: bool,\n        out: &mut [f32],",
        "pub fn rms_norm_f16w(\n        &mut self,\n        x: &[f32],\n        w_f16: &[u16],\n        eps: f32,\n        w_add_one: bool,\n        cache_key: &str,\n        out: &mut [f32],"
    );

    // Update rms_norm_f16w to use tensor_cache
    s = s.replace(
        "        ensure_buf_u16(&self.device, &mut self.w_buf, n)?;\n\n        let xb = self.x_buf.as_ref().unwrap();\n        let wb = self.w_buf.as_ref().unwrap();\n        let ob = self.out_buf.as_ref().unwrap();\n\n        write_f32(xb, x)?;\n        write_u16(wb, w_f16)?;",
        "        let wkey = format!(\"rmsnorm.w.{}\", cache_key);\n        if !self.tensor_cache.contains_key(&wkey) {\n            let buf = upload_u16(&self.device, w_f16).unwrap();\n            self.tensor_cache.insert(wkey.clone(), buf);\n        }\n        let wb = self.tensor_cache.get(&wkey).unwrap();\n\n        ensure_buf_f32(&self.device, &mut self.x_buf, n)?;\n        ensure_buf_f32(&self.device, &mut self.out_buf, n)?;\n\n        let xb = self.x_buf.as_ref().unwrap();\n        let ob = self.out_buf.as_ref().unwrap();\n\n        write_f32(xb, x)?;"
    );

    // Update fallback signature if it exists
    s = s.replace(
        "pub fn rms_norm_f16w(&mut self, _: &[f32], _: &[u16], _: f32, _: bool, _: &mut [f32])",
        "pub fn rms_norm_f16w(&mut self, _: &[f32], _: &[u16], _: f32, _: bool, _: &str, _: &mut [f32])"
    );

    fs::write("crates/cellm-kernels/src/metal.rs", s).unwrap();
}
