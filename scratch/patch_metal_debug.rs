use std::fs;
fn main() {
    let mut s = fs::read_to_string("crates/cellm-kernels/src/metal.rs").unwrap();
    s = s.replace(
        "let wbuf = upload_i8(&self.device, embed_i8)?;",
        "let wbuf = upload_i8(&self.device, embed_i8)?; if cache_key.contains(\"layers.14.mlp.down_proj\") { eprintln!(\"[metal-debug] uploading i8 weight: {} len={} sum={}\", cache_key, embed_i8.len(), embed_i8.iter().map(|&x| x as i64).sum::<i64>()); }"
    );
    s = s.replace(
        "let sbuf = upload_u16(&self.device, scales_f16)?;",
        "let sbuf = upload_u16(&self.device, scales_f16)?; if cache_key.contains(\"layers.14.mlp.down_proj\") { eprintln!(\"[metal-debug] uploading scales: {} len={} s[0]={}\", cache_key, scales_f16.len(), scales_f16[0]); }"
    );
    fs::write("crates/cellm-kernels/src/metal.rs", s).unwrap();
}
