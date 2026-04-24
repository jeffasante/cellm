use std::path::Path;
use cellm_model::CellmFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = CellmFile::load(Path::new("models/qwen3.5-0.8b.cellm"))?;
    let h = &file.header;
    println!("HEADER: hidden={}, layers={}, heads={}, kv_heads={}, vocab={}, head_dim={:?}, rope_theta={}", 
        h.hidden_dim, h.num_layers, h.num_heads, h.num_kv_heads, h.vocab_size, h.head_dim, h.rope_theta);
    
    for t in &h.tensors {
        if t.name.contains("q_proj.weight") {
            println!("{}: shape={:?}, dtype={}", t.name, t.shape, t.dtype);
        }
    }
    Ok(())
}
