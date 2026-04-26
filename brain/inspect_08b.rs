// Author: Jeffrey Asante (https://jeffasante.github.io/)
use cellm_model::CellmFile;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let p = Path::new("models/qwen3.5-0.8b.cellm");
    let f = CellmFile::open(p)?;
    println!("Model: {}", p.display());
    println!("Type: {}", f.header.model_type);
    println!("Layers: {}", f.header.num_layers);
    
    // Inspect a FullAttention layer (e.g. Layer 3)
    let layer = 3;
    let q_name = format!("language_model.model.layers.{}.self_attn.q_proj.weight", layer);
    let k_name = format!("language_model.model.layers.{}.self_attn.k_proj.weight", layer);
    let v_name = format!("language_model.model.layers.{}.self_attn.v_proj.weight", layer);
    
    for name in &[q_name, k_name, v_name] {
        if let Some(t) = f.tensor_index(name) {
            println!("{}: {:?}", name, f.header.tensors[t].shape);
        } else {
            println!("{} not found", name);
        }
    }
    
    // Inspect a LinearAttention layer (e.g. Layer 0)
    let layer = 0;
    let names = vec![
        format!("language_model.model.layers.{}.linear_attn.in_proj_qkv.weight", layer),
        format!("language_model.model.layers.{}.linear_attn.in_proj_a.weight", layer),
        format!("language_model.model.layers.{}.linear_attn.in_proj_b.weight", layer),
        format!("language_model.model.layers.{}.linear_attn.in_proj_z.weight", layer),
    ];
    for name in names {
        if let Some(t) = f.tensor_index(&name) {
            println!("{}: {:?}", name, f.header.tensors[t].shape);
        } else {
            // Try without the language_model prefix
            let alt = name.replace("language_model.", "");
            if let Some(t) = f.tensor_index(&alt) {
                println!("{}: {:?}", alt, f.header.tensors[t].shape);
            } else {
                println!("{} not found", name);
            }
        }
    }
    
    Ok(())
}
