// Author: Jeffrey Asante (https://jeffasante.github.io/)
use cellm_model::CellmFile;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        anyhow::bail!("usage: inspect_08b <model_path>");
    }
    let p = Path::new(&args[1]);
    let f = CellmFile::load(p)?;
    println!("Model: {}", p.display());
    println!("Type: {}", f.header.model_type);
    println!("Layers: {}", f.header.num_layers);
    
    println!("Tensors:");
    for t in &f.header.tensors {
        println!("{}: {:?} ({})", t.name, t.shape, t.dtype);
    }
    
    // Inspect a FullAttention layer (e.g. Layer 3)
    let layer = 3;
    let q_name = format!("language_model.model.layers.{}.self_attn.q_proj.weight", layer);
    let k_name = format!("language_model.model.layers.{}.self_attn.k_proj.weight", layer);
    let v_name = format!("language_model.model.layers.{}.self_attn.v_proj.weight", layer);
    
    for name in &[q_name, k_name, v_name] {
        if let Some(t) = f.tensor_index(name) {
            println!("{}: {:?}", name, t.shape);
        } else {
            println!("{} not found", name);
        }
    }
    
    // Inspect a LinearAttention layer (e.g. Layer 0)
    let layer = 0;
    let a_log_name = format!("model.language_model.layers.{}.linear_attn.A_log", layer);
    if let Some(t) = f.header.tensors.iter().find(|t| t.name == a_log_name) {
        let bytes = f.tensor_bytes(&a_log_name)?;
        let vals: &[u16] = bytemuck::cast_slice(bytes);
        println!("Layer {} A_log (f16):", layer);
        for &v in vals.iter().take(16) {
            print!("{:.4} ", half::f16::from_bits(v).to_f32());
        }
        println!();
    }
    
    let dt_bias_name = format!("model.language_model.layers.{}.linear_attn.dt_bias", layer);
    if let Some(t) = f.header.tensors.iter().find(|t| t.name == dt_bias_name) {
        let bytes = f.tensor_bytes(&dt_bias_name)?;
        let vals: &[u16] = bytemuck::cast_slice(bytes);
        println!("Layer {} dt_bias (f16):", layer);
        for &v in vals.iter().take(16) {
            print!("{:.4} ", half::f16::from_bits(v).to_f32());
        }
        println!();
    }
    
    // Inspect embeddings
    let embed_name = "model.language_model.embed_tokens.weight";
    if let Some(t) = f.header.tensors.iter().find(|t| t.name == embed_name) {
        let bytes = f.tensor_bytes(embed_name)?;
        let vals: &[u16] = bytemuck::cast_slice(bytes);
        println!("Embeddings (first 16):");
        for &v in vals.iter().take(16) {
            print!("{:.4} ", half::f16::from_bits(v).to_f32());
        }
        println!();
    } else {
        println!("{} not found", embed_name);
    }
    
    // Inspect Input Norm
    let norm_name = "model.language_model.norm.weight";
    if let Some(t) = f.header.tensors.iter().find(|t| t.name == norm_name) {
        let bytes = f.tensor_bytes(norm_name)?;
        let vals: &[u16] = bytemuck::cast_slice(bytes);
        println!("Final Norm Weight (first 16):");
        for &v in vals.iter().take(16) {
            print!("{:.4} ", half::f16::from_bits(v).to_f32());
        }
        println!();
    }
    
    Ok(())
}
