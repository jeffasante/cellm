// Author: Jeffrey Asante (https://jeffasante.github.io/)
use std::path::Path;
use cellm_model::CellmFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.cellm|.cellmd>", args[0]);
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);
    let file = CellmFile::load(path)?;

    println!("Header info:");
    println!("  model_type: {:?}", file.header.model_type);
    println!("  vocab_size: {}", file.header.vocab_size);
    println!("  hidden_dim: {}", file.header.hidden_dim);
    println!("  num_layers: {}", file.header.num_layers);
    println!("  text_tensor_prefix: {:?}", file.header.text_tensor_prefix);
    println!();

    println!("Tensors:");
    let mut tensors: Vec<_> = file.header.tensors.iter().collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    for t in tensors {
        println!("  {}: dtype={}, shape={:?}, nbytes={}",
            t.name, t.dtype, t.shape, t.nbytes);
    }

    // Specifically look for embed_tokens and lm_head
    println!();
    println!("Embedding/lm_head related tensors:");
    for t in &file.header.tensors {
        if t.name.contains("embed_tokens") || t.name.contains("lm_head") {
            println!("  {}: dtype={}, shape={:?}", t.name, t.dtype, t.shape);
        }
    }

    // Look for qscale tensors
    println!();
    println!("QScale tensors:");
    for t in &file.header.tensors {
        if t.name.contains("qscale") {
            println!("  {}: dtype={}, shape={:?}", t.name, t.dtype, t.shape);
        }
    }

    Ok(())
}
