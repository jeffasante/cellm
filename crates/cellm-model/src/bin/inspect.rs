use cellm_model::CellmFile;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: inspect <model_path>");
        std::process::exit(1);
    }
    let path = Path::new(&args[1]);
    let file = CellmFile::load(path)?;
    for meta in &file.header.tensors {
        if meta.name == "model.norm.weight" {
            let bytes = file.tensor_bytes(&meta.name)?;
            let f16_vals: &[u16] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, bytes.len() / 2) };
            print!("{}: ", meta.name);
            for i in 0..8.min(f16_vals.len()) {
                // simple f16 to f32 approximation for inspection
                let bits = f16_vals[i];
                println!("{:04x} ", bits);
            }
            println!();
        }
    }
    Ok(())
}
