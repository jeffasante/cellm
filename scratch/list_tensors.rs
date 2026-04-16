use std::path::Path;
use cellm_model::CellmFile;

fn main() {
    let path = Path::new("models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm");
    match CellmFile::load(path) {
        Ok(file) => {
            for (name, index) in &file.header.tensors {
                println!("{}: {:?}", name, index.dtype);
            }
        }
        Err(e) => eprintln!("Error loading file: {:?}", e),
    }
}
