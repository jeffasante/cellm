use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use serde_json::Value;

fn main() {
    let mut f = File::open("models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm").unwrap();
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).unwrap();
    let mut version = [0u8; 4];
    f.read_exact(&mut version).unwrap();
    let mut size_bytes = [0u8; 4];
    f.read_exact(&mut size_bytes).unwrap();
    let size = u32::from_le_bytes(size_bytes) as usize;
    let mut header_bytes = vec![0u8; size];
    f.read_exact(&mut header_bytes).unwrap();
    let header: Value = serde_json::from_slice(&header_bytes).unwrap();
    
    if let Some(tensors) = header["tensors"].as_array() {
        for t in tensors {
            let name = t["name"].as_str().unwrap();
            if name.contains("qscale") || name.contains("norm") {
                println!("{}: {}", name, t["dtype"]);
            }
        }
    }
}
