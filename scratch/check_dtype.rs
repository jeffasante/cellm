fn main() {
    let b = std::fs::read("models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm").unwrap();
    let magic = &b[0..8];
    let version = &b[8..12];
    let size = u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize;
    let header_bytes = &b[16..16+size];
    let s = String::from_utf8_lossy(header_bytes);
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    for t in v["tensors"].as_array().unwrap() {
        let name = t["name"].as_str().unwrap();
        if name.contains("norm") {
            println!("{} {}", name, t["dtype"]);
        }
    }
}
