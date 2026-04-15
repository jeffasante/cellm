use cellm_model::qwen::QwenRunner;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm";
    let mut runner = QwenRunner::load(Path::new(model_path))?;
    
    println!("--- Linear Parity ---");
    let tensors = ["language_model.model.layers.1.self_attn.o_proj.weight"];
    for t in tensors {
        let (max, mean) = runner.test_linear_parity(t, 896, 896)?;
        println!("{:<60} | Max: {:.6} | Mean: {:.6}", t, max, mean);
    }
    
    println!("\n--- RMSNorm Parity ---");
    let norm_name = "language_model.model.layers.1.input_layernorm.weight";
    let (max, mean) = runner.test_rmsnorm_parity(norm_name, 1e-6, false)?;
    println!("{:<60} | Max: {:.6} | Mean: {:.6}", norm_name, max, mean);
    
    println!("\n--- RoPE Parity ---");
    let (max, mean) = runner.test_rope_parity(14, 64, 64, 10, 1000000.0)?;
    println!("{:<60} | Max: {:.6} | Mean: {:.6}", "rope_half_f32", max, mean);
    
    println!("\n--- Attention Parity ---");
    let (max, mean) = runner.test_attn_parity(1)?;
    println!("{:<60} | Max: {:.6} | Mean: {:.6}", "attention_gqa_f32", max, mean);
    
    Ok(())
}
