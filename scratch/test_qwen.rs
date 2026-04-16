use cellm_model::qwen::QwenRunner;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm";
    let mut runner = QwenRunner::load(Path::new(model_path))?;
    let prompt: &[u32] = &[1, 2, 3];
    let mut x = vec![0.0f32; runner.config().hidden_size];
    
    println!("Embedding token 1");
    runner.embed_token_hidden(prompt[0], &mut x)?;
    let l2 = x.iter().fold(0.0_f32, |acc, &v| acc + v * v).sqrt();
    println!("Token 1 embed l2norm: {:.6}", l2);

    println!("Partial rotary factor: {}", runner.debug_partial_rotary_factor());
    println!("RMSNorm is offset: {}", runner.debug_rmsnorm_weight_is_offset());
    
    Ok(())
}
