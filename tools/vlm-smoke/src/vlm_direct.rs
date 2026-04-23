/// Direct test of describe_image_with_cellm_timed, bypassing the FFI 16-token limit.
use std::path::PathBuf;
use anyhow::Result;
use clap::Parser;
use cellm_sdk::vlm::{VlmRunConfig, describe_image_with_cellm_timed};
use cellm_sdk::BackendKind;

#[derive(Parser, Debug)]
#[command(name = "vlm-direct", about = "Direct VLM test without FFI token limit")]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long, default_value = "What is in this image?")]
    prompt: String,

    #[arg(long)]
    image: PathBuf,

    #[arg(long, default_value = "cpu")]
    backend: String,

    #[arg(long, default_value_t = 80)]
    tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let image_bytes = std::fs::read(&args.image)?;
    let backend = if args.backend.eq_ignore_ascii_case("metal") {
        BackendKind::Metal
    } else {
        BackendKind::Cpu
    };
    let cfg = VlmRunConfig {
        backend,
        tokens_per_block: 16,
        top_k: 1,
        temperature: 0.0,
        seed: 42,
        repeat_penalty: 1.0,
        repeat_window: 64,
        max_new_tokens: args.tokens,
        min_new_tokens: 1,
    };
    println!("Running VLM with max_new_tokens={}", args.tokens);
    let (text, timing) = describe_image_with_cellm_timed(&args.model, &image_bytes, &args.prompt, cfg)?;
    println!(
        "Timings: patch={:.1}ms encoder={:.1}ms decode={:.1}ms total={:.1}ms",
        timing.patch_ms, timing.encoder_ms, timing.decode_ms, timing.total_ms
    );
    println!("Output:\n{}", text);
    Ok(())
}
