/// Direct test of describe_audio_with_cellm_timed, bypassing the FFI 16-token limit.
use std::path::PathBuf;
use anyhow::Result;
use clap::Parser;
use cellm_sdk::vlm::{VlmRunConfig, describe_audio_with_cellm_timed};
use cellm_sdk::BackendKind;

#[derive(Parser, Debug)]
#[command(name = "audio-direct", about = "Direct audio VLM test without FFI token limit")]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long, default_value = "Describe what you hear in this audio.")]
    prompt: String,

    #[arg(long)]
    audio: PathBuf,

    /// Compute backend. Defaults to "metal" on macOS/iOS for practical
    /// performance — the audio conformer + bidir text decode over 100+
    /// audio tokens is impractically slow on CPU alone.
    #[arg(long, default_value_t = default_backend())]
    backend: String,

    #[arg(long, default_value_t = 80)]
    tokens: usize,
}

fn default_backend() -> String {
    if cfg!(any(target_os = "macos", target_os = "ios")) {
        "metal".to_string()
    } else {
        "cpu".to_string()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let audio_bytes = std::fs::read(&args.audio)?;
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
    println!("Running audio VLM with max_new_tokens={}", args.tokens);
    let (text, timing) = describe_audio_with_cellm_timed(&args.model, &audio_bytes, &args.prompt, cfg)?;
    println!(
        "Timings: mel={:.1}ms encoder={:.1}ms decode={:.1}ms total={:.1}ms",
        timing.patch_ms, timing.encoder_ms, timing.decode_ms, timing.total_ms
    );
    println!("Output:\n{}", text);
    Ok(())
}
