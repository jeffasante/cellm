// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! Detect and optionally redact PII spans using the privacy-filter model.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use cellm_model::PrivacyFilterRunner;
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(name = "pii", about = "PII span detection with a .cellm privacy filter")]
struct Args {
    /// Path to the .cellm model.
    model: PathBuf,
    /// Path to tokenizer.json.
    #[arg(long)]
    tokenizer: PathBuf,
    /// Text to scan. Repeat for multiple inputs.
    #[arg(long = "text", required = true)]
    texts: Vec<String>,
    /// Replace each detected span with `[LABEL]` and print the result.
    #[arg(long)]
    redact: bool,
    /// Append raw `[seq, num_labels]` logits to this file as little-endian
    /// f32, prefixed per text by two u32 (seq_len, num_labels). For parity
    /// checks against the Python reference.
    #[arg(long)]
    dump_logits: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let tok = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
    let model = PrivacyFilterRunner::load(&args.model).context("model load failed")?;

    let mut dump: Vec<u8> = Vec::new();

    for text in &args.texts {
        let enc = tok
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        let ids = enc.get_ids();
        if ids.is_empty() {
            continue;
        }
        let offsets: Vec<(usize, usize)> = enc.get_offsets().to_vec();

        let logits = model.forward(ids).context("forward failed")?;
        let spans = model.spans(&logits, &offsets);

        if args.dump_logits.is_some() {
            let n_labels = logits.len() / ids.len();
            dump.extend_from_slice(&(ids.len() as u32).to_le_bytes());
            dump.extend_from_slice(&(n_labels as u32).to_le_bytes());
            for v in &logits {
                dump.extend_from_slice(&v.to_le_bytes());
            }
        }

        println!("=== {text}");
        for (label, a, b) in &spans {
            println!("    {label:<16} [{a:4}:{b:4}]  {:?}", &text[*a..*b]);
        }

        if args.redact {
            let mut out = String::new();
            let mut cursor = 0usize;
            for (label, a, b) in &spans {
                if *a < cursor {
                    continue;
                }
                out.push_str(&text[cursor..*a]);
                out.push_str(&format!("[{}]", label.to_uppercase()));
                cursor = *b;
            }
            out.push_str(&text[cursor..]);
            println!("    redacted: {out}");
        }
    }

    if let Some(path) = &args.dump_logits {
        std::fs::write(path, &dump).context("logit dump failed")?;
    }
    Ok(())
}
