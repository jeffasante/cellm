// Author: Jeffrey Asante (https://jeffasante.github.io/)
/// cellm benchmark harness (runner-based).
///
/// Measures:
/// - Prefill latency (TTFT-ish)
/// - Decode throughput (tok/s)
///
/// Usage:
///   cargo run --release --bin bench -- \
///     --model models/smollm2-135m.cellm \
///     --tokenizer models/hf/smollm2-135m/tokenizer.json \
///     --prompt "Gravity is" --gen 32

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use cellm_cache::{KVCache, PageTable};
use cellm_core::KvCacheLayout;
use cellm_model::{gemma::GemmaRunner, lfm::LfmRunner, llama::LlamaRunner, qwen::QwenRunner, CellmFile};
use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "bench", about = "cellm benchmark (runner-based)")]
struct Args {
    /// Path to .cellm model file
    #[arg(long)]
    model: PathBuf,

    /// Optional path to HuggingFace `tokenizer.json` (required if --prompt is used)
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Prompt text (if set, uses tokenizer to encode)
    #[arg(long)]
    prompt: Option<String>,

    /// Prompt length (token ids 0..seq-1 unless --prompt provided)
    #[arg(long, default_value_t = 16)]
    seq: usize,

    /// Number of generated tokens
    #[arg(long, default_value_t = 32)]
    gen: usize,

    /// Run only the first N layers (debug)
    #[arg(long)]
    max_layers: Option<usize>,

    /// Tokens per KV block
    #[arg(long, default_value_t = 16)]
    tokens_per_block: usize,

    /// Total KV blocks (if omitted, computed from seq+gen)
    #[arg(long)]
    total_blocks: Option<usize>,

    /// Sample from the top K tokens
    #[arg(long, default_value_t = 40)]
    top_k: usize,
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();

    let file = CellmFile::load(&args.model)?;
    let header = file.header.clone();

    enum Runner {
        Llama(LlamaRunner),
        Gemma(GemmaRunner),
        Qwen(QwenRunner),
        Lfm(LfmRunner),
    }

    let mut runner = match header.model_type.as_str() {
        "llama" => Runner::Llama(LlamaRunner::load(&args.model)?),
        t if t.starts_with("gemma") => Runner::Gemma(GemmaRunner::load(&args.model)?),
        t if t.starts_with("qwen") => Runner::Qwen(QwenRunner::load(&args.model)?),
        "lfm" | "lfm2" => Runner::Lfm(LfmRunner::load(&args.model)?),
        _ => anyhow::bail!(
            "bench supports only llama/gemma/qwen/lfm right now (model_type={:?})",
            header.model_type
        ),
    };

    if let Some(n) = args.max_layers {
        match &mut runner {
            Runner::Llama(r) => r.set_max_layers(n),
            Runner::Gemma(r) => r.set_max_layers(n),
            Runner::Qwen(r) => r.set_max_layers(n),
            Runner::Lfm(r) => r.set_max_layers(n),
        }
    }

    let cfg = match &runner {
        Runner::Llama(r) => r.config().clone(),
        Runner::Gemma(r) => r.config().clone(),
        Runner::Qwen(r) => r.config().clone(),
        Runner::Lfm(r) => r.config().clone(),
    };

    let tokenizer = if args.prompt.is_some() {
        let tok_path = args
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--tokenizer is required when using --prompt"))?;
        Some(load_tokenizer(tok_path)?)
    } else {
        None
    };

    let prompt_tokens = if let Some(prompt) = args.prompt.as_deref() {
        let tok = tokenizer.as_ref().expect("tokenizer set when prompt set");
        let enc = tok
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        enc.get_ids().to_vec()
    } else {
        (0..args.seq as u32).collect()
    };
    let seq = prompt_tokens.len();

    let head_dim = match &runner {
        Runner::Llama(_) => cfg.hidden_size / cfg.num_attention_heads,
        Runner::Gemma(_) => infer_gemma_kv_head_dim(&file)?,
        Runner::Qwen(_) => infer_qwen_kv_head_dim(&file)?,
        Runner::Lfm(_) => infer_lfm_head_dim(&file)?,
    };
    let total_tokens = seq + args.gen + 8;
    let total_blocks = args
        .total_blocks
        .unwrap_or_else(|| (total_tokens + args.tokens_per_block - 1) / args.tokens_per_block);

    let layout = KvCacheLayout {
        total_blocks,
        tokens_per_block: args.tokens_per_block,
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim,
    };
    let mut kv_cache = KVCache::new(layout)?;
    let mut page_table = PageTable::new(1, args.tokens_per_block).unwrap();

    println!("Model: {:?}", args.model);
    println!(
        "Config: layers={} hidden={} heads={}/{} vocab={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size
    );
    println!("KV cache: blocks={} tokens_per_block={} head_dim={}", total_blocks, args.tokens_per_block, head_dim);

    // Prefill.
    let t0 = Instant::now();
    let mut next = 0u32;
    let mut recent: Vec<u32> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let cand = match &mut runner {
            Runner::Llama(r) => r.step_topk(tok, i, &mut page_table, &mut kv_cache, args.top_k)?,
            Runner::Gemma(r) => r.step_topk(tok, i, &mut page_table, &mut kv_cache, args.top_k)?,
            Runner::Qwen(r) => r.step_topk(tok, i, &mut page_table, &mut kv_cache, args.top_k)?,
            Runner::Lfm(r) => r.step_topk(tok, i, &mut page_table, &mut kv_cache, args.top_k)?,
        };
        recent.push(tok);
        next = cand[0].0;
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    println!("Prefill: {} tokens in {:.2}s (next={})", seq, prefill_s, next);

    // Decode (greedy).
    let t1 = Instant::now();
    let mut cur = next;
    for step in 0..args.gen {
        let pos = seq + step;
        let cand = match &mut runner {
            Runner::Llama(r) => r.step_topk(cur, pos, &mut page_table, &mut kv_cache, args.top_k)?,
            Runner::Gemma(r) => r.step_topk(cur, pos, &mut page_table, &mut kv_cache, args.top_k)?,
            Runner::Qwen(r) => r.step_topk(cur, pos, &mut page_table, &mut kv_cache, args.top_k)?,
            Runner::Lfm(r) => r.step_topk(cur, pos, &mut page_table, &mut kv_cache, args.top_k)?,
        };
        cur = cand[0].0;
        recent.push(cur);
    }
    let decode_s = t1.elapsed().as_secs_f64();
    println!("Decode: {} tokens in {:.2}s ({:.2} tok/s)", args.gen, decode_s, (args.gen as f64) / decode_s.max(1e-9));

    Ok(())
}

fn infer_qwen_kv_head_dim(file: &CellmFile) -> Result<usize> {
    let h = &file.header;
    let kv_heads = h.num_kv_heads.max(1);
    for t in &h.tensors {
        if t.name.contains(".self_attn.k_proj.weight") && t.shape.len() == 2 {
            let kv_dim = t.shape[0];
            if kv_dim % kv_heads == 0 {
                return Ok(kv_dim / kv_heads);
            }
        }
    }
    anyhow::bail!("unable to infer qwen KV head_dim (no self_attn.k_proj.weight found in tensor list)")
}

fn infer_gemma_kv_head_dim(file: &CellmFile) -> Result<usize> {
    let h = &file.header;
    let kv_heads = h.num_kv_heads.max(1);
    for t in &h.tensors {
        if t.name.contains(".self_attn.k_proj.weight") && t.shape.len() == 2 {
            let kv_dim = t.shape[0];
            if kv_dim % kv_heads == 0 {
                return Ok(kv_dim / kv_heads);
            }
        }
    }
    anyhow::bail!("unable to infer gemma KV head_dim (no self_attn.k_proj.weight found in tensor list)")
}

fn infer_lfm_head_dim(file: &CellmFile) -> Result<usize> {
    let h = &file.header;
    // First check if head_dim is explicitly stored
    if let Some(hd) = h.head_dim {
        return Ok(hd);
    }
    // Otherwise infer from k_proj tensor shape
    let kv_heads = h.num_kv_heads.max(1);
    for t in &h.tensors {
        if t.name.contains(".self_attn.k_proj.weight") && t.shape.len() == 2 {
            let kv_dim = t.shape[0];
            if kv_dim % kv_heads == 0 {
                return Ok(kv_dim / kv_heads);
            }
        }
    }
    // Fallback: use hidden / num_heads
    Ok(h.hidden_dim / h.num_heads)
}

fn load_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    match Tokenizer::from_file(path) {
        Ok(t) => Ok(t),
        Err(e) => {
            let normalized = try_normalize_tokenizer_json(path)?;
            if let Some(p) = normalized {
                return Tokenizer::from_file(&p).map_err(|e| {
                    anyhow::anyhow!("failed to load tokenizer {:?} (normalized {:?}): {e}", path, p)
                });
            }
            Err(anyhow::anyhow!("failed to load tokenizer {:?}: {e}", path))
        }
    }
}

fn try_normalize_tokenizer_json(path: &std::path::Path) -> Result<Option<std::path::PathBuf>> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("read tokenizer {:?} failed: {e}", path))?;
    let mut v: Value = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("parse tokenizer {:?} failed: {e}", path))?;

    let merges = v
        .get_mut("model")
        .and_then(|m| m.get_mut("merges"))
        .and_then(|m| m.as_array_mut());
    let Some(merges) = merges else {
        return Ok(None);
    };
    if merges.is_empty() {
        return Ok(None);
    }

    let looks_like_pairs = merges[0]
        .as_array()
        .is_some_and(|a| a.len() == 2 && a[0].is_string() && a[1].is_string());
    if !looks_like_pairs {
        return Ok(None);
    }

    let mut out: Vec<Value> = Vec::with_capacity(merges.len());
    for m in merges.iter() {
        let Some(pair) = m.as_array() else {
            return Ok(None);
        };
        if pair.len() != 2 {
            return Ok(None);
        }
        let Some(a) = pair[0].as_str() else {
            return Ok(None);
        };
        let Some(b) = pair[1].as_str() else {
            return Ok(None);
        };
        out.push(Value::String(format!("{a} {b}")));
    }
    *merges = out;

    let tmp = std::env::temp_dir().join(format!(
        "cellm_tokenizer_normalized_{}.json",
        std::process::id()
    ));
    std::fs::write(&tmp, serde_json::to_vec(&v)?)
        .map_err(|e| anyhow::anyhow!("write normalized tokenizer {:?} failed: {e}", tmp))?;
    Ok(Some(tmp))
}
