use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use cellm_cache::{KVCache, PageTable};
use cellm_core::KvCacheLayout;
use cellm_model::{llama::LlamaRunner, qwen::QwenRunner, CellmFile};
use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(clap::ValueEnum, Copy, Clone, Debug, Eq, PartialEq)]
enum ChatFormat {
    /// Prefer chat templates if the tokenizer config advertises one, otherwise fall back to plain
    Auto,
    /// ChatML-style: <|im_start|>role ... <|im_end|>
    Chatml,
    /// Plain: "System: ...\nUser: ...\nAssistant:"
    Plain,
}

#[derive(clap::ValueEnum, Copy, Clone, Debug, Eq, PartialEq)]
enum BackendKind {
    Cpu,
    Metal,
}

#[derive(Parser, Debug)]
#[command(name = "infer", about = "Smoke inference for a .cellm model (CPU, unoptimized)")]
struct Args {
    /// Path to .cellm model file
    #[arg(long)]
    model: PathBuf,

    /// Optional path to HuggingFace `tokenizer.json` (required if --prompt is used)
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Prompt text (if set, uses tokenizer to encode and decode)
    #[arg(long)]
    prompt: Option<String>,

    /// Wrap `--prompt` as a simple chat turn: `User: ...\nAssistant:`
    #[arg(long, default_value_t = false)]
    chat: bool,

    /// Chat formatting to use with --chat
    #[arg(long, value_enum, default_value_t = ChatFormat::Auto)]
    chat_format: ChatFormat,

    /// Optional system instruction (only used with --chat)
    #[arg(long)]
    system: Option<String>,

    /// Prompt length (token ids 0..seq-1 unless --tokens provided)
    #[arg(long, default_value_t = 8)]
    seq: usize,

    /// Number of generated tokens
    #[arg(long, default_value_t = 8)]
    gen: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Sample from the top K tokens (used when temperature > 0)
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// RNG seed for sampling (optional)
    #[arg(long)]
    seed: Option<u64>,

    /// Repetition penalty applied to tokens in the recent window (1.0 disables)
    #[arg(long, default_value_t = 1.05)]
    repeat_penalty: f32,

    /// Recent token window for repetition penalty
    #[arg(long, default_value_t = 64)]
    repeat_window: usize,

    /// Optional comma-separated token ids (overrides --seq)
    #[arg(long)]
    tokens: Option<String>,

    /// Run only the first N layers (for quick smoke tests)
    #[arg(long)]
    max_layers: Option<usize>,

    /// Tokens per KV block
    #[arg(long, default_value_t = 16)]
    tokens_per_block: usize,

    /// Total KV blocks (if omitted, computed from seq+gen)
    #[arg(long)]
    total_blocks: Option<usize>,

    /// Stop when EOS token is produced (only meaningful with tokenizer/config)
    #[arg(long, default_value_t = true)]
    stop_eos: bool,

    /// Inference backend selector.
    ///
    /// `metal` currently performs a Metal kernel smoke check and then falls back to CPU math paths.
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();
    select_backend(args.backend)?;

    let file = CellmFile::load(&args.model)?;
    let header = file.header.clone();

    enum Runner {
        Llama(LlamaRunner),
        Qwen(QwenRunner),
    }

    let text_model_type = effective_text_model_type(&header);
    let mut runner = match text_model_type.as_str() {
        "llama" => Runner::Llama(LlamaRunner::load(&args.model)?),
        t if t.starts_with("qwen") => Runner::Qwen(QwenRunner::load(&args.model)?),
        _ => {
            anyhow::bail!(
                "infer supports only llama/qwen right now. Detected model_type={:?} effective_text_model_type={:?} architectures={:?} quantization_config={:?}.",
                header.model_type,
                text_model_type,
                header.source_architectures,
                header.source_quantization_config
            );
        }
    };

    if let Some(n) = args.max_layers {
        match &mut runner {
            Runner::Llama(r) => r.set_max_layers(n),
            Runner::Qwen(r) => r.set_max_layers(n),
        }
    }

    let cfg = match &runner {
        Runner::Llama(r) => r.config().clone(),
        Runner::Qwen(r) => r.config().clone(),
    };
    println!("Model: {:?}", args.model);
    println!(
        "Config: layers={} hidden={} heads={}/{} vocab={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size
    );
    println!("RoPE theta: {}", cfg.rope_theta);
    let active_layers = match &runner {
        Runner::Llama(r) => r.max_layers(),
        Runner::Qwen(r) => r.max_layers(),
    };
    println!("Max layers (active): {}", active_layers);

    let tokenizer = if args.prompt.is_some() {
        let tok_path = args
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--tokenizer is required when using --prompt"))?;
        Some(load_tokenizer(tok_path)?)
    } else {
        None
    };

    let effective_chat_format = if args.chat {
        match args.chat_format {
            ChatFormat::Auto => {
                if let Some(tok_path) = args.tokenizer.as_ref() {
                    if tokenizer_config_chatml(tok_path) {
                        ChatFormat::Chatml
                    } else {
                        ChatFormat::Plain
                    }
                } else {
                    ChatFormat::Plain
                }
            }
            other => other,
        }
    } else {
        ChatFormat::Plain
    };

    let prompt_tokens = if let Some(prompt) = args.prompt.as_deref() {
        let tok = tokenizer.as_ref().expect("tokenizer set when prompt set");
        let prompt_text = build_prompt_text(
            tok,
            prompt,
            args.chat,
            args.system.as_deref(),
            effective_chat_format,
        );
        let enc = tok
            .encode(prompt_text, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        enc.get_ids().to_vec()
    } else if let Some(s) = args.tokens.as_deref() {
        parse_tokens(s)?
    } else {
        (0..args.seq as u32).collect()
    };
    let seq = prompt_tokens.len();

    let head_dim = match &runner {
        Runner::Llama(_) => cfg.hidden_size / cfg.num_attention_heads,
        Runner::Qwen(_) => infer_qwen_kv_head_dim(&file)?,
    };
    let total_tokens = seq + args.gen + 8;
    let total_blocks = args.total_blocks.unwrap_or_else(|| {
        (total_tokens + args.tokens_per_block - 1) / args.tokens_per_block
    });

    let layout = KvCacheLayout {
        total_blocks,
        tokens_per_block: args.tokens_per_block,
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim,
    };
    let mut kv_cache = KVCache::new(layout)?;
    let mut page_table = PageTable::new(1, args.tokens_per_block).unwrap();

    println!(
        "KV cache: blocks={} tokens_per_block={} f16_bytes={}",
        total_blocks,
        args.tokens_per_block,
        kv_cache.layout().total_bytes_f16()
    );

    let mut rng = XorShift64::new(args.seed.unwrap_or_else(seed_from_time));
    let chat_im_end = tokenizer
        .as_ref()
        .and_then(|t| {
            (args.chat && effective_chat_format == ChatFormat::Chatml)
                .then(|| t.token_to_id("<|im_end|>"))
                .flatten()
        });

    // Prefill.
    let t0 = Instant::now();
    let mut next = 0u32;
    let mut all_ids: Vec<u32> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let cand = match &mut runner {
            Runner::Llama(r) => r.step_topk(tok, i, &mut page_table, &mut kv_cache, args.top_k)?,
            Runner::Qwen(r) => r.step_topk(tok, i, &mut page_table, &mut kv_cache, args.top_k)?,
        };
        all_ids.push(tok);
        next = select_next(
            &cand,
            args.temperature,
            args.repeat_penalty,
            args.repeat_window,
            &all_ids,
            &mut rng,
        )?;
    }
    println!(
        "Prefill: {} tokens in {:.2}s (next={})",
        seq,
        t0.elapsed().as_secs_f64(),
        next
    );

    // Decode.
    let t1 = Instant::now();
    let mut cur = next;
    for step in 0..args.gen {
        let pos = seq + step;
        let cand = match &mut runner {
            Runner::Llama(r) => r.step_topk(cur, pos, &mut page_table, &mut kv_cache, args.top_k)?,
            Runner::Qwen(r) => r.step_topk(cur, pos, &mut page_table, &mut kv_cache, args.top_k)?,
        };

        all_ids.push(cur);
        if let Some(tok) = tokenizer.as_ref() {
            let piece = tok.decode(&[cur], true).unwrap_or_default();
            println!("step {:3}: token={} text={:?}", step, cur, piece);
        } else {
            println!("step {:3}: token={}", step, cur);
        }

        if args.stop_eos {
            let eos = match &runner {
                Runner::Llama(r) => r.eos_token_id(),
                Runner::Qwen(r) => r.eos_token_id(),
            };
            if let Some(eos) = eos {
                if cur == eos {
                    break;
                }
            }
            if let Some(im_end) = chat_im_end {
                if cur == im_end {
                    break;
                }
            }
        }

        cur = select_next(
            &cand,
            args.temperature,
            args.repeat_penalty,
            args.repeat_window,
            &all_ids,
            &mut rng,
        )?;
    }
    println!(
        "Decode: {} tokens in {:.2}s",
        args.gen,
        t1.elapsed().as_secs_f64()
    );

    if let Some(tok) = tokenizer.as_ref() {
        let text = if !prompt_tokens.is_empty() && all_ids.len() >= prompt_tokens.len() {
            tok.decode(&all_ids[prompt_tokens.len()..], true)
                .unwrap_or_default()
        } else {
            tok.decode(&all_ids, true).unwrap_or_default()
        };
        println!();
        println!("---");
        println!("{text}");
    }

    Ok(())
}

fn select_backend(backend: BackendKind) -> Result<()> {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    match backend {
        BackendKind::Cpu => {
            log::info!("Backend: cpu (host: {os}/{arch})");
            Ok(())
        }
        BackendKind::Metal => {
            match cellm_kernels::MetalKernels::smoke_test_add_f32() {
                Ok(_) => {
                    log::info!("Backend: metal (smoke ok). Forward path currently uses CPU math kernels.");
                    Ok(())
                }
                Err(e) => {
                    if arch == "x86_64" {
                        log::warn!("Backend: metal requested on Intel host ({os}/{arch}) but unavailable ({e}). Falling back to CPU.");
                    } else {
                        log::warn!("Backend: metal requested ({os}/{arch}) but unavailable ({e}). Falling back to CPU.");
                    }
                    Ok(())
                }
            }
        }
    }
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

fn build_prompt_text(
    _tok: &Tokenizer,
    prompt: &str,
    chat: bool,
    system: Option<&str>,
    chat_format: ChatFormat,
) -> String {
    if !chat {
        return prompt.to_string();
    }

    let sys = system.unwrap_or("").trim();

    match chat_format {
        ChatFormat::Chatml => {
            let mut s = String::new();
            if !sys.is_empty() {
                s.push_str("<|im_start|>system\n");
                s.push_str(sys);
                s.push_str("<|im_end|>\n");
            }
            s.push_str("<|im_start|>user\n");
            s.push_str(prompt);
            s.push_str("<|im_end|>\n");
            s.push_str("<|im_start|>assistant\n");
            s
        }
        ChatFormat::Auto | ChatFormat::Plain => match system {
            Some(sys) if !sys.trim().is_empty() => {
                format!("System: {sys}\nUser: {prompt}\nAssistant:")
            }
            _ => format!("User: {prompt}\nAssistant:"),
        },
    }
}

fn tokenizer_config_chatml(tokenizer_json_path: &std::path::Path) -> bool {
    let Some(dir) = tokenizer_json_path.parent() else {
        return false;
    };
    let cfg_path = dir.join("tokenizer_config.json");
    let Ok(bytes) = std::fs::read(&cfg_path) else {
        return false;
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    let Some(tpl) = v.get("chat_template").and_then(|x| x.as_str()) else {
        return false;
    };
    tpl.contains("<|im_start|>") && tpl.contains("<|im_end|>")
}

fn load_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    match Tokenizer::from_file(path) {
        Ok(t) => Ok(t),
        Err(e) => {
            // Some tokenizers (notably MLX exports) store BPE merges as `[["a","b"], ...]`
            // instead of `["a b", ...]`, which the `tokenizers` crate rejects.
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

fn select_next(
    candidates: &[(u32, f32)],
    temperature: f32,
    repeat_penalty: f32,
    repeat_window: usize,
    recent: &[u32],
    rng: &mut XorShift64,
) -> Result<u32> {
    if candidates.is_empty() {
        anyhow::bail!("no candidates");
    }
    if temperature <= 0.0 {
        return Ok(candidates[0].0);
    }

    let mut ids: Vec<u32> = Vec::with_capacity(candidates.len());
    let mut scores: Vec<f32> = Vec::with_capacity(candidates.len());
    for &(id, s) in candidates {
        ids.push(id);
        scores.push(s);
    }

    if repeat_penalty > 1.0 && repeat_window > 0 && !recent.is_empty() {
        let start = recent.len().saturating_sub(repeat_window);
        for i in 0..scores.len() {
            if recent[start..].contains(&ids[i]) {
                scores[i] /= repeat_penalty;
            }
        }
    }

    // Softmax with temperature over the candidate set.
    let mut max = f32::NEG_INFINITY;
    for &s in &scores {
        if s > max {
            max = s;
        }
    }
    let mut weights = Vec::with_capacity(scores.len());
    let mut sum = 0.0f32;
    for &s in &scores {
        let w = ((s - max) / temperature).exp();
        weights.push(w);
        sum += w;
    }
    if sum == 0.0 {
        return Ok(ids[0]);
    }

    let r = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for i in 0..weights.len() {
        acc += weights[i];
        if r <= acc {
            return Ok(ids[i]);
        }
    }
    Ok(*ids.last().unwrap())
}

fn seed_from_time() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    now.as_nanos() as u64 ^ (now.as_secs() << 32)
}

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as u32; // 24 bits
        (u as f32) / ((1u32 << 24) as f32)
    }
}

fn parse_tokens(s: &str) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        out.push(p.parse::<u32>()?);
    }
    if out.is_empty() {
        anyhow::bail!("--tokens parsed to empty list");
    }
    Ok(out)
}

fn effective_text_model_type(header: &cellm_model::CellmHeader) -> String {
    if let Some(Value::Object(obj)) = &header.source_text_config {
        if let Some(Value::String(mt)) = obj.get("model_type") {
            if !mt.is_empty() {
                return mt.clone();
            }
        }
    }
    header.model_type.clone()
}
