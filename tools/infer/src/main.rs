use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use cellm_cache::{KVCache, KvEncodingKind, PageTable};
use cellm_core::KvCacheLayout;
use cellm_model::{gemma::GemmaRunner, llama::LlamaRunner, qwen::QwenRunner, CellmFile};
use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(clap::ValueEnum, Copy, Clone, Debug, Eq, PartialEq)]
enum ChatFormat {
    /// Prefer chat templates if the tokenizer config advertises one, otherwise fall back to plain
    Auto,
    /// ChatML-style: <|im_start|>role ... <|im_end|>
    Chatml,
    /// Gemma turn-style: <start_of_turn>user ... <end_of_turn>
    Gemma,
    /// Gemma 4 turn-style: <|turn>user ... <turn|>
    Gemma4,
    /// Plain: "System: ...\nUser: ...\nAssistant:"
    Plain,
}

#[derive(clap::ValueEnum, Copy, Clone, Debug, Eq, PartialEq)]
enum BackendKind {
    Cpu,
    Metal,
}

#[derive(clap::ValueEnum, Copy, Clone, Debug, Eq, PartialEq)]
enum KvEncodingArg {
    F16,
    Turboquant,
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

    /// Optional image path (LiteRT proxy models only)
    #[arg(long)]
    image: Option<PathBuf>,

    /// Optional audio path (LiteRT proxy models only)
    #[arg(long)]
    audio: Option<PathBuf>,

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
    /// `metal` uses strict model Metal backend init; failures return an error.
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,

    /// KV cache encoding mode.
    ///
    /// `f16` is the current default and stable path.
    /// `turboquant` enables an experimental CPU int8+scale packed KV path.
    #[arg(long, value_enum, default_value_t = KvEncodingArg::F16)]
    kv_encoding: KvEncodingArg,
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();
    let t_startup = Instant::now();

    let t_stage = Instant::now();
    select_backend(args.backend)?;
    println!("Startup: backend check {:.2}s", t_stage.elapsed().as_secs_f64());

    let t_stage = Instant::now();
    let file = CellmFile::load(&args.model)?;
    println!(
        "Startup: model header load {:.2}s ({:?})",
        t_stage.elapsed().as_secs_f64(),
        args.model
    );
    let header = file.header.clone();
    if matches!(header.source_safetensors_format.as_deref(), Some("mlx")) {
        anyhow::bail!(
            "this .cellm was converted from safetensors format=\"mlx\", which is not supported for accurate text generation in cellm yet. \
Please convert from the original non-MLX Hugging Face checkpoint (e.g. Qwen/Qwen3.5-0.8B)."
        );
    }

    enum Runner {
        Llama(LlamaRunner),
        Gemma(GemmaRunner),
        Qwen(QwenRunner),
    }

    let text_model_type = effective_text_model_type(&header);
    if text_model_type == "litertlm_proxy" {
        if !allow_litert_proxy() {
            anyhow::bail!(
                "model_type=litertlm_proxy is disabled in Python-free mode. \
This model requires external litert-lm process execution. \
Use a native llama/gemma/qwen .cellm/.cellmd model, or set CELLM_ALLOW_LITERT_PROXY=1 to opt in explicitly."
            );
        }
        run_litertlm_proxy(&file, &args)?;
        return Ok(());
    }

    let t_stage = Instant::now();
    let mut runner = match text_model_type.as_str() {
        "llama" | "smollm3" => Runner::Llama(LlamaRunner::load(&args.model)?),
        t if t.starts_with("gemma") => Runner::Gemma(GemmaRunner::load(&args.model)?),
        t if t.starts_with("qwen") => Runner::Qwen(QwenRunner::load(&args.model)?),
        _ => {
            anyhow::bail!(
                "infer supports only llama/smollm3/gemma/qwen right now. Detected model_type={:?} effective_text_model_type={:?} architectures={:?} quantization_config={:?}.",
                header.model_type,
                text_model_type,
                header.source_architectures,
                header.source_quantization_config
            );
        }
    };
    println!(
        "Startup: runner init {:.2}s (type={})",
        t_stage.elapsed().as_secs_f64(),
        text_model_type
    );

    if args.backend == BackendKind::Metal {
        let t_stage = Instant::now();
        match &mut runner {
            Runner::Qwen(r) => {
                if r.enable_metal_full_backend() {
                    println!("LLM backend: metal (Qwen full acceleration)");
                } else {
                    anyhow::bail!("LLM backend: metal requested, but Qwen full-metal init failed");
                }
            }
            Runner::Gemma(r) => {
                if r.enable_metal_full_backend() {
                    println!("LLM backend: metal (Gemma full acceleration)");
                } else {
                    println!("LLM backend: metal requested, but Gemma is using the stable CPU math path");
                }
            }
            Runner::Llama(r) => {
                if r.enable_metal_full_backend() {
                    println!("LLM backend: metal (Llama full acceleration)");
                } else {
                    anyhow::bail!("LLM backend: metal requested, but Llama full-metal init failed");
                }
            }
        }
        println!("Startup: metal init {:.2}s", t_stage.elapsed().as_secs_f64());
    }

    if let Some(n) = args.max_layers {
        match &mut runner {
            Runner::Llama(r) => r.set_max_layers(n),
            Runner::Gemma(r) => r.set_max_layers(n),
            Runner::Qwen(r) => r.set_max_layers(n),
        }
    }

    let cfg = match &runner {
        Runner::Llama(r) => r.config().clone(),
        Runner::Gemma(r) => r.config().clone(),
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
        Runner::Gemma(r) => r.max_layers(),
        Runner::Qwen(r) => r.max_layers(),
    };
    println!("Max layers (active): {}", active_layers);

    let t_stage = Instant::now();
    let tokenizer = if args.prompt.is_some() {
        let tok_path = args
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--tokenizer is required when using --prompt"))?;
        Some(load_tokenizer(tok_path)?)
    } else {
        None
    };
    if args.prompt.is_some() {
        println!(
            "Startup: tokenizer load {:.2}s",
            t_stage.elapsed().as_secs_f64()
        );
    }
    let added_token_ids = if let Some(tok_path) = args.tokenizer.as_ref() {
        load_added_token_ids(tok_path)?
    } else {
        std::collections::HashMap::new()
    };

    let effective_chat_format = if args.chat {
        match args.chat_format {
            ChatFormat::Auto => {
                if let Some(tok_path) = args.tokenizer.as_ref() {
                    if tokenizer_config_chatml(tok_path) {
                        ChatFormat::Chatml
                    } else if tokenizer_config_gemma4_turn(tok_path) {
                        ChatFormat::Gemma4
                    } else if tokenizer_config_gemma_turn(tok_path) {
                        ChatFormat::Gemma
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

    let t_stage = Instant::now();
    let mut prompt_tokens = if let Some(prompt) = args.prompt.as_deref() {
        let tok = tokenizer.as_ref().expect("tokenizer set when prompt set");
        let tok_path = args
            .tokenizer
            .as_ref()
            .expect("tokenizer path set when prompt set");
        let include_think_prefill = args.chat
            && effective_chat_format == ChatFormat::Chatml
            && tokenizer_config_contains_think_prefill(tok_path);
        let prompt_text = build_prompt_text(
            tok,
            prompt,
            args.chat,
            args.system.as_deref(),
            effective_chat_format,
            include_think_prefill,
        );
        encode_with_explicit_added_tokens(tok, tok_path, &prompt_text)?
    } else if let Some(s) = args.tokens.as_deref() {
        parse_tokens(s)?
    } else {
        (0..args.seq as u32).collect()
    };
    if args.prompt.is_some() {
        if let (Some(tok_path), Some(bos_id)) = (args.tokenizer.as_ref(), file.header.bos_token_id)
        {
            let wants_bos = tokenizer_config_add_bos(tok_path);
            if wants_bos && prompt_tokens.first().copied() != Some(bos_id) {
                prompt_tokens.insert(0, bos_id);
            }
        }
    }
    if args.prompt.is_some() {
        println!(
            "Startup: prompt encode {:.2}s",
            t_stage.elapsed().as_secs_f64()
        );
    }
    let seq = prompt_tokens.len();

    let head_dim = match &runner {
        Runner::Llama(_) => cfg.hidden_size / cfg.num_attention_heads,
        Runner::Gemma(_) => infer_gemma_kv_head_dim(&file)?,
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
    drop(file);
    let kind = if args.backend == BackendKind::Metal {
        cellm_cache::KvStorageKind::Metal
    } else {
        cellm_cache::KvStorageKind::Cpu
    };
    let kv_encoding = match args.kv_encoding {
        KvEncodingArg::F16 => KvEncodingKind::F16,
        KvEncodingArg::Turboquant => KvEncodingKind::TurboQuant,
    };
    let t_stage = Instant::now();
    let mut kv_cache = KVCache::new_with_kind_and_encoding(layout, kind, kv_encoding)?;
    let mut page_table = PageTable::new(1, args.tokens_per_block).unwrap();
    println!(
        "Startup: KV init {:.2}s",
        t_stage.elapsed().as_secs_f64()
    );

    println!(
        "KV cache: blocks={} tokens_per_block={} f16_bytes={}",
        total_blocks,
        args.tokens_per_block,
        kv_cache.layout().total_bytes_f16()
    );
    println!("KV encoding: {:?}", kv_cache.encoding());
    println!(
        "Startup: total before prefill {:.2}s",
        t_startup.elapsed().as_secs_f64()
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
    let debug_logits = std::env::var("CELLM_DEBUG_LOGITS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let cand = match &mut runner {
            Runner::Llama(r) => {
                r.prefill(&prompt_tokens[..prompt_tokens.len() - 1], 0, &mut page_table, &mut kv_cache)?;
                let last_tok = *prompt_tokens.last().unwrap();
                r.step_topk(last_tok, prompt_tokens.len() - 1, &mut page_table, &mut kv_cache, args.top_k)?
            }
            Runner::Gemma(r) => r.step_topk(tok, i, &mut page_table, &mut kv_cache, args.top_k)?,
            Runner::Qwen(r) => r.step_topk(tok, i, &mut page_table, &mut kv_cache, args.top_k)?,
        };
        if debug_logits && i + 1 == prompt_tokens.len() {
            println!("Prefill top candidates:");
            for (rank, (id, score)) in cand.iter().take(10).enumerate() {
                let text = tokenizer
                    .as_ref()
                    .map(|t| t.decode(&[*id], true).unwrap_or_default())
                    .unwrap_or_default();
                println!("  {:2}: id={} score={:.6} text={:?}", rank, id, score, text);
            }
        }
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
            Runner::Gemma(r) => r.step_topk(cur, pos, &mut page_table, &mut kv_cache, args.top_k)?,
            Runner::Qwen(r) => r.step_topk(cur, pos, &mut page_table, &mut kv_cache, args.top_k)?,
        };
        if debug_logits && step == 0 {
            println!("Decode step0 top candidates:");
            for (rank, (id, score)) in cand.iter().take(10).enumerate() {
                let mut text = tokenizer
                    .as_ref()
                    .map(|t| t.decode(&[*id], true).unwrap_or_default())
                    .unwrap_or_default();
                if text.is_empty() {
                    for (t, tid) in &added_token_ids {
                        if *tid == *id {
                            text = t.clone();
                            break;
                        }
                    }
                }
                if text.is_empty() { text = format!("[ID:{}]", id); }
                println!("  {:2}: id={} score={:.6} text={:?}", rank, id, score, text);
            }
        }

        all_ids.push(cur);
        if let Some(tok) = tokenizer.as_ref() {
            let mut piece = tok.decode(&[cur], true).unwrap_or_default();
            if piece.is_empty() {
                for (text, id) in &added_token_ids {
                    if *id == cur {
                        piece = text.clone();
                        break;
                    }
                }
            }
            if piece.is_empty() { piece = format!("[ID:{}]", cur); }
            println!("step {:3}: token={} text={:?}", step, cur, piece);
        } else {
            println!("step {:3}: token={}", step, cur);
        }

        if args.stop_eos {
            let eos = match &runner {
                Runner::Llama(r) => r.eos_token_id(),
                Runner::Gemma(r) => r.eos_token_id(),
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
        let text = sanitize_assistant_text(&strip_think_blocks(&text));
        println!();
        println!("---");
        println!("{text}");
    }

    Ok(())
}


fn strip_think_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut in_think = false;
    while !rest.is_empty() {
        if !in_think {
            if let Some(idx) = rest.find("<think>") {
                out.push_str(&rest[..idx]);
                rest = &rest[idx + "<think>".len()..];
                in_think = true;
            } else {
                out.push_str(rest);
                break;
            }
        } else if let Some(end) = rest.find("</think>") {
            rest = &rest[end + "</think>".len()..];
            in_think = false;
        } else {
            break;
        }
    }
    out.replace("<think>", "").replace("</think>", "")
}

fn sanitize_assistant_text(text: &str) -> String {
    let s = text.trim_start();
    let lowered = s.to_ascii_lowercase();
    if lowered.starts_with("thought\n") {
        return s["thought\n".len()..].trim_start().to_string();
    }
    if lowered.starts_with("thought\r\n") {
        return s["thought\r\n".len()..].trim_start().to_string();
    }
    s.to_string()
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
                    log::info!("Backend: metal (smoke ok). Model-specific Metal acceleration enabled where available.");
                    Ok(())
                }
                Err(e) => {
                    if arch == "x86_64" {
                        anyhow::bail!("Backend: metal requested on Intel host ({os}/{arch}) but unavailable ({e}).");
                    } else {
                        anyhow::bail!("Backend: metal requested ({os}/{arch}) but unavailable ({e}).");
                    }
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

fn infer_gemma_kv_head_dim(file: &CellmFile) -> Result<usize> {
    let h = &file.header;
    let kv_heads = h.num_kv_heads.max(1);
    let mut max_head_dim = 0usize;
    for t in &h.tensors {
        if t.name.contains(".self_attn.k_proj.weight") && t.shape.len() == 2 {
            let kv_dim = t.shape[0];
            if kv_dim % kv_heads == 0 {
                max_head_dim = max_head_dim.max(kv_dim / kv_heads);
            }
        }
    }
    if max_head_dim > 0 {
        Ok(max_head_dim)
    } else {
        anyhow::bail!("unable to infer gemma KV head_dim (no self_attn.k_proj.weight found in tensor list)")
    }
}

fn build_prompt_text(
    _tok: &Tokenizer,
    prompt: &str,
    chat: bool,
    system: Option<&str>,
    chat_format: ChatFormat,
    include_think_prefill: bool,
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
            if include_think_prefill {
                s.push_str("<think>\n\n</think>\n\n");
            }
            s
        }
        ChatFormat::Gemma => {
            let user_text = if sys.is_empty() {
                prompt.to_string()
            } else {
                // Gemma 3-style instruction prompts typically do not support an explicit
                // `system` role, so fold system text into the first user turn.
                format!("{sys}\n\n{prompt}")
            };
            format!(
                "<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"
            )
        }
        ChatFormat::Gemma4 => {
            let user_text = if sys.is_empty() {
                prompt.to_string()
            } else {
                format!("{sys}\n\n{prompt}")
            };
            format!("<|turn>user\n{user_text}<turn|>\n<|turn>model\n")
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

fn tokenizer_config_gemma_turn(tokenizer_json_path: &std::path::Path) -> bool {
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
    tpl.contains("<start_of_turn>") && tpl.contains("<end_of_turn>")
}

fn tokenizer_config_gemma4_turn(tokenizer_json_path: &std::path::Path) -> bool {
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
    if let Some(tpl) = v.get("chat_template").and_then(|x| x.as_str()) {
        return tpl.contains("<|turn>") && tpl.contains("<turn|>");
    }
    // Newer Gemma4 tokenizer configs may omit `chat_template` but still expose
    // explicit turn tokens in config fields.
    let sot_ok = v
        .get("sot_token")
        .and_then(|x| x.as_str())
        .map(|s| s == "<|turn>")
        .unwrap_or(false);
    let eot_ok = v
        .get("eot_token")
        .and_then(|x| x.as_str())
        .map(|s| s == "<turn|>")
        .unwrap_or(false);
    sot_ok && eot_ok
}

fn tokenizer_config_contains_think_prefill(tokenizer_json_path: &std::path::Path) -> bool {
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
    tpl.contains("<think>") && tpl.contains("</think>")
}

fn tokenizer_config_add_bos(tokenizer_json_path: &std::path::Path) -> bool {
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
    v.get("add_bos_token")
        .and_then(|x| x.as_bool())
        .unwrap_or(false)
}

fn load_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    match Tokenizer::from_file(path) {
        Ok(t) => Ok(t),
        Err(e) => {
            // Some tokenizers (notably MLX exports) store BPE merges as `[["a","b"], ...]`
            // instead of `["a b", ...]`, which the `tokenizers` crate rejects.
            let normalized = try_normalize_tokenizer_json(path)?;
            if let Some(p) = normalized {
                let tok = Tokenizer::from_file(&p).map_err(|e| {
                    anyhow::anyhow!("failed to load tokenizer {:?} (normalized {:?}): {e}", path, p)
                })?;
                return Ok(tok);
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

fn encode_with_explicit_added_tokens(
    tok: &Tokenizer,
    tokenizer_json_path: &std::path::Path,
    text: &str,
) -> Result<Vec<u32>> {
    let added = load_added_token_ids(tokenizer_json_path)?;
    if added.is_empty() {
        let enc = tok
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        return Ok(enc.get_ids().to_vec());
    }

    let mut specials: Vec<(&str, u32)> = added.iter().map(|(k, &v)| (k.as_str(), v)).collect();
    specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut out: Vec<u32> = Vec::new();
    let mut i = 0usize;
    let mut chunk_start = 0usize;
    let bytes = text.as_bytes();

    while i < bytes.len() {
        let rest = &text[i..];
        let mut matched: Option<(&str, u32)> = None;
        for &(token, id) in &specials {
            if rest.starts_with(token) {
                matched = Some((token, id));
                break;
            }
        }

        if let Some((token, id)) = matched {
            if chunk_start < i {
                let chunk = &text[chunk_start..i];
                let enc = tok
                    .encode(chunk, false)
                    .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
                out.extend_from_slice(enc.get_ids());
            }
            out.push(id);
            i += token.len();
            chunk_start = i;
        } else {
            let ch = rest
                .chars()
                .next()
                .ok_or_else(|| anyhow::anyhow!("tokenize failed: invalid utf-8 boundary"))?;
            i += ch.len_utf8();
        }
    }

    if chunk_start < text.len() {
        let chunk = &text[chunk_start..];
        let enc = tok
            .encode(chunk, false)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        out.extend_from_slice(enc.get_ids());
    }

    Ok(out)
}

fn load_added_token_ids(tokenizer_json_path: &std::path::Path) -> Result<std::collections::HashMap<String, u32>> {
    let bytes = std::fs::read(tokenizer_json_path)
        .map_err(|e| anyhow::anyhow!("read tokenizer {:?} failed: {e}", tokenizer_json_path))?;
    let v: Value = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("parse tokenizer {:?} failed: {e}", tokenizer_json_path))?;

    let mut out = std::collections::HashMap::new();
    if let Some(arr) = v.get("added_tokens").and_then(|x| x.as_array()) {
        for item in arr {
            let Some(content) = item.get("content").and_then(|x| x.as_str()) else {
                continue;
            };
            let Some(id) = item.get("id").and_then(|x| x.as_u64()) else {
                continue;
            };
            out.insert(content.to_string(), id as u32);
        }
    }
    Ok(out)
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

fn run_litertlm_proxy(file: &CellmFile, args: &Args) -> Result<()> {
    let prompt = args
        .prompt
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("litertlm_proxy requires --prompt"))?;
    let litert_tensor_name = if file.tensor_index("litert.bundle").is_some() {
        "litert.bundle"
    } else {
        file.header
            .tensors
            .first()
            .map(|t| t.name.as_str())
            .ok_or_else(|| anyhow::anyhow!("litertlm_proxy model has no tensors"))?
    };
    let litert_bytes = file.tensor_bytes(litert_tensor_name)?;
    let temp_name = format!(
        "cellm_litert_proxy_{}_{}.litertlm",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );
    let temp_path = std::env::temp_dir().join(temp_name);
    std::fs::write(&temp_path, litert_bytes)
        .map_err(|e| anyhow::anyhow!("failed to write temporary litert model {:?}: {e}", temp_path))?;

    let litert_bin = resolve_tool_bin("CELLM_LITERT_LM_BIN", ".venv-hf/bin/litert-lm", "litert-lm");
    let backend = match args.backend {
        BackendKind::Cpu => "cpu",
        BackendKind::Metal => {
            let _ = std::fs::remove_file(&temp_path);
            anyhow::bail!(
                "litertlm_proxy gpu backend is disabled in cellm CLI because it routes through WebGPU. Use --backend cpu for litert proxy models, or use native non-proxy .cellm models for strict Metal execution."
            );
        }
    };

    println!("LLM backend: litert-lm ({backend})");
    println!("Model: {:?}", args.model);

    if args.image.is_some() || args.audio.is_some() {
        let _ = std::fs::remove_file(&temp_path);
        anyhow::bail!(
            "litertlm_proxy multimodal is not supported in Rust-only mode yet. Use text-only prompt for this model."
        );
    }

    let mut cmd = Command::new(&litert_bin);
    cmd.arg("run")
        .arg(&temp_path)
        .arg("--prompt")
        .arg(prompt)
        .arg("-b")
        .arg(backend);
    apply_litert_cpu_thread_tuning(&mut cmd, backend);
    let output = cmd
        .output()
        .map_err(|e| anyhow::anyhow!("failed to execute {:?}: {e}", litert_bin))?;
    let _ = std::fs::remove_file(&temp_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "litert-lm run failed with status {}: {}",
            output.status,
            stderr.trim()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout.trim_end());
    Ok(())
}

fn allow_litert_proxy() -> bool {
    matches!(
        std::env::var("CELLM_ALLOW_LITERT_PROXY").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn resolve_tool_bin(env_var: &str, local_rel_path: &str, fallback: &str) -> PathBuf {
    if let Ok(p) = std::env::var(env_var) {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return pb;
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join(local_rel_path);
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from(fallback)
}

fn apply_litert_cpu_thread_tuning(cmd: &mut Command, backend: &str) {
    if backend != "cpu" {
        return;
    }
    let has_tflite_threads = std::env::var_os("TFLITE_NUM_THREADS").is_some();
    let has_omp_threads = std::env::var_os("OMP_NUM_THREADS").is_some();
    if has_tflite_threads && has_omp_threads {
        return;
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(4, 32)
        .to_string();
    if !has_tflite_threads {
        cmd.env("TFLITE_NUM_THREADS", &threads);
    }
    if !has_omp_threads {
        cmd.env("OMP_NUM_THREADS", &threads);
    }
}
