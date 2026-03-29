use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use bytemuck::cast_slice;
use clap::Parser;
use cellm_cache::{KVCache, PageTable};
use cellm_core::KvCacheLayout;
use cellm_kernels::MetalKernels;
use cellm_model::{llama::LlamaRunner, CellmFile};
use half::f16;
use ndarray::{Array, Array2, Array3, Array4, Array5, ArrayD, Axis, IxDyn};
use ort::environment::Environment;
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue, SessionOutputs};
use ort::value::{Tensor, TensorElementType, ValueType};
use rand::prelude::*;
use serde_json::Value;
use tokenizers::Tokenizer;

#[cfg(target_os = "macos")]
#[allow(non_camel_case_types)]
type c_int = i32;

#[cfg(target_os = "macos")]
#[allow(non_camel_case_types)]
type c_float = f32;

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: c_int = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: c_int = 111;
#[cfg(target_os = "macos")]
const CBLAS_TRANS: c_int = 112;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );
}

#[derive(clap::ValueEnum, Copy, Clone, Debug, Eq, PartialEq)]
enum DecoderBackend {
    Onnx,
    Cellm,
}

#[derive(clap::ValueEnum, Copy, Clone, Debug, Eq, PartialEq)]
enum VisionBackend {
    Onnx,
    Cellm,
}

#[derive(clap::ValueEnum, Copy, Clone, Debug, Eq, PartialEq)]
enum BackendKind {
    Cpu,
    Metal,
}

#[derive(Parser, Debug)]
#[command(name = "vlm-infer", about = "Run SmolVLM-256M ONNX exports locally (CPU, unoptimized)")]
struct Args {
    /// Path to the HuggingFace model directory (containing tokenizer.json, config.json, onnx/*)
    #[arg(long)]
    model_dir: PathBuf,

    /// Image to query
    #[arg(long)]
    image: PathBuf,

    /// User prompt/question about the image
    #[arg(long, default_value = "Can you describe this image?")]
    prompt: String,

    /// Max new tokens to generate
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Minimum new tokens before EOS can stop generation
    #[arg(long, default_value_t = 16)]
    min_new_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Top-K sampling (used when temperature > 0)
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// RNG seed for sampling (optional)
    #[arg(long)]
    seed: Option<u64>,

    /// ONNX variant suffix to use (fp16/int8/q4f16/q4/bnb4/uint8/quantized)
    #[arg(long, default_value = "fp16")]
    onnx_variant: String,

    /// Use multiple threads in ONNX Runtime
    #[arg(long, default_value_t = 1)]
    threads: usize,

    /// Print ONNX model input/output names and exit
    #[arg(long, default_value_t = false)]
    dump_io: bool,

    /// Print per-step token ids and decoded pieces
    #[arg(long, default_value_t = false)]
    show_tokens: bool,

    /// Skip special tokens during final decode
    #[arg(long, default_value_t = true)]
    skip_special: bool,

    /// Prevent image-structure tokens (<row_*, <image>, etc.) from being generated
    #[arg(long)]
    ban_image_tokens: Option<bool>,

    /// Enable image splitting into local tiles (global image is always included)
    #[arg(long, default_value_t = false)]
    split_image: bool,

    /// Decoder backend: `onnx` (legacy) or `cellm` (native .cellm text stack).
    #[arg(long, value_enum, default_value_t = DecoderBackend::Onnx)]
    decoder_backend: DecoderBackend,

    /// Path to `.cellm` model used when `--decoder-backend cellm`.
    #[arg(long)]
    cellm_model: Option<PathBuf>,

    /// Vision backend: `onnx` (legacy) or `cellm` (experimental native path).
    #[arg(long, value_enum, default_value_t = VisionBackend::Onnx)]
    vision_backend: VisionBackend,

    /// Tokens per KV block for `--decoder-backend cellm`.
    #[arg(long, default_value_t = 16)]
    tokens_per_block: usize,

    /// Total KV blocks for `--decoder-backend cellm` (auto if omitted).
    #[arg(long)]
    total_blocks: Option<usize>,

    /// Backend selector for native codepaths.
    ///
    /// `metal` currently runs a Metal smoke check and then uses CPU math for inference.
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,
}

#[derive(Clone, Debug)]
struct SmolVlmConfig {
    image_token_id: i64,
    eos_token_id: i64,
    end_of_utterance_id: Option<i64>,
    num_hidden_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    image_seq_len: usize,
    banned_token_ids: Vec<i64>,
}

struct ProcessedImageInput {
    pixel_values: Array5<f32>,
    pixel_attention_mask: Array4<bool>,
    prompt_image_block: String,
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
            match MetalKernels::smoke_test_add_f32() {
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

fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();
    select_backend(args.backend)?;

    // ort uses a global environment.
    ort::init().with_name("cellm-vlm-onnx").commit();
    let _env = Environment::current()?;
    let cfg = load_config(&args.model_dir)?;
    let tok = load_tokenizer(&args.model_dir.join("tokenizer.json"))?;
    let banned_token_ids: Vec<i64> = if args.ban_image_tokens.unwrap_or(true) {
        let mut banned = cfg.banned_token_ids.clone();
        for token in ["<|im_start|>", "<|im_end|>"] {
            if let Some(id) = tok.token_to_id(token) {
                banned.push(id as i64);
            }
        }
        banned.sort_unstable();
        banned.dedup();
        banned
    } else {
        Vec::new()
    };

    let onnx_dir = args.model_dir.join("onnx");
    let vision_path = onnx_dir.join(format!("vision_encoder_{}.onnx", args.onnx_variant));
    let embed_path = onnx_dir.join(format!("embed_tokens_{}.onnx", args.onnx_variant));
    let decoder_path = onnx_dir.join(format!("decoder_model_merged_{}.onnx", args.onnx_variant));

    let mut vision = if args.vision_backend == VisionBackend::Onnx {
        Some(load_session(&vision_path, args.threads)?)
    } else {
        None
    };
    let mut embed = if args.decoder_backend == DecoderBackend::Onnx {
        Some(load_session(&embed_path, args.threads)?)
    } else {
        None
    };
    let mut decoder = if args.decoder_backend == DecoderBackend::Onnx {
        Some(load_session(&decoder_path, args.threads)?)
    } else {
        None
    };

    if args.dump_io {
        if let Some(vision) = vision.as_ref() {
            dump_io("vision", vision)?;
        } else {
            println!("== vision == (cellm backend selected; no ONNX session)");
        }
        if let Some(embed) = embed.as_ref() {
            dump_io("embed", embed)?;
        }
        if let Some(decoder) = decoder.as_ref() {
            dump_io("decoder", decoder)?;
        }
        return Ok(());
    }

    let image_input = preprocess_image_idefics3(&args.image, cfg.image_seq_len, args.split_image)?;

    let prompt_text = build_single_turn_prompt(&args.prompt, &image_input.prompt_image_block);
    let enc = tok
        // The chat template already contains special tokens like <|im_start|>.
        // Adding them again tends to corrupt the prompt.
        .encode(prompt_text, false)
        .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
    let input_ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
    let attention_mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();

    log::info!(
        "Prompt tokens: {} (image_token_id={} x{})",
        input_ids.len(),
        cfg.image_token_id,
        input_ids
            .iter()
            .filter(|&&t| t == cfg.image_token_id)
            .count()
    );

    // 1) Vision features
    let t0 = Instant::now();
    let image_features = match args.vision_backend {
        VisionBackend::Onnx => run_vision(
            vision
                .as_mut()
                .context("vision backend onnx selected but vision session missing")?,
            image_input.pixel_values,
            image_input.pixel_attention_mask,
        )?,
        VisionBackend::Cellm => run_vision_cellm(
            args.cellm_model
                .as_ref()
                .context("--vision-backend cellm requires --cellm-model <path>.")?,
            image_input.pixel_values,
            image_input.pixel_attention_mask,
        )?,
    };
    log::info!(
        "Vision: {:?} in {:.2}s",
        image_features.shape(),
        t0.elapsed().as_secs_f32()
    );

    let (generated, decode_secs) = match args.decoder_backend {
        DecoderBackend::Onnx => run_decode_onnx(
            embed
                .as_mut()
                .context("decoder backend onnx selected but embed session missing")?,
            decoder
                .as_mut()
                .context("decoder backend onnx selected but decoder session missing")?,
            &tok,
            &cfg,
            &args,
            input_ids,
            attention_mask,
            &image_features,
            &banned_token_ids,
        )?,
        DecoderBackend::Cellm => run_decode_cellm(
            &tok,
            &cfg,
            &args,
            input_ids,
            &image_features,
            &banned_token_ids,
        )?,
    };

    log::info!(
        "Decode: {} tokens in {:.2}s",
        generated.len(),
        decode_secs
    );

    let decoded = tok
        .decode(
            &generated.iter().map(|&t| t as u32).collect::<Vec<u32>>(),
            args.skip_special,
        )
        .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;
    println!("{decoded}");

    Ok(())
}

fn load_session(path: &Path, threads: usize) -> Result<Session> {
    let builder = Session::builder().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut builder = builder
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    builder
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("load onnx session: {path:?}"))
}

fn dump_io(name: &str, session: &Session) -> Result<()> {
    println!("== {name} ==");
    println!("inputs:");
    for (i, input) in session.inputs().iter().enumerate() {
        println!("  [{i}] {} {:?}", input.name(), input.dtype());
    }
    println!("outputs:");
    for (i, output) in session.outputs().iter().enumerate() {
        println!("  [{i}] {} {:?}", output.name(), output.dtype());
    }
    Ok(())
}

fn session_input_expects_f16(session: &Session, input_name: &str) -> bool {
    let Some(outlet) = session.inputs().iter().find(|o| o.name() == input_name) else {
        return false;
    };
    matches!(
        outlet.dtype(),
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        }
    )
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))
}

fn load_config(model_dir: &Path) -> Result<SmolVlmConfig> {
    let config_path = model_dir.join("config.json");
    let processor_path = model_dir.join("processor_config.json");
    let added_tokens_path = model_dir.join("added_tokens.json");

    let cfg_txt = fs::read_to_string(&config_path)
        .with_context(|| format!("read config: {config_path:?}"))?;
    let v: Value = serde_json::from_str(&cfg_txt).context("parse config.json")?;

    let image_token_id = v
        .get("image_token_id")
        .and_then(|x| x.as_i64())
        .context("config.image_token_id missing")?;
    let text_cfg = v.get("text_config").context("config.text_config missing")?;
    let eos_token_id = text_cfg
        .get("eos_token_id")
        .and_then(|x| x.as_i64())
        .context("config.text_config.eos_token_id missing")?;
    let num_hidden_layers = text_cfg
        .get("num_hidden_layers")
        .and_then(|x| x.as_u64())
        .context("config.text_config.num_hidden_layers missing")? as usize;
    let num_kv_heads = text_cfg
        .get("num_key_value_heads")
        .and_then(|x| x.as_u64())
        .context("config.text_config.num_key_value_heads missing")? as usize;
    let head_dim = text_cfg
        .get("head_dim")
        .and_then(|x| x.as_u64())
        .context("config.text_config.head_dim missing")? as usize;

    let image_seq_len = if processor_path.exists() {
        let p_txt = fs::read_to_string(&processor_path)
            .with_context(|| format!("read processor config: {processor_path:?}"))?;
        let pv: Value = serde_json::from_str(&p_txt).context("parse processor_config.json")?;
        pv.get("image_seq_len")
            .and_then(|x| x.as_u64())
            .unwrap_or(64) as usize
    } else {
        64
    };

    let (end_of_utterance_id, banned_token_ids) = if added_tokens_path.exists() {
        let t = fs::read_to_string(&added_tokens_path)
            .with_context(|| format!("read added_tokens: {added_tokens_path:?}"))?;
        let v: Value = serde_json::from_str(&t).context("parse added_tokens.json")?;
        let end_id = v.get("<end_of_utterance>").and_then(|x| x.as_i64());

        let mut banned = Vec::new();
        if let Some(obj) = v.as_object() {
            for (k, vv) in obj {
                let Some(id) = vv.as_i64() else { continue };
                if k == "<image>"
                    || k == "<fake_token_around_image>"
                    || k == "<global-img>"
                    || k.starts_with("<row_")
                {
                    banned.push(id);
                }
            }
        }

        (end_id, banned)
    } else {
        (None, Vec::new())
    };

    Ok(SmolVlmConfig {
        image_token_id,
        eos_token_id,
        end_of_utterance_id,
        num_hidden_layers,
        num_kv_heads,
        head_dim,
        image_seq_len,
        banned_token_ids,
    })
}

fn build_single_turn_prompt(user_text: &str, image_block: &str) -> String {
    // Matches the model's chat template for a single user message where the first content item is image.
    format!(
        "<|im_start|>User:{image_block}{user_text}<end_of_utterance>\nAssistant:"
    )
}

fn format_image_block(num_images: usize, image_seq_len: usize) -> String {
    let image_tokens = "<image>".repeat(image_seq_len);
    let mut block = String::new();
    for image_idx in 0..num_images {
        if image_idx == 0 {
            block.push_str("<fake_token_around_image><global-img>");
            block.push_str(&image_tokens);
            block.push_str("<fake_token_around_image>");
            continue;
        }
        let local_idx = image_idx - 1;
        let row = local_idx / 6 + 1;
        let col = local_idx % 6 + 1;
        block.push_str("<fake_token_around_image>");
        block.push_str(&format!("<row_{}_col_{}>", row, col));
        block.push_str(&image_tokens);
        block.push_str("<fake_token_around_image>");
    }
    block
}

struct ResizedPadded {
    image: image::RgbImage,
    valid_w: usize,
    valid_h: usize,
    offset_x: usize,
    offset_y: usize,
}

fn resize_and_pad_512(rgb: &image::RgbImage) -> ResizedPadded {
    let (w, h) = rgb.dimensions();
    let target = 512u32;
    let scale = target as f32 / (w.max(h) as f32);
    let new_w = ((w as f32) * scale).round().max(1.0) as u32;
    let new_h = ((h as f32) * scale).round().max(1.0) as u32;
    let resized = image::imageops::resize(rgb, new_w, new_h, image::imageops::FilterType::CatmullRom);
    let mut canvas = image::RgbImage::new(target, target);
    let x0 = (target - new_w) / 2;
    let y0 = (target - new_h) / 2;
    image::imageops::replace(&mut canvas, &resized, x0 as i64, y0 as i64);
    ResizedPadded {
        image: canvas,
        valid_w: new_w as usize,
        valid_h: new_h as usize,
        offset_x: x0 as usize,
        offset_y: y0 as usize,
    }
}

fn copy_rgb_to_nchw(dst: &mut Array5<f32>, image_idx: usize, rgb: &image::RgbImage) {
    let target = 512usize;
    for y in 0..target {
        for x in 0..target {
            let p = rgb.get_pixel(x as u32, y as u32).0;
            let r = (p[0] as f32) / 255.0 * 2.0 - 1.0;
            let g = (p[1] as f32) / 255.0 * 2.0 - 1.0;
            let b = (p[2] as f32) / 255.0 * 2.0 - 1.0;
            dst[[0, image_idx, 0, y, x]] = r;
            dst[[0, image_idx, 1, y, x]] = g;
            dst[[0, image_idx, 2, y, x]] = b;
        }
    }
}

fn fill_pixel_attention_mask(
    mask: &mut Array4<bool>,
    image_idx: usize,
    valid_w: usize,
    valid_h: usize,
    offset_x: usize,
    offset_y: usize,
) {
    for y in 0..valid_h {
        for x in 0..valid_w {
            mask[[0, image_idx, offset_y + y, offset_x + x]] = true;
        }
    }
}

fn preprocess_image_idefics3(
    path: &Path,
    image_seq_len: usize,
    split_image: bool,
) -> Result<ProcessedImageInput> {
    let img = image::open(path).with_context(|| format!("open image: {path:?}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();

    // Build one global resized view plus local tiles (up to 6x6 like the added row/col tokens).
    let tile = 512u32;
    let cols = ((w + tile - 1) / tile).clamp(1, 6);
    let rows = ((h + tile - 1) / tile).clamp(1, 6);
    let mut images = Vec::new();
    images.push(resize_and_pad_512(&rgb)); // global image
    if split_image {
        for row in 0..rows {
            for col in 0..cols {
                let x = col * tile;
                let y = row * tile;
                let cw = (w - x).min(tile);
                let ch = (h - y).min(tile);
                let crop = image::imageops::crop_imm(&rgb, x, y, cw, ch).to_image();
                images.push(resize_and_pad_512(&crop));
            }
        }
    }

    let num_images = images.len();
    let mut pixel_values = Array5::<f32>::zeros((1, num_images, 3, 512, 512));
    let mut pixel_attention_mask = Array4::<bool>::from_elem((1, num_images, 512, 512), false);
    for (idx, im) in images.iter().enumerate() {
        copy_rgb_to_nchw(&mut pixel_values, idx, &im.image);
        fill_pixel_attention_mask(
            &mut pixel_attention_mask,
            idx,
            im.valid_w,
            im.valid_h,
            im.offset_x,
            im.offset_y,
        );
    }
    let prompt_image_block = format_image_block(num_images, image_seq_len);

    Ok(ProcessedImageInput {
        pixel_values,
        pixel_attention_mask,
        prompt_image_block,
    })
}

fn position_ids_from_attention_mask(mask: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(mask.len());
    let mut acc = 0i64;
    for &m in mask {
        if m == 0 {
            out.push(1);
        } else {
            out.push(acc);
            acc += 1;
        }
    }
    out
}

fn last_position_id(mask: &[i64]) -> i64 {
    let mut acc = 0i64;
    for &m in mask {
        if m != 0 {
            acc += 1;
        }
    }
    (acc - 1).max(0)
}

fn run_decode_onnx(
    embed: &mut Session,
    decoder: &mut Session,
    tok: &Tokenizer,
    cfg: &SmolVlmConfig,
    args: &Args,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    image_features: &Array2<f32>,
    banned_token_ids: &[i64],
) -> Result<(Vec<i64>, f32)> {
    let mut rng: StdRng = match args.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };
    let mut cur_input_ids = input_ids;
    let mut cur_attention_mask = attention_mask;
    let mut cur_position_ids = position_ids_from_attention_mask(&cur_attention_mask);
    let past_expects_f16 = session_input_expects_f16(decoder, "past_key_values.0.key");
    log::info!(
        "Decoder KV dtype: {}",
        if past_expects_f16 { "f16" } else { "f32" }
    );
    let mut generated: Vec<i64> = Vec::new();
    let decode_t0 = Instant::now();

    if past_expects_f16 {
        let mut past =
            PastKeyValuesF16::zeros(1, cfg.num_hidden_layers, cfg.num_kv_heads, cfg.head_dim);
        let mut merged_image = false;
        for step in 0..args.max_new_tokens {
            let mut inputs_embeds = run_embed_f32(embed, &cur_input_ids)?;
            if !merged_image {
                let image_positions: Vec<usize> = cur_input_ids
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &t)| (t == cfg.image_token_id).then_some(i))
                    .collect();
                let expected = image_features.shape()[0];
                if image_positions.len() != expected {
                    anyhow::bail!(
                        "Image token count mismatch: prompt has {} <image> tokens but vision_encoder produced {} features.",
                        image_positions.len(),
                        expected
                    );
                }
                for (j, &pos) in image_positions.iter().enumerate() {
                    let src = image_features.index_axis(Axis(0), j);
                    inputs_embeds.index_axis_mut(Axis(1), pos).assign(&src);
                }
                merged_image = true;
            }
            let out = run_decoder_f32_past_f16(
                decoder,
                &inputs_embeds,
                &cur_attention_mask,
                &cur_position_ids,
                &past,
            )?;
            let last_index = out.logits.shape()[1] - 1;
            let last_logits = out.logits.index_axis(Axis(1), last_index).into_dyn();
            let next_token = sample_next_token(
                last_logits,
                args.temperature,
                args.top_k,
                banned_token_ids,
                &mut rng,
            )?;
            generated.push(next_token);
            if args.show_tokens {
                let piece = tok
                    .decode(&[next_token as u32], false)
                    .unwrap_or_else(|_| String::new());
                println!("step {step:4}: token={next_token} text={piece:?}");
            }
            if (next_token == cfg.eos_token_id || cfg.end_of_utterance_id == Some(next_token))
                && (step + 1) >= args.min_new_tokens
            {
                log::info!("EOS at step {step}");
                break;
            }
            cur_input_ids = vec![next_token];
            cur_attention_mask.push(1);
            cur_position_ids = vec![last_position_id(&cur_attention_mask)];
            past = out.present;
        }
    } else {
        let mut past = PastKeyValuesF32::zeros(1, cfg.num_hidden_layers, cfg.num_kv_heads, cfg.head_dim);
        let mut merged_image = false;
        for step in 0..args.max_new_tokens {
            let mut inputs_embeds = run_embed_f32(embed, &cur_input_ids)?;
            if !merged_image {
                let image_positions: Vec<usize> = cur_input_ids
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &t)| (t == cfg.image_token_id).then_some(i))
                    .collect();
                let expected = image_features.shape()[0];
                if image_positions.len() != expected {
                    anyhow::bail!(
                        "Image token count mismatch: prompt has {} <image> tokens but vision_encoder produced {} features.",
                        image_positions.len(),
                        expected
                    );
                }
                for (j, &pos) in image_positions.iter().enumerate() {
                    inputs_embeds
                        .index_axis_mut(Axis(1), pos)
                        .assign(&image_features.index_axis(Axis(0), j));
                }
                merged_image = true;
            }
            let out = run_decoder_f32(
                decoder,
                &inputs_embeds,
                &cur_attention_mask,
                &cur_position_ids,
                &past,
            )?;
            let last_index = out.logits.shape()[1] - 1;
            let last_logits = out.logits.index_axis(Axis(1), last_index).into_dyn();
            let next_token = sample_next_token(
                last_logits,
                args.temperature,
                args.top_k,
                banned_token_ids,
                &mut rng,
            )?;
            generated.push(next_token);
            if args.show_tokens {
                let piece = tok
                    .decode(&[next_token as u32], false)
                    .unwrap_or_else(|_| String::new());
                println!("step {step:4}: token={next_token} text={piece:?}");
            }
            if (next_token == cfg.eos_token_id || cfg.end_of_utterance_id == Some(next_token))
                && (step + 1) >= args.min_new_tokens
            {
                log::info!("EOS at step {step}");
                break;
            }
            cur_input_ids = vec![next_token];
            cur_attention_mask.push(1);
            cur_position_ids = vec![last_position_id(&cur_attention_mask)];
            past = out.present;
        }
    }
    Ok((generated, decode_t0.elapsed().as_secs_f32()))
}

fn run_decode_cellm(
    tok: &Tokenizer,
    cfg: &SmolVlmConfig,
    args: &Args,
    input_ids: Vec<i64>,
    image_features: &Array2<f32>,
    banned_token_ids: &[i64],
) -> Result<(Vec<i64>, f32)> {
    let model_path = args
        .cellm_model
        .as_ref()
        .context("--decoder-backend cellm requires --cellm-model <path>.")?;
    let mut runner = LlamaRunner::load(model_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let hidden = runner.hidden_size();
    if image_features.shape()[1] != hidden {
        anyhow::bail!(
            "image feature dim {} != cellm hidden {}",
            image_features.shape()[1],
            hidden
        );
    }
    let head_dim = runner.config().hidden_size / runner.config().num_attention_heads;
    let total_tokens = input_ids.len() + args.max_new_tokens + 8;
    let total_blocks = args
        .total_blocks
        .unwrap_or_else(|| (total_tokens + args.tokens_per_block - 1) / args.tokens_per_block);
    let layout = KvCacheLayout {
        total_blocks,
        tokens_per_block: args.tokens_per_block,
        num_layers: runner.config().num_hidden_layers,
        num_kv_heads: runner.config().num_key_value_heads,
        head_dim,
    };
    let mut kv_cache = KVCache::new(layout).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut page_table =
        PageTable::new(1, args.tokens_per_block).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut image_idx = 0usize;
    let mut x = vec![0.0f32; hidden];
    let mut rng: StdRng = match args.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };
    let mut next: i64 = 0;
    let decode_t0 = Instant::now();

    for (pos, &tok_id) in input_ids.iter().enumerate() {
        if tok_id == cfg.image_token_id {
            if image_idx >= image_features.shape()[0] {
                anyhow::bail!(
                    "image token count mismatch: prompt has more <image> tokens than vision features"
                );
            }
            let src = image_features.index_axis(Axis(0), image_idx);
            x.copy_from_slice(src.as_slice().context("vision feature row not contiguous")?);
            image_idx += 1;
        } else {
            runner
                .embed_token_hidden(tok_id as u32, &mut x)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        let cand = runner
            .step_topk_from_hidden(tok_id_to_hidden_input(&x), pos, &mut page_table, &mut kv_cache, args.top_k)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        next = sample_from_candidates(&cand, args.temperature, banned_token_ids, &mut rng)?;
    }
    if image_idx != image_features.shape()[0] {
        anyhow::bail!(
            "image token count mismatch: prompt has fewer <image> tokens ({image_idx}) than vision features ({})",
            image_features.shape()[0]
        );
    }

    let mut generated: Vec<i64> = Vec::new();
    for step in 0..args.max_new_tokens {
        generated.push(next);
        if args.show_tokens {
            let piece = tok
                .decode(&[next as u32], false)
                .unwrap_or_else(|_| String::new());
            println!("step {step:4}: token={next} text={piece:?}");
        }
        if (next == cfg.eos_token_id || cfg.end_of_utterance_id == Some(next))
            && (step + 1) >= args.min_new_tokens
        {
            log::info!("EOS at step {step}");
            break;
        }
        runner
            .embed_token_hidden(next as u32, &mut x)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let pos = input_ids.len() + step;
        let cand = runner
            .step_topk_from_hidden(tok_id_to_hidden_input(&x), pos, &mut page_table, &mut kv_cache, args.top_k)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        next = sample_from_candidates(&cand, args.temperature, banned_token_ids, &mut rng)?;
    }

    Ok((generated, decode_t0.elapsed().as_secs_f32()))
}

fn tok_id_to_hidden_input(x: &[f32]) -> &[f32] {
    x
}

fn sample_from_candidates(
    candidates: &[(u32, f32)],
    temperature: f32,
    banned_token_ids: &[i64],
    rng: &mut StdRng,
) -> Result<i64> {
    let mut filtered: Vec<(u32, f32)> = candidates
        .iter()
        .copied()
        .filter(|(id, _)| !banned_token_ids.contains(&(*id as i64)))
        .collect();
    if filtered.is_empty() {
        filtered = candidates.to_vec();
    }
    if filtered.is_empty() {
        anyhow::bail!("no candidates to sample from");
    }
    if temperature <= 0.0 {
        return Ok(filtered[0].0 as i64);
    }
    let temp = temperature.max(1e-5);
    let mut max = f32::NEG_INFINITY;
    for &(_, s) in &filtered {
        if s > max {
            max = s;
        }
    }
    let mut weights = Vec::with_capacity(filtered.len());
    let mut sum = 0.0f32;
    for &(_, s) in &filtered {
        let w = ((s - max) / temp).exp();
        weights.push(w);
        sum += w;
    }
    if sum <= 0.0 {
        return Ok(filtered[0].0 as i64);
    }
    let r = rng.gen::<f32>() * sum;
    let mut acc = 0.0f32;
    for (idx, w) in weights.iter().enumerate() {
        acc += *w;
        if r <= acc {
            return Ok(filtered[idx].0 as i64);
        }
    }
    Ok(filtered.last().unwrap().0 as i64)
}

fn run_vision(
    session: &mut Session,
    pixel_values: Array5<f32>,
    pixel_attention_mask: Array4<bool>,
) -> Result<Array2<f32>> {
    let out0_name = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .unwrap_or_else(|| "image_features".to_string());
    let outputs: SessionOutputs = session.run(ort::inputs! {
        "pixel_values" => Tensor::from_array(pixel_values)?,
        "pixel_attention_mask" => Tensor::from_array(pixel_attention_mask)?,
    })?;
    let v = outputs
        .get("image_features")
        .or_else(|| outputs.get(&out0_name))
        .context("vision output image_features missing")?;
    // Expect [num_images, 64, hidden] or [num_images*64, hidden]
    let arr: ArrayD<f32> = extract_array_f32(v)?;
    let arr = match arr.ndim() {
        3 => {
            let a3 = arr.into_dimensionality::<ndarray::Ix3>().unwrap();
            let n = a3.shape()[0];
            let t = a3.shape()[1];
            let h = a3.shape()[2];
            a3.into_shape((n * t, h))
                .context("flatten vision features")?
                .to_owned()
        }
        2 => arr.into_dimensionality::<ndarray::Ix2>().unwrap().to_owned(),
        other => anyhow::bail!("unexpected vision output rank: {other}"),
    };
    Ok(arr)
}

fn run_vision_cellm(
    model_path: &Path,
    pixel_values: Array5<f32>,
    _pixel_attention_mask: Array4<bool>,
) -> Result<Array2<f32>> {
    let file = CellmFile::load(model_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let vision_prefix = file
        .header
        .vision_tensor_prefix
        .clone()
        .unwrap_or_else(|| "model.vision_model.".to_string());
    let projector_prefix = file
        .header
        .projector_tensor_prefix
        .clone()
        .unwrap_or_else(|| "model.connector.".to_string());

    let patch_w_name = format!("{vision_prefix}embeddings.patch_embedding.weight");
    let patch_b_name = format!("{vision_prefix}embeddings.patch_embedding.bias");
    let pos_name = format!("{vision_prefix}embeddings.position_embedding.weight");
    let ln_w_name = format!("{vision_prefix}post_layernorm.weight");
    let ln_b_name = format!("{vision_prefix}post_layernorm.bias");
    let proj_name = format!("{projector_prefix}modality_projection.proj.weight");

    let patch_w_shape = tensor_shape(&file, &patch_w_name)?;
    let patch_b_shape = tensor_shape(&file, &patch_b_name)?;
    let pos_shape = tensor_shape(&file, &pos_name)?;
    let proj_shape = tensor_shape(&file, &proj_name)?;

    if patch_w_shape.len() != 4 || patch_w_shape[1] != 3 {
        anyhow::bail!("unexpected patch embedding shape: {patch_w_shape:?}");
    }
    if patch_b_shape.len() != 1 || patch_b_shape[0] != patch_w_shape[0] {
        anyhow::bail!("unexpected patch bias shape: {patch_b_shape:?}");
    }
    if pos_shape.len() != 2 || pos_shape[1] != patch_w_shape[0] {
        anyhow::bail!("unexpected position embedding shape: {pos_shape:?}");
    }
    if proj_shape.len() != 2 {
        anyhow::bail!("unexpected projector shape: {proj_shape:?}");
    }

    let hidden = patch_w_shape[0];
    let patch = patch_w_shape[2];
    if patch != patch_w_shape[3] || patch == 0 {
        anyhow::bail!("unexpected patch kernel shape: {patch_w_shape:?}");
    }
    let num_tokens = pos_shape[0];
    let grid = (num_tokens as f32).sqrt() as usize;
    if grid * grid != num_tokens {
        anyhow::bail!("vision position tokens not square: {num_tokens}");
    }
    let projector_in = proj_shape[1];
    if projector_in % hidden != 0 {
        anyhow::bail!("projector input {} not divisible by hidden {}", projector_in, hidden);
    }
    let packed = projector_in / hidden;
    let group_side = (packed as f32).sqrt() as usize;
    if group_side * group_side != packed {
        anyhow::bail!("projector packing {} is not square", packed);
    }
    if grid % group_side != 0 {
        anyhow::bail!("grid {} not divisible by group_side {}", grid, group_side);
    }
    let groups_per_side = grid / group_side;
    let out_tokens_per_image = groups_per_side * groups_per_side;
    let text_hidden = proj_shape[0];

    let patch_w = tensor_to_f32(&file, &patch_w_name)?;
    let patch_b = tensor_to_f32(&file, &patch_b_name)?;
    let pos = tensor_to_f32(&file, &pos_name)?;
    let post_ln_w = tensor_to_f32(&file, &ln_w_name)?;
    let post_ln_b = tensor_to_f32(&file, &ln_b_name)?;
    let proj = tensor_to_f32(&file, &proj_name)?;
    let eps = file
        .header
        .source_vision_config
        .as_ref()
        .and_then(|v| v.get("layer_norm_eps"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let num_layers = file
        .header
        .source_vision_config
        .as_ref()
        .and_then(|v| v.get("num_hidden_layers"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or_else(|| infer_vision_num_layers(&file, &vision_prefix));
    let num_heads = file
        .header
        .source_vision_config
        .as_ref()
        .and_then(|v| v.get("num_attention_heads"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or_else(|| hidden / 64);
    let intermediate = file
        .header
        .source_vision_config
        .as_ref()
        .and_then(|v| v.get("intermediate_size"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden * 4);

    if num_heads == 0 || hidden % num_heads != 0 {
        anyhow::bail!("invalid vision heads: hidden={hidden}, num_heads={num_heads}");
    }
    if num_layers == 0 {
        anyhow::bail!("vision layer count resolved to zero");
    }
    let head_dim = hidden / num_heads;

    let nimg = pixel_values.shape()[1];
    let h = pixel_values.shape()[3];
    let w = pixel_values.shape()[4];
    if h != grid * patch || w != grid * patch {
        anyhow::bail!(
            "pixel size {}x{} incompatible with vision grid {} and patch {}",
            h,
            w,
            grid,
            patch
        );
    }

    let mut out = Array2::<f32>::zeros((nimg * out_tokens_per_image, text_hidden));
    let mut tokens = vec![0.0f32; num_tokens * hidden];
    let mut norm1 = vec![0.0f32; num_tokens * hidden];
    let mut q = vec![0.0f32; num_tokens * hidden];
    let mut k = vec![0.0f32; num_tokens * hidden];
    let mut v = vec![0.0f32; num_tokens * hidden];
    let mut attn = vec![0.0f32; num_tokens * hidden];
    let mut proj_out = vec![0.0f32; num_tokens * hidden];
    let mut norm2 = vec![0.0f32; num_tokens * hidden];
    let mut mlp_up = vec![0.0f32; num_tokens * intermediate];
    let mut mlp_out = vec![0.0f32; num_tokens * hidden];
    let mut score_buf = vec![0.0f32; num_tokens];
    let mut prob_buf = vec![0.0f32; num_tokens];

    for img in 0..nimg {
        // patch embed + position
        for py in 0..grid {
            for px in 0..grid {
                let token_idx = py * grid + px;
                for oc in 0..hidden {
                    let mut acc = patch_b[oc];
                    for ic in 0..3usize {
                        for ky in 0..patch {
                            for kx in 0..patch {
                                let y = py * patch + ky;
                                let x = px * patch + kx;
                                let pv = pixel_values[[0, img, ic, y, x]];
                                let wi = (((oc * 3 + ic) * patch + ky) * patch) + kx;
                                acc += pv * patch_w[wi];
                            }
                        }
                    }
                    tokens[token_idx * hidden + oc] = acc + pos[token_idx * hidden + oc];
                }
            }
        }

        for layer in 0..num_layers {
            let prefix = format!("{vision_prefix}encoder.layers.{layer}.");
            let ln1_w = tensor_to_f32(&file, &format!("{prefix}layer_norm1.weight"))?;
            let ln1_b = tensor_to_f32(&file, &format!("{prefix}layer_norm1.bias"))?;
            let q_w = tensor_to_f32(&file, &format!("{prefix}self_attn.q_proj.weight"))?;
            let q_b = tensor_to_f32(&file, &format!("{prefix}self_attn.q_proj.bias"))?;
            let k_w = tensor_to_f32(&file, &format!("{prefix}self_attn.k_proj.weight"))?;
            let k_b = tensor_to_f32(&file, &format!("{prefix}self_attn.k_proj.bias"))?;
            let v_w = tensor_to_f32(&file, &format!("{prefix}self_attn.v_proj.weight"))?;
            let v_b = tensor_to_f32(&file, &format!("{prefix}self_attn.v_proj.bias"))?;
            let o_w = tensor_to_f32(&file, &format!("{prefix}self_attn.out_proj.weight"))?;
            let o_b = tensor_to_f32(&file, &format!("{prefix}self_attn.out_proj.bias"))?;
            let ln2_w = tensor_to_f32(&file, &format!("{prefix}layer_norm2.weight"))?;
            let ln2_b = tensor_to_f32(&file, &format!("{prefix}layer_norm2.bias"))?;
            let fc1_w = tensor_to_f32(&file, &format!("{prefix}mlp.fc1.weight"))?;
            let fc1_b = tensor_to_f32(&file, &format!("{prefix}mlp.fc1.bias"))?;
            let fc2_w = tensor_to_f32(&file, &format!("{prefix}mlp.fc2.weight"))?;
            let fc2_b = tensor_to_f32(&file, &format!("{prefix}mlp.fc2.bias"))?;

            layer_norm_rows(&tokens, num_tokens, hidden, &ln1_w, &ln1_b, eps, &mut norm1);
            linear_rows(&norm1, num_tokens, hidden, &q_w, hidden, Some(&q_b), &mut q);
            linear_rows(&norm1, num_tokens, hidden, &k_w, hidden, Some(&k_b), &mut k);
            linear_rows(&norm1, num_tokens, hidden, &v_w, hidden, Some(&v_b), &mut v);
            self_attention_full(
                &q,
                &k,
                &v,
                num_tokens,
                num_heads,
                head_dim,
                &mut score_buf,
                &mut prob_buf,
                &mut attn,
            );
            linear_rows(
                &attn,
                num_tokens,
                hidden,
                &o_w,
                hidden,
                Some(&o_b),
                &mut proj_out,
            );
            add_inplace(&mut tokens, &proj_out);

            layer_norm_rows(&tokens, num_tokens, hidden, &ln2_w, &ln2_b, eps, &mut norm2);
            linear_rows(
                &norm2,
                num_tokens,
                hidden,
                &fc1_w,
                intermediate,
                Some(&fc1_b),
                &mut mlp_up,
            );
            gelu_pytorch_tanh_inplace(&mut mlp_up);
            linear_rows(
                &mlp_up,
                num_tokens,
                intermediate,
                &fc2_w,
                hidden,
                Some(&fc2_b),
                &mut mlp_out,
            );
            add_inplace(&mut tokens, &mlp_out);
        }

        layer_norm_rows(
            &tokens,
            num_tokens,
            hidden,
            &post_ln_w,
            &post_ln_b,
            eps,
            &mut norm1,
        );
        tokens.copy_from_slice(&norm1);

        // pack groups and apply connector projection
        for gy in 0..groups_per_side {
            for gx in 0..groups_per_side {
                let out_tok = gy * groups_per_side + gx;
                for o in 0..text_hidden {
                    let mut acc = 0.0f32;
                    let proj_row = &proj[o * projector_in..(o + 1) * projector_in];
                    let mut proj_i = 0usize;
                    for dy in 0..group_side {
                        for dx in 0..group_side {
                            let ty = gy * group_side + dy;
                            let tx = gx * group_side + dx;
                            let token_idx = ty * grid + tx;
                            let src = &tokens[token_idx * hidden..(token_idx + 1) * hidden];
                            for &v in src {
                                acc += v * proj_row[proj_i];
                                proj_i += 1;
                            }
                        }
                    }
                    out[[img * out_tokens_per_image + out_tok, o]] = acc;
                }
            }
        }
    }

    Ok(out)
}

fn infer_vision_num_layers(file: &CellmFile, vision_prefix: &str) -> usize {
    let mut layers = 0usize;
    loop {
        let name = format!("{vision_prefix}encoder.layers.{layers}.layer_norm1.weight");
        if file.tensor_index(&name).is_some() {
            layers += 1;
        } else {
            break;
        }
    }
    layers
}

fn tensor_shape(file: &CellmFile, name: &str) -> Result<Vec<usize>> {
    file.tensor_index(name)
        .map(|t| t.shape.clone())
        .ok_or_else(|| anyhow::anyhow!("missing tensor: {name}"))
}

fn tensor_f16_to_f32(file: &CellmFile, name: &str) -> Result<Vec<f32>> {
    let bytes = file
        .tensor_bytes(name)
        .map_err(|e| anyhow::anyhow!("tensor bytes {name}: {e}"))?;
    if bytes.len() % 2 != 0 {
        anyhow::bail!("tensor {name} bytes not even");
    }
    let u16s: &[u16] = cast_slice(bytes);
    Ok(u16s.iter().map(|&b| f16::from_bits(b).to_f32()).collect())
}

fn tensor_to_f32(file: &CellmFile, name: &str) -> Result<Vec<f32>> {
    let index = file
        .tensor_index(name)
        .ok_or_else(|| anyhow::anyhow!("missing tensor: {name}"))?;
    match index.dtype.as_str() {
        "f16" => tensor_f16_to_f32(file, name),
        "f32" => {
            let bytes = file
                .tensor_bytes(name)
                .map_err(|e| anyhow::anyhow!("tensor bytes {name}: {e}"))?;
            if bytes.len() % 4 != 0 {
                anyhow::bail!("tensor {name} bytes not aligned for f32");
            }
            let vals: &[f32] = cast_slice(bytes);
            Ok(vals.to_vec())
        }
        "bf16" => {
            let bytes = file
                .tensor_bytes(name)
                .map_err(|e| anyhow::anyhow!("tensor bytes {name}: {e}"))?;
            if bytes.len() % 2 != 0 {
                anyhow::bail!("tensor {name} bytes not aligned for bf16");
            }
            let vals: &[u16] = cast_slice(bytes);
            Ok(vals
                .iter()
                .map(|&b| half::bf16::from_bits(b).to_f32())
                .collect())
        }
        "i8" => {
            let shape = &index.shape;
            if shape.len() != 2 {
                anyhow::bail!("int8 tensor {name} needs 2D shape, got {shape:?}");
            }
            let rows = shape[0];
            let cols = shape[1];
            let scale_name = format!("{name}.qscale");
            let scales = tensor_f16_to_f32(file, &scale_name).with_context(|| {
                format!("int8 tensor {name} requires per-row {scale_name} scales")
            })?;
            if scales.len() != rows {
                anyhow::bail!(
                    "int8 tensor {name} scale length mismatch: {} vs rows {}",
                    scales.len(),
                    rows
                );
            }
            let bytes = file
                .tensor_bytes(name)
                .map_err(|e| anyhow::anyhow!("tensor bytes {name}: {e}"))?;
            let vals: &[i8] = cast_slice(bytes);
            if vals.len() != rows * cols {
                anyhow::bail!(
                    "int8 tensor {name} data length mismatch: {} vs {}",
                    vals.len(),
                    rows * cols
                );
            }
            let mut out = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let s = scales[r];
                let src = &vals[r * cols..(r + 1) * cols];
                let dst = &mut out[r * cols..(r + 1) * cols];
                for c in 0..cols {
                    dst[c] = src[c] as f32 * s;
                }
            }
            Ok(out)
        }
        other => anyhow::bail!("unsupported tensor dtype for {name}: {other}"),
    }
}

fn layer_norm_rows(
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(weight.len(), cols);
    debug_assert_eq!(bias.len(), cols);
    debug_assert_eq!(input.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);
    for r in 0..rows {
        let x = &input[r * cols..(r + 1) * cols];
        let y = &mut out[r * cols..(r + 1) * cols];
        let mut mean = 0.0f32;
        for &v in x {
            mean += v;
        }
        mean /= cols as f32;
        let mut var = 0.0f32;
        for &v in x {
            let d = v - mean;
            var += d * d;
        }
        var /= cols as f32;
        let inv = 1.0f32 / (var + eps).sqrt();
        for i in 0..cols {
            y[i] = ((x[i] - mean) * inv) * weight[i] + bias[i];
        }
    }
}

fn linear_rows(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), rows * in_dim);
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(out.len(), rows * out_dim);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), out_dim);
    }

    if try_sgemm_input_weight_t(input, rows, in_dim, weight, out_dim, out).is_ok() {
        if let Some(b) = bias {
            add_bias_rows(out, rows, out_dim, b);
        }
        return;
    }

    for r in 0..rows {
        let x = &input[r * in_dim..(r + 1) * in_dim];
        let y = &mut out[r * out_dim..(r + 1) * out_dim];
        if let Some(b) = bias {
            y.copy_from_slice(b);
        } else {
            y.fill(0.0);
        }
        for o in 0..out_dim {
            let mut acc = y[o];
            let wrow = &weight[o * in_dim..(o + 1) * in_dim];
            for i in 0..in_dim {
                acc += x[i] * wrow[i];
            }
            y[o] = acc;
        }
    }
}

fn self_attention_full(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    num_heads: usize,
    head_dim: usize,
    score_buf: &mut [f32],
    prob_buf: &mut [f32],
    out: &mut [f32],
) {
    let hidden = num_heads * head_dim;
    debug_assert_eq!(q.len(), seq * hidden);
    debug_assert_eq!(k.len(), seq * hidden);
    debug_assert_eq!(v.len(), seq * hidden);
    debug_assert_eq!(out.len(), seq * hidden);
    debug_assert!(score_buf.len() >= seq);
    debug_assert!(prob_buf.len() >= seq);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    if try_self_attention_sgemm(q, k, v, seq, num_heads, head_dim, scale, out).is_ok() {
        return;
    }

    for h in 0..num_heads {
        let offset = h * head_dim;
        for i in 0..seq {
            for j in 0..seq {
                let qi = &q[i * hidden + offset..i * hidden + offset + head_dim];
                let kj = &k[j * hidden + offset..j * hidden + offset + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += qi[d] * kj[d];
                }
                score_buf[j] = dot * scale;
            }

            let mut mx = f32::NEG_INFINITY;
            for &s in score_buf[..seq].iter() {
                if s > mx {
                    mx = s;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..seq {
                let p = (score_buf[j] - mx).exp();
                prob_buf[j] = p;
                sum += p;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq {
                    let p = prob_buf[j] * inv;
                    acc += p * v[j * hidden + offset + d];
                }
                out[i * hidden + offset + d] = acc;
            }
        }
    }
}

fn add_bias_rows(out: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    for r in 0..rows {
        let row = &mut out[r * cols..(r + 1) * cols];
        for c in 0..cols {
            row[c] += bias[c];
        }
    }
}

fn try_sgemm_input_weight_t(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    out: &mut [f32],
) -> Result<()> {
    #[cfg(target_os = "macos")]
    return sgemm_row_major(
        input, rows, in_dim, weight, out_dim, false, true, 1.0, 0.0, out,
    );
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (input, rows, in_dim, weight, out_dim, out);
        anyhow::bail!("sgemm unavailable on this platform")
    }
}

fn try_self_attention_sgemm(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let hidden = num_heads * head_dim;
        let mut qh = vec![0.0f32; seq * head_dim];
        let mut kh = vec![0.0f32; seq * head_dim];
        let mut vh = vec![0.0f32; seq * head_dim];
        let mut scores = vec![0.0f32; seq * seq];
        let mut probs = vec![0.0f32; seq * seq];
        let mut head_out = vec![0.0f32; seq * head_dim];

        for h in 0..num_heads {
            let offset = h * head_dim;
            for t in 0..seq {
                let q_src = &q[t * hidden + offset..t * hidden + offset + head_dim];
                let k_src = &k[t * hidden + offset..t * hidden + offset + head_dim];
                let v_src = &v[t * hidden + offset..t * hidden + offset + head_dim];
                qh[t * head_dim..(t + 1) * head_dim].copy_from_slice(q_src);
                kh[t * head_dim..(t + 1) * head_dim].copy_from_slice(k_src);
                vh[t * head_dim..(t + 1) * head_dim].copy_from_slice(v_src);
            }

            sgemm_row_major(
                &qh,
                seq,
                head_dim,
                &kh,
                seq,
                false,
                true,
                scale,
                0.0,
                &mut scores,
            )?;

            for i in 0..seq {
                let row = &scores[i * seq..(i + 1) * seq];
                let mut maxv = f32::NEG_INFINITY;
                for &x in row {
                    if x > maxv {
                        maxv = x;
                    }
                }
                let mut sum = 0.0f32;
                let dst = &mut probs[i * seq..(i + 1) * seq];
                for j in 0..seq {
                    let p = (row[j] - maxv).exp();
                    dst[j] = p;
                    sum += p;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for p in dst.iter_mut() {
                    *p *= inv;
                }
            }

            sgemm_row_major(
                &probs,
                seq,
                seq,
                &vh,
                head_dim,
                false,
                false,
                1.0,
                0.0,
                &mut head_out,
            )?;

            for t in 0..seq {
                let dst = &mut out[t * hidden + offset..t * hidden + offset + head_dim];
                let src = &head_out[t * head_dim..(t + 1) * head_dim];
                dst.copy_from_slice(src);
            }
        }

        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (q, k, v, seq, num_heads, head_dim, scale, out);
        anyhow::bail!("sgemm attention unavailable on this platform")
    }
}

fn sgemm_row_major(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
    c: &mut [f32],
) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let (trans_a_flag, a_rows, a_cols) = if trans_a {
            (CBLAS_TRANS, k, m)
        } else {
            (CBLAS_NO_TRANS, m, k)
        };
        let (trans_b_flag, b_rows, b_cols) = if trans_b {
            (CBLAS_TRANS, n, k)
        } else {
            (CBLAS_NO_TRANS, k, n)
        };
        if a.len() != a_rows * a_cols {
            anyhow::bail!("sgemm A size mismatch: {} vs {}", a.len(), a_rows * a_cols);
        }
        if b.len() != b_rows * b_cols {
            anyhow::bail!("sgemm B size mismatch: {} vs {}", b.len(), b_rows * b_cols);
        }
        if c.len() != m * n {
            anyhow::bail!("sgemm C size mismatch: {} vs {}", c.len(), m * n);
        }
        let lda = if trans_a { m } else { k } as c_int;
        let ldb = if trans_b { k } else { n } as c_int;
        let ldc = n as c_int;
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                trans_a_flag,
                trans_b_flag,
                m as c_int,
                n as c_int,
                k as c_int,
                alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                beta,
                c.as_mut_ptr(),
                ldc,
            );
        }
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (a, m, k, b, n, trans_a, trans_b, alpha, beta, c);
        anyhow::bail!("sgemm unavailable on this platform")
    }
}

fn gelu_pytorch_tanh_inplace(x: &mut [f32]) {
    const A: f32 = 0.797_884_6;
    const B: f32 = 0.044_715;
    for v in x.iter_mut() {
        let t = A * (*v + B * *v * *v * *v);
        *v = 0.5 * *v * (1.0 + t.tanh());
    }
}

fn add_inplace(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn run_embed_f32(session: &mut Session, input_ids: &[i64]) -> Result<Array3<f32>> {
    let input = Array::from_shape_vec((1, input_ids.len()), input_ids.to_vec())?;
    let out0_name = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .unwrap_or_else(|| "inputs_embeds".to_string());
    let outputs: SessionOutputs = session.run(ort::inputs! {
        "input_ids" => Tensor::from_array(input)?,
    })?;
    let v = outputs
        .get("inputs_embeds")
        .or_else(|| outputs.get(&out0_name))
        .context("embed output missing")?;
    let arr = extract_array_f32(v)?;
    Ok(arr
        .into_dimensionality::<ndarray::Ix3>()
        .context("embed output shape")?)
}

fn run_embed_f16(session: &mut Session, input_ids: &[i64]) -> Result<Array3<f16>> {
    let input = Array::from_shape_vec((1, input_ids.len()), input_ids.to_vec())?;
    let out0_name = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .unwrap_or_else(|| "inputs_embeds".to_string());
    let outputs: SessionOutputs = session.run(ort::inputs! {
        "input_ids" => Tensor::from_array(input)?,
    })?;
    let v = outputs
        .get("inputs_embeds")
        .or_else(|| outputs.get(&out0_name))
        .context("embed output missing")?;
    let arr = if let Ok(a) = v.try_extract_array::<f16>() {
        a.to_owned()
    } else {
        extract_array_f32(v)?.mapv(f16::from_f32)
    };
    Ok(arr
        .into_dimensionality::<ndarray::Ix3>()
        .context("embed output shape")?)
}

struct DecoderOutF32 {
    logits: Array3<f32>,
    present: PastKeyValuesF32,
}

struct DecoderOutF16 {
    logits: Array3<f32>,
    present: PastKeyValuesF16,
}

fn run_decoder_f32(
    session: &mut Session,
    inputs_embeds: &Array3<f32>,
    attention_mask: &[i64],
    position_ids: &[i64],
    past: &PastKeyValuesF32,
) -> Result<DecoderOutF32> {
    let attn = Array::from_shape_vec((1, attention_mask.len()), attention_mask.to_vec())?;
    let pos = Array::from_shape_vec((1, position_ids.len()), position_ids.to_vec())?;

    let mut feed: Vec<(String, SessionInputValue)> = Vec::with_capacity(3 + past.tensors.len());
    feed.push((
        "inputs_embeds".to_string(),
        SessionInputValue::from(Tensor::from_array(inputs_embeds.to_owned())?),
    ));
    feed.push((
        "attention_mask".to_string(),
        SessionInputValue::from(Tensor::from_array(attn)?),
    ));
    feed.push((
        "position_ids".to_string(),
        SessionInputValue::from(Tensor::from_array(pos)?),
    ));

    let past_input_names: Vec<String> = session
        .inputs()
        .iter()
        .filter(|i| i.name().starts_with("past_key_values."))
        .map(|i| i.name().to_string())
        .collect();
    for (name, tensor) in past.as_named_tensors(&past_input_names)? {
        feed.push((name, SessionInputValue::from(tensor)));
    }

    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();

    let outputs: SessionOutputs = session.run(feed)?;

    let logits_val = outputs
        .get("logits")
        .or_else(|| output_names.first().and_then(|n| outputs.get(n)))
        .context("decoder output logits missing")?;
    let logits: ArrayD<f32> = extract_array_f32(logits_val)?;
    let logits: Array3<f32> = logits
        .into_dimensionality::<ndarray::Ix3>()
        .context("logits shape")?;

    let present = PastKeyValuesF32::from_outputs(&output_names, &outputs)?;
    Ok(DecoderOutF32 { logits, present })
}

fn run_decoder_f32_past_f16(
    session: &mut Session,
    inputs_embeds: &Array3<f32>,
    attention_mask: &[i64],
    position_ids: &[i64],
    past: &PastKeyValuesF16,
) -> Result<DecoderOutF16> {
    let attn = Array::from_shape_vec((1, attention_mask.len()), attention_mask.to_vec())?;
    let pos = Array::from_shape_vec((1, position_ids.len()), position_ids.to_vec())?;

    let mut feed: Vec<(String, SessionInputValue)> = Vec::with_capacity(3 + past.tensors.len());
    feed.push((
        "inputs_embeds".to_string(),
        SessionInputValue::from(Tensor::from_array(inputs_embeds.to_owned())?),
    ));
    feed.push((
        "attention_mask".to_string(),
        SessionInputValue::from(Tensor::from_array(attn)?),
    ));
    feed.push((
        "position_ids".to_string(),
        SessionInputValue::from(Tensor::from_array(pos)?),
    ));

    let past_input_names: Vec<String> = session
        .inputs()
        .iter()
        .filter(|i| i.name().starts_with("past_key_values."))
        .map(|i| i.name().to_string())
        .collect();
    for (name, tensor) in past.as_named_tensors(&past_input_names)? {
        feed.push((name, SessionInputValue::from(tensor)));
    }

    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();

    let outputs: SessionOutputs = session.run(feed)?;

    let logits_val = outputs
        .get("logits")
        .or_else(|| output_names.first().and_then(|n| outputs.get(n)))
        .context("decoder output logits missing")?;
    let logits: ArrayD<f32> = extract_array_f32(logits_val)?;
    let logits: Array3<f32> = logits
        .into_dimensionality::<ndarray::Ix3>()
        .context("logits shape")?;

    let present = PastKeyValuesF16::from_outputs(&output_names, &outputs)?;
    Ok(DecoderOutF16 { logits, present })
}

fn run_decoder_f16(
    session: &mut Session,
    inputs_embeds: &Array3<f16>,
    attention_mask: &[i64],
    position_ids: &[i64],
    past: &PastKeyValuesF16,
) -> Result<DecoderOutF16> {
    let attn = Array::from_shape_vec((1, attention_mask.len()), attention_mask.to_vec())?;
    let pos = Array::from_shape_vec((1, position_ids.len()), position_ids.to_vec())?;

    let mut feed: Vec<(String, SessionInputValue)> = Vec::with_capacity(3 + past.tensors.len());
    feed.push((
        "inputs_embeds".to_string(),
        SessionInputValue::from(Tensor::from_array(inputs_embeds.to_owned())?),
    ));
    feed.push((
        "attention_mask".to_string(),
        SessionInputValue::from(Tensor::from_array(attn)?),
    ));
    feed.push((
        "position_ids".to_string(),
        SessionInputValue::from(Tensor::from_array(pos)?),
    ));

    let past_input_names: Vec<String> = session
        .inputs()
        .iter()
        .filter(|i| i.name().starts_with("past_key_values."))
        .map(|i| i.name().to_string())
        .collect();
    for (name, tensor) in past.as_named_tensors(&past_input_names)? {
        feed.push((name, SessionInputValue::from(tensor)));
    }

    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();

    let outputs: SessionOutputs = session.run(feed)?;

    let logits_val = outputs
        .get("logits")
        .or_else(|| output_names.first().and_then(|n| outputs.get(n)))
        .context("decoder output logits missing")?;
    let logits: ArrayD<f32> = extract_array_f32(logits_val)?;
    let logits: Array3<f32> = logits
        .into_dimensionality::<ndarray::Ix3>()
        .context("logits shape")?;

    let present = PastKeyValuesF16::from_outputs(&output_names, &outputs)?;
    Ok(DecoderOutF16 { logits, present })
}

struct PastKeyValuesF32 {
    // layer0.key, layer0.value, layer1.key, layer1.value, ...
    tensors: Vec<ArrayD<f32>>,
}

impl PastKeyValuesF32 {
    fn zeros(batch: usize, num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let mut tensors = Vec::with_capacity(num_layers * 2);
        for _ in 0..num_layers {
            let k = ArrayD::<f32>::zeros(IxDyn(&[batch, num_kv_heads, 0, head_dim]));
            let v = ArrayD::<f32>::zeros(IxDyn(&[batch, num_kv_heads, 0, head_dim]));
            tensors.push(k);
            tensors.push(v);
        }
        Self { tensors }
    }

    fn as_named_tensors(&self, past_input_names: &[String]) -> Result<Vec<(String, Tensor<f32>)>> {
        if past_input_names.len() != self.tensors.len() {
            anyhow::bail!(
                "past tensors count mismatch: have {}, session expects {}",
                self.tensors.len(),
                past_input_names.len()
            );
        }
        let mut out: Vec<(String, Tensor<f32>)> = Vec::with_capacity(self.tensors.len());
        for (name, arr) in past_input_names.iter().cloned().zip(self.tensors.iter()) {
            out.push((name, Tensor::from_array(arr.to_owned())?));
        }
        Ok(out)
    }

    fn from_outputs(output_names: &[String], outputs: &SessionOutputs) -> Result<Self> {
        // Gather all present key/values in session output order after logits.
        let mut tensors = Vec::new();
        for name in output_names {
            if name == "logits" {
                continue;
            }
            let v = outputs.get(name).with_context(|| format!("missing decoder output {name}"))?;
            tensors.push(extract_array_f32(v)?);
        }
        Ok(Self { tensors })
    }
}

struct PastKeyValuesF16 {
    tensors: Vec<ArrayD<f16>>,
}

impl PastKeyValuesF16 {
    fn zeros(batch: usize, num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let mut tensors = Vec::with_capacity(num_layers * 2);
        for _ in 0..num_layers {
            let k = ArrayD::<f16>::from_shape_vec(
                IxDyn(&[batch, num_kv_heads, 0, head_dim]),
                Vec::<f16>::new(),
            )
            .expect("empty past k");
            let v = ArrayD::<f16>::from_shape_vec(
                IxDyn(&[batch, num_kv_heads, 0, head_dim]),
                Vec::<f16>::new(),
            )
            .expect("empty past v");
            tensors.push(k);
            tensors.push(v);
        }
        Self { tensors }
    }

    fn as_named_tensors(&self, past_input_names: &[String]) -> Result<Vec<(String, Tensor<f16>)>> {
        if past_input_names.len() != self.tensors.len() {
            anyhow::bail!(
                "past tensors count mismatch: have {}, session expects {}",
                self.tensors.len(),
                past_input_names.len()
            );
        }
        let mut out: Vec<(String, Tensor<f16>)> = Vec::with_capacity(self.tensors.len());
        for (name, arr) in past_input_names.iter().cloned().zip(self.tensors.iter()) {
            out.push((name, Tensor::from_array(arr.to_owned())?));
        }
        Ok(out)
    }

    fn from_outputs(output_names: &[String], outputs: &SessionOutputs) -> Result<Self> {
        let mut tensors = Vec::new();
        for name in output_names {
            if name == "logits" {
                continue;
            }
            let v = outputs.get(name).with_context(|| format!("missing decoder output {name}"))?;
            let arr = if let Ok(a) = v.try_extract_array::<f16>() {
                a.to_owned()
            } else {
                // Some quantized variants return f32 present K/V; convert down to f16 to match decoder inputs.
                extract_array_f32(v)?.mapv(f16::from_f32)
            };
            tensors.push(arr);
        }
        Ok(Self { tensors })
    }
}

fn extract_array_f32(v: &ort::value::Value) -> Result<ArrayD<f32>> {
    if let Ok(a) = v.try_extract_array::<f32>() {
        return Ok(a.to_owned());
    }
    let a = v.try_extract_array::<f16>().context("extract tensor as f16")?;
    Ok(a.mapv(|x| x.to_f32()))
}

fn sample_next_token(
    last_logits: ndarray::ArrayViewD<'_, f32>, // [1, vocab]
    temperature: f32,
    top_k: usize,
    banned_token_ids: &[i64],
    rng: &mut StdRng,
) -> Result<i64> {
    let last_logits = last_logits
        .into_dimensionality::<ndarray::Ix2>()
        .context("last logits shape")?;
    let row = last_logits.index_axis(Axis(0), 0);

    // Apply a hard ban on image-structure tokens. This prevents the model from "spilling" internal image markers
    // into the assistant response (common when the prompt packing differs slightly from HF processors).
    let mut row_logits: Vec<f32> = row.to_vec();
    for &tid in banned_token_ids {
        if tid >= 0 {
            let i = tid as usize;
            if i < row_logits.len() {
                row_logits[i] = f32::NEG_INFINITY;
            }
        }
    }

    if temperature <= 0.0 {
        let mut best_i = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in row_logits.iter().enumerate() {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        return Ok(best_i as i64);
    }

    let k = top_k.min(row.len()).max(1);
    let mut pairs: Vec<(usize, f32)> = row_logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    pairs.truncate(k);

    let max = pairs
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs = Vec::with_capacity(k);
    let mut sum = 0f32;
    for &(_, v) in &pairs {
        let p = ((v - max) / temperature).exp();
        probs.push(p);
        sum += p;
    }
    if sum == 0.0 || !sum.is_finite() {
        return Ok(pairs[0].0 as i64);
    }
    for p in &mut probs {
        *p /= sum;
    }

    let mut r = rng.gen::<f32>();
    for (i, p) in probs.iter().enumerate() {
        if r <= *p {
            return Ok(pairs[i].0 as i64);
        }
        r -= *p;
    }
    Ok(pairs[pairs.len() - 1].0 as i64)
}
