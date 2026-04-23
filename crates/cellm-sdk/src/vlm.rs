use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Instant;
use rayon::prelude::*;

use anyhow::{Context, Result};
use bytemuck::cast_slice;
use cellm_cache::{KVCache, PageTable};
use cellm_core::KvCacheLayout;
use cellm_kernels::metal::MetalBuffer;
use cellm_kernels::metal::MetalMatmul;
use cellm_kernels::MetalKernels;
use cellm_model::{gemma::GemmaRunner, llama::LlamaRunner, CellmFile};
use half::f16;
use image::RgbImage;
use ndarray::{Array2, Array3, Array4, Array5, Axis};
use rand::prelude::*;
use serde_json::Value;
use tokenizers::Tokenizer;
use crate::BackendKind;

#[derive(Debug, Clone, Copy)]
pub struct VlmRunConfig {
    pub backend: BackendKind,
    pub tokens_per_block: usize,
    pub top_k: usize,
    pub temperature: f32,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub max_new_tokens: usize,
    pub min_new_tokens: usize,
}

impl Default for VlmRunConfig {
    fn default() -> Self {
        Self {
            backend: BackendKind::Cpu,
            tokens_per_block: 16,
            top_k: 40,
            temperature: 0.7,
            seed: 1,
            repeat_penalty: 1.05,
            repeat_window: 64,
            max_new_tokens: 96,
            min_new_tokens: 16,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VlmTimingBreakdown {
    pub patch_ms: f64,
    pub encoder_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub encoder_layer_ms: Vec<f64>,
}

struct PreparedImage {
    pixel_values: Array5<f32>,
    gemma4_patch_values: Option<Array3<f32>>,
    gemma4_position_ids: Option<Array3<i32>>,
    gemma4_num_soft_tokens: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Copy)]
struct VlmProcessorHints {
    image_seq_len: Option<usize>,
    do_image_splitting: bool,
}

struct VisionLayerWeights {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    o_w: Vec<f32>,
    o_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
}

struct Gemma4VisionLayerWeights {
    input_ln_w: Vec<f32>,
    q_w: Vec<f32>,
    k_w: Vec<f32>,
    v_w: Vec<f32>,
    o_w: Vec<f32>,
    q_norm_w: Vec<f32>,
    k_norm_w: Vec<f32>,
    post_attn_ln_w: Vec<f32>,
    pre_ffn_ln_w: Vec<f32>,
    gate_w: Vec<f32>,
    up_w: Vec<f32>,
    down_w: Vec<f32>,
    post_ffn_ln_w: Vec<f32>,
    q_clip_in: Option<(f32, f32)>,
    q_clip_out: Option<(f32, f32)>,
    k_clip_in: Option<(f32, f32)>,
    k_clip_out: Option<(f32, f32)>,
    v_clip_in: Option<(f32, f32)>,
    v_clip_out: Option<(f32, f32)>,
    o_clip_in: Option<(f32, f32)>,
    o_clip_out: Option<(f32, f32)>,
    gate_clip_in: Option<(f32, f32)>,
    gate_clip_out: Option<(f32, f32)>,
    up_clip_in: Option<(f32, f32)>,
    up_clip_out: Option<(f32, f32)>,
    down_clip_in: Option<(f32, f32)>,
    down_clip_out: Option<(f32, f32)>,
}

pub fn describe_image_with_cellm(
    model_path: &Path,
    image_bytes: &[u8],
    user_prompt: &str,
    cfg: VlmRunConfig,
) -> Result<String> {
    let (text, _) = describe_image_with_cellm_timed(model_path, image_bytes, user_prompt, cfg)?;
    Ok(text)
}

pub fn describe_image_with_cellm_timed(
    model_path: &Path,
    image_bytes: &[u8],
    user_prompt: &str,
    cfg: VlmRunConfig,
) -> Result<(String, VlmTimingBreakdown)> {
    let total_start = Instant::now();
    let tokenizer_path = resolve_tokenizer_path(model_path)?;
    let tok = load_tokenizer(&tokenizer_path)?;
    let processor_hints = load_vlm_processor_hints(model_path, &tokenizer_path, &tok);
    let file = CellmFile::load(model_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let is_gemma4_vision = file
        .tensor_index("model.vision_tower.patch_embedder.input_proj.weight")
        .is_some()
        && file
            .tensor_index("model.embed_vision.embedding_projection.weight")
            .is_some();
    if !is_gemma4_vision {
        anyhow::bail!("This model does not support image input");
    }
    let model_type = effective_text_model_type(&file.header);
    let gemma4_mode = std::env::var("CELLM_VLM_GEMMA4_MODE").unwrap_or_else(|_| "placeholder".to_string());
    let gemma4_prefix_image = is_gemma4_vision && model_type.starts_with("gemma4")
        && gemma4_mode.eq_ignore_ascii_case("prefix");
    let image_input = preprocess_image_for_model(
        image_bytes,
        processor_hints.do_image_splitting,
        is_gemma4_vision,
        processor_hints.image_seq_len.unwrap_or(280),
    )?;
    let num_images = image_input.pixel_values.shape()[1];
    let (mut image_features, image_seq_len, patch_ms, encoder_ms, encoder_layer_ms) =
        run_vision_cellm(
            &file,
            &image_input,
            cfg.backend,
            processor_hints.image_seq_len,
        )?;
    if std::env::var("CELLM_VLM_DEBUG_FEATURE_STATS").is_ok() {
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut n = 0usize;
        for v in image_features.iter().copied() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            sum += v as f64;
            sumsq += (v as f64) * (v as f64);
            n += 1;
        }
        let mean = if n > 0 { sum / n as f64 } else { 0.0 };
        let var = if n > 0 {
            (sumsq / n as f64) - mean * mean
        } else {
            0.0
        };
        eprintln!(
            "CELLM_VLM_DEBUG_FEATURE_STATS rows={} cols={} min={:.6} max={:.6} mean={:.6} std={:.6}",
            image_features.shape()[0],
            image_features.shape()[1],
            min_v,
            max_v,
            mean,
            var.max(0.0).sqrt()
        );
    }
    let (image_token_id, image_token_text, use_idefics_wrappers, boi_token_text, eoi_token_text) =
        resolve_image_token(&tok)?;
    let eos_token_id = file
        .header
        .eos_token_id
        .map(|v| v as i64)
        .or_else(|| {
            file.header
                .source_text_config
                .as_ref()
                .and_then(|v| v.get("eos_token_id"))
                .and_then(|v| v.as_i64())
        })
        .unwrap_or(2);
    // For Gemma4, <turn|> is end-of-turn and also effectively EOS for generation.
    let end_of_utt = tok.token_to_id("<turn|>")
        .map(|v| v as i64)
        .or_else(|| tok.token_to_id("<end_of_utterance>").map(|v| v as i64));
    let banned_token_ids = banned_token_ids(&tok);

    if let Some(expected) = processor_hints.image_seq_len {
        if expected != image_seq_len {
            anyhow::bail!(
                "processor/image_seq_len mismatch: processor_config={} vision_projector={}",
                expected,
                image_seq_len
            );
        }
    }
    let image_block = if gemma4_prefix_image {
        String::new()
    } else {
        format_image_block(
            num_images,
            image_seq_len,
            image_token_text.as_str(),
            use_idefics_wrappers,
            boi_token_text.as_deref(),
            eoi_token_text.as_deref(),
        )
    };
    let prompt = if gemma4_prefix_image {
        if image_block.is_empty() {
            user_prompt.to_string()
        } else {
            format!("{image_block}\n{user_prompt}")
        }
    } else {
        build_single_turn_prompt(
            user_prompt,
            &image_block,
            tok.token_to_id("<|turn>").is_some() && tok.token_to_id("<turn|>").is_some(),
        )
    };
    let enc_ids = encode_with_explicit_added_tokens(&tok, &tokenizer_path, &prompt)?;
    let mut input_ids: Vec<i64> = enc_ids.into_iter().map(|x| x as i64).collect();
    if gemma4_prefix_image {
        if let Some(bos) = file.header.bos_token_id.map(|v| v as i64) {
            if input_ids.first().copied() != Some(bos) {
                input_ids.insert(0, bos);
            }
        }
    }

    let (generated, decode_ms) = run_decode_cellm(
        model_path,
        image_token_id,
        eos_token_id,
        end_of_utt,
        cfg,
        input_ids,
        &image_features,
        gemma4_prefix_image,
        &banned_token_ids,
    )?;
    let text = tok
        .decode(
            &generated.into_iter().map(|t| t as u32).collect::<Vec<u32>>(),
            true,
        )
        .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;
    let timing = VlmTimingBreakdown {
        patch_ms,
        encoder_ms,
        decode_ms,
        total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        encoder_layer_ms,
    };
    Ok((text.trim().to_string(), timing))
}

/// Transcribe / describe audio using the Gemma 4 audio tower.
pub fn describe_audio_with_cellm_timed(
    model_path: &Path,
    audio_bytes: &[u8],
    user_prompt: &str,
    cfg: VlmRunConfig,
) -> Result<(String, VlmTimingBreakdown)> {
    let total_start = Instant::now();
    let tokenizer_path = resolve_tokenizer_path(model_path)?;
    let tok = load_tokenizer(&tokenizer_path)?;
    let file = CellmFile::load(model_path).map_err(|e| anyhow::anyhow!("{e}"))?;

    if file.tensor_index("model.audio_tower.output_proj.weight").is_none() {
        anyhow::bail!("Model does not have an audio tower");
    }

    // Resolve audio token IDs from the tokenizer
    let audio_token_id: i64 = tok
        .token_to_id("<|audio|>")
        .map(|id| id as i64)
        .unwrap_or(258881);
    let boa_str = if tok.token_to_id("<|audio>").is_some() { "<|audio>" } else { "" };
    let eoa_str = if tok.token_to_id("<audio|>").is_some() { "<audio|>" } else { "" };

    // Parse WAV → PCM i16 samples
    let t_mel = Instant::now();
    let (pcm_i16, sample_rate) = parse_wav_pcm16(audio_bytes)?;
    if sample_rate != 16000 {
        anyhow::bail!("Audio must be 16 kHz, got {} Hz. Resample before calling.", sample_rate);
    }

    // Compute mel spectrogram: [T_frames, 128]
    let (mel_features, t_frames) = compute_mel_spectrogram(
        &pcm_i16,
        320,     // frame_length
        160,     // hop_length
        512,     // fft_length
        128,     // num_mel_bins
        0.0f64,  // min_frequency
        8000.0f64, // max_frequency
        0.001f32,  // mel_floor
    );
    let mel_ms = t_mel.elapsed().as_secs_f64() * 1000.0;

    // Run the audio conformer encoder
    let t_enc = Instant::now();
    let audio_features = run_audio_cellm_gemma4(&file, &mel_features, t_frames, 128)?;
    let encoder_ms = t_enc.elapsed().as_secs_f64() * 1000.0;
    let n_audio_tokens = audio_features.shape()[0];

    if std::env::var("CELLM_VLM_DEBUG_FEATURE_STATS").is_ok() {
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let n = audio_features.len();
        for &v in audio_features.iter() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            sum += v as f64;
            sumsq += (v as f64) * (v as f64);
        }
        let mean = sum / n as f64;
        let std = ((sumsq / n as f64) - mean * mean).max(0.0).sqrt();
        eprintln!(
            "CELLM_AUDIO_FEATURE_STATS tokens={} cols={} min={:.6} max={:.6} mean={:.6} std={:.6}",
            n_audio_tokens, audio_features.shape()[1], min_v, max_v, mean, std
        );
    }

    // Build prompt
    // Gemma4 chat template for audio: <|turn>user\n{audio_tokens}{user_text}<turn|>\n<|turn>model\n
    // (no extra newlines around audio, unlike image which uses \n\n<|image|>\n\n)
    let audio_token_text = tok
        .id_to_token(audio_token_id as u32)
        .unwrap_or_else(|| "<|audio|>".to_string());
    let audio_placeholders = audio_token_text.repeat(n_audio_tokens);
    let audio_block = format!("{boa_str}{audio_placeholders}{eoa_str}");
    let use_gemma4_turn = tok.token_to_id("<|turn>").is_some() && tok.token_to_id("<turn|>").is_some();
    let prompt = if use_gemma4_turn {
        // Template: bos + <|turn>user\n + audio_tokens + user_text + <turn|>\n + <|turn>model\n
        format!("<bos><|turn>user\n{audio_block}{user_prompt}<turn|>\n<|turn>model\n")
    } else {
        build_single_turn_prompt(user_prompt, &audio_block, false)
    };

    let enc_ids = encode_with_explicit_added_tokens(&tok, &tokenizer_path, &prompt)?;
    let input_ids: Vec<i64> = enc_ids.into_iter().map(|x| x as i64).collect();

    if std::env::var("CELLM_AUDIO_DEBUG").is_ok() {
        let n_audio_in_ids = input_ids.iter().filter(|&&t| t == audio_token_id).count();
        eprintln!(
            "AUDIO_DECODE_DEBUG prompt={:?} n_input_ids={} audio_token_id={} n_audio_in_ids={} n_audio_features={}",
            &prompt[..prompt.len().min(200)],
            input_ids.len(), audio_token_id, n_audio_in_ids, n_audio_tokens
        );
        eprintln!("  first 10 input_ids: {:?}", &input_ids[..input_ids.len().min(10)]);
        eprintln!("  last 10 input_ids: {:?}", &input_ids[input_ids.len().saturating_sub(10)..]);
    }

    let eos_token_id = file
        .header
        .eos_token_id
        .map(|v| v as i64)
        .or_else(|| {
            file.header
                .source_text_config
                .as_ref()
                .and_then(|v| v.get("eos_token_id"))
                .and_then(|v| v.as_i64())
        })
        .unwrap_or(2);
    // For Gemma4, <turn|> = end-of-turn is also effectively EOS for generation.
    let end_of_utt = tok.token_to_id("<turn|>")
        .map(|v| v as i64)
        .or_else(|| tok.token_to_id("<end_of_utterance>").map(|v| v as i64));
    let banned = banned_token_ids(&tok);

    let (generated, decode_ms) = run_decode_cellm(
        model_path,
        audio_token_id,
        eos_token_id,
        end_of_utt,
        cfg,
        input_ids,
        &audio_features,
        false, // never prefix; always bidir
        &banned,
    )?;
    let text = tok
        .decode(
            &generated.into_iter().map(|t| t as u32).collect::<Vec<u32>>(),
            true,
        )
        .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

    let timing = VlmTimingBreakdown {
        patch_ms: mel_ms,
        encoder_ms,
        decode_ms,
        total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        encoder_layer_ms: vec![],
    };
    Ok((text.trim().to_string(), timing))
}

fn run_decode_cellm(
    model_path: &Path,
    image_token_id: i64,
    eos_token_id: i64,
    end_of_utterance_id: Option<i64>,
    cfg: VlmRunConfig,
    input_ids: Vec<i64>,
    image_features: &Array2<f32>,
    prefix_image_features: bool,
    banned_token_ids: &[i64],
) -> Result<(Vec<i64>, f64)> {
    let decode_start = Instant::now();
    enum DecodeRunner {
        Llama(LlamaRunner),
        Gemma(GemmaRunner),
    }

    let file = CellmFile::load(model_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let model_type = effective_text_model_type(&file.header);
    let mut runner = match model_type.as_str() {
        "llama" | "smollm3" => {
            DecodeRunner::Llama(LlamaRunner::load(model_path).map_err(|e| anyhow::anyhow!("{e}"))?)
        }
        t if t.starts_with("gemma") => {
            DecodeRunner::Gemma(GemmaRunner::load(model_path).map_err(|e| anyhow::anyhow!("{e}"))?)
        }
        other => anyhow::bail!("unsupported text model for VLM decode: {other}"),
    };

    // Enable Metal acceleration for the text decode path when requested.
    // Without this the decode runner uses CPU-only math, which makes
    // multimodal inference with 100+ audio/image tokens impractically slow.
    if cfg.backend == BackendKind::Metal {
        match &mut runner {
            DecodeRunner::Llama(r) => { r.enable_metal_full_backend(); }
            DecodeRunner::Gemma(r) => { r.enable_metal_full_backend(); }
        }
    }

    let (hidden, num_layers, num_kv_heads, num_heads) = match &runner {
        DecodeRunner::Llama(r) => {
            let c = r.config();
            (c.hidden_size, c.num_hidden_layers, c.num_key_value_heads, c.num_attention_heads)
        }
        DecodeRunner::Gemma(r) => {
            let c = r.config();
            (c.hidden_size, c.num_hidden_layers, c.num_key_value_heads, c.num_attention_heads)
        }
    };
    if image_features.shape()[1] != hidden {
        anyhow::bail!(
            "image feature dim {} != text hidden {}",
            image_features.shape()[1],
            hidden
        );
    }

    let head_dim = if model_type.starts_with("gemma") {
        infer_gemma_kv_head_dim(&file)?
    } else {
        hidden / num_heads.max(1)
    };
    let prefix_tokens = if prefix_image_features {
        image_features.shape()[0]
    } else {
        0
    };
    let total_tokens = prefix_tokens + input_ids.len() + cfg.max_new_tokens + 8;
    let total_blocks = (total_tokens + cfg.tokens_per_block - 1) / cfg.tokens_per_block;
    let layout = KvCacheLayout {
        total_blocks,
        tokens_per_block: cfg.tokens_per_block,
        num_layers,
        num_kv_heads,
        head_dim,
    };
    let kv_storage_kind = if cfg.backend == BackendKind::Metal {
        cellm_cache::KvStorageKind::Metal
    } else {
        cellm_cache::KvStorageKind::Cpu
    };
    let mut kv_cache = KVCache::new_with_kind(layout, kv_storage_kind)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut page_table =
        PageTable::new(1, cfg.tokens_per_block).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut image_idx = 0usize;
    let mut x = vec![0.0f32; hidden];
    let mut rng = StdRng::seed_from_u64(cfg.seed.max(1));
    let mut recent: Vec<u32> = Vec::new();
    let mut next: i64 = 0;
    let is_gemma4_text = model_type.starts_with("gemma4");
    let enable_gemma4_image_bidir = std::env::var("CELLM_VLM_GEMMA4_BIDIR_IMAGE")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no")))
        .unwrap_or(true);

    let mut pos = 0usize;
    if prefix_image_features {
        for i in 0..image_features.shape()[0] {
            if std::env::var("CELLM_VLM_ZERO_IMAGE_FEATURES").is_ok() {
                x.fill(0.0);
            } else {
                let src = image_features.index_axis(Axis(0), i);
                x.copy_from_slice(src.as_slice().context("vision feature row not contiguous")?);
            }
            let cand = match &mut runner {
                DecodeRunner::Llama(r) => r
                    .step_topk_from_hidden(&x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                    .map_err(|e| anyhow::anyhow!("{e}"))?,
                DecodeRunner::Gemma(r) => r
                    .step_topk_from_hidden(&x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                    .map_err(|e| anyhow::anyhow!("{e}"))?,
            };
            next = sample_from_candidates(
                &cand,
                cfg.temperature,
                cfg.repeat_penalty,
                cfg.repeat_window,
                banned_token_ids,
                &recent,
                &mut rng,
            )?;
            pos += 1;
            image_idx += 1;
        }
    }
    if !prefix_image_features && is_gemma4_text && enable_gemma4_image_bidir {
        let img_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| if t == image_token_id { Some(i) } else { None })
            .collect();
        let mut image_pos_to_feature = vec![None; input_ids.len()];
        for (feat_idx, &p) in img_positions.iter().enumerate() {
            if p < image_pos_to_feature.len() {
                image_pos_to_feature[p] = Some(feat_idx);
            }
        }
        if let (Some(&img_start), Some(&img_end)) = (img_positions.first(), img_positions.last()) {
            // Phase A: causal prefill before image block.
            for &tok_id in input_ids.iter().take(img_start) {
                match &runner {
                    DecodeRunner::Llama(r) => r
                        .embed_token_hidden(tok_id as u32, &mut x)
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                    DecodeRunner::Gemma(r) => r
                        .embed_token_hidden(tok_id as u32, &mut x)
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                }
                recent.push(tok_id as u32);
                let cand = match &mut runner {
                    DecodeRunner::Llama(r) => r
                        .step_topk_from_hidden(&x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                    DecodeRunner::Gemma(r) => r
                        .step_topk_from_hidden_with_token(tok_id as u32, &x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                };
                next = sample_from_candidates(
                    &cand,
                    cfg.temperature,
                    cfg.repeat_penalty,
                    cfg.repeat_window,
                    banned_token_ids,
                    &recent,
                    &mut rng,
                )?;
                pos += 1;
            }

            // Reserve all image positions so replays can use pos < token_count.
            while page_table.token_count() < (img_end + 1) {
                page_table
                    .append_token(kv_cache.allocator_mut())
                    .map_err(|e| anyhow::anyhow!("gemma image reserve append failed: {e}"))?;
            }

            // Phase B1: fill image K/V once.
            for p in img_start..=img_end {
                if let Some(feat_idx) = image_pos_to_feature[p] {
                    if feat_idx >= image_features.shape()[0] {
                        anyhow::bail!("image token/feature mismatch at pos={p}");
                    }
                    if std::env::var("CELLM_VLM_ZERO_IMAGE_FEATURES").is_ok() {
                        x.fill(0.0);
                    } else {
                        let src = image_features.index_axis(Axis(0), feat_idx);
                        x.copy_from_slice(src.as_slice().context("vision feature row not contiguous")?);
                    }
                    let _ = match &mut runner {
                        DecodeRunner::Llama(r) => r
                            .step_topk_from_hidden(&x, p, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                            .map_err(|e| anyhow::anyhow!("{e}"))?,
                        DecodeRunner::Gemma(r) => r
                            .step_topk_from_hidden(&x, p, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                            .map_err(|e| anyhow::anyhow!("{e}"))?,
                    };
                }
            }

            // Phase B2: replay image block with all image K/V present.
            for p in img_start..=img_end {
                if let Some(feat_idx) = image_pos_to_feature[p] {
                    if feat_idx >= image_features.shape()[0] {
                        anyhow::bail!("image token/feature mismatch at pos={p}");
                    }
                    if std::env::var("CELLM_VLM_ZERO_IMAGE_FEATURES").is_ok() {
                        x.fill(0.0);
                    } else {
                        let src = image_features.index_axis(Axis(0), feat_idx);
                        x.copy_from_slice(src.as_slice().context("vision feature row not contiguous")?);
                    }
                    let _ = match &mut runner {
                        DecodeRunner::Llama(r) => r
                            .step_topk_from_hidden(&x, p, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                            .map_err(|e| anyhow::anyhow!("{e}"))?,
                        DecodeRunner::Gemma(r) => r
                            .step_topk_from_hidden(&x, p, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                            .map_err(|e| anyhow::anyhow!("{e}"))?,
                    };
                    recent.push(image_token_id as u32);
                }
            }
            image_idx = img_positions.len();
            pos = img_end + 1;

            // Phase C: causal prefill after image block.
            for &tok_id in input_ids.iter().skip(img_end + 1) {
                match &runner {
                    DecodeRunner::Llama(r) => r
                        .embed_token_hidden(tok_id as u32, &mut x)
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                    DecodeRunner::Gemma(r) => r
                        .embed_token_hidden(tok_id as u32, &mut x)
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                }
                recent.push(tok_id as u32);
                let cand = match &mut runner {
                    DecodeRunner::Llama(r) => r
                        .step_topk_from_hidden(&x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                    DecodeRunner::Gemma(r) => r
                        .step_topk_from_hidden_with_token(tok_id as u32, &x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                };
                next = sample_from_candidates(
                    &cand,
                    cfg.temperature,
                    cfg.repeat_penalty,
                    cfg.repeat_window,
                    banned_token_ids,
                    &recent,
                    &mut rng,
                )?;
                pos += 1;
            }
        } else {
            // No image placeholders found, fall back to standard loop.
            for &tok_id in input_ids.iter() {
                match &runner {
                    DecodeRunner::Llama(r) => r
                        .embed_token_hidden(tok_id as u32, &mut x)
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                    DecodeRunner::Gemma(r) => r
                        .embed_token_hidden(tok_id as u32, &mut x)
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                }
                recent.push(tok_id as u32);
                let cand = match &mut runner {
                    DecodeRunner::Llama(r) => r
                        .step_topk_from_hidden(&x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                    DecodeRunner::Gemma(r) => r
                        .step_topk_from_hidden_with_token(tok_id as u32, &x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                };
                next = sample_from_candidates(
                    &cand,
                    cfg.temperature,
                    cfg.repeat_penalty,
                    cfg.repeat_window,
                    banned_token_ids,
                    &recent,
                    &mut rng,
                )?;
                pos += 1;
            }
        }
    } else {
        for &tok_id in input_ids.iter() {
            if !prefix_image_features && tok_id == image_token_id {
                if image_idx >= image_features.shape()[0] {
                    anyhow::bail!("image token count mismatch: prompt has more <image> tokens than vision features");
                }
                if std::env::var("CELLM_VLM_ZERO_IMAGE_FEATURES").is_ok() {
                    x.fill(0.0);
                } else {
                    let src = image_features.index_axis(Axis(0), image_idx);
                    x.copy_from_slice(src.as_slice().context("vision feature row not contiguous")?);
                }
                image_idx += 1;
            } else {
                match &runner {
                    DecodeRunner::Llama(r) => r
                        .embed_token_hidden(tok_id as u32, &mut x)
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                    DecodeRunner::Gemma(r) => r
                        .embed_token_hidden(tok_id as u32, &mut x)
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                }
                recent.push(tok_id as u32);
            }
            let cand = match &mut runner {
                DecodeRunner::Llama(r) => r
                    .step_topk_from_hidden(&x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                    .map_err(|e| anyhow::anyhow!("{e}"))?,
                DecodeRunner::Gemma(r) => {
                    let is_img = !prefix_image_features && tok_id == image_token_id;
                    if is_img {
                        r.step_topk_from_hidden(&x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                    } else {
                        r.step_topk_from_hidden_with_token(tok_id as u32, &x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                    }
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                },
            };
            next = sample_from_candidates(
                &cand,
                cfg.temperature,
                cfg.repeat_penalty,
                cfg.repeat_window,
                banned_token_ids,
                &recent,
                &mut rng,
            )?;
            pos += 1;
        }
    }
    if image_idx != image_features.shape()[0] {
        anyhow::bail!(
            "image token count mismatch: prompt has fewer <image> tokens ({image_idx}) than vision features ({})",
            image_features.shape()[0]
        );
    }

    let debug_gen = std::env::var("CELLM_GEN_DEBUG").is_ok();
    if debug_gen { eprintln!("GEN_DEBUG: first token (from prefill) = {}", next); }
    let mut generated = Vec::new();
    let mut same_token_run = 0usize;
    let mut last_token: Option<i64> = None;
    for step in 0..cfg.max_new_tokens {
        generated.push(next);
        if debug_gen { eprintln!("GEN_DEBUG[{}] = {}", step, next); }
        if Some(next) == last_token {
            same_token_run += 1;
        } else {
            same_token_run = 1;
            last_token = Some(next);
        }
        if (next == eos_token_id || end_of_utterance_id == Some(next))
            && (step + 1) >= cfg.min_new_tokens
        {
            break;
        }
        if same_token_run >= 6 && (step + 1) >= cfg.min_new_tokens {
            break;
        }
        if is_alternating_two_token_loop(&generated, 10) && (step + 1) >= cfg.min_new_tokens {
            break;
        }
        match &runner {
            DecodeRunner::Llama(r) => r
                .embed_token_hidden(next as u32, &mut x)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            DecodeRunner::Gemma(r) => r
                .embed_token_hidden(next as u32, &mut x)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        }
        let pos = pos + step;
        let cand = match &mut runner {
            DecodeRunner::Llama(r) => r
                .step_topk_from_hidden(&x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            DecodeRunner::Gemma(r) => r
                .step_topk_from_hidden_with_token(next as u32, &x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        };
        next = sample_from_candidates(
            &cand,
            cfg.temperature,
            cfg.repeat_penalty,
            cfg.repeat_window,
            banned_token_ids,
            &recent,
            &mut rng,
        )?;
        recent.push(generated[generated.len() - 1] as u32);
    }

    Ok((generated, decode_start.elapsed().as_secs_f64() * 1000.0))
}

fn effective_text_model_type(header: &cellm_model::CellmHeader) -> String {
    let mut mt = header.model_type.clone();
    if mt == "llava" || mt == "llava_qwen" || mt == "llava_idefics3" || mt == "llava_smolvlm" || mt == "vlm" {
        if let Some(st) = header
            .source_text_config
            .as_ref()
            .and_then(|v| v.get("model_type"))
            .and_then(|v| v.as_str())
        {
            mt = st.to_string();
        } else if let Some(archs) = header.source_architectures.as_ref() {
            if archs.iter().any(|a| a.to_lowercase().contains("qwen")) {
                mt = "qwen".to_string();
            } else if archs.iter().any(|a| a.to_lowercase().contains("llama")) {
                mt = "llama".to_string();
            }
        }
    }
    mt
}

fn infer_gemma_kv_head_dim(file: &CellmFile) -> Result<usize> {
    let kv_heads = file.header.num_kv_heads.max(1);
    let mut max_head_dim = 0usize;
    for t in &file.header.tensors {
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
        anyhow::bail!("failed to infer gemma kv head dim from k_proj weights")
    }
}

fn is_alternating_two_token_loop(tokens: &[i64], min_len: usize) -> bool {
    if tokens.len() < min_len || min_len < 4 {
        return false;
    }
    let recent = &tokens[tokens.len() - min_len..];
    let a = recent[0];
    let b = recent[1];
    if a == b {
        return false;
    }
    for (i, &tok) in recent.iter().enumerate() {
        let expect = if i % 2 == 0 { a } else { b };
        if tok != expect {
            return false;
        }
    }
    true
}

fn sample_from_candidates(
    candidates: &[(u32, f32)],
    temperature: f32,
    repeat_penalty: f32,
    repeat_window: usize,
    banned_token_ids: &[i64],
    recent: &[u32],
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
    if repeat_penalty > 1.0 && repeat_window > 0 && !recent.is_empty() {
        let start = recent.len().saturating_sub(repeat_window);
        for (id, score) in &mut filtered {
            if recent[start..].contains(id) {
                *score /= repeat_penalty;
            }
        }
    }
    filtered.sort_by(|a, b| b.1.total_cmp(&a.1));

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
    Ok(filtered.last().expect("filtered non-empty").0 as i64)
}

fn run_vision_cellm(
    file: &CellmFile,
    image_input: &PreparedImage,
    backend: BackendKind,
    target_image_seq_len: Option<usize>,
) -> Result<(Array2<f32>, usize, f64, f64, Vec<f64>)> {
    if file
        .tensor_index("model.vision_tower.patch_embedder.input_proj.weight")
        .is_some()
        && file
            .tensor_index("model.embed_vision.embedding_projection.weight")
            .is_some()
    {
        return run_vision_cellm_gemma4(file, image_input, target_image_seq_len, backend);
    }
    let pixel_values = image_input.pixel_values.clone();

    let mut linear_backend = match backend {
        BackendKind::Metal => {
            let ctx = MetalKernels::create_matmul()
                .map_err(|e| anyhow::anyhow!("VLM Metal backend requested but unavailable: {e}"))?;
            LinearBackend::Metal {
                ctx,
                weight_t_cache: HashMap::new(),
            }
        }
        BackendKind::Cpu => LinearBackend::Cpu,
    };
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

    let patch_w_shape = tensor_shape(file, &patch_w_name)?;
    let patch_b_shape = tensor_shape(file, &patch_b_name)?;
    let pos_shape = tensor_shape(file, &pos_name)?;
    let proj_shape = tensor_shape(file, &proj_name)?;

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

    let patch_w = tensor_to_f32(file, &patch_w_name)?;
    let patch_b = tensor_to_f32(file, &patch_b_name)?;
    let pos = tensor_to_f32(file, &pos_name)?;
    let post_ln_w = tensor_to_f32(file, &ln_w_name)?;
    let post_ln_b = tensor_to_f32(file, &ln_b_name)?;
    let proj = tensor_to_f32(file, &proj_name)?;
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
        .unwrap_or_else(|| infer_vision_num_layers(file, &vision_prefix));
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

    let mut layers = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let prefix = format!("{vision_prefix}encoder.layers.{layer}.");
        layers.push(VisionLayerWeights {
            ln1_w: tensor_to_f32(file, &format!("{prefix}layer_norm1.weight"))?,
            ln1_b: tensor_to_f32(file, &format!("{prefix}layer_norm1.bias"))?,
            q_w: tensor_to_f32(file, &format!("{prefix}self_attn.q_proj.weight"))?,
            q_b: tensor_to_f32(file, &format!("{prefix}self_attn.q_proj.bias"))?,
            k_w: tensor_to_f32(file, &format!("{prefix}self_attn.k_proj.weight"))?,
            k_b: tensor_to_f32(file, &format!("{prefix}self_attn.k_proj.bias"))?,
            v_w: tensor_to_f32(file, &format!("{prefix}self_attn.v_proj.weight"))?,
            v_b: tensor_to_f32(file, &format!("{prefix}self_attn.v_proj.bias"))?,
            o_w: tensor_to_f32(file, &format!("{prefix}self_attn.out_proj.weight"))?,
            o_b: tensor_to_f32(file, &format!("{prefix}self_attn.out_proj.bias"))?,
            ln2_w: tensor_to_f32(file, &format!("{prefix}layer_norm2.weight"))?,
            ln2_b: tensor_to_f32(file, &format!("{prefix}layer_norm2.bias"))?,
            fc1_w: tensor_to_f32(file, &format!("{prefix}mlp.fc1.weight"))?,
            fc1_b: tensor_to_f32(file, &format!("{prefix}mlp.fc1.bias"))?,
            fc2_w: tensor_to_f32(file, &format!("{prefix}mlp.fc2.weight"))?,
            fc2_b: tensor_to_f32(file, &format!("{prefix}mlp.fc2.bias"))?,
        });
    }

    let mut patch_ms = 0.0f64;
    let mut encoder_ms = 0.0f64;
    let mut encoder_layer_ms = vec![0.0f64; num_layers];

    for img in 0..nimg {
        let patch_start = Instant::now();
        patch_embed_rows(
            &pixel_values,
            img,
            grid,
            patch,
            hidden,
            &patch_w,
            &patch_b,
            &pos,
            &mut tokens,
            &mut linear_backend,
        );
        patch_ms += patch_start.elapsed().as_secs_f64() * 1000.0;

        let encoder_start = Instant::now();
        for (layer_idx, layer) in layers.iter().enumerate() {
            let layer_start = Instant::now();
            layer_norm_rows(
                &tokens, num_tokens, hidden, &layer.ln1_w, &layer.ln1_b, eps, &mut norm1,
            );
            linear_rows(
                &norm1,
                num_tokens,
                hidden,
                &layer.q_w,
                hidden,
                Some(&layer.q_b),
                &mut q,
                &mut linear_backend,
            );
            linear_rows(
                &norm1,
                num_tokens,
                hidden,
                &layer.k_w,
                hidden,
                Some(&layer.k_b),
                &mut k,
                &mut linear_backend,
            );
            linear_rows(
                &norm1,
                num_tokens,
                hidden,
                &layer.v_w,
                hidden,
                Some(&layer.v_b),
                &mut v,
                &mut linear_backend,
            );
            self_attention_full(
                &q,
                &k,
                &v,
                num_tokens,
                num_heads,
                head_dim,
                None,
                &mut score_buf,
                &mut prob_buf,
                &mut attn,
                None,
                &mut linear_backend,
            );
            linear_rows(
                &attn,
                num_tokens,
                hidden,
                &layer.o_w,
                hidden,
                Some(&layer.o_b),
                &mut proj_out,
                &mut linear_backend,
            );
            add_inplace(&mut tokens, &proj_out);

            layer_norm_rows(
                &tokens, num_tokens, hidden, &layer.ln2_w, &layer.ln2_b, eps, &mut norm2,
            );
            linear_rows(
                &norm2,
                num_tokens,
                hidden,
                &layer.fc1_w,
                intermediate,
                Some(&layer.fc1_b),
                &mut mlp_up,
                &mut linear_backend,
            );
            gelu_pytorch_tanh_inplace(&mut mlp_up);
            linear_rows(
                &mlp_up,
                num_tokens,
                intermediate,
                &layer.fc2_w,
                hidden,
                Some(&layer.fc2_b),
                &mut mlp_out,
                &mut linear_backend,
            );
            add_inplace(&mut tokens, &mlp_out);
            encoder_layer_ms[layer_idx] += layer_start.elapsed().as_secs_f64() * 1000.0;
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
                            for &val in src {
                                acc += val * proj_row[proj_i];
                                proj_i += 1;
                            }
                        }
                    }
                    out[[img * out_tokens_per_image + out_tok, o]] = acc;
                }
            }
        }
        encoder_ms += encoder_start.elapsed().as_secs_f64() * 1000.0;
    }

    Ok((out, out_tokens_per_image, patch_ms, encoder_ms, encoder_layer_ms))
}

fn run_vision_cellm_gemma4(
    file: &CellmFile,
    image_input: &PreparedImage,
    target_image_seq_len: Option<usize>,
    backend: BackendKind,
) -> Result<(Array2<f32>, usize, f64, f64, Vec<f64>)> {
    let mut linear_backend = match backend {
        BackendKind::Metal => {
            let ctx = MetalKernels::create_matmul()
                .map_err(|e| anyhow::anyhow!("Gemma4 VLM Metal backend unavailable: {e}"))?;
            LinearBackend::Metal {
                ctx,
                weight_t_cache: HashMap::new(),
            }
        }
        BackendKind::Cpu => LinearBackend::Cpu,
    };

    let patch_w_name = "model.vision_tower.patch_embedder.input_proj.weight";
    let pos_name = "model.vision_tower.patch_embedder.position_embedding_table";
    let proj_name = "model.embed_vision.embedding_projection.weight";

    let patch_w_shape = tensor_shape(file, patch_w_name)?;
    let pos_shape = tensor_shape(file, pos_name)?;
    let proj_shape = tensor_shape(file, proj_name)?;
    if patch_w_shape.len() != 2 || patch_w_shape[0] == 0 || patch_w_shape[1] == 0 {
        anyhow::bail!("unexpected Gemma4 patch projector shape: {patch_w_shape:?}");
    }
    if pos_shape.len() != 3 || pos_shape[2] != patch_w_shape[0] {
        anyhow::bail!("unexpected Gemma4 position table shape: {pos_shape:?}");
    }
    if proj_shape.len() != 2 || proj_shape[1] != patch_w_shape[0] {
        anyhow::bail!("unexpected Gemma4 image projection shape: {proj_shape:?}");
    }

    let hidden = patch_w_shape[0];
    let patch_in = patch_w_shape[1];
    let pos_table = tensor_to_f32(file, pos_name)?;
    let pos_rows = pos_shape[1];
    let patches = image_input
        .gemma4_patch_values
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Gemma4 preprocessing missing patch vectors"))?;
    let pos_ids = image_input
        .gemma4_position_ids
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Gemma4 preprocessing missing position ids"))?;
    let soft_tokens = image_input
        .gemma4_num_soft_tokens
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Gemma4 preprocessing missing soft-token counts"))?;
    let nimg = patches.shape()[0];
    let num_tokens = patches.shape()[1];
    if patches.shape()[2] != patch_in {
        anyhow::bail!(
            "Gemma4 patch vector dim mismatch: expected {patch_in}, got {}",
            patches.shape()[2]
        );
    }
    if pos_ids.shape()[0] != nimg || pos_ids.shape()[1] != num_tokens || pos_ids.shape()[2] != 2 {
        anyhow::bail!("Gemma4 position id tensor shape mismatch");
    }
    if num_tokens > pos_rows {
        anyhow::bail!("Gemma4 position rows too small: need {num_tokens}, have {pos_rows}");
    }
    let patch_w = tensor_to_f32(file, patch_w_name)?;
    let proj = tensor_to_f32(file, proj_name)?;
    let eps = file
        .header
        .source_vision_config
        .as_ref()
        .and_then(|v| v.get("rms_norm_eps"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let num_layers = file
        .header
        .source_vision_config
        .as_ref()
        .and_then(|v| v.get("num_hidden_layers"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or_else(|| infer_gemma4_vision_num_layers(file));
    let num_heads = file
        .header
        .source_vision_config
        .as_ref()
        .and_then(|v| v.get("num_attention_heads"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(12);
    let intermediate = file
        .header
        .source_vision_config
        .as_ref()
        .and_then(|v| v.get("intermediate_size"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden * 4);
    if num_layers == 0 {
        anyhow::bail!("Gemma4 vision layer count resolved to zero");
    }
    if num_heads == 0 || hidden % num_heads != 0 {
        anyhow::bail!("invalid Gemma4 vision heads: hidden={hidden}, num_heads={num_heads}");
    }
    let head_dim = hidden / num_heads;
    let rope_theta = file
        .header
        .source_vision_config
        .as_ref()
        .and_then(|v| v.get("rope_parameters"))
        .and_then(|v| v.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(100.0) as f32;
    if head_dim % 4 != 0 {
        anyhow::bail!("Gemma4 vision head_dim must be divisible by 4, got {head_dim}");
    }
    let spatial_dim = head_dim / 2;
    let rope_half = spatial_dim / 2;
    let mut rope_inv_freq = vec![0.0f32; rope_half];
    for (i, rf) in rope_inv_freq.iter_mut().enumerate() {
        let exp = (2 * i) as f32 / spatial_dim as f32;
        *rf = rope_theta.powf(-exp);
    }

    let mut layers = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let prefix = format!("model.vision_tower.encoder.layers.{layer}.");
        let (q_clip_in, q_clip_out) = load_linear_clip_ranges(file, &format!("{prefix}self_attn.q_proj"))?;
        let (k_clip_in, k_clip_out) = load_linear_clip_ranges(file, &format!("{prefix}self_attn.k_proj"))?;
        let (v_clip_in, v_clip_out) = load_linear_clip_ranges(file, &format!("{prefix}self_attn.v_proj"))?;
        let (o_clip_in, o_clip_out) = load_linear_clip_ranges(file, &format!("{prefix}self_attn.o_proj"))?;
        let (gate_clip_in, gate_clip_out) = load_linear_clip_ranges(file, &format!("{prefix}mlp.gate_proj"))?;
        let (up_clip_in, up_clip_out) = load_linear_clip_ranges(file, &format!("{prefix}mlp.up_proj"))?;
        let (down_clip_in, down_clip_out) = load_linear_clip_ranges(file, &format!("{prefix}mlp.down_proj"))?;
        layers.push(Gemma4VisionLayerWeights {
            input_ln_w: tensor_to_f32(file, &format!("{prefix}input_layernorm.weight"))?,
            q_w: tensor_to_f32(file, &format!("{prefix}self_attn.q_proj.linear.weight"))?,
            k_w: tensor_to_f32(file, &format!("{prefix}self_attn.k_proj.linear.weight"))?,
            v_w: tensor_to_f32(file, &format!("{prefix}self_attn.v_proj.linear.weight"))?,
            o_w: tensor_to_f32(file, &format!("{prefix}self_attn.o_proj.linear.weight"))?,
            q_norm_w: tensor_to_f32(file, &format!("{prefix}self_attn.q_norm.weight"))?,
            k_norm_w: tensor_to_f32(file, &format!("{prefix}self_attn.k_norm.weight"))?,
            post_attn_ln_w: tensor_to_f32(file, &format!("{prefix}post_attention_layernorm.weight"))?,
            pre_ffn_ln_w: tensor_to_f32(file, &format!("{prefix}pre_feedforward_layernorm.weight"))?,
            gate_w: tensor_to_f32(file, &format!("{prefix}mlp.gate_proj.linear.weight"))?,
            up_w: tensor_to_f32(file, &format!("{prefix}mlp.up_proj.linear.weight"))?,
            down_w: tensor_to_f32(file, &format!("{prefix}mlp.down_proj.linear.weight"))?,
            post_ffn_ln_w: tensor_to_f32(file, &format!("{prefix}post_feedforward_layernorm.weight"))?,
            q_clip_in,
            q_clip_out,
            k_clip_in,
            k_clip_out,
            v_clip_in,
            v_clip_out,
            o_clip_in,
            o_clip_out,
            gate_clip_in,
            gate_clip_out,
            up_clip_in,
            up_clip_out,
            down_clip_in,
            down_clip_out,
        });
    }
    let text_hidden = proj_shape[0];
    if nimg != 1 {
        anyhow::bail!("Gemma4 VLM currently supports exactly 1 image per prompt, got {nimg}");
    }
    let out_tokens_per_image = target_image_seq_len
        .unwrap_or_else(|| soft_tokens[0].max(1))
        .min(soft_tokens[0].max(1));

    let mut tokens = vec![0.0f32; num_tokens * hidden];
    let mut norm0 = vec![0.0f32; num_tokens * hidden];
    let mut norm1 = vec![0.0f32; num_tokens * hidden];
    let mut q = vec![0.0f32; num_tokens * hidden];
    let mut k = vec![0.0f32; num_tokens * hidden];
    let mut v = vec![0.0f32; num_tokens * hidden];
    let mut attn = vec![0.0f32; num_tokens * hidden];
    let mut proj_out = vec![0.0f32; num_tokens * hidden];
    let mut gate = vec![0.0f32; num_tokens * intermediate];
    let mut up = vec![0.0f32; num_tokens * intermediate];
    let mut mlp_out = vec![0.0f32; num_tokens * hidden];
    let mut score_buf = vec![0.0f32; num_tokens];
    let mut prob_buf = vec![0.0f32; num_tokens];
    let mut rope_cos = vec![0.0f32; num_tokens * (2 * rope_half)];
    let mut rope_sin = vec![0.0f32; num_tokens * (2 * rope_half)];
    let mut valid_mask = vec![false; num_tokens];
    let mut patch_ms = 0.0f64;
    let mut encoder_ms = 0.0f64;
    let mut encoder_layer_ms = vec![0.0f64; num_layers];
    let mut out = Array2::<f32>::zeros((nimg * out_tokens_per_image, text_hidden));

    for img in 0..nimg {
        let t_patch = Instant::now();
        let mut patch_rows = vec![0.0f32; num_tokens * patch_in];
        for t in 0..num_tokens {
            for i in 0..patch_in {
                patch_rows[t * patch_in + i] = 2.0 * (patches[[img, t, i]] - 0.5);
            }
        }
        linear_rows(
            &patch_rows,
            num_tokens,
            patch_in,
            &patch_w,
            hidden,
            None,
            &mut tokens,
            &mut linear_backend,
        );
        let pos_axis = &pos_table[..pos_rows * hidden];
        valid_mask.fill(false);
        for t in 0..num_tokens {
            let x = pos_ids[[img, t, 0]];
            let y = pos_ids[[img, t, 1]];
            if x < 0 || y < 0 {
                let row = &mut tokens[t * hidden..(t + 1) * hidden];
                row.fill(0.0);
                continue;
            }
            let xi = x as usize;
            let yi = y as usize;
            if xi >= pos_rows || yi >= pos_rows {
                anyhow::bail!("Gemma4 position id out of range: x={xi} y={yi} max={pos_rows}");
            }
            valid_mask[t] = true;
            let row = &mut tokens[t * hidden..(t + 1) * hidden];
            let xemb = &pos_axis[xi * hidden..(xi + 1) * hidden];
            let yemb = &pos_table[pos_rows * hidden + yi * hidden..pos_rows * hidden + (yi + 1) * hidden];
            for i in 0..hidden {
                row[i] += xemb[i] + yemb[i];
            }
            let base = t * (2 * rope_half);
            for i in 0..rope_half {
                let ax = xi as f32 * rope_inv_freq[i];
                let ay = yi as f32 * rope_inv_freq[i];
                rope_cos[base + i] = ax.cos();
                rope_sin[base + i] = ax.sin();
                rope_cos[base + rope_half + i] = ay.cos();
                rope_sin[base + rope_half + i] = ay.sin();
            }
        }
        patch_ms += t_patch.elapsed().as_secs_f64() * 1000.0;

        let t_enc = Instant::now();
        for (layer_idx, layer) in layers.iter().enumerate() {
            let layer_start = Instant::now();
            rms_norm_rows(&tokens, num_tokens, hidden, &layer.input_ln_w, eps, &mut norm0);

            linear_rows_clipped(
                &norm0,
                num_tokens,
                hidden,
                &layer.q_w,
                hidden,
                None,
                &mut q,
                layer.q_clip_in,
                layer.q_clip_out,
                &mut linear_backend,
            );
            linear_rows_clipped(
                &norm0,
                num_tokens,
                hidden,
                &layer.k_w,
                hidden,
                None,
                &mut k,
                layer.k_clip_in,
                layer.k_clip_out,
                &mut linear_backend,
            );
            linear_rows_clipped(
                &norm0,
                num_tokens,
                hidden,
                &layer.v_w,
                hidden,
                None,
                &mut v,
                layer.v_clip_in,
                layer.v_clip_out,
                &mut linear_backend,
            );

            for t in 0..num_tokens {
                let q_row = &mut q[t * hidden..(t + 1) * hidden];
                let k_row = &mut k[t * hidden..(t + 1) * hidden];
                let v_row = &mut v[t * hidden..(t + 1) * hidden];
                let rope_base = t * (2 * rope_half);
                let cos = &rope_cos[rope_base..rope_base + (2 * rope_half)];
                let sin = &rope_sin[rope_base..rope_base + (2 * rope_half)];
                for hidx in 0..num_heads {
                    let start = hidx * head_dim;
                    let end = start + head_dim;
                    rms_norm_inplace_segment(&mut q_row[start..end], &layer.q_norm_w, eps);
                    rms_norm_inplace_segment(&mut k_row[start..end], &layer.k_norm_w, eps);
                    rms_norm_inplace_noscale(&mut v_row[start..end], eps);
                    apply_multidim_rope_inplace(&mut q_row[start..end], cos, sin);
                    apply_multidim_rope_inplace(&mut k_row[start..end], cos, sin);
                }
            }

            self_attention_full(
                &q,
                &k,
                &v,
                num_tokens,
                num_heads,
                head_dim,
                Some(1.0),
                &mut score_buf,
                &mut prob_buf,
                &mut attn,
                Some(&valid_mask),
                &mut linear_backend,
            );
            linear_rows_clipped(
                &attn,
                num_tokens,
                hidden,
                &layer.o_w,
                hidden,
                None,
                &mut proj_out,
                layer.o_clip_in,
                layer.o_clip_out,
                &mut linear_backend,
            );
            rms_norm_rows(
                &proj_out,
                num_tokens,
                hidden,
                &layer.post_attn_ln_w,
                eps,
                &mut norm0,
            );
            add_inplace(&mut tokens, &norm0);

            rms_norm_rows(
                &tokens,
                num_tokens,
                hidden,
                &layer.pre_ffn_ln_w,
                eps,
                &mut norm1,
            );
            linear_rows_clipped(
                &norm1,
                num_tokens,
                hidden,
                &layer.gate_w,
                intermediate,
                None,
                &mut gate,
                layer.gate_clip_in,
                layer.gate_clip_out,
                &mut linear_backend,
            );
            linear_rows_clipped(
                &norm1,
                num_tokens,
                hidden,
                &layer.up_w,
                intermediate,
                None,
                &mut up,
                layer.up_clip_in,
                layer.up_clip_out,
                &mut linear_backend,
            );
            gelu_pytorch_tanh_inplace(&mut gate);
            mul_inplace(&mut gate, &up);
            linear_rows_clipped(
                &gate,
                num_tokens,
                intermediate,
                &layer.down_w,
                hidden,
                None,
                &mut mlp_out,
                layer.down_clip_in,
                layer.down_clip_out,
                &mut linear_backend,
            );
            rms_norm_rows(
                &mlp_out,
                num_tokens,
                hidden,
                &layer.post_ffn_ln_w,
                eps,
                &mut norm0,
            );
            add_inplace(&mut tokens, &norm0);
            encoder_layer_ms[layer_idx] += layer_start.elapsed().as_secs_f64() * 1000.0;
        }

        let mut max_x = 0usize;
        for t in 0..num_tokens {
            let x = pos_ids[[img, t, 0]];
            if x >= 0 {
                max_x = max_x.max(x as usize);
            }
        }
        let pooled_w = (max_x + 1) / 3;
        let mut pooled = vec![vec![0.0f32; hidden]; out_tokens_per_image];
        let mut pooled_seen = vec![false; out_tokens_per_image];
        for t in 0..num_tokens {
            if !valid_mask[t] {
                continue;
            }
            let x = pos_ids[[img, t, 0]] as usize;
            let y = pos_ids[[img, t, 1]] as usize;
            let idx = (x / 3) + pooled_w * (y / 3);
            if idx >= out_tokens_per_image {
                continue;
            }
            pooled_seen[idx] = true;
            let src = &tokens[t * hidden..(t + 1) * hidden];
            for i in 0..hidden {
                pooled[idx][i] += src[i] / 9.0;
            }
        }
        for out_tok in 0..out_tokens_per_image {
            if !pooled_seen[out_tok] {
                continue;
            }
            let mut src_norm = pooled[out_tok].clone();
            rms_norm_inplace_noscale(&mut src_norm, eps);
            for o in 0..text_hidden {
                let mut acc = 0.0f32;
                let wrow = &proj[o * hidden..(o + 1) * hidden];
                for i in 0..hidden {
                    acc += src_norm[i] * wrow[i];
                }
                out[[img * out_tokens_per_image + out_tok, o]] = acc;
            }
        }
        encoder_ms += t_enc.elapsed().as_secs_f64() * 1000.0;
    }

    Ok((out, out_tokens_per_image, patch_ms, encoder_ms, encoder_layer_ms))
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

fn infer_gemma4_vision_num_layers(file: &CellmFile) -> usize {
    let mut layers = 0usize;
    loop {
        let name = format!("model.vision_tower.encoder.layers.{layers}.input_layernorm.weight");
        if file.tensor_index(&name).is_some() {
            layers += 1;
        } else {
            break;
        }
    }
    layers
}

//  Audio encoder

/// Parse a WAV file, returning (i16_samples, sample_rate).
/// Handles PCM-16 mono/stereo; stereo is mixed to mono.
fn parse_wav_pcm16(bytes: &[u8]) -> Result<(Vec<i16>, u32)> {
    if bytes.len() < 44 {
        anyhow::bail!("WAV file too short");
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("Not a valid RIFF/WAVE file");
    }
    let mut pos = 12usize;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut data_start = 0usize;
    let mut data_len = 0usize;
    while pos + 8 <= bytes.len() {
        let tag = &bytes[pos..pos + 4];
        let chunk_len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        if tag == b"fmt " {
            let audio_fmt = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap());
            if audio_fmt != 1 {
                anyhow::bail!("Only PCM (format=1) WAV supported, got {audio_fmt}");
            }
            channels = u16::from_le_bytes(bytes[pos + 2..pos + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(bytes[pos + 14..pos + 16].try_into().unwrap());
        } else if tag == b"data" {
            data_start = pos;
            data_len = chunk_len;
        }
        pos += chunk_len;
    }
    if sample_rate == 0 || data_start == 0 {
        anyhow::bail!("Invalid WAV: missing fmt or data chunk");
    }
    if bits_per_sample != 16 {
        anyhow::bail!("Only 16-bit PCM supported, got {bits_per_sample}-bit");
    }
    let end = (data_start + data_len).min(bytes.len());
    let raw = &bytes[data_start..end];
    let n_samples = raw.len() / (2 * channels as usize);
    let mut mono = Vec::with_capacity(n_samples);
    for s in 0..n_samples {
        let base = s * channels as usize * 2;
        if channels == 1 {
            mono.push(i16::from_le_bytes(raw[base..base + 2].try_into().unwrap()));
        } else {
            // Mix stereo to mono by averaging
            let l = i16::from_le_bytes(raw[base..base + 2].try_into().unwrap()) as i32;
            let r = i16::from_le_bytes(raw[base + 2..base + 4].try_into().unwrap()) as i32;
            mono.push(((l + r) / 2) as i16);
        }
    }
    Ok((mono, sample_rate))
}

/// Build the HTK mel filterbank matrix [fft_bins, num_mels] using linear frequency in FFT domain
/// and the HTK mel scale.
fn build_mel_filterbank(
    fft_length: usize,
    num_mels: usize,
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
) -> Vec<Vec<f32>> {
    let fft_bins = fft_length / 2 + 1;
    // HTK mel scale
    let hz_to_mel = |hz: f64| -> f64 { 2595.0 * (1.0 + hz / 700.0).log10() };
    let mel_to_hz = |m: f64| -> f64 { 700.0 * (10.0_f64.powf(m / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(min_freq);
    let mel_max = hz_to_mel(max_freq);

    // num_mels + 2 linearly-spaced points in mel space
    let n_points = num_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let bin_pts: Vec<f64> = hz_points
        .iter()
        .map(|&hz| hz / (sample_rate / 2.0) * (fft_bins - 1) as f64)
        .collect();

    let mut filters = vec![vec![0.0f32; num_mels]; fft_bins];
    for m in 0..num_mels {
        let left = bin_pts[m];
        let center = bin_pts[m + 1];
        let right = bin_pts[m + 2];
        for f in 0..fft_bins {
            let fv = f as f64;
            let val = if fv >= left && fv < center {
                (fv - left) / (center - left)
            } else if fv >= center && fv <= right {
                (right - fv) / (right - center)
            } else {
                0.0
            };
            if val > 0.0 {
                filters[f][m] = val as f32;
            }
        }
    }
    filters
}

/// Compute mel spectrogram from PCM i16 samples.
/// Returns (features [T, num_mels], num_frames).
fn compute_mel_spectrogram(
    samples: &[i16],
    frame_length: usize,
    hop_length: usize,
    fft_length: usize,
    num_mels: usize,
    min_freq: f64,
    max_freq: f64,
    mel_floor: f32,
) -> (Vec<f32>, usize) {
    // Normalise i16 → f32 in [-1, 1]
    let signal: Vec<f64> = samples.iter().map(|&s| s as f64 / 32768.0).collect();

    // Hann window (periodic) — window_function in HF uses periodic Hann: w[n] = 0.5 - 0.5*cos(2π*n/N)
    let window: Vec<f64> = (0..frame_length)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / frame_length as f64).cos())
        .collect();

    // Semicausal padding: prepend frame_length//2 zeros
    let pad_left = frame_length / 2;
    let mut padded = vec![0.0f64; pad_left + signal.len()];
    padded[pad_left..].copy_from_slice(&signal);

    // The feature extractor uses frame_size = frame_length + 1 for unfold, then takes [:-1]
    // This effectively gives frame_length samples per frame
    let frame_size_unfold = frame_length + 1;
    let num_frames = if padded.len() >= frame_size_unfold {
        (padded.len() - frame_size_unfold) / hop_length + 1
    } else {
        0
    };

    let mel_filters = build_mel_filterbank(fft_length, num_mels, 16000.0, min_freq, max_freq);
    let fft_bins = fft_length / 2 + 1;

    let mut features = vec![0.0f32; num_frames * num_mels];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        // Take frame_size_unfold samples, apply pre-emphasis=0 (just take first frame_length)
        let frame: Vec<f64> = (0..frame_length)
            .map(|i| padded.get(start + i).copied().unwrap_or(0.0) * window[i])
            .collect();

        // Real FFT via DFT (power of 2 length)
        let mut re = vec![0.0f64; fft_length];
        let mut im = vec![0.0f64; fft_length];
        for (i, &v) in frame.iter().enumerate().take(fft_length) {
            re[i] = v;
        }
        // Simple DFT for small sizes; use FFT for real use
        fft_real_inplace(&mut re, &mut im, fft_length);

        let row = &mut features[frame_idx * num_mels..(frame_idx + 1) * num_mels];
        for m in 0..num_mels {
            let mut mel_val = 0.0f32;
            for f in 0..fft_bins {
                let mag = (re[f] * re[f] + im[f] * im[f]).sqrt() as f32;
                mel_val += mag * mel_filters[f][m];
            }
            row[m] = (mel_val + mel_floor).ln();
        }
    }
    (features, num_frames)
}

/// Radix-2 Cooley-Tukey FFT in-place (power-of-2 length).
fn fft_real_inplace(re: &mut [f64], im: &mut [f64], n: usize) {
    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    // Cooley-Tukey butterfly
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let wr = ang.cos();
        let wi = ang.sin();
        let mut i = 0;
        while i < n {
            let mut w_re = 1.0f64;
            let mut w_im = 0.0f64;
            for k in 0..len / 2 {
                let u_re = re[i + k];
                let u_im = im[i + k];
                let v_re = re[i + k + len / 2] * w_re - im[i + k + len / 2] * w_im;
                let v_im = re[i + k + len / 2] * w_im + im[i + k + len / 2] * w_re;
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + len / 2] = u_re - v_re;
                im[i + k + len / 2] = u_im - v_im;
                let new_w_re = w_re * wr - w_im * wi;
                w_im = w_re * wi + w_im * wr;
                w_re = new_w_re;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Compute Conv2d (no bias, stride 2, padding 1, kernel 3×3).
/// Input shape: [in_c, H, W].  Weight: [out_c, in_c, 3, 3].
/// Output: [out_c, ceil(H/2), ceil(W/2)].
fn conv2d_3x3_stride2_pad1(
    input: &[f32],
    in_c: usize,
    height: usize,
    width: usize,
    weight: &[f32],
    out_c: usize,
) -> Vec<f32> {
    let out_h = (height + 1) / 2;
    let out_w = (width + 1) / 2;
    let mut output = vec![0.0f32; out_c * out_h * out_w];
    for oc in 0..out_c {
        for oh in 0..out_h {
            let ih_base = oh as isize * 2 - 1;
            for ow in 0..out_w {
                let iw_base = ow as isize * 2 - 1;
                let mut acc = 0.0f32;
                for ic in 0..in_c {
                    for kh in 0..3i32 {
                        let ih = ih_base + kh as isize;
                        if ih < 0 || ih >= height as isize { continue; }
                        for kw in 0..3i32 {
                            let iw = iw_base + kw as isize;
                            if iw < 0 || iw >= width as isize { continue; }
                            let inp_val = input[ic * height * width + ih as usize * width + iw as usize];
                            let w_val = weight[oc * in_c * 9 + ic * 9 + kh as usize * 3 + kw as usize];
                            acc += inp_val * w_val;
                        }
                    }
                }
                output[oc * out_h * out_w + oh * out_w + ow] = acc;
            }
        }
    }
    output
}

/// Compute sinusoidal relative positional embeddings for the audio conformer.
/// Returns [context_size, hidden] where positions go from (context_size-1) down to 0.
/// context_size = chunk_size + context_left - 1 + context_right = 12+13-1+0 = 24.
fn compute_audio_rel_pos_embed(context_size: usize, hidden: usize) -> Vec<f32> {
    let num_timescales = hidden / 2;
    let max_timescale = 10000.0f64;
    let log_inc = max_timescale.ln() / (num_timescales - 1).max(1) as f64;
    let inv_timescales: Vec<f64> = (0..num_timescales)
        .map(|k| (-log_inc * k as f64).exp())
        .collect();

    let mut out = vec![0.0f32; context_size * hidden];
    // Ascending: rel_pos_embed[i] = embedding for pos_id i (= distance i).
    // Lookup by distance d: rel_k_buf[d] uses pos_id d, matching HF's rel_shift result.
    for i in 0..context_size {
        let p = i as f64;
        for k in 0..num_timescales {
            let scaled = p * inv_timescales[k];
            out[i * hidden + k] = scaled.sin() as f32;
            out[i * hidden + num_timescales + k] = scaled.cos() as f32;
        }
    }
    out
}

/// SiLU activation: x * sigmoid(x)
#[inline(always)]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// Run the Gemma 4 audio tower + embedder.
/// Returns an Array2 of shape [n_audio_tokens, text_hidden=1536].
fn audio_stats(tag: &str, buf: &[f32]) {
    if std::env::var("CELLM_AUDIO_DEBUG").is_ok() {
        let mut nan_count = 0usize;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        for &v in buf {
            if v.is_nan() || v.is_infinite() { nan_count += 1; continue; }
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            sum += v as f64;
        }
        let mean = sum / (buf.len() - nan_count).max(1) as f64;
        eprintln!("AUDIO_STATS {:40} n={} nan/inf={} min={:.4} max={:.4} mean={:.4}", tag, buf.len(), nan_count, min_v, max_v, mean);
    }
}

fn run_audio_cellm_gemma4(
    file: &CellmFile,
    mel_features: &[f32],   // [T_frames, 128] row-major
    t_frames: usize,
    num_mels: usize,        // 128
) -> Result<Array2<f32>> {
    let eps = 1e-6f32;
    if std::env::var("CELLM_DUMP_AUDIO_TENSORS").is_ok() {
        for t in &file.header.tensors {
            if t.name.contains("audio") || t.name.contains("embed_proj") {
                eprintln!("TENSOR {} {:?} {}", t.name, t.shape, t.dtype);
            }
        }
        if let Some(cfg) = &file.header.source_text_config {
            eprintln!("TEXT hidden_size={:?} model_type={:?}", cfg.get("hidden_size"), cfg.get("model_type"));
        }
    }

    //  Config
    let hidden = 1024usize;
    let num_heads = 8usize;
    let head_dim = hidden / num_heads; // 128
    let ffw_intermediate = hidden * 4; // 4096
    let text_hidden = 1536usize;
    let chunk_size = 12usize;
    let context_left = 13usize;
    let context_right = 0usize;
    let context_size = chunk_size + context_left - 1 + context_right; // 24
    let conv_kernel = 5usize;
    let residual_weight = 0.5f32;
    let attn_cap = 50.0f32;
    let attn_invalid = -1.0e9f32;
    let q_scale = (head_dim as f32).powf(-0.5) / (2.0f32).ln();
    let k_scale = (1.0 + std::f32::consts::E).ln() / (2.0f32).ln();
    let num_audio_layers = 12usize;
    let sub_ch0 = 128usize;
    let sub_ch1 = 32usize;

    // Use Metal for large conformer matmuls when available — the CPU rayon path
    // achieves only ~1 GFLOPS (scalar, not SIMD-vectorised by LLVM at this loop
    // shape) while Metal gives ~50–100× speedup for the 144×1024×4096 ops that
    // dominate each conformer layer.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    let mut backend = MetalKernels::create_matmul()
        .map(|ctx| LinearBackend::Metal {
            ctx,
            weight_t_cache: std::collections::HashMap::new(),
        })
        .unwrap_or(LinearBackend::Cpu);
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    let mut backend = LinearBackend::Cpu;

    //  A. Subsampling Conv Projection
    // Input mel: [T, 128] → treat as image [B, 1, T, 128] where T=time, 128=mel bins
    // layer0: Conv2d(1→128, 3×3, stride=2, pad=1), LayerNorm(128), ReLU
    audio_stats("mel_features", mel_features);
    let conv0_w = load_audio_weight_f32(file, "model.audio_tower.subsample_conv_projection.layer0.conv.weight")?;
    let norm0_w = load_audio_weight_f32(file, "model.audio_tower.subsample_conv_projection.layer0.norm.weight")?;
    // Correct orientation: H=t_frames (time), W=num_mels (frequency)
    // conv2d_3x3_stride2_pad1(input, in_c, height, width, weight, out_c)
    let conv0_out = conv2d_3x3_stride2_pad1(mel_features, 1, t_frames, num_mels, &conv0_w, sub_ch0);
    audio_stats("conv0_out", &conv0_out);
    // conv0_out: [sub_ch0=128, h0=ceil(T/2), w0=ceil(128/2)=64]
    // With H=t_frames, W=num_mels:
    let h0 = (t_frames + 1) / 2; // ceil(T/2) — time dimension
    let w0 = (num_mels + 1) / 2; // 64 — frequency dimension
    // Apply LayerNorm on last dim (channels) after permuting [C,H,W]→[H,W,C]
    // We permute by iterating h,w,c
    let mut after_norm0 = vec![0.0f32; sub_ch0 * h0 * w0];
    for hi in 0..h0 {
        for wi in 0..w0 {
            // Compute mean and var over channels
            let mut mean = 0.0f32;
            for c in 0..sub_ch0 {
                mean += conv0_out[c * h0 * w0 + hi * w0 + wi];
            }
            mean /= sub_ch0 as f32;
            let mut var = 0.0f32;
            for c in 0..sub_ch0 {
                let d = conv0_out[c * h0 * w0 + hi * w0 + wi] - mean;
                var += d * d;
            }
            var /= sub_ch0 as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..sub_ch0 {
                let v = (conv0_out[c * h0 * w0 + hi * w0 + wi] - mean) * inv * norm0_w[c];
                after_norm0[c * h0 * w0 + hi * w0 + wi] = v.max(0.0); // ReLU
            }
        }
    }
    // layer1: Conv2d(128→32, 3×3, stride=2, pad=1), LayerNorm(32), ReLU
    let conv1_w = load_audio_weight_f32(file, "model.audio_tower.subsample_conv_projection.layer1.conv.weight")?;
    let norm1_w = load_audio_weight_f32(file, "model.audio_tower.subsample_conv_projection.layer1.norm.weight")?;
    let conv1_out = conv2d_3x3_stride2_pad1(&after_norm0, sub_ch0, h0, w0, &conv1_w, sub_ch1);
    let h1 = (h0 + 1) / 2; // ceil(T/4) — time steps after two stride-2 convs
    let w1 = (w0 + 1) / 2; // 32 — frequency after two stride-2 convs
    let t_sub = h1; // time steps (sequence length for conformer)
    let mut after_norm1 = vec![0.0f32; sub_ch1 * h1 * w1];
    for hi in 0..h1 {
        for wi in 0..w1 {
            let mut mean = 0.0f32;
            for c in 0..sub_ch1 {
                mean += conv1_out[c * h1 * w1 + hi * w1 + wi];
            }
            mean /= sub_ch1 as f32;
            let mut var = 0.0f32;
            for c in 0..sub_ch1 {
                let d = conv1_out[c * h1 * w1 + hi * w1 + wi] - mean;
                var += d * d;
            }
            var /= sub_ch1 as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..sub_ch1 {
                let v = (conv1_out[c * h1 * w1 + hi * w1 + wi] - mean) * inv * norm1_w[c];
                after_norm1[c * h1 * w1 + hi * w1 + wi] = v.max(0.0); // ReLU
            }
        }
    }

    // Reshape: permute [C, H, W] → [T, H*C] i.e. [w1, h1*sub_ch1]
    // In HF: hidden_states.permute(0,2,3,1).reshape(batch, seq_len, -1)
    // Tensor is [B=1, C=sub_ch1, H=t_sub, W=w1] → permute → [1, H=t_sub, W=w1, C=sub_ch1] → [1, t_sub, w1*sub_ch1]
    let proj_in_dim = w1 * sub_ch1; // 32*32 = 1024
    let mut sub_out = vec![0.0f32; t_sub * proj_in_dim];
    for t in 0..t_sub {          // H dimension = time
        for wi in 0..w1 {        // W dimension = frequency
            for c in 0..sub_ch1 { // C dimension = channels
                sub_out[t * proj_in_dim + wi * sub_ch1 + c] = after_norm1[c * t_sub * w1 + t * w1 + wi];
            }
        }
    }

    // input_proj_linear: [1024, 1024]
    let proj_w = load_audio_weight_f32(file, "model.audio_tower.subsample_conv_projection.input_proj_linear.weight")?;
    let mut hidden_states = vec![0.0f32; t_sub * hidden];
    linear_rows(&sub_out, t_sub, proj_in_dim, &proj_w, hidden, None, &mut hidden_states, &mut backend);
    audio_stats("hidden_states_init", &hidden_states);
    let t_subsample_done = std::time::Instant::now();
    eprintln!("[audio_timing] t_sub={t_sub} subsample_done");

    //  B. Relative positional embeddings
    // positions 12 down to 0 (context_size = 24, but only 13 unique positions for left context)
    let n_rel_pos = context_left; // 13
    let rel_pos_embed = compute_audio_rel_pos_embed(n_rel_pos, hidden);

    //  C. 12 Conformer Layers
    let t_conformer_start = std::time::Instant::now();
    let mut norm_buf = vec![0.0f32; t_sub * hidden];
    let mut norm2_buf = vec![0.0f32; t_sub * hidden];
    let mut ffw1_buf = vec![0.0f32; t_sub * ffw_intermediate];
    let mut ffw2_buf = vec![0.0f32; t_sub * hidden];
    let mut q_buf = vec![0.0f32; t_sub * hidden];
    let mut k_buf = vec![0.0f32; t_sub * hidden];
    let mut v_buf = vec![0.0f32; t_sub * hidden];
    let mut rel_k_buf = vec![0.0f32; n_rel_pos * hidden];
    let mut score_buf = vec![0.0f32; context_size.max(t_sub)];
    let mut prob_buf = vec![0.0f32; context_size.max(t_sub)];
    let mut attn_out = vec![0.0f32; t_sub * hidden];
    let mut lconv_lin_buf = vec![0.0f32; t_sub * hidden * 2];
    let mut lconv_out = vec![0.0f32; t_sub * hidden];
    let mut conv_state = vec![0.0f32; (conv_kernel - 1) * hidden]; // causal conv padding

    for layer in 0..num_audio_layers {
        let t_layer_start = std::time::Instant::now();
        let p = format!("model.audio_tower.layers.{layer}.");

        //  feed_forward1
        {
            let t0 = std::time::Instant::now();
            let pre_w = load_audio_weight_f32(file, &format!("{p}feed_forward1.pre_layer_norm.weight"))?;
            let ff1_w = load_audio_weight_f32(file, &format!("{p}feed_forward1.ffw_layer_1.linear.weight"))?;
            let ff2_w = load_audio_weight_f32(file, &format!("{p}feed_forward1.ffw_layer_2.linear.weight"))?;
            let post_w = load_audio_weight_f32(file, &format!("{p}feed_forward1.post_layer_norm.weight"))?;
            if layer == 0 { eprintln!("[audio_timing] L0 ff1 weight_load={:.1}ms", t0.elapsed().as_secs_f64()*1000.0); }
            let (ff1_in_clip, ff1_out_clip) = load_linear_clip_ranges(file, &format!("{p}feed_forward1.ffw_layer_1"))?;
            let (ff2_in_clip, ff2_out_clip) = load_linear_clip_ranges(file, &format!("{p}feed_forward1.ffw_layer_2"))?;

            let residual = hidden_states.clone();
            rms_norm_rows(&hidden_states, t_sub, hidden, &pre_w, eps, &mut norm_buf);
            if let Some((mn, mx)) = ff1_in_clip { for v in &mut norm_buf[..t_sub*hidden] { *v = v.clamp(mn, mx); } }
            linear_rows(&norm_buf, t_sub, hidden, &ff1_w, ffw_intermediate, None, &mut ffw1_buf, &mut backend);
            if let Some((mn, mx)) = ff1_out_clip { for v in &mut ffw1_buf[..t_sub*ffw_intermediate] { *v = v.clamp(mn, mx); } }
            silu_inplace(&mut ffw1_buf);
            if let Some((mn, mx)) = ff2_in_clip { for v in &mut ffw1_buf[..t_sub*ffw_intermediate] { *v = v.clamp(mn, mx); } }
            linear_rows(&ffw1_buf, t_sub, ffw_intermediate, &ff2_w, hidden, None, &mut ffw2_buf, &mut backend);
            if let Some((mn, mx)) = ff2_out_clip { for v in &mut ffw2_buf[..t_sub*hidden] { *v = v.clamp(mn, mx); } }
            rms_norm_rows(&ffw2_buf, t_sub, hidden, &post_w, eps, &mut norm_buf);
            for i in 0..t_sub * hidden {
                hidden_states[i] = residual[i] + norm_buf[i] * residual_weight;
            }
            if layer == 0 { audio_stats("L0_after_ffn1", &hidden_states[..t_sub*hidden]); }
            if layer == 0 { eprintln!("[audio_timing] L0 ff1 total={:.1}ms", t0.elapsed().as_secs_f64()*1000.0); }
        }

        //  self_attn (chunked local with relative pos bias)
        {
            let t0 = std::time::Instant::now();
            let t_wload = std::time::Instant::now();
            let norm_pre_w = load_audio_weight_f32(file, &format!("{p}norm_pre_attn.weight"))?;
            let norm_post_w = load_audio_weight_f32(file, &format!("{p}norm_post_attn.weight"))?;
            let q_w = load_audio_weight_f32(file, &format!("{p}self_attn.q_proj.linear.weight"))?;
            let k_w = load_audio_weight_f32(file, &format!("{p}self_attn.k_proj.linear.weight"))?;
            let v_w = load_audio_weight_f32(file, &format!("{p}self_attn.v_proj.linear.weight"))?;
            let post_w = load_audio_weight_f32(file, &format!("{p}self_attn.post.linear.weight"))?;
            let rel_k_w = load_audio_weight_f32(file, &format!("{p}self_attn.relative_k_proj.weight"))?;
            let per_dim_scale = load_audio_weight_f32(file, &format!("{p}self_attn.per_dim_scale"))?;
            let (qkv_in_clip, qkv_out_clip) = load_linear_clip_ranges(file, &format!("{p}self_attn.q_proj"))?;
            let (post_in_clip, post_out_clip) = load_linear_clip_ranges(file, &format!("{p}self_attn.post"))?;
            if layer == 0 { eprintln!("[audio_timing] L0 attn weight_load={:.1}ms", t_wload.elapsed().as_secs_f64()*1000.0); }
            let t_qkv = std::time::Instant::now();

            // softplus(per_dim_scale) element-wise
            let pds: Vec<f32> = per_dim_scale.iter().map(|&v| (1.0 + v.exp()).ln()).collect();

            let residual = hidden_states.clone();
            rms_norm_rows(&hidden_states, t_sub, hidden, &norm_pre_w, eps, &mut norm_buf);
            if let Some((mn, mx)) = qkv_in_clip { for v in &mut norm_buf[..t_sub*hidden] { *v = v.clamp(mn, mx); } }

            linear_rows(&norm_buf, t_sub, hidden, &q_w, hidden, None, &mut q_buf, &mut backend);
            linear_rows(&norm_buf, t_sub, hidden, &k_w, hidden, None, &mut k_buf, &mut backend);
            linear_rows(&norm_buf, t_sub, hidden, &v_w, hidden, None, &mut v_buf, &mut backend);
            if let Some((mn, mx)) = qkv_out_clip {
                for v in &mut q_buf[..t_sub*hidden] { *v = v.clamp(mn, mx); }
                for v in &mut k_buf[..t_sub*hidden] { *v = v.clamp(mn, mx); }
                for v in &mut v_buf[..t_sub*hidden] { *v = v.clamp(mn, mx); }
            }

            // Scale q and k
            for t in 0..t_sub {
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        q_buf[t * hidden + h * head_dim + d] *= q_scale * pds[d];
                        k_buf[t * hidden + h * head_dim + d] *= k_scale;
                    }
                }
            }
            if layer == 0 {
                audio_stats("L0_q_scaled", &q_buf[..t_sub*hidden]);
                audio_stats("L0_k_scaled", &k_buf[..t_sub*hidden]);
                audio_stats("L0_v_buf", &v_buf[..t_sub*hidden]);
                let pds_min = pds.iter().cloned().fold(f32::INFINITY, f32::min);
                let pds_max = pds.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if std::env::var("CELLM_AUDIO_DEBUG").is_ok() {
                    eprintln!("L0_per_dim_scale softplus min={:.4} max={:.4} q_scale={:.6} k_scale={:.6}", pds_min, pds_max, q_scale, k_scale);
                }
            }

            // rel_k_proj on pos embeddings: [n_rel_pos, hidden] → [n_rel_pos, hidden]
            linear_rows(&rel_pos_embed, n_rel_pos, hidden, &rel_k_w, hidden, None, &mut rel_k_buf, &mut backend);

            // Chunked local attention with relative position bias
            let n_blocks = (t_sub + chunk_size - 1) / chunk_size;
            let padded_t = n_blocks * chunk_size;
            attn_out.resize(padded_t * hidden, 0.0);

            for b in 0..n_blocks {
                let q_start = b * chunk_size;
                for bq in 0..chunk_size {
                    let q_tok = q_start + bq;
                    if q_tok >= t_sub {
                        break;
                    }
                    for h in 0..num_heads {
                        let q_off = q_tok * hidden + h * head_dim;
                        let mut acc = vec![0.0f32; head_dim];

                        // Compute scores against context window (left-causal only)
                        let kv_start = (q_start as isize - context_left as isize + 1).max(0) as usize;
                        let kv_end = (q_tok + context_right + 1).min(t_sub);
                        let n_ctx = kv_end.saturating_sub(kv_start);

                        score_buf[..context_size].fill(attn_invalid);
                        // Matrix AC (content score)
                        for (ctx_i, kv_tok) in (kv_start..kv_end).enumerate() {
                            let k_off = kv_tok * hidden + h * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q_buf[q_off + d] * k_buf[k_off + d];
                            }
                            score_buf[ctx_i] = dot;
                        }

                        // Matrix BD (relative position score)
                        // rel_k_buf[d] corresponds to pos_id d (ascending), so lookup by distance is direct.
                        for (ctx_i, kv_tok) in (kv_start..kv_end).enumerate() {
                            if score_buf[ctx_i] <= attn_invalid + 1.0 { continue; }
                            let rel_pos_idx = (q_tok as isize - kv_tok as isize).min(n_rel_pos as isize - 1).max(0) as usize;
                            let rk_off = rel_pos_idx * hidden + h * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q_buf[q_off + d] * rel_k_buf[rk_off + d];
                            }
                            score_buf[ctx_i] += dot;
                        }

                        // Softcap scores first, then softmax
                        // First pass: apply softcap to all scores
                        for i in 0..n_ctx {
                            score_buf[i] = (score_buf[i] / attn_cap).tanh() * attn_cap;
                        }
                        // Find max of capped scores
                        let mut mx = f32::NEG_INFINITY;
                        for i in 0..n_ctx { if score_buf[i] > mx { mx = score_buf[i]; } }
                        if mx == f32::NEG_INFINITY { mx = 0.0; }
                        let mut sum = 0.0f32;
                        for i in 0..n_ctx {
                            let p = (score_buf[i] - mx).exp();
                            prob_buf[i] = p;
                            sum += p;
                        }
                        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };

                        // Weighted sum of values
                        acc.fill(0.0);
                        for (ctx_i, kv_tok) in (kv_start..kv_end).enumerate() {
                            let p = prob_buf[ctx_i] * inv;
                            let v_off = kv_tok * hidden + h * head_dim;
                            for d in 0..head_dim {
                                acc[d] += p * v_buf[v_off + d];
                            }
                        }
                        let out_off = q_tok * hidden + h * head_dim;
                        attn_out[out_off..out_off + head_dim].copy_from_slice(&acc);
                    }
                }
            }
            attn_out.truncate(t_sub * hidden);

            if layer == 0 {
                audio_stats("L0_attn_out_raw", &attn_out[..t_sub*hidden]);
                if std::env::var("CELLM_AUDIO_DEBUG").is_ok() {
                    // Find first NaN to understand context
                    'find_nan: for qt in 0..t_sub {
                        for h in 0..num_heads {
                            let off = qt * hidden + h * head_dim;
                            if attn_out[off..off+head_dim].iter().any(|v| v.is_nan()) {
                                let q_start_for_qt = (qt / chunk_size) * chunk_size;
                                let kv_start = (q_start_for_qt as isize - context_left as isize + 1).max(0) as usize;
                                let kv_end = (qt + context_right + 1).min(t_sub);
                                let n = kv_end.saturating_sub(kv_start);
                                eprintln!("L0_first_nan tok={qt} h={h} q_start={q_start_for_qt} kv_start={kv_start} kv_end={kv_end} n_ctx={n}");
                                let v_off = qt * hidden + h * head_dim;
                                let v_s = &v_buf[v_off..v_off+4];
                                eprintln!("  v_buf[tok,h,0..4]={:.4} {:.4} {:.4} {:.4}", v_s[0],v_s[1],v_s[2],v_s[3]);
                                break 'find_nan;
                            }
                        }
                    }
                }
            }

            // post projection
            if let Some((mn, mx)) = post_in_clip { for v in &mut attn_out[..t_sub*hidden] { *v = v.clamp(mn, mx); } }
            linear_rows(&attn_out, t_sub, hidden, &post_w, hidden, None, &mut norm_buf, &mut backend);
            if let Some((mn, mx)) = post_out_clip { for v in &mut norm_buf[..t_sub*hidden] { *v = v.clamp(mn, mx); } }
            // norm_post_attn + add residual
            rms_norm_rows(&norm_buf, t_sub, hidden, &norm_post_w, eps, &mut norm2_buf);
            for i in 0..t_sub * hidden {
                hidden_states[i] = residual[i] + norm2_buf[i];
            }
            if layer == 0 { audio_stats("L0_after_attn", &hidden_states[..t_sub*hidden]); }
            if layer == 0 { eprintln!("[audio_timing] L0 attn qkv+rest={:.1}ms total={:.1}ms", t_qkv.elapsed().as_secs_f64()*1000.0, t0.elapsed().as_secs_f64()*1000.0); }
        }

        //  lconv1d
        {
            let t0 = std::time::Instant::now();
            let pre_w = load_audio_weight_f32(file, &format!("{p}lconv1d.pre_layer_norm.weight"))?;
            let lin_start_w = load_audio_weight_f32(file, &format!("{p}lconv1d.linear_start.linear.weight"))?;
            let dw_w = load_audio_weight_f32(file, &format!("{p}lconv1d.depthwise_conv1d.weight"))?;
            let conv_norm_w = load_audio_weight_f32(file, &format!("{p}lconv1d.conv_norm.weight"))?;
            let lin_end_w = load_audio_weight_f32(file, &format!("{p}lconv1d.linear_end.linear.weight"))?;
            let (ls_in_clip, ls_out_clip) = load_linear_clip_ranges(file, &format!("{p}lconv1d.linear_start"))?;
            let (le_in_clip, le_out_clip) = load_linear_clip_ranges(file, &format!("{p}lconv1d.linear_end"))?;

            let residual = hidden_states.clone();
            rms_norm_rows(&hidden_states, t_sub, hidden, &pre_w, eps, &mut norm_buf);
            if let Some((mn, mx)) = ls_in_clip { for v in &mut norm_buf[..t_sub*hidden] { *v = v.clamp(mn, mx); } }

            // linear_start: [hidden, hidden*2]
            lconv_lin_buf.resize(t_sub * hidden * 2, 0.0);
            linear_rows(&norm_buf, t_sub, hidden, &lin_start_w, hidden * 2, None, &mut lconv_lin_buf, &mut backend);
            if let Some((mn, mx)) = ls_out_clip { for v in &mut lconv_lin_buf[..t_sub*hidden*2] { *v = v.clamp(mn, mx); } }

            // GLU: split into two halves, output = first_half * sigmoid(second_half)
            lconv_out.resize(t_sub * hidden, 0.0);
            for t in 0..t_sub {
                for c in 0..hidden {
                    let a = lconv_lin_buf[t * hidden * 2 + c];
                    let b = lconv_lin_buf[t * hidden * 2 + hidden + c];
                    lconv_out[t * hidden + c] = a / (1.0 + (-b).exp());
                }
            }

            // Depthwise causal conv1d: kernel=5, groups=hidden
            // Weight shape: [1024, 1, 5] → for each channel, convolve with 5-tap kernel
            // Causal: left-pad by (kernel-1)=4
            conv_state.resize((conv_kernel - 1) * hidden, 0.0);
            conv_state.fill(0.0); // reset state for each layer (non-streaming)
            let mut conv_result = vec![0.0f32; t_sub * hidden];
            // Build padded input [pad + t_sub, hidden]
            let pad_size = conv_kernel - 1;
            let mut padded_input = vec![0.0f32; (pad_size + t_sub) * hidden];
            padded_input[pad_size * hidden..].copy_from_slice(&lconv_out[..t_sub * hidden]);
            for t in 0..t_sub {
                for c in 0..hidden {
                    let mut acc = 0.0f32;
                    for k in 0..conv_kernel {
                        acc += padded_input[(t + k) * hidden + c] * dw_w[c * conv_kernel + k];
                    }
                    conv_result[t * hidden + c] = acc;
                }
            }

            // conv_norm (RMSNorm with scale)
            rms_norm_rows(&conv_result, t_sub, hidden, &conv_norm_w, eps, &mut norm_buf);
            // SiLU
            silu_inplace(&mut norm_buf[..t_sub * hidden]);
            // linear_end: [hidden, hidden]
            if let Some((mn, mx)) = le_in_clip { for v in &mut norm_buf[..t_sub*hidden] { *v = v.clamp(mn, mx); } }
            linear_rows(&norm_buf, t_sub, hidden, &lin_end_w, hidden, None, &mut lconv_out, &mut backend);
            if let Some((mn, mx)) = le_out_clip { for v in &mut lconv_out[..t_sub*hidden] { *v = v.clamp(mn, mx); } }
            for i in 0..t_sub * hidden {
                hidden_states[i] = residual[i] + lconv_out[i];
            }
            if layer == 0 { audio_stats("L0_after_lconv", &hidden_states[..t_sub*hidden]); }
            if layer == 0 { eprintln!("[audio_timing] L0 lconv total={:.1}ms", t0.elapsed().as_secs_f64()*1000.0); }
        }

        //  feed_forward2
        {
            let t0 = std::time::Instant::now();
            let pre_w = load_audio_weight_f32(file, &format!("{p}feed_forward2.pre_layer_norm.weight"))?;
            let ff1_w = load_audio_weight_f32(file, &format!("{p}feed_forward2.ffw_layer_1.linear.weight"))?;
            let ff2_w = load_audio_weight_f32(file, &format!("{p}feed_forward2.ffw_layer_2.linear.weight"))?;
            let post_w = load_audio_weight_f32(file, &format!("{p}feed_forward2.post_layer_norm.weight"))?;
            let (ff1_in_clip, ff1_out_clip) = load_linear_clip_ranges(file, &format!("{p}feed_forward2.ffw_layer_1"))?;
            let (ff2_in_clip, ff2_out_clip) = load_linear_clip_ranges(file, &format!("{p}feed_forward2.ffw_layer_2"))?;

            let residual = hidden_states.clone();
            rms_norm_rows(&hidden_states, t_sub, hidden, &pre_w, eps, &mut norm_buf);
            if let Some((mn, mx)) = ff1_in_clip { for v in &mut norm_buf[..t_sub*hidden] { *v = v.clamp(mn, mx); } }
            linear_rows(&norm_buf, t_sub, hidden, &ff1_w, ffw_intermediate, None, &mut ffw1_buf, &mut backend);
            if let Some((mn, mx)) = ff1_out_clip { for v in &mut ffw1_buf[..t_sub*ffw_intermediate] { *v = v.clamp(mn, mx); } }
            silu_inplace(&mut ffw1_buf);
            if let Some((mn, mx)) = ff2_in_clip { for v in &mut ffw1_buf[..t_sub*ffw_intermediate] { *v = v.clamp(mn, mx); } }
            linear_rows(&ffw1_buf, t_sub, ffw_intermediate, &ff2_w, hidden, None, &mut ffw2_buf, &mut backend);
            if let Some((mn, mx)) = ff2_out_clip { for v in &mut ffw2_buf[..t_sub*hidden] { *v = v.clamp(mn, mx); } }
            rms_norm_rows(&ffw2_buf, t_sub, hidden, &post_w, eps, &mut norm_buf);
            for i in 0..t_sub * hidden {
                hidden_states[i] = residual[i] + norm_buf[i] * residual_weight;
            }
            if layer == 0 { eprintln!("[audio_timing] L0 ff2 total={:.1}ms", t0.elapsed().as_secs_f64()*1000.0); }
        }

        //  norm_out
        {
            let norm_out_w = load_audio_weight_f32(file, &format!("{p}norm_out.weight"))?;
            rms_norm_rows(&hidden_states.clone(), t_sub, hidden, &norm_out_w, eps, &mut norm_buf);
            hidden_states[..t_sub * hidden].copy_from_slice(&norm_buf[..t_sub * hidden]);
        }
        if layer < 2 || layer == 11 {
            audio_stats(&format!("after_layer_{layer}"), &hidden_states[..t_sub * hidden]);
        }
        eprintln!("[audio_timing] layer {layer} done in {:.1}ms", t_layer_start.elapsed().as_secs_f64() * 1000.0);
    }
    eprintln!("[audio_timing] all conformer layers done in {:.1}ms", t_conformer_start.elapsed().as_secs_f64() * 1000.0);

    //  D. Output projection [1024 → 1536]
    let t_outproj_start = std::time::Instant::now();
    let out_proj_w = load_audio_weight_f32(file, "model.audio_tower.output_proj.weight")?;
    let out_proj_b = load_audio_weight_f32(file, "model.audio_tower.output_proj.bias")?;
    let mut proj_out = vec![0.0f32; t_sub * text_hidden];
    linear_rows(&hidden_states, t_sub, hidden, &out_proj_w, text_hidden, Some(&out_proj_b), &mut proj_out, &mut backend);
    audio_stats("D_out_proj", &proj_out[..t_sub * text_hidden]);

    eprintln!("[audio_timing] output_proj done in {:.1}ms", t_outproj_start.elapsed().as_secs_f64() * 1000.0);
    //  E. Embedder (RMSNorm no-scale + Linear 1536→1536)
    let embed_proj_w = load_audio_weight_f32(file, "model.embed_audio.embedding_projection.weight")?;
    let mut embed_norm = vec![0.0f32; t_sub * text_hidden];
    // RMSNorm with no scale (with_scale=False)
    for t in 0..t_sub {
        let x = &proj_out[t * text_hidden..(t + 1) * text_hidden];
        let y = &mut embed_norm[t * text_hidden..(t + 1) * text_hidden];
        let mut ms = 0.0f32;
        for &v in x { ms += v * v; }
        ms /= text_hidden as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..text_hidden { y[i] = x[i] * inv; }
    }
    audio_stats("E_after_rmsnorm", &embed_norm[..t_sub * text_hidden]);
    let mut embed_out = vec![0.0f32; t_sub * text_hidden];
    linear_rows(&embed_norm, t_sub, text_hidden, &embed_proj_w, text_hidden, None, &mut embed_out, &mut backend);
    audio_stats("E_embed_out", &embed_out[..t_sub * text_hidden]);

    let out = Array2::from_shape_vec((t_sub, text_hidden), embed_out)
        .map_err(|e| anyhow::anyhow!("audio features shape error: {e}"))?;
    Ok(out)
}

fn tensor_shape(file: &CellmFile, name: &str) -> Result<Vec<usize>> {
    file.tensor_index(name)
        .map(|t| t.shape.clone())
        .ok_or_else(|| anyhow::anyhow!("missing tensor: {name}"))
}

/// Load an audio tower weight as f32, handling both f16 and i4 quantized dtypes.
///
/// The Gemma 4 audio conformer stores its linear weights as i4 with a per-row
/// f16 qscale tensor at `{name}.qscale`. Reading them naively with
/// `tensor_f16_to_f32` yields 4× too few values (i4 packs 2 values per byte;
/// interpreted as f16 that is 8× smaller per element than f32, not 4×), causing
/// out-of-bounds panics in `linear_rows` around row 256.
fn load_audio_weight_f32(file: &CellmFile, name: &str) -> Result<Vec<f32>> {
    let index = file
        .tensor_index(name)
        .ok_or_else(|| anyhow::anyhow!("missing audio weight: {name}"))?;
    match index.dtype.as_str() {
        "f16" => tensor_f16_to_f32(file, name),
        "i4" => {
            let bytes = file
                .tensor_bytes(name)
                .map_err(|e| anyhow::anyhow!("tensor bytes {name}: {e}"))?;
            let scale_name = format!("{name}.qscale");
            let scale_bytes = file
                .tensor_bytes(&scale_name)
                .map_err(|e| anyhow::anyhow!("qscale bytes {scale_name}: {e}"))?;
            let scales_u16: &[u16] = cast_slice(scale_bytes);

            // Treat the weight as [out_dim, in_dim] regardless of how many
            // dimensions it has: the product of all but the first is in_dim.
            let out_dim = index.shape[0];
            let in_dim: usize = index.shape[1..].iter().product::<usize>().max(1);

            // Parallel dequantization: each output row is independent.
            let scales_f32: Vec<f32> = scales_u16.iter().map(|&s| f16::from_bits(s).to_f32()).collect();
            let mut result = vec![0.0f32; out_dim * in_dim];
            result.par_chunks_mut(in_dim).enumerate().for_each(|(r, row)| {
                let scale = scales_f32[r];
                for c in 0..in_dim {
                    let flat = r * in_dim + c;
                    let byte = bytes[flat / 2];
                    // Lower nibble → even index; upper nibble → odd index.
                    let nibble = if flat % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                    // Sign-extend 4-bit value: range [-8, 7].
                    let s = (nibble as i8) - (if nibble >= 8 { 16i8 } else { 0i8 });
                    row[c] = s as f32 * scale;
                }
            });
            Ok(result)
        }
        other => anyhow::bail!(
            "unsupported audio weight dtype '{other}' for '{name}' \
             (expected f16 or i4)"
        ),
    }
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

fn tensor_scalar_f32_optional(file: &CellmFile, name: &str) -> Result<Option<f32>> {
    if file.tensor_index(name).is_none() {
        return Ok(None);
    }
    let vals = tensor_to_f32(file, name)?;
    if vals.len() != 1 {
        anyhow::bail!("expected scalar tensor for {name}, got len={}", vals.len());
    }
    Ok(Some(vals[0]))
}

fn load_linear_clip_ranges(
    file: &CellmFile,
    prefix: &str,
) -> Result<(Option<(f32, f32)>, Option<(f32, f32)>)> {
    let in_min = tensor_scalar_f32_optional(file, &format!("{prefix}.input_min"))?;
    let in_max = tensor_scalar_f32_optional(file, &format!("{prefix}.input_max"))?;
    let out_min = tensor_scalar_f32_optional(file, &format!("{prefix}.output_min"))?;
    let out_max = tensor_scalar_f32_optional(file, &format!("{prefix}.output_max"))?;
    let in_clip = match (in_min, in_max) {
        (Some(mn), Some(mx)) if mn.is_finite() && mx.is_finite() => Some((mn, mx)),
        _ => None,
    };
    let out_clip = match (out_min, out_max) {
        (Some(mn), Some(mx)) if mn.is_finite() && mx.is_finite() => Some((mn, mx)),
        _ => None,
    };
    Ok((in_clip, out_clip))
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
    for r in 0..rows {
        let x = &input[r * cols..(r + 1) * cols];
        let y = &mut out[r * cols..(r + 1) * cols];
        let mut mean = 0.0f32;
        for &val in x {
            mean += val;
        }
        mean /= cols as f32;
        let mut var = 0.0f32;
        for &val in x {
            let d = val - mean;
            var += d * d;
        }
        var /= cols as f32;
        let inv = 1.0f32 / (var + eps).sqrt();
        for i in 0..cols {
            y[i] = ((x[i] - mean) * inv) * weight[i] + bias[i];
        }
    }
}

fn rms_norm_rows(
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    for r in 0..rows {
        let x = &input[r * cols..(r + 1) * cols];
        let y = &mut out[r * cols..(r + 1) * cols];
        let mut ms = 0.0f32;
        for &val in x {
            ms += val * val;
        }
        ms /= cols as f32;
        let inv = 1.0f32 / (ms + eps).sqrt();
        for i in 0..cols {
            y[i] = x[i] * inv * weight[i];
        }
    }
}

fn rms_norm_inplace_segment(x: &mut [f32], weight: &[f32], eps: f32) {
    let mut ms = 0.0f32;
    for &v in x.iter() {
        ms += v * v;
    }
    ms /= x.len() as f32;
    let inv = 1.0f32 / (ms + eps).sqrt();
    for i in 0..x.len() {
        x[i] = x[i] * inv * weight[i];
    }
}

fn rms_norm_inplace_noscale(x: &mut [f32], eps: f32) {
    let mut ms = 0.0f32;
    for &v in x.iter() {
        ms += v * v;
    }
    ms /= x.len() as f32;
    let inv = 1.0f32 / (ms + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn apply_rope_1d_inplace(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos[i] - b * sin[i];
        x[i + half] = b * cos[i] + a * sin[i];
    }
}

fn apply_multidim_rope_inplace(x: &mut [f32], cos_xy: &[f32], sin_xy: &[f32]) {
    let spatial = x.len() / 2;
    let (x_part, y_part) = x.split_at_mut(spatial);
    let (cos_x, cos_y) = cos_xy.split_at(spatial / 2);
    let (sin_x, sin_y) = sin_xy.split_at(spatial / 2);
    apply_rope_1d_inplace(x_part, cos_x, sin_x);
    apply_rope_1d_inplace(y_part, cos_y, sin_y);
}

fn patch_embed_rows(
    pixel_values: &Array5<f32>,
    img: usize,
    grid: usize,
    patch: usize,
    hidden: usize,
    patch_w: &[f32],
    patch_b: &[f32],
    pos: &[f32],
    tokens_out: &mut [f32],
    backend: &mut LinearBackend,
) {
    let rows = grid * grid;
    let in_dim = 3 * patch * patch;
    let mut im2col = vec![0.0f32; rows * in_dim];
    for py in 0..grid {
        for px in 0..grid {
            let token_idx = py * grid + px;
            let mut col_i = 0usize;
            for ic in 0..3usize {
                for ky in 0..patch {
                    for kx in 0..patch {
                        let y = py * patch + ky;
                        let x = px * patch + kx;
                        im2col[token_idx * in_dim + col_i] = pixel_values[[0, img, ic, y, x]];
                        col_i += 1;
                    }
                }
            }
        }
    }

    linear_rows(
        &im2col,
        rows,
        in_dim,
        patch_w,
        hidden,
        Some(patch_b),
        tokens_out,
        backend,
    );
    for t in 0..rows {
        let row = &mut tokens_out[t * hidden..(t + 1) * hidden];
        let p = &pos[t * hidden..(t + 1) * hidden];
        for i in 0..hidden {
            row[i] += p[i];
        }
    }
}

fn patch_embed_rows_linear(
    pixel_values: &Array5<f32>,
    img: usize,
    grid: usize,
    patch: usize,
    hidden: usize,
    patch_w: &[f32],
    pos: &[f32],
    tokens_out: &mut [f32],
    backend: &mut LinearBackend,
) {
    let rows = grid * grid;
    let in_dim = 3 * patch * patch;
    let mut im2col = vec![0.0f32; rows * in_dim];
    for py in 0..grid {
        for px in 0..grid {
            let token_idx = py * grid + px;
            let mut col_i = 0usize;
            for ic in 0..3usize {
                for ky in 0..patch {
                    for kx in 0..patch {
                        let y = py * patch + ky;
                        let x = px * patch + kx;
                        im2col[token_idx * in_dim + col_i] = pixel_values[[0, img, ic, y, x]];
                        col_i += 1;
                    }
                }
            }
        }
    }

    linear_rows(
        &im2col,
        rows,
        in_dim,
        patch_w,
        hidden,
        None,
        tokens_out,
        backend,
    );
    for t in 0..rows {
        let row = &mut tokens_out[t * hidden..(t + 1) * hidden];
        let p = &pos[t * hidden..(t + 1) * hidden];
        for i in 0..hidden {
            row[i] += p[i];
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
    backend: &mut LinearBackend,
) {
    if let LinearBackend::Metal {
        ctx,
        weight_t_cache,
    } = backend
    {
        let key = (weight.as_ptr() as usize, in_dim, out_dim);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let maybe_ok = {
            if !weight_t_cache.contains_key(&key) {
                let mut wt = vec![0.0f32; in_dim * out_dim];
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        wt[i * out_dim + o] = weight[o * in_dim + i];
                    }
                }
                if let Ok(buf) = ctx.upload_f32(&wt) {
                    weight_t_cache.insert(key, buf);
                }
            }

            if let Some(b_buf) = weight_t_cache.get(&key) {
                if let Ok(a_buf) = ctx.upload_f32(input) {
                    ctx.matmul_row_major_f32_with_b_buffer(&a_buf, rows, in_dim, b_buf, out_dim, out)
                        .is_ok()
                } else {
                    false
                }
            } else {
                false
            }
        };

        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let maybe_ok = false;

        if maybe_ok {
            if let Some(b) = bias {
                add_bias_rows(out, rows, out_dim, b);
            }
            return;
        }
    }
    // For large matrices (audio conformer, vision encoder) use rayon to
    // parallelise over input rows.  The threshold avoids spawning threads for
    // the tiny projections used during token-by-token decode.
    let total_ops = rows * in_dim * out_dim;
    if total_ops >= 1 << 20 {
        let out_slice = &mut out[..rows * out_dim];
        let inp_slice = &input[..rows * in_dim];
        out_slice
            .par_chunks_mut(out_dim)
            .zip(inp_slice.par_chunks(in_dim))
            .for_each(|(y, x)| {
                if let Some(b) = bias {
                    y.copy_from_slice(b);
                } else {
                    y.fill(0.0);
                }
                for o in 0..out_dim {
                    let wrow = &weight[o * in_dim..(o + 1) * in_dim];
                    let mut acc = y[o];
                    for i in 0..in_dim {
                        acc += x[i] * wrow[i];
                    }
                    y[o] = acc;
                }
            });
    } else {
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
}

fn linear_rows_clipped(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    input_clip: Option<(f32, f32)>,
    output_clip: Option<(f32, f32)>,
    backend: &mut LinearBackend,
) {
    if let Some((mn, mx)) = input_clip {
        let mut clipped = vec![0.0f32; input.len()];
        for (d, &s) in clipped.iter_mut().zip(input.iter()) {
            *d = s.clamp(mn, mx);
        }
        linear_rows(&clipped, rows, in_dim, weight, out_dim, bias, out, backend);
    } else {
        linear_rows(input, rows, in_dim, weight, out_dim, bias, out, backend);
    }
    if let Some((mn, mx)) = output_clip {
        for v in out.iter_mut() {
            *v = v.clamp(mn, mx);
        }
    }
}

enum LinearBackend {
    Cpu,
    Metal {
        ctx: MetalMatmul,
        weight_t_cache: HashMap<(usize, usize, usize), MetalBuffer>,
    },
}

fn add_bias_rows(out: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    for r in 0..rows {
        let row = &mut out[r * cols..(r + 1) * cols];
        for c in 0..cols {
            row[c] += bias[c];
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
    scale_override: Option<f32>,
    score_buf: &mut [f32],
    prob_buf: &mut [f32],
    out: &mut [f32],
    valid_mask: Option<&[bool]>,
    backend: &mut LinearBackend,
) {
    let hidden = num_heads * head_dim;
    let scale = scale_override.unwrap_or(1.0f32 / (head_dim as f32).sqrt());

    if valid_mask.is_none() {
        if let LinearBackend::Metal { ctx, .. } = backend {
            if self_attention_full_metal(
                ctx, q, k, v, seq, num_heads, head_dim, scale, score_buf, prob_buf, out,
            )
            .is_ok()
            {
                return;
            }
        }
    }

    if let Some(mask) = valid_mask {
        for h in 0..num_heads {
            let offset = h * head_dim;
            for i in 0..seq {
                if !mask[i] {
                    for d in 0..head_dim {
                        out[i * hidden + offset + d] = 0.0;
                    }
                    continue;
                }
                for j in 0..seq {
                    if !mask[j] {
                        score_buf[j] = -1.0e30;
                        continue;
                    }
                    let qi = &q[i * hidden + offset..i * hidden + offset + head_dim];
                    let kj = &k[j * hidden + offset..j * hidden + offset + head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += qi[d] * kj[d];
                    }
                    score_buf[j] = dot * scale;
                }
                let mut mx = f32::NEG_INFINITY;
                for j in 0..seq {
                    if score_buf[j] > mx {
                        mx = score_buf[j];
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
        return;
    }

    if let LinearBackend::Metal { ctx, .. } = backend {
        if self_attention_full_metal(
            ctx, q, k, v, seq, num_heads, head_dim, scale, score_buf, prob_buf, out,
        )
        .is_ok()
        {
            return;
        }
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

fn self_attention_full_metal(
    ctx: &MetalMatmul,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    score_buf: &mut [f32],
    prob_buf: &mut [f32],
    out: &mut [f32],
) -> Result<()> {
    let hidden = num_heads * head_dim;
    let mut qh = vec![0.0f32; seq * head_dim];
    let mut kh_t = vec![0.0f32; head_dim * seq];
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
            vh[t * head_dim..(t + 1) * head_dim].copy_from_slice(v_src);
            for d in 0..head_dim {
                kh_t[d * seq + t] = k_src[d];
            }
        }

        ctx.matmul_row_major_f32(&qh, seq, head_dim, &kh_t, seq, &mut scores)?;
        for val in &mut scores {
            *val *= scale;
        }

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

        ctx.matmul_row_major_f32(&probs, seq, seq, &vh, head_dim, &mut head_out)?;

        for t in 0..seq {
            let dst = &mut out[t * hidden + offset..t * hidden + offset + head_dim];
            let src = &head_out[t * head_dim..(t + 1) * head_dim];
            dst.copy_from_slice(src);
        }
    }
    let _ = (score_buf, prob_buf);
    Ok(())
}

fn gelu_pytorch_tanh_inplace(x: &mut [f32]) {
    const A: f32 = 0.797_884_6;
    const B: f32 = 0.044_715;
    for v in x.iter_mut() {
        let t = A * (*v + B * *v * *v * *v);
        *v = 0.5 * *v * (1.0 + t.tanh());
    }
}

fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

fn mul_inplace(dst: &mut [f32], rhs: &[f32]) {
    for (d, r) in dst.iter_mut().zip(rhs.iter()) {
        *d *= *r;
    }
}

fn add_inplace(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn build_single_turn_prompt(user_text: &str, image_block: &str, use_gemma4_turn: bool) -> String {
    if use_gemma4_turn {
        if image_block.is_empty() {
            return format!("<bos><|turn>user\n{user_text}<turn|>\n<|turn>model\n");
        }
        // Match Gemma4 processor chat template layout more closely:
        // BOS, user turn header, blank lines, image block, blank lines, user text.
        return format!("<bos><|turn>user\n\n\n{image_block}\n\n{user_text}<turn|>\n<|turn>model\n");
    }
    format!("<|im_start|>User:{image_block}{user_text}<end_of_utterance>\nAssistant:")
}

fn format_image_block(
    num_images: usize,
    image_seq_len: usize,
    image_token_text: &str,
    use_idefics_wrappers: bool,
    boi_token_text: Option<&str>,
    eoi_token_text: Option<&str>,
) -> String {
    let image_tokens = image_token_text.repeat(image_seq_len);
    let mut block = String::new();
    let wrap_boi_eoi = !use_idefics_wrappers && boi_token_text.is_some() && eoi_token_text.is_some();
    for image_idx in 0..num_images {
        if image_idx == 0 && use_idefics_wrappers {
            block.push_str("<fake_token_around_image><global-img>");
            block.push_str(&image_tokens);
            block.push_str("<fake_token_around_image>");
            continue;
        }
        if image_idx == 0 {
            if wrap_boi_eoi {
                block.push_str(boi_token_text.unwrap());
            }
            block.push_str(&image_tokens);
            if wrap_boi_eoi {
                block.push_str(eoi_token_text.unwrap());
            }
            continue;
        }
        if !use_idefics_wrappers {
            if wrap_boi_eoi {
                block.push_str(boi_token_text.unwrap());
            }
            block.push_str(&image_tokens);
            if wrap_boi_eoi {
                block.push_str(eoi_token_text.unwrap());
            }
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

fn resolve_image_token(tok: &Tokenizer) -> Result<(i64, String, bool, Option<String>, Option<String>)> {
    let boi = tok
        .token_to_id("<|image>")
        .map(|_| "<|image>".to_string());
    let eoi = tok
        .token_to_id("<image|>")
        .map(|_| "<image|>".to_string());
    let is_gemma4_turn = tok.token_to_id("<|turn>").is_some() && tok.token_to_id("<turn|>").is_some();
    if is_gemma4_turn {
        if let Some(id) = tok.token_to_id("<|image|>") {
            return Ok((id as i64, "<|image|>".to_string(), false, boi, eoi));
        }
    }
    if let Some(id) = tok.token_to_id("<image>") {
        return Ok((id as i64, "<image>".to_string(), true, boi, eoi));
    }
    if let Some(id) = tok.token_to_id("<image_soft_token>") {
        return Ok((id as i64, "<image_soft_token>".to_string(), false, boi, eoi));
    }
    if let Some(id) = tok.token_to_id("<|image|>") {
        return Ok((id as i64, "<|image|>".to_string(), false, boi, eoi));
    }
    if let Some(id) = tok.token_to_id("<|image>") {
        return Ok((id as i64, "<|image>".to_string(), false, boi, eoi));
    }
    if let Some(id) = tok.token_to_id("<image|>") {
        return Ok((id as i64, "<image|>".to_string(), false, boi, eoi));
    }
    anyhow::bail!(
        "tokenizer missing supported image token (<image>, <image_soft_token>, <|image|>, <|image>, <image|>)"
    )
}

struct ResizedPadded {
    image: RgbImage,
    valid_w: usize,
    valid_h: usize,
    offset_x: usize,
    offset_y: usize,
}

fn resize_and_pad_512(rgb: &RgbImage) -> ResizedPadded {
    let (w, h) = rgb.dimensions();
    let target = 512u32;
    let scale = target as f32 / (w.max(h) as f32);
    let new_w = ((w as f32) * scale).round().max(1.0) as u32;
    let new_h = ((h as f32) * scale).round().max(1.0) as u32;
    let resized = image::imageops::resize(rgb, new_w, new_h, image::imageops::FilterType::CatmullRom);
    let mut canvas = RgbImage::new(target, target);
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

fn resize_longest_edge(rgb: &RgbImage, target_longest: u32) -> RgbImage {
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return rgb.clone();
    }
    let longest = w.max(h);
    if longest == target_longest {
        return rgb.clone();
    }
    let scale = target_longest as f32 / longest as f32;
    let new_w = ((w as f32) * scale).round().max(1.0) as u32;
    let new_h = ((h as f32) * scale).round().max(1.0) as u32;
    image::imageops::resize(rgb, new_w, new_h, image::imageops::FilterType::CatmullRom)
}

fn copy_rgb_to_nchw(dst: &mut Array5<f32>, image_idx: usize, rgb: &RgbImage) {
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

fn preprocess_image_for_model(
    image_bytes: &[u8],
    split_image: bool,
    is_gemma4_vision: bool,
    max_soft_tokens: usize,
) -> Result<PreparedImage> {
    if is_gemma4_vision {
        let max_soft_tokens = std::env::var("CELLM_VLM_MAX_SOFT_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(max_soft_tokens);
        return preprocess_image_gemma4(image_bytes, max_soft_tokens);
    }
    preprocess_image_idefics3(image_bytes, split_image)
}

fn get_aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize)> {
    let total_px = (height * width) as f32;
    let target_px = (max_patches * patch_size * patch_size) as f32;
    let factor = (target_px / total_px).sqrt();
    let ideal_h = factor * height as f32;
    let ideal_w = factor * width as f32;
    let side_mult = pooling_kernel_size * patch_size;
    let mut target_h = ((ideal_h / side_mult as f32).floor() as usize) * side_mult;
    let mut target_w = ((ideal_w / side_mult as f32).floor() as usize) * side_mult;
    if target_h == 0 && target_w == 0 {
        anyhow::bail!("Gemma4 resize collapsed to 0x0");
    }
    let max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_h == 0 {
        target_h = side_mult;
        target_w = ((width / height).max(1)) * side_mult;
        target_w = target_w.min(max_side).max(side_mult);
    } else if target_w == 0 {
        target_w = side_mult;
        target_h = ((height / width).max(1)) * side_mult;
        target_h = target_h.min(max_side).max(side_mult);
    }
    Ok((target_h, target_w))
}

fn preprocess_image_gemma4(image_bytes: &[u8], max_soft_tokens: usize) -> Result<PreparedImage> {
    let img = image::load_from_memory(image_bytes).context("decode image bytes failed")?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let patch_size = 16usize;
    let pooling = 3usize;
    let max_patches = max_soft_tokens * pooling * pooling;
    let (target_h, target_w) =
        get_aspect_ratio_preserving_size(h as usize, w as usize, patch_size, max_patches, pooling)?;
    let resized = image::imageops::resize(
        &rgb,
        target_w as u32,
        target_h as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let patch_h = target_h / patch_size;
    let patch_w = target_w / patch_size;
    let valid_patches = patch_h * patch_w;
    let mut patch_values = Array3::<f32>::zeros((1, max_patches, 3 * patch_size * patch_size));
    let mut pos_ids = Array3::<i32>::from_elem((1, max_patches, 2), -1);
    for py in 0..patch_h {
        for px in 0..patch_w {
            let t = py * patch_w + px;
            pos_ids[[0, t, 0]] = px as i32;
            pos_ids[[0, t, 1]] = py as i32;
            let mut idx = 0usize;
            for c in 0..3 {
                for ky in 0..patch_size {
                    for kx in 0..patch_size {
                        let p = resized
                            .get_pixel((px * patch_size + kx) as u32, (py * patch_size + ky) as u32)
                            .0[c];
                        patch_values[[0, t, idx]] = p as f32 / 255.0;
                        idx += 1;
                    }
                }
            }
        }
    }
    let num_soft = valid_patches / (pooling * pooling);
    let mut pixel_values = Array5::<f32>::zeros((1, 1, 3, 1, 1));
    pixel_values[[0, 0, 0, 0, 0]] = 0.0;
    Ok(PreparedImage {
        pixel_values,
        gemma4_patch_values: Some(patch_values),
        gemma4_position_ids: Some(pos_ids),
        gemma4_num_soft_tokens: Some(vec![num_soft]),
    })
}

fn preprocess_image_idefics3(image_bytes: &[u8], split_image: bool) -> Result<PreparedImage> {
    let img = image::load_from_memory(image_bytes).context("decode image bytes failed")?;
    let rgb = img.to_rgb8();
    let mut images = Vec::new();
    images.push(resize_and_pad_512(&rgb));
    if split_image {
        // Match Idefics3-style processor behavior: split patches from a
        // resized longest-edge image (2048), then each patch is padded/resized to 512.
        let split_base = resize_longest_edge(&rgb, 2048);
        let (w, h) = split_base.dimensions();
        let tile = 512u32;
        let cols = ((w + tile - 1) / tile).clamp(1, 6);
        let rows = ((h + tile - 1) / tile).clamp(1, 6);
        let mut coords = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                coords.push((row, col));
            }
        }
        let max_tiles = max_split_tiles();
        let selected = select_spread_tiles(&coords, max_tiles);
        for (row, col) in selected {
            let x = col * tile;
            let y = row * tile;
            let cw = (w - x).min(tile);
            let ch = (h - y).min(tile);
            let crop = image::imageops::crop_imm(&split_base, x, y, cw, ch).to_image();
            images.push(resize_and_pad_512(&crop));
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
    Ok(PreparedImage {
        pixel_values,
        gemma4_patch_values: None,
        gemma4_position_ids: None,
        gemma4_num_soft_tokens: None,
    })
}

fn max_split_tiles() -> usize {
    std::env::var("CELLM_VLM_MAX_SPLIT_TILES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|v| v.clamp(1, 36))
        .unwrap_or(4)
}

fn select_spread_tiles(coords: &[(u32, u32)], max_tiles: usize) -> Vec<(u32, u32)> {
    if coords.is_empty() || max_tiles == 0 {
        return Vec::new();
    }
    if coords.len() <= max_tiles {
        return coords.to_vec();
    }
    let mut out = Vec::with_capacity(max_tiles);
    let span = (coords.len() - 1) as f32;
    for i in 0..max_tiles {
        let t = if max_tiles == 1 {
            0.5
        } else {
            i as f32 / (max_tiles - 1) as f32
        };
        let idx = (t * span).round() as usize;
        let pick = coords[idx];
        if !out.contains(&pick) {
            out.push(pick);
        }
    }
    if out.len() < max_tiles {
        for &c in coords {
            if !out.contains(&c) {
                out.push(c);
                if out.len() == max_tiles {
                    break;
                }
            }
        }
    }
    out
}

fn load_vlm_processor_hints(model_path: &Path, tokenizer_path: &Path, tok: &Tokenizer) -> VlmProcessorHints {
    let default_split = tok.token_to_id("<global-img>").is_some() && tok.token_to_id("<row_1_col_1>").is_some();
    let mut hints = VlmProcessorHints {
        image_seq_len: None,
        do_image_splitting: default_split,
    };

    if let Ok(v) = std::env::var("CELLM_VLM_SPLIT_IMAGE") {
        let lowered = v.trim().to_ascii_lowercase();
        if lowered == "0" || lowered == "false" || lowered == "no" {
            hints.do_image_splitting = false;
        } else if lowered == "1" || lowered == "true" || lowered == "yes" {
            hints.do_image_splitting = true;
        }
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = tokenizer_path.parent() {
        candidates.push(dir.join("processor_config.json"));
        candidates.push(dir.join("preprocessor_config.json"));
        candidates.push(dir.join("processor_config-smolvlm-256m.json"));
        candidates.push(dir.join("preprocessor_config-smolvlm-256m.json"));
    }
    if let Some(dir) = model_path.parent() {
        candidates.push(dir.join("processor_config.json"));
        candidates.push(dir.join("preprocessor_config.json"));
        candidates.push(dir.join("processor_config-smolvlm-256m.json"));
        candidates.push(dir.join("preprocessor_config-smolvlm-256m.json"));
    }

    for path in candidates {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Ok(json) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        if hints.image_seq_len.is_none() {
            hints.image_seq_len = json
                .get("image_seq_len")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            if hints.image_seq_len.is_none() {
                hints.image_seq_len = json
                    .get("image_seq_length")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if hints.image_seq_len.is_none() {
                hints.image_seq_len = json
                    .get("image_processor")
                    .and_then(|v| v.get("image_seq_length"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
        }
        if let Some(split) = json.get("do_image_splitting").and_then(|v| v.as_bool()) {
            hints.do_image_splitting = split;
        }
    }

    hints
}

fn encode_with_explicit_added_tokens(
    tok: &Tokenizer,
    tokenizer_json_path: &Path,
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

fn load_added_token_ids(tokenizer_json_path: &Path) -> Result<HashMap<String, u32>> {
    let bytes = std::fs::read(tokenizer_json_path).map_err(|e| {
        anyhow::anyhow!("read tokenizer {:?} failed: {e}", tokenizer_json_path)
    })?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| {
        anyhow::anyhow!("parse tokenizer {:?} failed: {e}", tokenizer_json_path)
    })?;

    let mut out = HashMap::new();
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

fn banned_token_ids(tok: &Tokenizer) -> Vec<i64> {
    let mut banned = Vec::new();
    for token in ["<|im_start|>", "<|im_end|>", "<image>", "<fake_token_around_image>", "<global-img>"] {
        if let Some(id) = tok.token_to_id(token) {
            banned.push(id as i64);
        }
    }
    for row in 1..=6 {
        for col in 1..=6 {
            let token = format!("<row_{row}_col_{col}>");
            if let Some(id) = tok.token_to_id(&token) {
                banned.push(id as i64);
            }
        }
    }
    banned.sort_unstable();
    banned.dedup();
    banned
}

fn resolve_tokenizer_path(model_path: &Path) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("CELLM_VLM_TOKENIZER") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }

    let dir = model_path
        .parent()
        .context("model path has no parent directory")?;
    let stem = model_path
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or("model");
    let mut candidates = vec![
        dir.join("tokenizer-smolvlm-256m.json"),
        dir.join("tokenizer.json"),
        dir.join(format!("{stem}.tokenizer.json")),
    ];
    if stem.ends_with("-int8") {
        candidates.push(dir.join(format!("{}.tokenizer.json", stem.trim_end_matches("-int8"))));
    }
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    anyhow::bail!(
        "missing tokenizer for VLM. Place tokenizer JSON next to model (tried tokenizer-smolvlm-256m.json/tokenizer.json) or set CELLM_VLM_TOKENIZER"
    )
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    match Tokenizer::from_file(path) {
        Ok(t) => Ok(t),
        Err(e) => {
            let normalized = try_normalize_tokenizer_json(path)?;
            if let Some(p) = normalized {
                return Tokenizer::from_file(&p)
                    .map_err(|e| anyhow::anyhow!("load tokenizer failed (normalized {:?}): {e}", p));
            }
            Err(anyhow::anyhow!("load tokenizer failed: {e}"))
        }
    }
}

fn try_normalize_tokenizer_json(path: &Path) -> Result<Option<PathBuf>> {
    let bytes = std::fs::read(path).context("read tokenizer failed")?;
    let mut v: Value = serde_json::from_slice(&bytes).context("parse tokenizer failed")?;

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

    let mut out = Vec::with_capacity(merges.len());
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

    let filename = path
        .file_name()
        .map(|v| v.to_os_string())
        .unwrap_or_else(|| OsString::from("tokenizer.json"));
    let tmp = std::env::temp_dir().join(format!(
        "cellm_tokenizer_normalized_{}_{}",
        std::process::id(),
        filename.to_string_lossy()
    ));
    std::fs::write(
        &tmp,
        serde_json::to_vec(&v).context("json serialize failed")?,
    )
    .context("write normalized tokenizer failed")?;
    Ok(Some(tmp))
}
