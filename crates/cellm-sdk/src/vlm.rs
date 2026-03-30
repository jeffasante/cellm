use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use bytemuck::cast_slice;
use cellm_cache::{KVCache, PageTable};
use cellm_core::KvCacheLayout;
use cellm_kernels::metal::MetalBuffer;
use cellm_kernels::metal::MetalMatmul;
use cellm_kernels::MetalKernels;
use cellm_model::{llama::LlamaRunner, CellmFile};
use half::f16;
use image::RgbImage;
use ndarray::{Array2, Array4, Array5, Axis};
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

    let image_input = preprocess_image_idefics3(image_bytes, false)?;
    let file = CellmFile::load(model_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let (image_features, image_seq_len, patch_ms, encoder_ms, encoder_layer_ms) =
        run_vision_cellm(&file, image_input.pixel_values, cfg.backend)?;
    let image_token_id = tok
        .token_to_id("<image>")
        .context("tokenizer missing <image> token")? as i64;
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
    let end_of_utt = tok.token_to_id("<end_of_utterance>").map(|v| v as i64);
    let banned_token_ids = banned_token_ids(&tok);

    let image_block = format_image_block(1, image_seq_len);
    let prompt = build_single_turn_prompt(user_prompt, &image_block);
    let enc = tok
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
    let input_ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();

    let (generated, decode_ms) = run_decode_cellm(
        model_path,
        image_token_id,
        eos_token_id,
        end_of_utt,
        cfg,
        input_ids,
        &image_features,
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

fn run_decode_cellm(
    model_path: &Path,
    image_token_id: i64,
    eos_token_id: i64,
    end_of_utterance_id: Option<i64>,
    cfg: VlmRunConfig,
    input_ids: Vec<i64>,
    image_features: &Array2<f32>,
    banned_token_ids: &[i64],
) -> Result<(Vec<i64>, f64)> {
    let decode_start = Instant::now();
    let mut runner = LlamaRunner::load(model_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let hidden = runner.hidden_size();
    if image_features.shape()[1] != hidden {
        anyhow::bail!(
            "image feature dim {} != text hidden {}",
            image_features.shape()[1],
            hidden
        );
    }

    let head_dim = runner.config().hidden_size / runner.config().num_attention_heads;
    let total_tokens = input_ids.len() + cfg.max_new_tokens + 8;
    let total_blocks = (total_tokens + cfg.tokens_per_block - 1) / cfg.tokens_per_block;
    let layout = KvCacheLayout {
        total_blocks,
        tokens_per_block: cfg.tokens_per_block,
        num_layers: runner.config().num_hidden_layers,
        num_kv_heads: runner.config().num_key_value_heads,
        head_dim,
    };
    let mut kv_cache = KVCache::new(layout).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut page_table =
        PageTable::new(1, cfg.tokens_per_block).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut image_idx = 0usize;
    let mut x = vec![0.0f32; hidden];
    let mut rng = StdRng::seed_from_u64(cfg.seed.max(1));
    let mut recent: Vec<u32> = Vec::new();
    let mut next: i64 = 0;

    for (pos, &tok_id) in input_ids.iter().enumerate() {
        if tok_id == image_token_id {
            if image_idx >= image_features.shape()[0] {
                anyhow::bail!("image token count mismatch: prompt has more <image> tokens than vision features");
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
            .step_topk_from_hidden(&x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        next = sample_from_candidates(
            &cand,
            cfg.temperature,
            cfg.repeat_penalty,
            cfg.repeat_window,
            banned_token_ids,
            &recent,
            &mut rng,
        )?;
        recent.push(tok_id as u32);
    }
    if image_idx != image_features.shape()[0] {
        anyhow::bail!(
            "image token count mismatch: prompt has fewer <image> tokens ({image_idx}) than vision features ({})",
            image_features.shape()[0]
        );
    }

    let mut generated = Vec::new();
    for step in 0..cfg.max_new_tokens {
        generated.push(next);
        if (next == eos_token_id || end_of_utterance_id == Some(next))
            && (step + 1) >= cfg.min_new_tokens
        {
            break;
        }
        runner
            .embed_token_hidden(next as u32, &mut x)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let pos = input_ids.len() + step;
        let cand = runner
            .step_topk_from_hidden(&x, pos, &mut page_table, &mut kv_cache, cfg.top_k.max(1))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
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
    pixel_values: Array5<f32>,
    backend: BackendKind,
) -> Result<(Array2<f32>, usize, f64, f64, Vec<f64>)> {
    let mut linear_backend = match backend {
        BackendKind::Metal => match MetalKernels::create_matmul() {
            Ok(ctx) => LinearBackend::Metal {
                ctx,
                weight_t_cache: HashMap::new(),
            },
            Err(_) => LinearBackend::Cpu,
        },
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
                &mut score_buf,
                &mut prob_buf,
                &mut attn,
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
    score_buf: &mut [f32],
    prob_buf: &mut [f32],
    out: &mut [f32],
    backend: &mut LinearBackend,
) {
    let hidden = num_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

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

fn add_inplace(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn build_single_turn_prompt(user_text: &str, image_block: &str) -> String {
    format!("<|im_start|>User:{image_block}{user_text}<end_of_utterance>\nAssistant:")
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

fn preprocess_image_idefics3(image_bytes: &[u8], split_image: bool) -> Result<PreparedImage> {
    let img = image::load_from_memory(image_bytes).context("decode image bytes failed")?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();

    let tile = 512u32;
    let cols = ((w + tile - 1) / tile).clamp(1, 6);
    let rows = ((h + tile - 1) / tile).clamp(1, 6);
    let mut images = Vec::new();
    images.push(resize_and_pad_512(&rgb));
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
    Ok(PreparedImage { pixel_values })
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
