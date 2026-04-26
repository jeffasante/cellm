// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! LFM2 (Liquid Foundation Model 2) runner.
//!
//! LFM2.5 uses a hybrid architecture with:
//! - LIV (Linear Input-Varying) convolution blocks for short-range dependencies
//! - Grouped Query Attention (GQA) for long-range dependencies
//! - SwiGLU feedforward networks
//! - RMSNorm normalization
//!
//! Layer layout (16 total):
//! conv, conv, full_attention, conv, conv, full_attention, conv, conv,
//! full_attention, conv, full_attention, conv, full_attention, conv, full_attention, conv

use std::path::Path;
use std::collections::HashMap;
use std::sync::Mutex;

use rayon::prelude::*;
use cellm_cache::{KVCache, PageTable};
use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::{rms_norm_f32, rope_non_interleaved_inplace_f32};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use cellm_kernels::metal::MetalOps;
use half::f16;
use serde_json::Value;

use crate::{CellmFile, ModelConfig};

/// Maximum weight cache entries before LRU eviction (approx 500MB with typical layer sizes)
const MAX_CACHE_ENTRIES: usize = 128;

pub struct LfmRunner {
    file: CellmFile,
    cfg: ModelConfig,
    max_layers: usize,
    eos_token_id: Option<u32>,
    /// Layer types: "conv" or "full_attention"
    layer_types: Vec<String>,
    /// Conv kernel size (L_cache)
    conv_kernel_size: usize,
    /// Conv state cache for LIV convolution [layer][batch][position][dim]
    conv_states: Vec<Vec<f32>>,
    /// Dequantized weight cache: (name, out_dim, in_dim) -> dequantized f32 weights
    weight_cache: HashMap<(String, usize, usize), Vec<f32>>,
    /// LRU tracking: list of cache keys in access order (most recent at end)
    lru_order: Vec<(String, usize, usize)>,
}

impl LfmRunner {
    pub fn load(path: &Path) -> Result<Self, CoreError> {
        let file = CellmFile::load(path)?;
        let h = file.header.clone();

        // Parse layer types from source_text_config if available
        let layer_types: Vec<String> = h.source_text_config
            .as_ref()
            .and_then(|cfg: &Value| cfg.get("layer_types"))
            .and_then(|v: &Value| v.as_array())
            .map(|arr: &Vec<Value>| {
                arr.iter()
                    .filter_map(|v: &Value| v.as_str().map(|s: &str| s.to_string()))
                    .collect()
            })
            .unwrap_or_else(|| {
                // Default LFM2.5-350M pattern: 16 layers
                vec![
                    "conv", "conv", "full_attention", "conv", "conv", "full_attention",
                    "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                    "full_attention", "conv", "full_attention", "conv",
                ]
                .into_iter()
                .map(|s: &str| s.to_string())
                .collect()
            });

        // Get conv kernel size (L_cache in config)
        let conv_kernel_size: usize = h.source_text_config
            .as_ref()
            .and_then(|cfg: &Value| cfg.get("conv_L_cache"))
            .and_then(|v: &Value| v.as_u64())
            .map(|v: u64| v as usize)
            .unwrap_or(3);

        let cfg = ModelConfig {
            vocab_size: h.vocab_size,
            hidden_size: h.hidden_dim,
            num_hidden_layers: h.num_layers,
            num_attention_heads: h.num_heads,
            num_key_value_heads: h.num_kv_heads,
            head_dim: h.head_dim.unwrap_or_else(|| {
                // Infer from k_proj if possible
                for t in &h.tensors {
                    if t.name.contains("self_attn.k_proj.weight") && t.shape.len() == 2 {
                        let kv_dim = t.shape[0];
                        let kv_heads = h.num_kv_heads.max(1);
                        if kv_dim % kv_heads == 0 {
                            return kv_dim / kv_heads;
                        }
                    }
                }
                h.hidden_dim / h.num_heads
            }),
            intermediate_size: h.intermediate_size,
            rms_norm_eps: h.rms_norm_eps,
            rope_theta: h.rope_theta,
        };

        // Initialize conv state cache
        // For each conv layer, store the last kernel_size Bx vectors for causal conv
        let num_conv_layers = layer_types.iter().filter(|t| *t == "conv").count();
        let conv_states: Vec<Vec<f32>> = (0..num_conv_layers)
            .map(|_| vec![0.0f32; conv_kernel_size * cfg.hidden_size])
            .collect();

        Ok(Self {
            file,
            cfg: cfg.clone(),
            max_layers: cfg.num_hidden_layers,
            eos_token_id: h.eos_token_id,
            layer_types,
            conv_kernel_size,
            conv_states,
            weight_cache: HashMap::new(),
            lru_order: Vec::new(),
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn set_max_layers(&mut self, n: usize) {
        self.max_layers = n.min(self.cfg.num_hidden_layers).max(1);
    }

    pub fn max_layers(&self) -> usize {
        self.max_layers
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    pub fn embed_token_hidden(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        self.embed_token(token, out)
    }

    pub fn step_topk(
        &mut self,
        token: u32,
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>, CoreError> {
        let mut x = vec![0.0f32; self.cfg.hidden_size];
        self.embed_token(token, &mut x)?;
        self.step_topk_from_hidden(&x, pos, page_table, kv_cache, top_k)
    }

    pub fn prefill(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
    ) -> Result<(), CoreError> {
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            if pos == page_table.token_count() {
                page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                    CoreError::Backend(format!("lfm prefill: page_table append_token failed: {e}"))
                })?;
            }
            let mut x = vec![0.0f32; self.cfg.hidden_size];
            self.embed_token(tok, &mut x)?;
            self.step_inner(&x, pos, page_table, kv_cache, false)?;
        }
        Ok(())
    }

    pub fn step_topk_from_hidden(
        &mut self,
        x0: &[f32],
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>, CoreError> {
        let logits = self.step_inner(x0, pos, page_table, kv_cache, true)?;
        self.topk_from_logits(&logits, top_k)
    }

    fn step_inner(
        &mut self,
        x0: &[f32],
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        return_logits: bool,
    ) -> Result<Vec<f32>, CoreError> {
        let cfg = self.cfg.clone();
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let attn_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Ensure pagetable covers this token position
        if pos == page_table.token_count() {
            page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                CoreError::Backend(format!("lfm step: page_table append_token failed: {e}"))
            })?;
        } else if pos > page_table.token_count() {
            return Err(CoreError::Backend(format!(
                "lfm step: non-contiguous pos {pos} (token_count={})",
                page_table.token_count()
            )));
        }

        let block_id = page_table.block_for_token(pos).map_err(|e| {
            CoreError::Backend(format!("lfm step: page_table block_for_token failed: {e}"))
        })?;
        let token_off = page_table.offset_in_block(pos).map_err(|e| {
            CoreError::Backend(format!("lfm step: page_table offset_in_block failed: {e}"))
        })?;

        if x0.len() != hidden {
            return Err(CoreError::Backend(format!(
                "lfm step_from_hidden: hidden len mismatch {} != {}",
                x0.len(),
                hidden
            )));
        }
        let mut x = x0.to_vec();

        // Per-layer scratch buffers
        let mut op_norm_w = vec![0.0f32; hidden];
        let mut x_norm = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; attn_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; attn_dim];
        let mut attn_proj = vec![0.0f32; hidden];

        let mut ffn_norm_w = vec![0.0f32; hidden];
        let mut mlp_in = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; cfg.intermediate_size];
        let mut up = vec![0.0f32; cfg.intermediate_size];
        let mut down = vec![0.0f32; hidden];

        // Conv buffers
        let mut conv_in = vec![0.0f32; hidden];
        let mut conv_out = vec![0.0f32; hidden];

        let mut gather_bases: Vec<usize> = Vec::new();
        let mut conv_layer_idx = 0usize;

        for layer in 0..self.max_layers {
            let layer_type = self.layer_types.get(layer).map(|s| s.as_str()).unwrap_or("conv");

            // Operator norm (replaces input_layernorm)
            self.rmsnorm_weight(
                &format!("model.layers.{layer}.operator_norm.weight"),
                &mut op_norm_w,
            )?;
            rms_norm_f32(&x, &op_norm_w, cfg.rms_norm_eps, &mut x_norm);

            match layer_type {
                "conv" => {
                    // ShortConv block (LFM2 style):
                    //   B, C, x = in_proj(input)
                    //   x = B * x
                    //   x = causal_conv1d(x)
                    //   x = C * x
                    //   x = out_proj(x)
                    let expanded_dim = hidden * 3;
                    let mut bcx = vec![0.0f32; expanded_dim];

                    self.linear_f16_out_in(
                        &x_norm,
                        &format!("model.layers.{layer}.conv.in_proj.weight"),
                        expanded_dim,
                        hidden,
                        &mut bcx,
                    )?;

                    // Split into B, C, x
                    let b_part = &bcx[0..hidden];
                    let c_part = &bcx[hidden..2*hidden];
                    let x_part = &bcx[2*hidden..3*hidden];

                    // Compute Bx = B * x (element-wise gating)
                    let mut bx = vec![0.0f32; hidden];
                    for i in 0..hidden {
                        bx[i] = b_part[i] * x_part[i];
                    }

                    // Causal depthwise conv1d with sliding window state.
                    // conv_states[conv_layer_idx] holds the last kernel_size
                    // Bx vectors (oldest first). Layout: [slot0 * hidden, slot1 * hidden, ...]
                    let ks = self.conv_kernel_size;
                    let state = &mut self.conv_states[conv_layer_idx];

                    // Shift state: move slots 1..ks to 0..ks-1, then write current Bx into last slot
                    if ks > 1 {
                        state.copy_within(hidden..(ks * hidden), 0);
                    }
                    state[(ks - 1) * hidden..ks * hidden].copy_from_slice(&bx);

                    // Load conv kernel: shape [hidden, kernel_size, 1] stored as f16
                    let conv_kernel_name = format!("model.layers.{layer}.conv.conv.weight");
                    let conv_kernel_bytes = self.file.tensor_bytes(&conv_kernel_name)?;
                    let conv_kernel_u16: &[u16] = bytemuck::cast_slice(conv_kernel_bytes);

                    // Depthwise causal conv: for each channel, dot product with kernel
                    let mut conv_result = vec![0.0f32; hidden];
                    for i in 0..hidden {
                        let mut acc = 0.0f32;
                        let kernel_base = i * ks;
                        for k in 0..ks {
                            let s = state[k * hidden + i];
                            let w = f16::from_bits(conv_kernel_u16[kernel_base + k]).to_f32();
                            acc += s * w;
                        }
                        conv_result[i] = acc;
                    }

                    // Second gating: y = C * conv_result
                    let mut y = vec![0.0f32; hidden];
                    for i in 0..hidden {
                        y[i] = c_part[i] * conv_result[i];
                    }

                    // out_proj: hidden -> hidden
                    self.linear_f16_out_in(
                        &y,
                        &format!("model.layers.{layer}.conv.out_proj.weight"),
                        hidden,
                        hidden,
                        &mut attn_proj,
                    )?;

                    conv_layer_idx += 1;
                }
                "full_attention" | "attention" => {
                    // Grouped Query Attention
                    let fused_qkv = self.linear_qkv_f16_out_in(
                        &x_norm,
                        &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                        attn_dim,
                        &format!("model.layers.{layer}.self_attn.k_proj.weight"),
                        kv_dim,
                        &format!("model.layers.{layer}.self_attn.v_proj.weight"),
                        kv_dim,
                        hidden,
                        &mut q,
                        &mut k,
                        &mut v,
                    )?;
                    if !fused_qkv {
                        self.linear_f16_out_in(
                            &x_norm,
                            &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                            attn_dim,
                            hidden,
                            &mut q,
                        )?;
                        self.linear_f16_out_in(
                            &x_norm,
                            &format!("model.layers.{layer}.self_attn.k_proj.weight"),
                            kv_dim,
                            hidden,
                            &mut k,
                        )?;
                        self.linear_f16_out_in(
                            &x_norm,
                            &format!("model.layers.{layer}.self_attn.v_proj.weight"),
                            kv_dim,
                            hidden,
                            &mut v,
                        )?;
                    }

                    // Apply Q/K layernorm (LFM2 specific - applied per-head)
                    let mut q_normed = vec![0.0f32; attn_dim];
                    let mut k_normed = vec![0.0f32; kv_dim];
                    if let Ok(q_norm_w) = self.tensor_f16(&format!("model.layers.{layer}.self_attn.q_layernorm.weight")) {
                        let q_norm_w_f32: Vec<f32> = q_norm_w.iter().map(|&x| f16::from_bits(x).to_f32()).collect();
                        // Apply per-head: q is [n_heads * head_dim], norm each head independently
                        for h in 0..n_heads {
                            let h_start = h * head_dim;
                            let h_end = h_start + head_dim;
                            rms_norm_f32(&q[h_start..h_end], &q_norm_w_f32, cfg.rms_norm_eps, &mut q_normed[h_start..h_end]);
                        }
                        q.copy_from_slice(&q_normed);
                    }
                    if let Ok(k_norm_w) = self.tensor_f16(&format!("model.layers.{layer}.self_attn.k_layernorm.weight")) {
                        let k_norm_w_f32: Vec<f32> = k_norm_w.iter().map(|&x| f16::from_bits(x).to_f32()).collect();
                        // Apply per-head for KV heads
                        for h in 0..n_kv_heads {
                            let h_start = h * head_dim;
                            let h_end = h_start + head_dim;
                            rms_norm_f32(&k[h_start..h_end], &k_norm_w_f32, cfg.rms_norm_eps, &mut k_normed[h_start..h_end]);
                        }
                        k.copy_from_slice(&k_normed);
                    }

                    // Apply RoPE (non-interleaved/split layout for LFM2)
                    rope_non_interleaved_inplace_f32(&mut q, n_heads, head_dim, head_dim, pos, cfg.rope_theta);
                    rope_non_interleaved_inplace_f32(&mut k, n_kv_heads, head_dim, head_dim, pos, cfg.rope_theta);

                    // Write K/V to cache
                    {
                        let mut cv = kv_cache.view_mut();
                        cv.write_token(block_id, layer, token_off, &k, &v)?;
                    }

                    // Gather and compute attention
                    let seq = page_table.token_count();
                    let cr = kv_cache.view();
                    gather_bases.clear();
                    gather_bases.reserve(seq);
                    for tpos in 0..seq {
                        let b = page_table.block_for_token(tpos).map_err(|e| {
                            CoreError::Backend(format!("lfm: block_for_token failed: {e}"))
                        })?;
                        let o = page_table.offset_in_block(tpos).map_err(|e| {
                            CoreError::Backend(format!("lfm: offset_in_block failed: {e}"))
                        })?;
                        gather_bases.push(cr.layout.token_base_elem(b, layer, o)?);
                    }
                    cr.attention_single_token_gqa_from_bases(
                        &gather_bases,
                        &q,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        None,
                        None,
                        &mut attn_out,
                    )?;

                    // Output projection
                    self.linear_f16_out_in(
                        &attn_out,
                        &format!("model.layers.{layer}.self_attn.out_proj.weight"),
                        hidden,
                        attn_dim,
                        &mut attn_proj,
                    )?;
                }
                _ => {
                    return Err(CoreError::Backend(format!(
                        "lfm: unknown layer type '{layer_type}' at layer {layer}"
                    )));
                }
            }

            // Residual connection
            for i in 0..hidden {
                x[i] += attn_proj[i];
            }

            // FFN norm
            self.rmsnorm_weight(
                &format!("model.layers.{layer}.ffn_norm.weight"),
                &mut ffn_norm_w,
            )?;
            rms_norm_f32(&x, &ffn_norm_w, cfg.rms_norm_eps, &mut mlp_in);

            // SwiGLU MLP: w1=gate, w3=up, w2=down
            self.linear_f16_out_in(
                &mlp_in,
                &format!("model.layers.{layer}.feed_forward.w1.weight"),
                cfg.intermediate_size,
                hidden,
                &mut gate,
            )?;
            self.linear_f16_out_in(
                &mlp_in,
                &format!("model.layers.{layer}.feed_forward.w3.weight"),
                cfg.intermediate_size,
                hidden,
                &mut up,
            )?;

            // Swish activation: silu(gate) = gate * sigmoid(gate)
            for g in gate.iter_mut() {
                let s = 1.0 / (1.0 + (-*g).exp());
                *g = *g * s;
            }
            for i in 0..gate.len() {
                gate[i] *= up[i];
            }

            self.linear_f16_out_in(
                &gate,
                &format!("model.layers.{layer}.feed_forward.w2.weight"),
                hidden,
                cfg.intermediate_size,
                &mut down,
            )?;

            // Residual connection
            for i in 0..hidden {
                x[i] += down[i];
            }
        }

        // Final embedding norm
        let mut final_norm_w = vec![0.0f32; hidden];
        let mut x_final = vec![0.0f32; hidden];
        self.rmsnorm_weight(
            "model.embedding_norm.weight",
            &mut final_norm_w,
        )?;
        rms_norm_f32(&x, &final_norm_w, cfg.rms_norm_eps, &mut x_final);

        if return_logits {
            let mut logits = vec![0.0f32; cfg.vocab_size];
            // Output projection: use embeddings as transposed linear layer
            // embeddings are [vocab_size, hidden], we need [hidden, vocab_size]^T @ x_final
            // This is equivalent to x_final^T @ embeddings^T = x_final @ embeddings (row-wise dot products)
            
            // Check if embeddings are quantized
            let dtype = self.tensor_dtype("model.embed_tokens.weight").unwrap_or_else(|| "f16".to_string());
            let scales_name = "model.embed_tokens.scales".to_string();
            let biases_name = "model.embed_tokens.biases".to_string();
            
            if dtype == "u32" && self.file.has_tensor(&scales_name) && self.file.has_tensor(&biases_name) {
                // Quantized embeddings - need to dequantize each row and dot with x_final
                let weight_bytes = self.file.tensor_bytes("model.embed_tokens.weight")?;
                let scales_bytes = self.file.tensor_bytes(&scales_name)?;
                let biases_bytes = self.file.tensor_bytes(&biases_name)?;
                
                let weight_u32: &[u32] = bytemuck::cast_slice(weight_bytes);
                let scales_f32: &[f32] = bytemuck::cast_slice(scales_bytes);
                let biases_f32: &[f32] = bytemuck::cast_slice(biases_bytes);
                
                let group_size = 64usize;
                let groups_per_row = (hidden + group_size - 1) / group_size;
                let packed_in = hidden / 8;
                
                // Compute logits in parallel
                logits.par_iter_mut().enumerate().for_each(|(vocab_idx, logit)| {
                    let row_offset = vocab_idx * packed_in;
                    let mut acc = 0.0f32;
                    
                    for g in 0..groups_per_row {
                        let g_start = g * group_size;
                        let g_end = ((g + 1) * group_size).min(hidden);
                        let scale_idx = vocab_idx * groups_per_row + g;
                        let scale = scales_f32.get(scale_idx).copied().unwrap_or(1.0);
                        let bias = biases_f32.get(scale_idx).copied().unwrap_or(0.0);
                        
                        for j in g_start..g_end {
                            let packed_idx = row_offset + (j / 8);
                            let nibble_pos = j % 8;
                            
                            if packed_idx < weight_u32.len() {
                                let packed = weight_u32[packed_idx];
                                let nibble = ((packed >> (nibble_pos * 4)) & 0xF) as i32;
                                let q = nibble as f32;
                                let w = q * scale + bias;
                                acc += w * x_final[j];
                            }
                        }
                    }
                    
                    *logit = acc;
                });
            } else {
                // F16 embeddings
                let emb = self.tensor_f16("model.embed_tokens.weight")?;
                
                logits.par_iter_mut().enumerate().for_each(|(vocab_idx, logit)| {
                    let row_start = vocab_idx * hidden;
                    let mut acc = 0.0f32;
                    
                    for j in 0..hidden {
                        let w = f16::from_bits(emb[row_start + j]).to_f32();
                        acc += w * x_final[j];
                    }
                    
                    *logit = acc;
                });
            }
            
            Ok(logits)
        } else {
            Ok(vec![])
        }
    }

    /// Apply LIV (Linear Input-Varying) convolution
    /// This implements a depthwise separable convolution with learned gates
    fn apply_liv_convolution(
        &self,
        layer: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        let kernel_size = self.conv_kernel_size;

        // Load conv weights [out_channels, in_channels, kernel_size]
        let conv_weight_name = format!("model.layers.{layer}.conv.conv.weight");
        let conv_weight = self.tensor_f16(&conv_weight_name)?;

        // Convert f16 weights to f32
        let weight_f32: Vec<f32> = conv_weight.iter().map(|&x| f16::from_bits(x).to_f32()).collect();

        // Depthwise convolution: each channel has its own kernel
        // Weight layout: [hidden, 1, kernel_size] for depthwise conv
        for d in 0..hidden {
            let mut acc = 0.0f32;
            for k in 0..kernel_size {
                let weight_idx = d * kernel_size + k;
                if weight_idx < weight_f32.len() {
                    acc += input[d] * weight_f32[weight_idx];
                }
            }
            output[d] = acc;
        }

        // Apply gating (LIV specific: double gate mechanism)
        // Simplified: apply sigmoid gate
        for d in 0..hidden {
            let gate = 1.0 / (1.0 + (-output[d]).exp());
            output[d] *= gate;
        }

        Ok(())
    }

    fn topk_from_logits(&self, logits: &[f32], k: usize) -> Result<Vec<(u32, f32)>, CoreError> {
        let mut indexed: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, if v.is_finite() { v } else { f32::NEG_INFINITY }))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        Ok(indexed)
    }

    fn embed_token(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;

        if (token as usize) >= vocab {
            return Err(CoreError::Backend(format!(
                "embed_token: token {token} >= vocab {vocab}"
            )));
        }

        // Check if embeddings are quantized (u32) or f16
        let dtype = self.tensor_dtype("model.embed_tokens.weight").unwrap_or_else(|| "f16".to_string());
        let scales_name = "model.embed_tokens.scales".to_string();
        let biases_name = "model.embed_tokens.biases".to_string();
        
        if dtype == "u32" && self.file.has_tensor(&scales_name) && self.file.has_tensor(&biases_name) {
            // 4-bit quantized embeddings (MLX format)
            let weight_bytes = self.file.tensor_bytes("model.embed_tokens.weight")?;
            let scales_bytes = self.file.tensor_bytes(&scales_name)?;
            let biases_bytes = self.file.tensor_bytes(&biases_name)?;
            
            let weight_u32: &[u32] = bytemuck::cast_slice(weight_bytes);
            let scales_f32: &[f32] = bytemuck::cast_slice(scales_bytes);
            let biases_f32: &[f32] = bytemuck::cast_slice(biases_bytes);
            
            let group_size = 64usize;
            let groups_per_row = (hidden + group_size - 1) / group_size;
            let packed_in = hidden / 8;  // Each uint32 holds 8 nibbles
            
            let row_offset = (token as usize) * packed_in;
            
            for g in 0..groups_per_row {
                let g_start = g * group_size;
                let g_end = ((g + 1) * group_size).min(hidden);
                let scale_idx = (token as usize) * groups_per_row + g;
                let scale = scales_f32.get(scale_idx).copied().unwrap_or(1.0);
                let bias = biases_f32.get(scale_idx).copied().unwrap_or(0.0);
                
                for j in g_start..g_end {
                    let packed_idx = row_offset + (j / 8);
                    let nibble_pos = j % 8;
                    
                    if packed_idx >= weight_u32.len() {
                        out[j] = 0.0;
                        continue;
                    }
                    
                    let packed = weight_u32[packed_idx];
                    let nibble = ((packed >> (nibble_pos * 4)) & 0xF) as i32;
                    // MLX uses zero_point=0, so q = nibble (0-15 range)
                    // The bias term handles centering
                    let q = nibble as f32;
                    
                    out[j] = q * scale + bias;
                }
            }
        } else {
            // Standard f16 embeddings
            let emb = self.tensor_f16("model.embed_tokens.weight")?;
            let row_start = (token as usize) * hidden;
            let row_end = row_start + hidden;

            if row_end > emb.len() {
                return Err(CoreError::Backend(
                    "embed_tokens.weight shape mismatch".into(),
                ));
            }

            for i in 0..hidden {
                out[i] = f16::from_bits(emb[row_start + i]).to_f32();
            }
        }
        Ok(())
    }

    fn rmsnorm_weight(&self, name: &str, out: &mut [f32]) -> Result<(), CoreError> {
        let w = self.tensor_f16(name)?;
        if w.len() != out.len() {
            return Err(CoreError::Backend(format!(
                "rmsnorm_weight: shape mismatch for {name}: {} vs {}",
                w.len(),
                out.len()
            )));
        }
        for i in 0..w.len() {
            out[i] = f16::from_bits(w[i]).to_f32();
        }
        Ok(())
    }

    fn tensor_f16(&self, name: &str) -> Result<&[u16], CoreError> {
        let bytes = self.file.tensor_bytes(name)?;
        if bytes.len() % 2 != 0 {
            return Err(CoreError::Backend(format!("tensor {name} nbytes not even")));
        }
        Ok(bytemuck::cast_slice(bytes))
    }

    /// Get tensor dtype from header
    fn tensor_dtype(&self, name: &str) -> Option<String> {
        self.file.header.tensors.iter()
            .find(|t| t.name == name)
            .map(|t| t.dtype.clone())
    }

    /// Dequantize int4 weights and perform matmul: out = weight @ input
    /// Handles both pre-quantized (from MLX) and standard f16 weights
    fn linear_f16_out_in(
        &mut self,
        input: &[f32],
        weight_name: &str,
        out_dim: usize,
        in_dim: usize,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        // Check tensor dtype from metadata
        let dtype = self.tensor_dtype(weight_name).unwrap_or_else(|| "f16".to_string());
        
        // Check for pre-quantized weights (uint32 dtype with .scales/.biases)
        // MLX format: scales/biases are named {base}.scales where base is weight name without .weight
        let base_name = weight_name.trim_end_matches(".weight");
        let scales_name = format!("{}.scales", base_name);
        let biases_name = format!("{}.biases", base_name);
        
        let has_scales = self.file.has_tensor(&scales_name);
        let has_biases = self.file.has_tensor(&biases_name);

        if dtype == "u32" && has_scales && has_biases {
            // Pre-quantized int4 path
            return self.linear_i4_out_in(input, weight_name, out_dim, in_dim, out);
        }

        // Standard f16 path
        let w = self.tensor_f16(weight_name)?;

        // Validate weight shape: [out_dim, in_dim] -> out_dim * in_dim elements
        let expected_len = out_dim * in_dim * 2; // f16 = 2 bytes
        if w.len() != expected_len {
            return Err(CoreError::Backend(format!(
                "linear_f16_out_in: weight shape mismatch for {weight_name}: got {} bytes, expected {} ({}x{} f16)",
                w.len(), expected_len, out_dim, in_dim
            )));
        }

        // matmul: out[i] = sum_j weight[i,j] * input[j]
        // w is already &[u16] from tensor_f16
        for i in 0..out_dim {
            let mut acc = 0.0f32;
            let row_start = i * in_dim;
            for j in 0..in_dim {
                let w_f32 = f16::from_bits(w[row_start + j]).to_f32();
                acc += w_f32 * input[j];
            }
            out[i] = acc;
        }

        Ok(())
    }

    /// Dequantize MLX-style int4 weights and perform matmul
    /// Uses weight cache to avoid repeated dequantization
    fn linear_i4_out_in(
        &mut self,
        input: &[f32],
        weight_name: &str,
        out_dim: usize,
        in_dim: usize,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        // Check cache first
        let cache_key = (weight_name.to_string(), out_dim, in_dim);
        
        if !self.weight_cache.contains_key(&cache_key) {
            // Dequantize and cache
            let base_name = weight_name.trim_end_matches(".weight");
            let weight_bytes = self.file.tensor_bytes(weight_name)?;
            let scales_bytes = self.file.tensor_bytes(&format!("{}.scales", base_name))?;
            let biases_bytes = self.file.tensor_bytes(&format!("{}.biases", base_name))?;

            let weight_u32: &[u32] = bytemuck::cast_slice(weight_bytes);
            let scales_f32: &[f32] = bytemuck::cast_slice(scales_bytes);
            let biases_f32: &[f32] = bytemuck::cast_slice(biases_bytes);

            let group_size = 64usize;
            let groups_per_row = (in_dim + group_size - 1) / group_size;
            let packed_in = in_dim / 8;

            let mut dequant = vec![0.0f32; out_dim * in_dim];

            for i in 0..out_dim {
                let row_offset = i * packed_in;
                let out_offset = i * in_dim;
                
                for g in 0..groups_per_row {
                    let g_start = g * group_size;
                    let g_end = ((g + 1) * group_size).min(in_dim);
                    let scale_idx = i * groups_per_row + g;
                    let scale = scales_f32.get(scale_idx).copied().unwrap_or(1.0);
                    let bias = biases_f32.get(scale_idx).copied().unwrap_or(0.0);
                    
                    for j in g_start..g_end {
                        let packed_idx = row_offset + (j / 8);
                        let nibble_pos = j % 8;
                        
                        if packed_idx >= weight_u32.len() {
                            continue;
                        }
                        
                        let packed = weight_u32[packed_idx];
                        let nibble = ((packed >> (nibble_pos * 4)) & 0xF) as i32;
                        // MLX uses zero_point=0
                        let q = nibble as f32;
                        
                        dequant[out_offset + j] = q * scale + bias;
                    }
                }
            }
            
            // LRU eviction: if at capacity, remove oldest entry
            if self.weight_cache.len() >= MAX_CACHE_ENTRIES {
                if let Some(oldest_key) = self.lru_order.first().cloned() {
                    self.weight_cache.remove(&oldest_key);
                    self.lru_order.remove(0);
                }
            }
            
            self.weight_cache.insert(cache_key.clone(), dequant);
            self.lru_order.push(cache_key.clone());
        } else {
            // Cache hit: move key to end (most recent)
            if let Some(pos) = self.lru_order.iter().position(|k| k == &cache_key) {
                let key = self.lru_order.remove(pos);
                self.lru_order.push(key);
            }
        }
        
        // Use cached weights for matmul with 8x unrolled loop + parallel rows
        let weights = self.weight_cache.get(&cache_key).unwrap();
        
        // Parallelize across output rows using rayon
        out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
            let row_start = i * in_dim;
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;
            let mut acc4 = 0.0f32;
            let mut acc5 = 0.0f32;
            let mut acc6 = 0.0f32;
            let mut acc7 = 0.0f32;
            
            // 8x unrolled loop for better instruction-level parallelism
            let chunks = in_dim / 8;
            for c in 0..chunks {
                let offset = row_start + c * 8;
                acc0 += weights[offset] * input[c * 8];
                acc1 += weights[offset + 1] * input[c * 8 + 1];
                acc2 += weights[offset + 2] * input[c * 8 + 2];
                acc3 += weights[offset + 3] * input[c * 8 + 3];
                acc4 += weights[offset + 4] * input[c * 8 + 4];
                acc5 += weights[offset + 5] * input[c * 8 + 5];
                acc6 += weights[offset + 6] * input[c * 8 + 6];
                acc7 += weights[offset + 7] * input[c * 8 + 7];
            }
            
            // Handle remainder
            let mut acc = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
            for j in (chunks * 8)..in_dim {
                acc += weights[row_start + j] * input[j];
            }
            
            *out_val = acc;
        });

        Ok(())
    }

    /// Try to fuse QKV projections for efficiency
    fn linear_qkv_f16_out_in(
        &mut self,
        input: &[f32],
        q_name: &str,
        q_dim: usize,
        k_name: &str,
        k_dim: usize,
        v_name: &str,
        v_dim: usize,
        in_dim: usize,
        q_out: &mut [f32],
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<bool, CoreError> {
        // Check if we can fuse (all weights present and contiguous)
        if !self.file.has_tensor(q_name) || !self.file.has_tensor(k_name) || !self.file.has_tensor(v_name) {
            return Ok(false);
        }

        // Fall back to individual projections
        Ok(false)
    }
}
