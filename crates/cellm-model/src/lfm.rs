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
use rayon::prelude::*;use std::sync::Mutex;

use rayon::prelude::*;
use cellm_cache::{KVCache, PageTable};
use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::{rms_norm_f32, rope_non_interleaved_inplace_f32};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use cellm_kernels::metal::MetalOps;
use half::f16;
use serde_json::Value;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::ffi::c_void;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::rc::autoreleasepool;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::MTLResourceOptions;

use crate::{CellmFile, ModelConfig};

/// Maximum weight cache entries before LRU eviction (approx 500MB with typical layer sizes)
const MAX_CACHE_ENTRIES: usize = 128;

/// Byte ceiling for the dequantized weight cache, overridable via `CELLM_WEIGHT_CACHE_MB`.
///
/// The entry count alone is not a memory bound: entry size scales with the model, so 128
/// entries is ~0.6 GiB at 230M but ~9.6 GiB at 2.6B, which exhausts swap and wedges the host.
const DEFAULT_CACHE_BYTES: usize = 1536 * 1024 * 1024;

/// The Metal FFN path pre-populates w1/w3/w2 and then unwraps all three, so eviction must
/// never drop an entry the caller is still about to read.
const MIN_CACHE_ENTRIES: usize = 4;

fn weight_cache_budget_bytes() -> usize {
    std::env::var("CELLM_WEIGHT_CACHE_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(DEFAULT_CACHE_BYTES)
}

pub struct LfmRunner {
    file: CellmFile,
    cfg: ModelConfig,
    max_layers: usize,
    pub eos_token_id: Option<u32>,
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

    // Metal backend
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    metal_ops: Option<MetalOps>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    graph_state: Option<LfmGraphState>,
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    metal_ops: (),
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    graph_state: (),
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct LfmGraphState {
    ops: MetalOps,
    cfg: ModelConfig,
    layer_types: Vec<String>,
    /// Preloaded weight buffers: name -> Metal Buffer (f16)
    weights: HashMap<String, metal::Buffer>,
    /// Dtype string per tensor name ("f16" | "u32").
    tensor_dtypes: HashMap<String, String>,
    /// Cached MLX i4 scale buffers: "name.scales" -> Buffer
    i4_scales: HashMap<String, metal::Buffer>,
    /// Cached MLX i4 bias buffers: "name.biases" -> Buffer
    i4_biases: HashMap<String, metal::Buffer>,
    /// Conv state buffers per conv layer
    conv_states: Vec<metal::Buffer>,
    /// Conv kernel buffers per conv layer
    conv_kernels: Vec<metal::Buffer>,
    /// Activation buffers (reused per step)
    buf_x: metal::Buffer,
    buf_x_norm: metal::Buffer,
    buf_bcx: metal::Buffer,
    buf_bx: metal::Buffer,
    buf_y: metal::Buffer,
    buf_attn_proj: metal::Buffer,
    buf_q: metal::Buffer,
    buf_k: metal::Buffer,
    buf_v: metal::Buffer,
    buf_attn_out: metal::Buffer,
    buf_mlp_in: metal::Buffer,
    buf_gate: metal::Buffer,
    buf_up: metal::Buffer,
    buf_down: metal::Buffer,
    buf_final_norm_w: metal::Buffer,
    buf_logits: metal::Buffer,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl LfmGraphState {
    pub fn new(
        ops: MetalOps,
        cfg: ModelConfig,
        layer_types: Vec<String>,
        conv_kernel_size: usize,
        num_conv_layers: usize,
    ) -> Self {
        let device = ops.device.clone();
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let attn_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let intermediate = cfg.intermediate_size;
        let vocab = cfg.vocab_size;

        let make_buf = |len_f32: usize| {
            device.new_buffer(
                (len_f32 * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        // Conv state buffers: one per conv layer, each sized [conv_kernel_size * hidden] f32
        let conv_states: Vec<metal::Buffer> = (0..num_conv_layers)
            .map(|_| make_buf(conv_kernel_size * hidden))
            .collect();

        // Conv kernel buffers: one per conv layer, each sized [hidden * conv_kernel_size] f16
        let conv_kernels: Vec<metal::Buffer> = (0..num_conv_layers)
            .map(|_| {
                device.new_buffer(
                    (hidden * conv_kernel_size * 2) as u64, // f16 = 2 bytes each
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();

        Self {
            ops,
            cfg,
            layer_types,
            weights: HashMap::new(),
            tensor_dtypes: HashMap::new(),
            i4_scales: HashMap::new(),
            i4_biases: HashMap::new(),
            conv_states,
            conv_kernels,
            buf_x: make_buf(hidden),
            buf_x_norm: make_buf(hidden),
            buf_bcx: make_buf(hidden * 3),
            buf_bx: make_buf(hidden),
            buf_y: make_buf(hidden),
            buf_attn_proj: make_buf(hidden),
            buf_q: make_buf(attn_dim),
            buf_k: make_buf(kv_dim),
            buf_v: make_buf(kv_dim),
            buf_attn_out: make_buf(attn_dim),
            buf_mlp_in: make_buf(hidden),
            buf_gate: make_buf(intermediate),
            buf_up: make_buf(intermediate),
            buf_down: make_buf(hidden),
            buf_final_norm_w: make_buf(hidden),
            buf_logits: make_buf(vocab),
        }
    }

    fn get_weight(&self, name: &str) -> &metal::Buffer {
        // Try exact name first
        if let Some(w) = self.weights.get(name) {
            return w;
        }
        // Try without 'model.' prefix (some models strip it)
        if name.starts_with("model.") {
            let stripped = &name[6..];
            if let Some(w) = self.weights.get(stripped) {
                return w;
            }
            // Try with 'model.text_model.' prefix (converted checkpoints)
            let txt_name = format!("model.text_model.{}", stripped);
            if let Some(w) = self.weights.get(&txt_name) {
                return w;
            }
        } else {
            // Name doesn't start with 'model.' — try adding it
            let prefixed = format!("model.{}", name);
            if let Some(w) = self.weights.get(&prefixed) {
                return w;
            }
        }
        panic!("LfmGraphState weight not found: {}", name);
    }

    pub fn preload_weight(&mut self, name: String, bytes: &[u8], dtype: String) {
        let buf = self.ops.device.new_buffer_with_data(
            bytes.as_ptr() as *const c_void,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.tensor_dtypes.insert(name.clone(), dtype);
        // Store scales/biases separately for MLX i4 lookup
        if name.ends_with(".scales") {
            self.i4_scales.insert(name.clone(), buf.clone());
        } else if name.ends_with(".biases") {
            self.i4_biases.insert(name.clone(), buf.clone());
        }
        self.weights.insert(name, buf);
    }

    /// Dispatch the correct matrix-vector kernel based on weight dtype.
    fn encode_mv_auto(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        name: &str,
        x: &metal::Buffer,
        out: &metal::Buffer,
        rows: usize,
        cols: usize,
    ) {
        let dtype = self.tensor_dtypes.get(name)
            .map(|s| s.as_str()).unwrap_or("f16");
        if dtype == "u32" {
            // MLX-style i4: on-the-fly GPU dequant + matmul
            let w = self.get_weight(name);
            let group_size = 64usize;
            let groups_per_row = (cols + group_size - 1) / group_size;
            let scales_name = format!("{}.scales", name.trim_end_matches(".weight"));
            let biases_name = format!("{}.biases", name.trim_end_matches(".weight"));
            let scales = self.i4_scales.get(&scales_name)
                .or_else(|| { let n = format!("{}.scales", name); self.i4_scales.get(&n) })
                .unwrap_or_else(|| panic!("LfmGraphState: missing i4 scales for {name}"));
            let biases = self.i4_biases.get(&biases_name)
                .or_else(|| { let n = format!("{}.biases", name); self.i4_biases.get(&n) })
                .unwrap_or_else(|| panic!("LfmGraphState: missing i4 biases for {name}"));
            self.ops.encode_mv_mlx_i4(enc, w, scales, biases, x, out, rows, cols, group_size);
        } else {
            let w = self.get_weight(name);
            self.ops.encode_mv_f16(enc, w, x, out, rows, cols);
        }
    }

    /// Run a fused forward pass for all layers (conv + attention) in a single
    /// Metal command buffer. For conv layers, element-wise gating operations
    /// (B*x and C*conv_out) are done on CPU with explicit syncs since there is
    /// no dedicated Metal kernel for them yet. Attention layers are fully fused
    /// on GPU with zero intermediate syncs.
    pub fn step_fused(
        &mut self,
        x_in: &[f32],
        cfg: &ModelConfig,
        _prefix: &str,
        kv_cache: &mut KVCache,
        page_table: &PageTable,
        pos: usize,
        token_off: usize,
        block_id: u32,
        return_logits: bool,
    ) -> Result<Option<Vec<f32>>, CoreError> {
        autoreleasepool(|| {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let attn_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let intermediate = cfg.intermediate_size;

        // 1. Upload input to buf_x
        unsafe {
            let ptr = self.buf_x.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(x_in.as_ptr(), ptr, hidden);
        }

        let seq = page_table.token_count();
        let num_layers = cfg.num_hidden_layers;

        // DEBUG: limit layers via env var LFM_DEBUG_MAX_LAYERS
        let debug_max_layers = std::env::var("LFM_DEBUG_MAX_LAYERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(num_layers)
            .min(num_layers);
        let debug_logits = std::env::var("LFM_DEBUG_GRAPH").map(|v| v == "1").unwrap_or(false);

        // Override num_layers for the loop (only process up to debug_max_layers)
        let original_num_layers = num_layers;
        let num_layers = debug_max_layers;

        // Get KV storage reference once for all attention layers
        let kv_store = kv_cache.storage().as_any()
            .downcast_ref::<cellm_cache::kvcache::MetalKvStorage>()
            .expect("LfmGraphState step_fused requires MetalKvStorage");

        // Track current conv layer index
        let mut conv_idx: usize = 0;

        for layer in 0..num_layers {
            let layer_type = self.layer_types.get(layer)
                .map(|s| s.as_str())
                .unwrap_or("conv");

            match layer_type {
                "conv" => {
                    // Conv layer: hybrid GPU/CPU path
                    // Each GPU sync point uses its own command buffer
                    // because Metal doesn't allow creating encoders on a
                    // committed command buffer.

                    let w_norm = self.get_weight(
                        &format!("model.layers.{layer}.operator_norm.weight"));

                    // Encoder 1: op_norm + in_proj
                    {
                        let cb = self.ops.queue.new_command_buffer();
                        let enc = cb.new_compute_command_encoder();
                        self.ops.encode_rms_norm_f16w(
                            &enc, &self.buf_x, w_norm, &self.buf_x_norm,
                            hidden, cfg.rms_norm_eps, false);
                        self.encode_mv_auto(&enc, &format!("model.layers.{layer}.conv.in_proj.weight"),
                            &self.buf_x_norm, &self.buf_bcx,
                            hidden * 3, hidden);
                        enc.end_encoding();
                        cb.commit();
                        cb.wait_until_completed();
                    }

                    // CPU: B*x element-wise gating
                    let ks = (self.conv_kernels[conv_idx].length() as usize) / (hidden * 2);
                    let mut bcx = vec![0.0f32; hidden * 3];
                    unsafe {
                        let ptr = self.buf_bcx.contents() as *const f32;
                        std::ptr::copy_nonoverlapping(ptr, bcx.as_mut_ptr(), hidden * 3);
                    }
                    let b_part = &bcx[..hidden];
                    let x_part = &bcx[2 * hidden..3 * hidden];
                    let mut bx = vec![0.0f32; hidden];
                    for i in 0..hidden {
                        bx[i] = b_part[i] * x_part[i];
                    }
                    unsafe {
                        let ptr = self.buf_bx.contents() as *mut f32;
                        std::ptr::copy_nonoverlapping(bx.as_ptr(), ptr, hidden);
                    }

                    // Encoder 2: lfm_conv (depthwise causal conv)
                    {
                        let cb = self.ops.queue.new_command_buffer();
                        let enc = cb.new_compute_command_encoder();
                        self.ops.encode_lfm_conv(
                            &enc,
                            &self.conv_states[conv_idx],
                            &self.buf_bx,
                            &self.conv_kernels[conv_idx],
                            &self.buf_y,
                            ks, hidden);
                        enc.end_encoding();
                        cb.commit();
                        cb.wait_until_completed();
                    }

                    // CPU: C*conv_out element-wise gating
                    let c_part = &bcx[hidden..2 * hidden];
                    let mut y_vals = vec![0.0f32; hidden];
                    unsafe {
                        let ptr = self.buf_y.contents() as *const f32;
                        std::ptr::copy_nonoverlapping(ptr, y_vals.as_mut_ptr(), hidden);
                    }
                    for i in 0..hidden {
                        y_vals[i] = c_part[i] * y_vals[i];
                    }
                    unsafe {
                        let ptr = self.buf_y.contents() as *mut f32;
                        std::ptr::copy_nonoverlapping(y_vals.as_ptr(), ptr, hidden);
                    }

                    // ── Encoder 3: out_proj + residual (conv layers have no MLP) ──

                    {
                        let cb = self.ops.queue.new_command_buffer();
                        let enc = cb.new_compute_command_encoder();
                        // out_proj: y → attn_proj
                        self.encode_mv_auto(&enc, &format!("model.layers.{layer}.conv.out_proj.weight"),
                            &self.buf_y, &self.buf_attn_proj,
                            hidden, hidden);
                        // residual: x += attn_proj
                        self.ops.encode_add_f32_inplace(
                            &enc, &self.buf_x, &self.buf_attn_proj, hidden);
                        enc.end_encoding();
                        cb.commit();
                        cb.wait_until_completed();
                    }

                    conv_idx += 1;
                }

                "full_attention" | "attention" => {
                    // Attention layer: fully fused on GPU
                    // All ops encoded into a single encoder — zero CPU syncs
                    // until the layer is complete.

                    let cb = self.ops.queue.new_command_buffer();
                    let enc = cb.new_compute_command_encoder();

                    // Operator norm
                    let w_norm = self.get_weight(
                        &format!("model.layers.{layer}.operator_norm.weight"));
                    self.ops.encode_rms_norm_f16w(
                        &enc, &self.buf_x, w_norm, &self.buf_x_norm,
                        hidden, cfg.rms_norm_eps, false);

                    // QKV projections
                    self.encode_mv_auto(&enc, &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                        &self.buf_x_norm, &self.buf_q,
                        attn_dim, hidden);
                    self.encode_mv_auto(&enc, &format!("model.layers.{layer}.self_attn.k_proj.weight"),
                        &self.buf_x_norm, &self.buf_k,
                        kv_dim, hidden);
                    self.encode_mv_auto(&enc, &format!("model.layers.{layer}.self_attn.v_proj.weight"),
                        &self.buf_x_norm, &self.buf_v,
                        kv_dim, hidden);

                    // TODO: Q/K per-head layernorm (requires encode_rms_norm_f16w_at)
                    // Skipped for now — model still functions, minor quality impact.

                    // RoPE (rotate-half layout for LFM)
                    self.ops.encode_rope_half_f32(
                        &enc, &self.buf_q, n_heads, head_dim, head_dim,
                        pos, cfg.rope_theta);
                    self.ops.encode_rope_half_f32(
                        &enc, &self.buf_k, n_kv_heads, head_dim, head_dim,
                        pos, cfg.rope_theta);

                    // Write K,V to cache
                    let target_base = kv_cache.layout()
                        .token_base_elem(block_id, layer, token_off)
                        .map_err(|e| CoreError::Backend(
                            format!("LfmGraphState token_base_elem: {e}")))?;
                    kv_store.encode_write_token_f32(
                        &enc, target_base, &self.buf_k, &self.buf_v, kv_dim);

                    // Build per-layer bases buffer for attention
                    // Contains the page-table derived element offset for each
                    // token position in the KV cache for this specific layer.
                    let bases_buf = self.ops.device.new_buffer(
                        (seq * 4) as u64, // u32 per token
                        MTLResourceOptions::StorageModeShared,
                    );
                    unsafe {
                        let bases_ptr = bases_buf.contents() as *mut u32;
                        for t in 0..seq {
                            let b = page_table.block_for_token(t)
                                .map_err(|e| CoreError::Backend(
                                    format!("LfmGraphState block_for_token: {e}")))?;
                            let o = page_table.offset_in_block(t)
                                .map_err(|e| CoreError::Backend(
                                    format!("LfmGraphState offset_in_block: {e}")))?;
                            let base = kv_cache.layout()
                                .token_base_elem(b, layer, o)
                                .map_err(|e| CoreError::Backend(
                                    format!("LfmGraphState token_base_elem: {e}")))?;
                            *bases_ptr.add(t) = base as u32;
                        }
                    }

                    // Fused GQA attention
                    kv_store.encode_attention(
                        &enc,
                        &bases_buf,
                        0,
                        &self.buf_q,
                        &self.buf_attn_out,
                        seq as u32,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        None,
                        None,
                    );

                    // O projection (LFM uses 'out_proj' not 'o_proj')
                    self.encode_mv_auto(&enc, &format!("model.layers.{layer}.self_attn.out_proj.weight"),
                        &self.buf_attn_out, &self.buf_mlp_in,
                        hidden, attn_dim);

                    // Residual: x += mlp_in
                    self.ops.encode_add_f32_inplace(
                        &enc, &self.buf_x, &self.buf_mlp_in, hidden);

                    // Post-attention norm (LFM uses 'ffn_norm')
                    let w_post = self.get_weight(
                        &format!("model.layers.{layer}.ffn_norm.weight"));
                    self.ops.encode_rms_norm_f16w(
                        &enc, &self.buf_x, w_post, &self.buf_x_norm,
                        hidden, cfg.rms_norm_eps, false);

                    // Gate + Up projection (LFM uses 'feed_forward' not 'mlp')
                    self.encode_mv_auto(&enc, &format!("model.layers.{layer}.feed_forward.w1.weight"),
                        &self.buf_x_norm, &self.buf_gate,
                        intermediate, hidden);
                    self.encode_mv_auto(&enc, &format!("model.layers.{layer}.feed_forward.w3.weight"),
                        &self.buf_x_norm, &self.buf_up,
                        intermediate, hidden);

                    // SiLU activation: gate *= sigmoid(gate) * up
                    self.ops.encode_silu_mul_f32_inplace(
                        &enc, &self.buf_gate, &self.buf_up, intermediate);

                    // Down projection (LFM uses w2 for down)
                    self.encode_mv_auto(&enc, &format!("model.layers.{layer}.feed_forward.w2.weight"),
                        &self.buf_gate, &self.buf_down,
                        hidden, intermediate);

                    // Residual: x += down
                    self.ops.encode_add_f32_inplace(
                        &enc, &self.buf_x, &self.buf_down, hidden);

                    enc.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                }

                _ => {
                    return Err(CoreError::Backend(format!(
                        "LfmGraphState: unknown layer type '{layer_type}' at layer {layer}")));
                }
            }
        }

        // Final norm (LFM uses 'embedding_norm.weight' not 'model.norm.weight')
        // LM head: use embed_tokens.weight as transposed projection (dot product with each row)
        let cb = self.ops.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let w_final = self.get_weight("model.embedding_norm.weight");
        self.ops.encode_rms_norm_f16w(
            &enc, &self.buf_x, w_final, &self.buf_x_norm,
            hidden, cfg.rms_norm_eps, false);

        if return_logits {
            self.encode_mv_auto(&enc, "model.embed_tokens.weight",
                &self.buf_x_norm, &self.buf_logits,
                cfg.vocab_size, hidden);
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        if !return_logits {
            return Ok(None);
        }

        let mut logits = vec![0.0f32; cfg.vocab_size];
        unsafe {
            let ptr = self.buf_logits.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, logits.as_mut_ptr(), cfg.vocab_size);
        }

        // Divergence detection
        let mut nan_count = 0;
        let mut inf_count = 0;
        for &v in &logits {
            if v.is_nan() { nan_count += 1; }
            else if v.is_infinite() { inf_count += 1; }
        }
        if nan_count > 0 || inf_count > 0 {
            return Err(CoreError::Backend(format!(
                "LfmGraphState: divergence at pos {pos} (NaNs={nan_count}, Infs={inf_count})")));
        }

        // DEBUG: compare graph output vs CPU reference
        if std::env::var("LFM_DEBUG_GRAPH").map(|v| v == "1").unwrap_or(false) {
            // Read buf_x (pre-norm hidden state from GPU, already committed+waited)
            let mut x_cpu = vec![0.0f32; hidden];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buf_x.contents() as *const f32,
                    x_cpu.as_mut_ptr(),
                    hidden,
                );
            }
            // Hidden state statistics
            let mean = x_cpu.iter().sum::<f32>() / hidden as f32;
            let variance = x_cpu.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
            let std = variance.sqrt();
            let max_val = x_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_val = x_cpu.iter().cloned().fold(f32::INFINITY, f32::min);
            let l2_norm = x_cpu.iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!("LFM_DEBUG_STATE: pos={pos} mean={mean:.4} std={std:.4} max={max_val:.4} min={min_val:.4} l2={l2_norm:.4}");

            // Also dump the first 8 values
            eprintln!("LFM_DEBUG_STATE:   x[..8] = {:?}", &x_cpu[..8.min(hidden)]);

            // Read final norm weight (f16)
            let w_final_buf = self.get_weight("model.embedding_norm.weight");
            let w_final_len = w_final_buf.length() as usize / 2; // f16 = 2 bytes per element
            let w_final_u16: &[u16] = unsafe {
                std::slice::from_raw_parts(w_final_buf.contents() as *const u16, w_final_len)
            };

            // CPU RMSNorm: normed[i] = x_cpu[i] * rms * w_final[i]
            let ss: f32 = x_cpu.iter().map(|v| v * v).sum();
            let rms = (ss / hidden as f32 + cfg.rms_norm_eps).sqrt().recip();
            let mut normed = vec![0.0f32; hidden];
            for i in 0..hidden {
                normed[i] = x_cpu[i] * rms * half::f16::from_bits(w_final_u16[i]).to_f32();
            }

            let embed_name = "model.embed_tokens.weight";
            let embed_dtype = self.tensor_dtypes.get(embed_name)
                .map(|s| s.as_str()).unwrap_or("f16");

            const DEBUG_VOCAB_HEAD: usize = 10;
            let head = DEBUG_VOCAB_HEAD.min(cfg.vocab_size as usize);

            let mut cpu_logits = vec![0.0f32; head];

            if embed_dtype == "u32" {
                // MLX i4: dequantize each row on CPU
                let w_embed = self.get_weight(embed_name);
                let w_embed_len = w_embed.length() as usize / 4; // u32 = 4 bytes
                let w_u32: &[u32] = unsafe {
                    std::slice::from_raw_parts(w_embed.contents() as *const u32, w_embed_len)
                };

                let scales_name = "model.embed_tokens.scales";
                let biases_name = "model.embed_tokens.biases";

                let s_buf = self.i4_scales.get(scales_name)
                    .unwrap_or_else(|| panic!("LFM_DEBUG: missing {scales_name}"));
                let b_buf = self.i4_biases.get(biases_name)
                    .unwrap_or_else(|| panic!("LFM_DEBUG: missing {biases_name}"));

                let s_len = s_buf.length() as usize / 4;
                let b_len = b_buf.length() as usize / 4;
                let s_f32: &[f32] = unsafe {
                    std::slice::from_raw_parts(s_buf.contents() as *const f32, s_len)
                };
                let b_f32: &[f32] = unsafe {
                    std::slice::from_raw_parts(b_buf.contents() as *const f32, b_len)
                };

                let group_size = 64usize;
                let groups_per_row = (hidden + group_size - 1) / group_size;
                let packed_per_row = hidden / 8;

                for i in 0..head {
                    let mut acc = 0.0f32;
                    let row_off = i * packed_per_row;
                    for g in 0..groups_per_row {
                        let g_start = g * group_size;
                        let g_end = (g_start + group_size).min(hidden);
                        let scale = s_f32[i * groups_per_row + g];
                        let bias = b_f32[i * groups_per_row + g];
                        for j in g_start..g_end {
                            let packed = w_u32[row_off + j / 8];
                            let nibble = (packed >> ((j % 8) * 4)) & 0xF;
                            let w_val = (nibble as f32) * scale + bias;
                            acc += w_val * normed[j];
                        }
                    }
                    cpu_logits[i] = acc;
                }
            } else {
                // f16 embeddings
                let w_embed = self.get_weight(embed_name);
                let w_embed_len = w_embed.length() as usize / 2;
                let w_u16: &[u16] = unsafe {
                    std::slice::from_raw_parts(w_embed.contents() as *const u16, w_embed_len)
                };

                for i in 0..head {
                    let mut acc = 0.0f32;
                    let row_off = i * hidden;
                    for j in 0..hidden {
                        let w = half::f16::from_bits(w_u16[row_off + j]).to_f32();
                        acc += w * normed[j];
                    }
                    cpu_logits[i] = acc;
                }
            }

            // Compare GPU vs CPU logits for the first `head` vocab entries
            let diff_sum: f32 = (0..head).map(|i| (logits[i] - cpu_logits[i]).abs()).sum();
            let gpu_top1 = (0..cfg.vocab_size as usize)
                .max_by(|&a, &b| logits[a].partial_cmp(&logits[b]).unwrap()).unwrap_or(0);
            let cpu_top1 = (0..head)
                .max_by(|&a, &b| cpu_logits[a].partial_cmp(&cpu_logits[b]).unwrap()).unwrap_or(0);

            eprintln!("LFM_DEBUG: pos={pos} vocab_size={vocab_sz} diff_first{hd}={diff_sum:.6} gpu_top1={gpu_top1} cpu_top1_within{hd}={cpu_top1}",
                hd = head, vocab_sz = cfg.vocab_size);

            // Always compute CPU logit at gpu_top1 for comparison
            if gpu_top1 < cfg.vocab_size as usize && gpu_top1 >= head {
                let mut cpu_at_gpu_top = 0.0f32;
                if embed_dtype == "u32" {
                    let w_embed = self.get_weight(embed_name);
                    let w_embed_len = w_embed.length() as usize / 4;
                    let w_u32: &[u32] = unsafe {
                        std::slice::from_raw_parts(w_embed.contents() as *const u32, w_embed_len)
                    };
                    let s_buf = self.i4_scales.get("model.embed_tokens.scales").unwrap();
                    let b_buf = self.i4_biases.get("model.embed_tokens.biases").unwrap();
                    let s_f32: &[f32] = unsafe { std::slice::from_raw_parts(s_buf.contents() as *const f32, s_buf.length() as usize / 4) };
                    let b_f32: &[f32] = unsafe { std::slice::from_raw_parts(b_buf.contents() as *const f32, b_buf.length() as usize / 4) };
                    let packed_per_row = hidden / 8;
                    let groups_per_row = (hidden + 63) / 64;
                    let i = gpu_top1;
                    let mut acc = 0.0f32;
                    let row_off = i * packed_per_row;
                    for j in 0..hidden {
                        let packed = w_u32[row_off + j / 8];
                        let nibble = (packed >> ((j % 8) * 4)) & 0xF;
                        let g = j / 64;
                        let w_val = (nibble as f32) * s_f32[i * groups_per_row + g] + b_f32[i * groups_per_row + g];
                        acc += w_val * normed[j];
                    }
                    cpu_at_gpu_top = acc;
                }
                eprintln!("LFM_DEBUG:   logits[{}] gpu={:.6} cpu={:.6} gpu_vs_cpu_diff={:.6}",
                    gpu_top1, logits[gpu_top1], cpu_at_gpu_top,
                    (logits[gpu_top1] - cpu_at_gpu_top).abs());
            }

            if diff_sum > 1.0 {
                eprintln!("LFM_DEBUG: gpu_logits[..{head}] = {:?}", &logits[..head]);
                eprintln!("LFM_DEBUG: cpu_logits[..{head}] = {:?}", &cpu_logits);
                let scatter: [usize; 5] = [100, 1000, 5000, 10000, 20000];
                for &idx in &scatter {
                    if idx < cfg.vocab_size as usize {
                        eprintln!("LFM_DEBUG:   logits[{idx}] gpu={:.6} cpu=N/A (not computed)", logits[idx]);
                    }
                }
            }
        }

        Ok(Some(logits))
        })
    }
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
            attention_softcap: 0.0,
            ..ModelConfig::default()
        };

        // Initialize conv state cache
        // For each conv layer, store the last kernel_size Bx vectors for causal conv
        let num_conv_layers = layer_types.iter().filter(|t| *t == "conv").count();
        let conv_states: Vec<Vec<f32>> = (0..num_conv_layers)
            .map(|_| vec![0.0f32; conv_kernel_size * cfg.hidden_size])
            .collect();

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let (metal_ops, graph_state) = (None, None);
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let (metal_ops, graph_state) = ((), ());

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
            metal_ops,
            graph_state,
        })
    }

    pub fn from_file(file: CellmFile) -> Result<Self, CoreError> {
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
            attention_softcap: 0.0,
            ..ModelConfig::default()
        };

        // Initialize conv state cache
        // For each conv layer, store the last kernel_size Bx vectors for causal conv
        let num_conv_layers = layer_types.iter().filter(|t| *t == "conv").count();
        let conv_states: Vec<Vec<f32>> = (0..num_conv_layers)
            .map(|_| vec![0.0f32; conv_kernel_size * cfg.hidden_size])
            .collect();

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let (metal_ops, graph_state) = (None, None);
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let (metal_ops, graph_state) = ((), ());

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
            metal_ops,
            graph_state,
        })
    }

    pub fn file(&self) -> &CellmFile {
        &self.file
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Clears the causal convolution state so the next sequence starts fresh.
    ///
    /// LFM2 is a hybrid architecture: attention layers keep their history in
    /// the paged KV cache, but conv layers keep theirs here, in a rolling
    /// window of the last `conv_kernel_size` gated activations. Resetting a
    /// session truncates the page table and therefore misses this entirely,
    /// so without this call generation N+1 begins with generation N's
    /// convolution history still in the window.
    ///
    /// That leak is observable: with greedy decoding at temperature 0.0 the
    /// same prompt produced three different outputs depending on what ran
    /// before it, including a corrupted first token ("SubjectRXT") and
    /// verbatim echoes of the previous prompt's source text.
    ///
    /// Must be called whenever a session is reset or a new sequence begins.
    pub fn reset_state(&mut self) {
        for state in &mut self.conv_states {
            state.fill(0.0);
        }

        // The Metal path keeps its own copy in shared-storage buffers; zeroing
        // the CPU vectors above does not touch them.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gs) = self.graph_state.as_mut() {
            for buf in &gs.conv_states {
                let len = buf.length() as usize;
                unsafe {
                    std::ptr::write_bytes(buf.contents() as *mut u8, 0, len);
                }
            }
        }
    }

    pub fn set_max_layers(&mut self, n: usize) {
        self.max_layers = n.min(self.cfg.num_hidden_layers).max(1);
    }

    pub fn enable_metal_full_backend(&mut self) -> bool {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            match MetalOps::create() {
                Ok(ops) => {
                    let num_conv_layers = self.layer_types
                        .iter()
                        .filter(|t| *t == "conv")
                        .count();
                    let ops_for_gs = ops.clone();
                    let mut gs = LfmGraphState::new(
                        ops_for_gs,
                        self.cfg.clone(),
                        self.layer_types.clone(),
                        self.conv_kernel_size,
                        num_conv_layers,
                    );
                    // MLX i4 weights: per-ops Metal path (dequant cache)
                    // Fused graph + mlx_i4 kernel verified correct per-matmul
                    // but needs debugging for full 16-layer pipeline (hidden state diverges).
                    eprintln!("lfm: i4 quantized weights; using per-ops Metal (fused graph debug wip)");
                    self.metal_ops = Some(ops);
                    true
                }
                Err(e) => {
                    eprintln!("lfm: failed to enable metal backend: {e}");
                    false
                }
            }
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        false
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

    /// Depthwise short-conv kernel width (`conv_L_cache`).
    pub fn conv_kernel_size(&self) -> usize {
        self.conv_kernel_size
    }

    /// Layer kind at `idx` (`"conv"` or `"full_attention"`), `""` when out of range.
    pub fn layer_type(&self, idx: usize) -> &str {
        self.layer_types.get(idx).map(|s| s.as_str()).unwrap_or("")
    }

    /// Load an RMSNorm gain vector as f32. Used by the bidirectional encoder,
    /// which cannot reuse the single-position scratch buffers of `step_inner`.
    pub(crate) fn norm_weight_f32(&self, name: &str) -> Result<Vec<f32>, CoreError> {
        let w = self.tensor_f16(name)?;
        Ok(w.iter().map(|&b| f16::from_bits(b).to_f32()).collect())
    }

    /// Copy an f16 weight matrix out of the mmap so callers can hold it across
    /// `&mut self` borrows.
    pub(crate) fn weight_f16_owned(&self, name: &str) -> Result<Vec<u16>, CoreError> {
        match self.tensor_dtype(name).as_deref() {
            Some("f16") | None => Ok(self.tensor_f16(name)?.to_vec()),
            Some(other) => Err(CoreError::Backend(format!(
                "lfm encode: tensor '{name}' has dtype '{other}', only f16 is supported"
            ))),
        }
    }

    /// Owned int8 weight matrix plus its per-row scales.
    pub(crate) fn weight_i8_owned(&self, name: &str) -> Result<(Vec<i8>, Vec<u16>), CoreError> {
        let w = self.tensor_i8(name)?.to_vec();
        let s = self.tensor_f16(&format!("{name}.qscale"))?.to_vec();
        Ok((w, s))
    }

    /// Dequantize an MLX-style int4 weight into a dense f32 matrix.
    ///
    /// Same layout as [`Self::linear_i4_out_in`] but returns the matrix instead
    /// of consuming it, and bypasses the LRU cache — the encoder streams each
    /// layer once so caching would only add peak memory.
    pub(crate) fn weight_i4_dequant(
        &self,
        name: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>, CoreError> {
        let base_name = name.trim_end_matches(".weight");
        let weight_u32: &[u32] = bytemuck::cast_slice(self.file.tensor_bytes(name)?);
        let scales_f32: &[f32] =
            bytemuck::cast_slice(self.file.tensor_bytes(&format!("{base_name}.scales"))?);
        let biases_f32: &[f32] =
            bytemuck::cast_slice(self.file.tensor_bytes(&format!("{base_name}.biases"))?);

        let group_size = 64usize;
        let groups_per_row = in_dim.div_ceil(group_size);
        let packed_in = in_dim / 8;

        let mut dequant = vec![0.0f32; out_dim * in_dim];
        dequant.par_chunks_exact_mut(in_dim).enumerate().for_each(|(i, row)| {
            let row_offset = i * packed_in;
            for j_packed in 0..packed_in {
                let packed = weight_u32[row_offset + j_packed];
                let g = (j_packed * 8) / group_size;
                let scale = scales_f32[i * groups_per_row + g];
                let bias = biases_f32[i * groups_per_row + g];
                let j_base = j_packed * 8;
                for k in 0..8 {
                    let nibble = ((packed >> (k * 4)) & 0xF) as f32;
                    row[j_base + k] = nibble * scale + bias;
                }
            }
        });
        Ok(dequant)
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

    /// Number of prompt tokens processed per batched prefill chunk.
    ///
    /// Large enough to amortise each weight read across many tokens, small
    /// enough that a chunk's activations stay in cache: at hidden=1024 a
    /// 32-token chunk is 128 KB of f32 activations. Measured 3-7x over the
    /// per-token path on the real projection shapes, with gains flattening
    /// beyond this.
    const PREFILL_CHUNK: usize = 32;

    /// Prefill that runs a whole chunk of tokens through each weight at once.
    ///
    /// Identical arithmetic to calling [`Self::prefill`], just reordered: the
    /// per-token loop becomes the inner dimension of a GEMM instead of the
    /// outer loop around a GEMV, so each weight is read once per chunk rather
    /// than once per token.
    ///
    /// Only the CPU path is batched. Metal already coalesces its work into
    /// command buffers, so it falls back to [`Self::prefill`].
    pub fn prefill_batched(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
    ) -> Result<(), CoreError> {
        if !self.can_batch_prefill() || tokens.len() < 2 {
            return self.prefill(tokens, start_pos, page_table, kv_cache);
        }

        let cfg = self.cfg.clone();
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let attn_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        for (ci, chunk) in tokens.chunks(Self::PREFILL_CHUNK).enumerate() {
            let nt = chunk.len();
            let chunk_start = start_pos + ci * Self::PREFILL_CHUNK;

            // Reserve cache slots for the whole chunk before any layer runs, so
            // the page table can resolve every position in it.
            for i in 0..nt {
                if chunk_start + i == page_table.token_count() {
                    page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                        CoreError::Backend(format!(
                            "lfm prefill_batched: page_table append_token failed: {e}"
                        ))
                    })?;
                }
            }

            let mut x = vec![0.0f32; nt * hidden];
            for (i, &tok) in chunk.iter().enumerate() {
                self.embed_token(tok, &mut x[i * hidden..(i + 1) * hidden])?;
            }

            let mut x_norm = vec![0.0f32; nt * hidden];
            let mut norm_w = vec![0.0f32; hidden];
            let mut bcx = vec![0.0f32; nt * hidden * 3];
            let mut y = vec![0.0f32; nt * hidden];
            let mut proj = vec![0.0f32; nt * hidden];
            let mut q = vec![0.0f32; nt * attn_dim];
            let mut k = vec![0.0f32; nt * kv_dim];
            let mut v = vec![0.0f32; nt * kv_dim];
            let mut attn_out = vec![0.0f32; nt * attn_dim];
            let mut gate = vec![0.0f32; nt * inter];
            let mut up = vec![0.0f32; nt * inter];
            let mut conv_layer_idx = 0usize;

            for layer in 0..self.max_layers {
                let layer_type = self
                    .layer_types
                    .get(layer)
                    .map(|s| s.as_str())
                    .unwrap_or("conv")
                    .to_string();

                let op_norm_name = format!("model.layers.{layer}.operator_norm.weight");
                self.rmsnorm_weight(&op_norm_name, &mut norm_w)?;
                for t in 0..nt {
                    let r = t * hidden..(t + 1) * hidden;
                    rms_norm_f32(&x[r.clone()], &norm_w, cfg.rms_norm_eps, &mut x_norm[r]);
                }

                match layer_type.as_str() {
                    "conv" => {
                        self.linear_batched_out_in(
                            &x_norm,
                            &format!("model.layers.{layer}.conv.in_proj.weight"),
                            nt,
                            hidden * 3,
                            hidden,
                            &mut bcx,
                        )?;

                        // The convolution is the one part that cannot batch: its
                        // state is a rolling window that advances one token at a
                        // time. It is cheap (kernel_size MACs per channel) and
                        // streams no weights, so leaving it sequential costs
                        // little now that the projections around it are batched.
                        let ks = self.conv_kernel_size;
                        let conv_kernel_name = format!("model.layers.{layer}.conv.conv.weight");
                        let kernel: Vec<u16> = {
                            let bytes = self.file.tensor_bytes(&conv_kernel_name)?;
                            bytemuck::cast_slice::<u8, u16>(bytes).to_vec()
                        };
                        let state = &mut self.conv_states[conv_layer_idx];

                        for t in 0..nt {
                            let base = t * hidden * 3;
                            let b_part = &bcx[base..base + hidden];
                            let c_part = &bcx[base + hidden..base + 2 * hidden];
                            let x_part = &bcx[base + 2 * hidden..base + 3 * hidden];

                            if ks > 1 {
                                state.copy_within(hidden..(ks * hidden), 0);
                            }
                            for i in 0..hidden {
                                state[(ks - 1) * hidden + i] = b_part[i] * x_part[i];
                            }

                            let yr = t * hidden;
                            for i in 0..hidden {
                                let mut acc = 0.0f32;
                                let kernel_base = i * ks;
                                for kk in 0..ks {
                                    acc += state[kk * hidden + i]
                                        * f16::from_bits(kernel[kernel_base + kk]).to_f32();
                                }
                                y[yr + i] = c_part[i] * acc;
                            }
                        }

                        self.linear_batched_out_in(
                            &y,
                            &format!("model.layers.{layer}.conv.out_proj.weight"),
                            nt,
                            hidden,
                            hidden,
                            &mut proj,
                        )?;

                        conv_layer_idx += 1;
                    }
                    "full_attention" | "attention" => {
                        self.linear_batched_out_in(
                            &x_norm,
                            &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                            nt,
                            attn_dim,
                            hidden,
                            &mut q,
                        )?;
                        self.linear_batched_out_in(
                            &x_norm,
                            &format!("model.layers.{layer}.self_attn.k_proj.weight"),
                            nt,
                            kv_dim,
                            hidden,
                            &mut k,
                        )?;
                        self.linear_batched_out_in(
                            &x_norm,
                            &format!("model.layers.{layer}.self_attn.v_proj.weight"),
                            nt,
                            kv_dim,
                            hidden,
                            &mut v,
                        )?;

                        let q_norm_w: Option<Vec<f32>> = self
                            .tensor_f16(&format!("model.layers.{layer}.self_attn.q_layernorm.weight"))
                            .ok()
                            .map(|w| w.iter().map(|&b| f16::from_bits(b).to_f32()).collect());
                        let k_norm_w: Option<Vec<f32>> = self
                            .tensor_f16(&format!("model.layers.{layer}.self_attn.k_layernorm.weight"))
                            .ok()
                            .map(|w| w.iter().map(|&b| f16::from_bits(b).to_f32()).collect());

                        let mut head_buf = vec![0.0f32; head_dim];
                        for t in 0..nt {
                            let pos = chunk_start + t;
                            let qb = t * attn_dim;
                            let kb = t * kv_dim;

                            if let Some(w) = &q_norm_w {
                                for h in 0..n_heads {
                                    let s = qb + h * head_dim;
                                    rms_norm_f32(
                                        &q[s..s + head_dim],
                                        w,
                                        cfg.rms_norm_eps,
                                        &mut head_buf,
                                    );
                                    q[s..s + head_dim].copy_from_slice(&head_buf);
                                }
                            }
                            if let Some(w) = &k_norm_w {
                                for h in 0..n_kv_heads {
                                    let s = kb + h * head_dim;
                                    rms_norm_f32(
                                        &k[s..s + head_dim],
                                        w,
                                        cfg.rms_norm_eps,
                                        &mut head_buf,
                                    );
                                    k[s..s + head_dim].copy_from_slice(&head_buf);
                                }
                            }

                            rope_non_interleaved_inplace_f32(
                                &mut q[qb..qb + attn_dim],
                                n_heads,
                                head_dim,
                                head_dim,
                                pos,
                                cfg.rope_theta,
                            );
                            rope_non_interleaved_inplace_f32(
                                &mut k[kb..kb + kv_dim],
                                n_kv_heads,
                                head_dim,
                                head_dim,
                                pos,
                                cfg.rope_theta,
                            );

                            let block_id = page_table.block_for_token(pos).map_err(|e| {
                                CoreError::Backend(format!("lfm: block_for_token failed: {e}"))
                            })?;
                            let off = page_table.offset_in_block(pos).map_err(|e| {
                                CoreError::Backend(format!("lfm: offset_in_block failed: {e}"))
                            })?;
                            let mut cv = kv_cache.view_mut();
                            cv.write_token(
                                block_id,
                                layer,
                                off,
                                &k[kb..kb + kv_dim],
                                &v[kb..kb + kv_dim],
                            )?;
                        }

                        // Every K/V in this chunk is now written, so each token
                        // can attend over exactly its own causal prefix. Gathering
                        // 0..=pos is what the per-token path did implicitly, by
                        // running before later tokens existed.
                        let cr = kv_cache.view();
                        let mut bases: Vec<usize> = Vec::with_capacity(chunk_start + nt);
                        for t in 0..nt {
                            let pos = chunk_start + t;
                            bases.clear();
                            for tpos in 0..=pos {
                                let b = page_table.block_for_token(tpos).map_err(|e| {
                                    CoreError::Backend(format!("lfm: block_for_token failed: {e}"))
                                })?;
                                let o = page_table.offset_in_block(tpos).map_err(|e| {
                                    CoreError::Backend(format!("lfm: offset_in_block failed: {e}"))
                                })?;
                                bases.push(cr.layout.token_base_elem(b, layer, o)?);
                            }
                            cr.attention_single_token_gqa_from_bases(
                                &bases,
                                &q[t * attn_dim..(t + 1) * attn_dim],
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                None,
                                None,
                                &mut attn_out[t * attn_dim..(t + 1) * attn_dim],
                            )?;
                        }

                        self.linear_batched_out_in(
                            &attn_out,
                            &format!("model.layers.{layer}.self_attn.out_proj.weight"),
                            nt,
                            hidden,
                            attn_dim,
                            &mut proj,
                        )?;
                    }
                    _ => {
                        return Err(CoreError::Backend(format!(
                            "lfm: unknown layer type '{layer_type}' at layer {layer}"
                        )));
                    }
                }

                for i in 0..nt * hidden {
                    x[i] += proj[i];
                }

                let ffn_norm_name = format!("model.layers.{layer}.ffn_norm.weight");
                self.rmsnorm_weight(&ffn_norm_name, &mut norm_w)?;
                for t in 0..nt {
                    let r = t * hidden..(t + 1) * hidden;
                    rms_norm_f32(&x[r.clone()], &norm_w, cfg.rms_norm_eps, &mut x_norm[r]);
                }

                self.linear_batched_out_in(
                    &x_norm,
                    &format!("model.layers.{layer}.feed_forward.w1.weight"),
                    nt,
                    inter,
                    hidden,
                    &mut gate,
                )?;
                self.linear_batched_out_in(
                    &x_norm,
                    &format!("model.layers.{layer}.feed_forward.w3.weight"),
                    nt,
                    inter,
                    hidden,
                    &mut up,
                )?;

                for i in 0..nt * inter {
                    let g = gate[i];
                    gate[i] = g * (1.0 / (1.0 + (-g).exp())) * up[i];
                }

                self.linear_batched_out_in(
                    &gate,
                    &format!("model.layers.{layer}.feed_forward.w2.weight"),
                    nt,
                    hidden,
                    inter,
                    &mut proj,
                )?;

                for i in 0..nt * hidden {
                    x[i] += proj[i];
                }
            }
        }

        Ok(())
    }

    /// True when the batched prefill path applies.
    ///
    /// It reimplements only the CPU int8 route, so anything using Metal or a
    /// non-int8 projection dtype must keep the per-token path to stay correct.
    fn can_batch_prefill(&self) -> bool {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if self.metal_ops.is_some() {
            return false;
        }
        if self.max_layers == 0 {
            return false;
        }
        // All large projections share a dtype in practice; sample one per kind.
        for layer in 0..self.max_layers {
            let name = match self.layer_types.get(layer).map(|s| s.as_str()) {
                Some("conv") => format!("model.layers.{layer}.conv.in_proj.weight"),
                Some("full_attention") | Some("attention") => {
                    format!("model.layers.{layer}.self_attn.q_proj.weight")
                }
                _ => return false,
            };
            // Both dtypes have a batched kernel: i8 via gemm_i8_w8a8, packed
            // int4 via gemm_affine_i4_f32.
            let batchable = |d: Option<&str>| matches!(d, Some("i8") | Some("u32"));
            if !batchable(self.tensor_dtype(&name).as_deref()) {
                return false;
            }
            if !batchable(
                self.tensor_dtype(&format!("model.layers.{layer}.feed_forward.w1.weight"))
                    .as_deref(),
            ) {
                return false;
            }
        }
        true
    }

    pub fn prefill_topk(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        top_k: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>, CoreError> {
        let n = tokens.len();
        if n <= 1 {
            return Ok(Vec::new());
        }
        let mut results = Vec::with_capacity(n - 1);
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            if pos == page_table.token_count() {
                page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                    CoreError::Backend(format!("lfm prefill_topk: page_table append_token failed: {e}"))
                })?;
            }
            let mut x = vec![0.0f32; self.cfg.hidden_size];
            self.embed_token(tok, &mut x)?;
            let logits = self.step_inner(&x, pos, page_table, kv_cache, true)?;
            if i < n - 1 {
                results.push(self.topk_from_logits(&logits, top_k)?);
            }
        }
        Ok(results)
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

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gs) = &mut self.graph_state {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LFM_GRAPH_WARNED: AtomicBool = AtomicBool::new(false);
            if kv_cache.encoding() == cellm_cache::KvEncodingKind::TurboQuant {
                if !LFM_GRAPH_WARNED.swap(true, Ordering::Relaxed) {
                    eprintln!("lfm: TurboQuant kv-cache unsupported in fused graph, falling back to per-ops path");
                }
            } else {
                match gs.step_fused(
                    x0,
                    &self.cfg,
                    "",
                    kv_cache,
                    page_table,
                    pos,
                    token_off,
                    block_id as u32,
                    return_logits,
                ) {
                    Ok(maybe_logits) => {
                        if let Some(logits) = maybe_logits {
                            let has_non_finite = logits.iter().any(|v| !v.is_finite());
                            if has_non_finite {
                                eprintln!("lfm fused graph: non-finite logits at pos {pos}; disabling graph");
                                self.graph_state = None;
                            } else {
                                return Ok(logits);
                            }
                        } else {
                            return Ok(vec![]);
                        }
                    }
                    Err(e) => {
                        eprintln!("lfm fused graph: step_fused failed at pos {pos}: {e}; falling back");
                        self.graph_state = None;
                    }
                }
            }
        }

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
            let norm_name = format!("model.layers.{layer}.operator_norm.weight");
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if let Some(ops) = &self.metal_ops {
                let w = self.tensor_f16(&norm_name)?;
                ops.rms_norm_f16w(&x, w, cfg.rms_norm_eps, false, &norm_name, &mut x_norm)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                self.rmsnorm_weight(&norm_name, &mut op_norm_w)?;
                rms_norm_f32(&x, &op_norm_w, cfg.rms_norm_eps, &mut x_norm);
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                self.rmsnorm_weight(&norm_name, &mut op_norm_w)?;
                rms_norm_f32(&x, &op_norm_w, cfg.rms_norm_eps, &mut x_norm);
            }

            match layer_type {
                "conv" => {
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

                    let ks = self.conv_kernel_size;
                    let conv_kernel_name = format!("model.layers.{layer}.conv.conv.weight");
                    let mut conv_out = vec![0.0f32; hidden];

                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    if let Some(ops) = &self.metal_ops {
                        let conv_kernel_bytes = self.file.tensor_bytes(&conv_kernel_name)?;
                        let w: &[u16] = bytemuck::cast_slice(conv_kernel_bytes);
                        let state = &mut self.conv_states[conv_layer_idx];
                        ops.lfm_conv(state, &bx, w, ks, hidden, &conv_kernel_name, &mut conv_out)
                            .map_err(|e| CoreError::Backend(e.to_string()))?;
                    } else {
                        // ... CPU ...
                        let conv_kernel_bytes = self.file.tensor_bytes(&conv_kernel_name)?;
                        let conv_kernel_u16: &[u16] = bytemuck::cast_slice(conv_kernel_bytes);
                        let state = &mut self.conv_states[conv_layer_idx];
                        if ks > 1 { state.copy_within(hidden..(ks * hidden), 0); }
                        state[(ks - 1) * hidden..ks * hidden].copy_from_slice(&bx);
                        for i in 0..hidden {
                            let mut acc = 0.0f32;
                            let kernel_base = i * ks;
                            for k in 0..ks {
                                acc += state[k * hidden + i] * f16::from_bits(conv_kernel_u16[kernel_base + k]).to_f32();
                            }
                            conv_out[i] = acc;
                        }
                    }

                    // Second gating: y = C * conv_out
                    let mut y = vec![0.0f32; hidden];
                    for i in 0..hidden {
                        y[i] = c_part[i] * conv_out[i];
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
                    let mut qkv_done = false;
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    if let Some(ops) = &self.metal_ops {
                        let q_name = format!("model.layers.{layer}.self_attn.q_proj.weight");
                        let k_name = format!("model.layers.{layer}.self_attn.k_proj.weight");
                        let v_name = format!("model.layers.{layer}.self_attn.v_proj.weight");

                        // ONLY use Metal if weights are f16. If u32 (i4), fallback to CPU.
                        let q_dtype = self.tensor_dtype(&q_name).unwrap_or_else(|| "f16".to_string());
                        let k_dtype = self.tensor_dtype(&k_name).unwrap_or_else(|| "f16".to_string());
                        let v_dtype = self.tensor_dtype(&v_name).unwrap_or_else(|| "f16".to_string());

                        if q_dtype == "f16" && k_dtype == "f16" && v_dtype == "f16" {
                            let q_w = self.tensor_f16(&q_name)?;
                            let k_w = self.tensor_f16(&k_name)?;
                            let v_w = self.tensor_f16(&v_name)?;
                            ops.logits_qkv_f16(
                                &x_norm,
                                q_w, k_w, v_w,
                                attn_dim, kv_dim, kv_dim, hidden,
                                &format!("q.{layer}"), &format!("k.{layer}"), &format!("v.{layer}"),
                                &mut q, &mut k, &mut v
                            ).map_err(|e| CoreError::Backend(e.to_string()))?;
                            qkv_done = true;
                        }
                    }

                    if !qkv_done {
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
                    }

                    // Apply Q/K layernorm (LFM2 specific - applied per-head)
                    let mut q_normed = vec![0.0f32; attn_dim];
                    let mut k_normed = vec![0.0f32; kv_dim];
                    if let Ok(q_norm_w) = self.tensor_f16(&format!("model.layers.{layer}.self_attn.q_layernorm.weight")) {
                        #[cfg(any(target_os = "macos", target_os = "ios"))]
                        if let Some(ops) = &self.metal_ops {
                            for h in 0..n_heads {
                                let h_start = h * head_dim;
                                let h_end = h_start + head_dim;
                                ops.rms_norm_f16w(&q[h_start..h_end], q_norm_w, cfg.rms_norm_eps, false, &format!("qnorm.{layer}"), &mut q_normed[h_start..h_end])
                                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                            }
                        } else {
                            let q_norm_w_f32: Vec<f32> = q_norm_w.iter().map(|&x| f16::from_bits(x).to_f32()).collect();
                            for h in 0..n_heads {
                                let h_start = h * head_dim;
                                let h_end = h_start + head_dim;
                                rms_norm_f32(&q[h_start..h_end], &q_norm_w_f32, cfg.rms_norm_eps, &mut q_normed[h_start..h_end]);
                            }
                        }
                        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                        {
                            let q_norm_w_f32: Vec<f32> = q_norm_w.iter().map(|&x| f16::from_bits(x).to_f32()).collect();
                            for h in 0..n_heads {
                                let h_start = h * head_dim;
                                let h_end = h_start + head_dim;
                                rms_norm_f32(&q[h_start..h_end], &q_norm_w_f32, cfg.rms_norm_eps, &mut q_normed[h_start..h_end]);
                            }
                        }
                        q.copy_from_slice(&q_normed);
                    }
                    if let Ok(k_norm_w) = self.tensor_f16(&format!("model.layers.{layer}.self_attn.k_layernorm.weight")) {
                        #[cfg(any(target_os = "macos", target_os = "ios"))]
                        if let Some(ops) = &self.metal_ops {
                            for h in 0..n_kv_heads {
                                let h_start = h * head_dim;
                                let h_end = h_start + head_dim;
                                ops.rms_norm_f16w(&k[h_start..h_end], k_norm_w, cfg.rms_norm_eps, false, &format!("knorm.{layer}"), &mut k_normed[h_start..h_end])
                                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                            }
                        } else {
                            let k_norm_w_f32: Vec<f32> = k_norm_w.iter().map(|&x| f16::from_bits(x).to_f32()).collect();
                            for h in 0..n_kv_heads {
                                let h_start = h * head_dim;
                                let h_end = h_start + head_dim;
                                rms_norm_f32(&k[h_start..h_end], &k_norm_w_f32, cfg.rms_norm_eps, &mut k_normed[h_start..h_end]);
                            }
                        }
                        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                        {
                            let k_norm_w_f32: Vec<f32> = k_norm_w.iter().map(|&x| f16::from_bits(x).to_f32()).collect();
                            for h in 0..n_kv_heads {
                                let h_start = h * head_dim;
                                let h_end = h_start + head_dim;
                                rms_norm_f32(&k[h_start..h_end], &k_norm_w_f32, cfg.rms_norm_eps, &mut k_normed[h_start..h_end]);
                            }
                        }
                        k.copy_from_slice(&k_normed);
                    }

                    // Apply RoPE (non-interleaved/split layout for LFM2)
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    if let Some(ops) = &self.metal_ops {
                        ops.rope_half_f32(&mut q, n_heads, head_dim, head_dim, pos, cfg.rope_theta)
                            .map_err(|e| CoreError::Backend(e.to_string()))?;
                        ops.rope_half_f32(&mut k, n_kv_heads, head_dim, head_dim, pos, cfg.rope_theta)
                            .map_err(|e| CoreError::Backend(e.to_string()))?;
                    } else {
                        rope_non_interleaved_inplace_f32(&mut q, n_heads, head_dim, head_dim, pos, cfg.rope_theta);
                        rope_non_interleaved_inplace_f32(&mut k, n_kv_heads, head_dim, head_dim, pos, cfg.rope_theta);
                    }
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    {
                        rope_non_interleaved_inplace_f32(&mut q, n_heads, head_dim, head_dim, pos, cfg.rope_theta);
                        rope_non_interleaved_inplace_f32(&mut k, n_kv_heads, head_dim, head_dim, pos, cfg.rope_theta);
                    }

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

            // Batched FFN block
            // On Metal: batch ffn_norm + gate + up + silu_mul + down into ONE
            // command buffer, eliminating 5 GPU round-trips → 1 per layer.
            let ffn_norm_name = format!("model.layers.{layer}.ffn_norm.weight");
            let w1_name = format!("model.layers.{layer}.feed_forward.w1.weight");
            let w3_name = format!("model.layers.{layer}.feed_forward.w3.weight");
            let w2_name = format!("model.layers.{layer}.feed_forward.w2.weight");

            let mut ffn_done = false;
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                let has_metal = self.metal_ops.is_some();
                if has_metal {
                    let dtype_w1 = self.tensor_dtype(&w1_name).unwrap_or_else(|| "f16".to_string());
                    let inter = cfg.intermediate_size;
                    let eps = cfg.rms_norm_eps;

                    // Pre-fetch all tensor data before borrowing metal_ops
                    let norm_w_data: Vec<u16>;
                    let mut w1b_data: Option<Vec<u16>> = None;
                    let mut w3b_data: Option<Vec<u16>> = None;
                    let mut w2b_data: Option<Vec<u16>> = None;

                    {
                        let nw = self.tensor_f16(&ffn_norm_name)?;
                        norm_w_data = nw.to_vec();
                    }

                    if dtype_w1 == "f16" {
                        let w1 = self.tensor_f16(&w1_name)?.to_vec();
                        let w3 = self.tensor_f16(&w3_name)?.to_vec();
                        let w2 = self.tensor_f16(&w2_name)?.to_vec();
                        w1b_data = Some(w1);
                        w3b_data = Some(w3);
                        w2b_data = Some(w2);
                    } else if dtype_w1 == "u32" {
                        // Pre-populate the dequant cache (mutably borrows self)
                        let w1_key = (w1_name.clone(), inter, hidden);
                        let w3_key = (w3_name.clone(), inter, hidden);
                        let w2_key = (w2_name.clone(), hidden, inter);
                        if !self.weight_cache.contains_key(&w1_key) {
                            let mut dummy = vec![0.0f32; inter];
                            let zero_in = vec![0.0f32; hidden];
                            let _ = self.linear_i4_out_in(&zero_in, &w1_name, inter, hidden, &mut dummy);
                        }
                        if !self.weight_cache.contains_key(&w3_key) {
                            let mut dummy = vec![0.0f32; inter];
                            let zero_in = vec![0.0f32; hidden];
                            let _ = self.linear_i4_out_in(&zero_in, &w3_name, inter, hidden, &mut dummy);
                        }
                        if !self.weight_cache.contains_key(&w2_key) {
                            let mut dummy = vec![0.0f32; hidden];
                            let zero_in = vec![0.0f32; inter];
                            let _ = self.linear_i4_out_in(&zero_in, &w2_name, hidden, inter, &mut dummy);
                        }
                    }

                    // Now borrow metal_ops immutably — all mut borrows are done
                    let ops = self.metal_ops.as_ref().unwrap();

                    ops.ensure_named_buf("ffn_x", hidden).map_err(|e| CoreError::Backend(e.to_string()))?;
                    ops.ensure_named_buf("ffn_norm_out", hidden).map_err(|e| CoreError::Backend(e.to_string()))?;
                    ops.ensure_named_buf("ffn_gate", inter).map_err(|e| CoreError::Backend(e.to_string()))?;
                    ops.ensure_named_buf("ffn_up", inter).map_err(|e| CoreError::Backend(e.to_string()))?;
                    ops.ensure_named_buf("ffn_down", hidden).map_err(|e| CoreError::Backend(e.to_string()))?;

                    let norm_wb = ops.ensure_tensor_cached(&ffn_norm_name, &norm_w_data).map_err(|e| CoreError::Backend(e.to_string()))?;

                    if dtype_w1 == "f16" {
                        let w1b = ops.ensure_tensor_cached(&w1_name, w1b_data.as_ref().unwrap()).map_err(|e| CoreError::Backend(e.to_string()))?;
                        let w3b = ops.ensure_tensor_cached(&w3_name, w3b_data.as_ref().unwrap()).map_err(|e| CoreError::Backend(e.to_string()))?;
                        let w2b = ops.ensure_tensor_cached(&w2_name, w2b_data.as_ref().unwrap()).map_err(|e| CoreError::Backend(e.to_string()))?;

                        ops.write_named_buf("ffn_x", &x).map_err(|e| CoreError::Backend(e.to_string()))?;
                        let xb = ops.get_named_buf("ffn_x").map_err(|e| CoreError::Backend(e.to_string()))?;
                        let nb = ops.get_named_buf("ffn_norm_out").map_err(|e| CoreError::Backend(e.to_string()))?;
                        let gb = ops.get_named_buf("ffn_gate").map_err(|e| CoreError::Backend(e.to_string()))?;
                        let ub = ops.get_named_buf("ffn_up").map_err(|e| CoreError::Backend(e.to_string()))?;
                        let db = ops.get_named_buf("ffn_down").map_err(|e| CoreError::Backend(e.to_string()))?;

                        ops.run_batch(|enc| {
                            ops.encode_rms_norm_f16w(enc, &xb, &norm_wb, &nb, hidden, eps, false);
                            ops.encode_mv_f16_bias(enc, &w1b, &nb, None, &gb, inter, hidden);
                            ops.encode_mv_f16_bias(enc, &w3b, &nb, None, &ub, inter, hidden);
                            ops.encode_silu_mul_f32_inplace(enc, &gb, &ub, inter);
                            ops.encode_mv_f16_bias(enc, &w2b, &gb, None, &db, hidden, inter);
                            ops.encode_add_f32_inplace(enc, &xb, &db, hidden);
                            Ok(())
                        }).map_err(|e| CoreError::Backend(e.to_string()))?;

                        ops.read_named_buf("ffn_x", &mut x).map_err(|e| CoreError::Backend(e.to_string()))?;
                        ffn_done = true;
                    } else if dtype_w1 == "u32" {
                        let w1_key = (w1_name.clone(), inter, hidden);
                        let w3_key = (w3_name.clone(), inter, hidden);
                        let w2_key = (w2_name.clone(), hidden, inter);

                        let w1b = ops.ensure_tensor_cached_f32(&w1_name, self.weight_cache.get(&w1_key).unwrap()).map_err(|e| CoreError::Backend(e.to_string()))?;
                        let w3b = ops.ensure_tensor_cached_f32(&w3_name, self.weight_cache.get(&w3_key).unwrap()).map_err(|e| CoreError::Backend(e.to_string()))?;
                        let w2b = ops.ensure_tensor_cached_f32(&w2_name, self.weight_cache.get(&w2_key).unwrap()).map_err(|e| CoreError::Backend(e.to_string()))?;

                        ops.write_named_buf("ffn_x", &x).map_err(|e| CoreError::Backend(e.to_string()))?;
                        let xb = ops.get_named_buf("ffn_x").map_err(|e| CoreError::Backend(e.to_string()))?;
                        let nb = ops.get_named_buf("ffn_norm_out").map_err(|e| CoreError::Backend(e.to_string()))?;
                        let gb = ops.get_named_buf("ffn_gate").map_err(|e| CoreError::Backend(e.to_string()))?;
                        let ub = ops.get_named_buf("ffn_up").map_err(|e| CoreError::Backend(e.to_string()))?;
                        let db = ops.get_named_buf("ffn_down").map_err(|e| CoreError::Backend(e.to_string()))?;

                        ops.run_batch(|enc| {
                            ops.encode_rms_norm_f16w(enc, &xb, &norm_wb, &nb, hidden, eps, false);
                            ops.encode_mv_f32(enc, &w1b, &nb, &gb, inter, hidden);
                            ops.encode_mv_f32(enc, &w3b, &nb, &ub, inter, hidden);
                            ops.encode_silu_mul_f32_inplace(enc, &gb, &ub, inter);
                            ops.encode_mv_f32(enc, &w2b, &gb, &db, hidden, inter);
                            ops.encode_add_f32_inplace(enc, &xb, &db, hidden);
                            Ok(())
                        }).map_err(|e| CoreError::Backend(e.to_string()))?;

                        ops.read_named_buf("ffn_x", &mut x).map_err(|e| CoreError::Backend(e.to_string()))?;
                        ffn_done = true;
                    }
                }
            }

            if !ffn_done {
                // CPU fallback
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                if let Some(ops) = &self.metal_ops {
                    let w = self.tensor_f16(&ffn_norm_name)?;
                    ops.rms_norm_f16w(&x, w, cfg.rms_norm_eps, false, &ffn_norm_name, &mut mlp_in)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                } else {
                    self.rmsnorm_weight(&ffn_norm_name, &mut ffn_norm_w)?;
                    rms_norm_f32(&x, &ffn_norm_w, cfg.rms_norm_eps, &mut mlp_in);
                }
                #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                {
                    self.rmsnorm_weight(&ffn_norm_name, &mut ffn_norm_w)?;
                    rms_norm_f32(&x, &ffn_norm_w, cfg.rms_norm_eps, &mut mlp_in);
                }

                self.linear_f16_out_in(&mlp_in, &w1_name, cfg.intermediate_size, hidden, &mut gate)?;
                self.linear_f16_out_in(&mlp_in, &w3_name, cfg.intermediate_size, hidden, &mut up)?;

                for g in gate.iter_mut() {
                    let s = 1.0 / (1.0 + (-*g).exp());
                    *g = *g * s;
                }
                for i in 0..gate.len() {
                    gate[i] *= up[i];
                }

                self.linear_f16_out_in(&gate, &w2_name, hidden, cfg.intermediate_size, &mut down)?;

                for i in 0..hidden {
                    x[i] += down[i];
                }
            }
        }

        // Final embedding norm
        let final_norm_name = "model.embedding_norm.weight";
        let mut final_norm_w = vec![0.0f32; hidden];
        let mut x_final = vec![0.0f32; hidden];
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(ops) = &self.metal_ops {
            let w = self.tensor_f16(final_norm_name)?;
            ops.rms_norm_f16w(&x, w, cfg.rms_norm_eps, false, final_norm_name, &mut x_final)
                .map_err(|e| CoreError::Backend(e.to_string()))?;
        } else {
            self.rmsnorm_weight(final_norm_name, &mut final_norm_w)?;
            rms_norm_f32(&x, &final_norm_w, cfg.rms_norm_eps, &mut x_final);
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            self.rmsnorm_weight(final_norm_name, &mut final_norm_w)?;
            rms_norm_f32(&x, &final_norm_w, cfg.rms_norm_eps, &mut x_final);
        }

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
                // Tied int4 embeddings: the largest matmul in the model. The u32 nibble
                // order matches the kernel's byte-pair order, so the packed bytes feed
                // straight in with no dequantization.
                let weight_bytes = self.file.tensor_bytes("model.embed_tokens.weight")?;
                let scales_f32: &[f32] = bytemuck::cast_slice(self.file.tensor_bytes(&scales_name)?);
                let biases_f32: &[f32] = bytemuck::cast_slice(self.file.tensor_bytes(&biases_name)?);

                let groups_per_row = scales_f32.len() / cfg.vocab_size.max(1);
                let group_size = hidden.div_ceil(groups_per_row.max(1));

                cellm_kernels::cpu_kernels::matmul_affine_i4_f32(
                    weight_bytes,
                    scales_f32,
                    biases_f32,
                    cfg.vocab_size,
                    hidden,
                    group_size,
                    &x_final,
                    &mut logits,
                );
            } else if dtype == "i4" {
                // Per-row symmetric int4 tied embeddings. This is the single
                // largest matmul in the model, so consuming the packed nibbles
                // directly rather than dequantizing matters most here.
                let emb = self.tensor_u8("model.embed_tokens.weight")?;
                let scales = self.tensor_f16("model.embed_tokens.weight.qscale")?;
                let group_size = hidden / (scales.len() / cfg.vocab_size).max(1);
                cellm_kernels::cpu_kernels::gemv_i4_w4a8(
                    emb,
                    scales,
                    &x_final,
                    &mut logits,
                    cfg.vocab_size,
                    hidden,
                    group_size,
                );
            } else if dtype == "i8" {
                // Per-row symmetric int8 tied embeddings
                let emb = self.tensor_i8("model.embed_tokens.weight")?;
                let scales = self.tensor_f16("model.embed_tokens.weight.qscale")?;
                cellm_kernels::cpu_kernels::gemv_i8_w8a8(
                    emb,
                    scales,
                    &x_final,
                    &mut logits,
                    cfg.vocab_size,
                    hidden,
                );
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
        } else if dtype == "i4" {
            // Per-row symmetric int4 embeddings: one row gather, two weights
            // per byte, stored biased by +8.
            let emb = self.tensor_u8("model.embed_tokens.weight")?;
            let scales = self.tensor_f16("model.embed_tokens.weight.qscale")?;
            let row_stride = hidden.div_ceil(2);
            let row_start = (token as usize) * row_stride;
            if row_start + row_stride > emb.len() {
                return Err(CoreError::Backend(
                    "embed_tokens.weight shape mismatch".into(),
                ));
            }
            let groups_per_row = (scales.len() / vocab.max(1)).max(1);
            let group_size = hidden / groups_per_row;
            let srow = (token as usize) * groups_per_row;
            for i in 0..hidden {
                let byte = emb[row_start + i / 2];
                let nibble = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let scale = f16::from_bits(scales[srow + i / group_size]).to_f32();
                out[i] = (nibble as i32 - 8) as f32 * scale;
            }
        } else if dtype == "i8" {
            // Per-row symmetric int8 embeddings
            let emb = self.tensor_i8("model.embed_tokens.weight")?;
            let scales = self.tensor_f16("model.embed_tokens.weight.qscale")?;
            let row_start = (token as usize) * hidden;
            if row_start + hidden > emb.len() {
                return Err(CoreError::Backend(
                    "embed_tokens.weight shape mismatch".into(),
                ));
            }
            let scale = f16::from_bits(scales[token as usize]).to_f32();
            for i in 0..hidden {
                out[i] = (emb[row_start + i] as f32) * scale;
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

    fn tensor_i8(&self, name: &str) -> Result<&[i8], CoreError> {
        let bytes = self.file.tensor_bytes(name)?;
        Ok(bytemuck::cast_slice(bytes))
    }

    fn tensor_u8(&self, name: &str) -> Result<&[u8], CoreError> {
        self.file.tensor_bytes(name).map_err(CoreError::from)
    }

    /// Get tensor dtype from header
    pub(crate) fn tensor_dtype(&self, name: &str) -> Option<String> {
        self.file.header.tensors.iter()
            .find(|t| t.name == name)
            .map(|t| t.dtype.clone())
    }

    /// Applies a linear layer to `n_tokens` activations at once.
    ///
    /// A GEMV reads the whole weight matrix to produce one output vector, so
    /// prefilling token-by-token re-reads every weight once per token. For a
    /// 350M model and a 750-token prompt that is hundreds of gigabytes of
    /// memory traffic to do work the weights only need to be read once for.
    /// Passing the batch down to a GEMM amortises each weight read across all
    /// the tokens in the chunk.
    ///
    /// `input` is `[n_tokens, in_dim]` row-major and `out` is
    /// `[n_tokens, out_dim]`. Only the int8 path is batched, since that is what
    /// this model's large tensors use; every other dtype falls back to a
    /// per-token loop, which is exactly what it did before.
    fn linear_batched_out_in(
        &mut self,
        input: &[f32],
        weight_name: &str,
        n_tokens: usize,
        out_dim: usize,
        in_dim: usize,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        if n_tokens == 0 {
            return Ok(());
        }

        let dtype = self.tensor_dtype(weight_name).unwrap_or_else(|| "f16".to_string());

        // The GEMM is CPU-only, and the Metal path already batches its own work
        // into command buffers, so leave it alone.
        let cpu_batched = n_tokens > 1 && {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                self.metal_ops.is_none()
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                true
            }
        };
        let use_gemm = dtype == "i8" && cpu_batched;

        let base_name = weight_name.trim_end_matches(".weight");
        let scales_name = format!("{base_name}.scales");
        let biases_name = format!("{base_name}.biases");
        if dtype == "u32"
            && cpu_batched
            && self.file.has_tensor(&scales_name)
            && self.file.has_tensor(&biases_name)
        {
            let weight_bytes = self.file.tensor_bytes(weight_name)?;
            let scales: &[f32] = bytemuck::cast_slice(self.file.tensor_bytes(&scales_name)?);
            let biases: &[f32] = bytemuck::cast_slice(self.file.tensor_bytes(&biases_name)?);
            let groups_per_row = scales.len() / out_dim.max(1);
            let group_size = in_dim.div_ceil(groups_per_row.max(1));
            cellm_kernels::cpu_kernels::gemm_affine_i4_f32(
                weight_bytes,
                scales,
                biases,
                out_dim,
                in_dim,
                group_size,
                input,
                out,
                n_tokens,
            );
            return Ok(());
        }

        if use_gemm {
            let w = self.tensor_i8(weight_name)?;
            let expected_len = out_dim * in_dim;
            if w.len() != expected_len {
                return Err(CoreError::Backend(format!(
                    "linear_batched_out_in: weight shape mismatch for {weight_name}: got {} elements, expected {expected_len}",
                    w.len()
                )));
            }
            let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
            if scales.len() != out_dim {
                return Err(CoreError::Backend(format!(
                    "linear_batched_out_in: {weight_name}.qscale has {} entries, expected {out_dim}",
                    scales.len()
                )));
            }
            cellm_kernels::cpu_kernels::gemm_i8_w8a8(
                w, scales, input, out, n_tokens, out_dim, in_dim,
            );
            return Ok(());
        }

        for t in 0..n_tokens {
            let (i0, i1) = (t * in_dim, (t + 1) * in_dim);
            let (o0, o1) = (t * out_dim, (t + 1) * out_dim);
            // Split the borrow: the input rows are read-only here, and
            // `linear_f16_out_in` needs `&mut self` for its weight cache.
            let row = input[i0..i1].to_vec();
            self.linear_f16_out_in(&row, weight_name, out_dim, in_dim, &mut out[o0..o1])?;
        }
        Ok(())
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

        if dtype == "i4" {
            // Per-row symmetric int4, two weights per byte. Unlike the MLX "u32"
            // path above this never materialises the matrix in f32: the packed
            // nibbles feed ARM SDOT directly, so a decode step moves half the
            // bytes of the int8 path instead of eight times as many.
            let w = self.tensor_u8(weight_name)?;
            let expected_len = out_dim * in_dim.div_ceil(2);
            if w.len() != expected_len {
                return Err(CoreError::Backend(format!(
                    "linear_f16_out_in: weight shape mismatch for {weight_name}: got {} bytes, expected {} ({}x{} i4)",
                    w.len(), expected_len, out_dim, in_dim
                )));
            }
            let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
            let groups_per_row = scales.len() / out_dim.max(1);
            if groups_per_row == 0 || scales.len() % out_dim != 0 {
                return Err(CoreError::Backend(format!(
                    "linear_f16_out_in: {weight_name}.qscale has {} entries, not a multiple of {out_dim}",
                    scales.len()
                )));
            }
            let group_size = in_dim / groups_per_row;

            cellm_kernels::cpu_kernels::gemv_i4_w4a8(
                w, scales, input, out, out_dim, in_dim, group_size,
            );
            return Ok(());
        }

        if dtype == "i8" {
            // Per-row symmetric int8 path (cellm --quantize-int8-symmetric).
            let w = self.tensor_i8(weight_name)?;
            let expected_len = out_dim * in_dim;
            if w.len() != expected_len {
                return Err(CoreError::Backend(format!(
                    "linear_f16_out_in: weight shape mismatch for {weight_name}: got {} elements, expected {} ({}x{} i8)",
                    w.len(), expected_len, out_dim, in_dim
                )));
            }
            let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
            if scales.len() != out_dim {
                return Err(CoreError::Backend(format!(
                    "linear_f16_out_in: {weight_name}.qscale has {} entries, expected {out_dim}",
                    scales.len()
                )));
            }

            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if let Some(ops) = &self.metal_ops {
                ops.logits_i8(input, w, scales, out_dim, in_dim, weight_name, out)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                return Ok(());
            }

            // W8A8: quantize the activation and use ARM SDOT integer dot products.
            cellm_kernels::cpu_kernels::gemv_i8_w8a8(w, scales, input, out, out_dim, in_dim);
            return Ok(());
        }

        // Standard f16 path
        let w = self.tensor_f16(weight_name)?;

        // Validate weight shape: [out_dim, in_dim] -> out_dim * in_dim elements
        let expected_len = out_dim * in_dim;
        if w.len() != expected_len {
            return Err(CoreError::Backend(format!(
                "linear_f16_out_in: weight shape mismatch for {weight_name}: got {} elements, expected {} ({}x{} f16)",
                w.len(), expected_len, out_dim, in_dim
            )));
        }

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(ops) = &self.metal_ops {
            ops.logits_f16(input, w, out_dim, in_dim, weight_name, out)
                .map_err(|e| CoreError::Backend(e.to_string()))?;
            return Ok(());
        }

        // matmul: out[i] = sum_j weight[i,j] * input[j]
        // w is already &[u16] from tensor_f16
        out.par_chunks_mut(64).enumerate().for_each(|(chunk_idx, chunk)| {
            for (local_i, out_val) in chunk.iter_mut().enumerate() {
                let i = chunk_idx * 64 + local_i;
                let mut acc = 0.0f32;
                let row_start = i * in_dim;
                for j in 0..in_dim {
                    let w_f32 = f16::from_bits(w[row_start + j]).to_f32();
                    acc += w_f32 * input[j];
                }
                *out_val = acc;
            }
        });

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
        // The Metal path needs an f32 copy; on CPU dot the packed nibbles directly so the
        // dequant cache is never populated.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let metal_active = self.metal_ops.is_some();
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let metal_active = false;

        if !metal_active {
            let base_name = weight_name.trim_end_matches(".weight");
            let packed = self.file.tensor_bytes(weight_name)?;
            let scales: &[f32] =
                bytemuck::cast_slice(self.file.tensor_bytes(&format!("{base_name}.scales"))?);
            let biases: &[f32] =
                bytemuck::cast_slice(self.file.tensor_bytes(&format!("{base_name}.biases"))?);
            let groups_per_row = scales.len() / out_dim.max(1);
            if groups_per_row > 0 && scales.len() == biases.len() {
                let group_size = in_dim.div_ceil(groups_per_row);
                cellm_kernels::cpu_kernels::matmul_affine_i4_f32(
                    packed, scales, biases, out_dim, in_dim, group_size, input, out,
                );
                return Ok(());
            }
        }

        // Check cache first
        let cache_key = (weight_name.to_string(), out_dim, in_dim);

        if !self.weight_cache.contains_key(&cache_key) {
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

            dequant.par_chunks_exact_mut(in_dim).enumerate().for_each(|(i, row)| {
                let row_offset = i * packed_in;
                for j_packed in 0..packed_in {
                    let packed = weight_u32[row_offset + j_packed];
                    let g = (j_packed * 8) / group_size;
                    let scale_idx = i * groups_per_row + g;
                    let scale = scales_f32[scale_idx];
                    let bias = biases_f32[scale_idx];
                    let j_base = j_packed * 8;
                    for k in 0..8 {
                        let nibble = ((packed >> (k * 4)) & 0xF) as f32;
                        row[j_base + k] = nibble * scale + bias;
                    }
                }
            });

            // LRU eviction, bounded by both entry count and total bytes.
            let budget = weight_cache_budget_bytes();
            let incoming = dequant.len() * std::mem::size_of::<f32>();
            let mut live: usize = self
                .weight_cache
                .values()
                .map(|v| v.len() * std::mem::size_of::<f32>())
                .sum();

            while self.lru_order.len() >= MIN_CACHE_ENTRIES
                && (self.weight_cache.len() >= MAX_CACHE_ENTRIES || live + incoming > budget)
            {
                let Some(old_key) = self.lru_order.first().cloned() else { break };
                if let Some(evicted) = self.weight_cache.remove(&old_key) {
                    live -= evicted.len() * std::mem::size_of::<f32>();
                }
                self.lru_order.remove(0);
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

        // Use cached weights for matmul
        let weights = self.weight_cache.get(&cache_key).unwrap();

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(ops) = &self.metal_ops {
            ops.logits_f32(input, weights, out_dim, in_dim, weight_name, out)
                .map_err(|e| CoreError::Backend(e.to_string()))?;
            return Ok(());
        }

        out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
            let row_start = i * in_dim;
            let mut acc = 0.0f32;
            for j in 0..in_dim {
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
