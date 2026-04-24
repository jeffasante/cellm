//! Gemma3 fused Metal graph — single command-buffer decode.
//!
//! Handles the 4-norm branch-residual structure, per-head Q/K norms,
//! GELU activation, rotate-half RoPE, and sliding-window attention.

#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::collections::HashMap;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use cellm_core::CoreError;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use cellm_kernels::metal::MetalOps;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use cellm_cache::{KVCache, PageTable};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

/// Per-layer attention geometry stored in the graph state.
#[derive(Clone, Debug)]
pub struct GemmaGraphLayerSpec {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub ffn_dim: usize,
}

/// Fused Metal inference graph for Gemma3 text models.
///
/// Encodes one full forward pass (all layers) into a single `CommandBuffer`
/// with a single `wait_until_completed` sync point, matching the Llama graph approach.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct GemmaGraphState {
    pub device: Device,
    pub queue: CommandQueue,
    pub ops: MetalOps,

    /// Model weights pre-uploaded to Metal shared memory.
    pub weights: HashMap<String, Buffer>,
    /// Dtype string per tensor name ("f16" | "i8").
    pub tensor_dtypes: HashMap<String, String>,

    // Persistent activation buffers.
    pub x_buf: Buffer,
    pub x_norm_buf: Buffer,
    pub q_buf: Buffer,
    pub k_buf: Buffer,
    pub v_buf: Buffer,
    pub attn_out_buf: Buffer,
    pub mlp_in_buf: Buffer,
    pub gate_buf: Buffer,
    pub up_buf: Buffer,
    pub down_buf: Buffer,
    pub logits_buf: Buffer,

    /// KV-bases index buffer, fixed-stride layout [num_layers][bases_stride] u32.
    pub bases_buf: Option<Buffer>,
    pub bases_stride: usize,       // allocated tokens-per-layer (power of two, >= seq)
    pub bases_last_session: u64,   // session that last wrote to this buffer
    pub bases_last_seq: usize,     // token_count when buffer was last updated

    // Static model properties.
    pub layer_specs: Vec<GemmaGraphLayerSpec>,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub is_gemma3: bool,
    pub rope_theta_sliding: f32,
    pub rmsnorm_weight_is_offset: bool,
    pub tensor_prefix: String,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl GemmaGraphState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        vocab_size: usize,
        max_q_dim: usize,
        max_kv_dim: usize,
        max_ffn_dim: usize,
        layer_specs: Vec<GemmaGraphLayerSpec>,
        sliding_window: usize,
        sliding_window_pattern: usize,
        is_gemma3: bool,
        rope_theta_sliding: f32,
        rmsnorm_weight_is_offset: bool,
        tensor_prefix: String,
        ops: MetalOps,
    ) -> Result<Self, CoreError> {
        let device = ops.device.clone();
        let queue = ops.queue.clone();

        let mk = |n: usize| {
            device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
        };

        // Pre-compute all buffers before the struct literal so the `mk` borrow of
        // `device` ends before `device` is moved into `Self`.
        let x_buf        = mk(hidden_size);
        let x_norm_buf   = mk(hidden_size);
        let q_buf        = mk(max_q_dim);
        let k_buf        = mk(max_kv_dim);
        let v_buf        = mk(max_kv_dim);
        let attn_out_buf = mk(max_q_dim);
        let mlp_in_buf   = mk(hidden_size);
        let gate_buf     = mk(max_ffn_dim);
        let up_buf       = mk(max_ffn_dim);
        let down_buf     = mk(hidden_size);
        let logits_buf   = mk(vocab_size);
        let _ = mk; // release borrow of `device` before it is moved

        Ok(Self {
            device,
            queue,
            ops,
            weights: HashMap::new(),
            tensor_dtypes: HashMap::new(),
            x_buf,
            x_norm_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            mlp_in_buf,
            gate_buf,
            up_buf,
            down_buf,
            logits_buf,
            bases_buf: None,
            bases_stride: 0,
            bases_last_session: 0,
            bases_last_seq: 0,
            layer_specs,
            sliding_window,
            sliding_window_pattern,
            is_gemma3,
            rope_theta_sliding,
            rmsnorm_weight_is_offset,
            tensor_prefix,
        })
    }

    pub fn preload_weight(&mut self, name: String, bytes: &[u8], dtype: String) {
        let buf = self.device.new_buffer_with_data(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.tensor_dtypes.insert(name.clone(), dtype);
        self.weights.insert(name, buf);
    }

    fn get_w(&self, name: &str) -> &Buffer {
        self.weights.get(name)
            .or_else(|| if name.starts_with("model.") { self.weights.get(&name[6..]) } else { None })
            .or_else(|| self.weights.get(&format!("model.{name}")))
            .unwrap_or_else(|| panic!("GemmaGraph: weight not found: {name}"))
    }

    fn try_get_w(&self, name: &str) -> Option<&Buffer> {
        self.weights.get(name)
            .or_else(|| if name.starts_with("model.") { self.weights.get(&name[6..]) } else { None })
            .or_else(|| self.weights.get(&format!("model.{name}")))
    }

    /// Whether layer `l` uses full (global) attention.
    /// Gemma3 pattern: every `sliding_window_pattern`-th layer (1-indexed) is full-attention.
    fn is_full_attention_layer(&self, layer: usize) -> bool {
        if self.is_gemma3 {
            self.sliding_window_pattern != 0 && (layer + 1) % self.sliding_window_pattern == 0
        } else {
            true
        }
    }

    /// Single-token fused decode.
    ///
    /// `token_off` and `block_id` are for the current token position (already appended
    /// to the page table by the caller).  `pos` is the 0-based sequence position.
    ///
    /// Returns `Some(raw_logits)` when `return_logits=true`, `None` otherwise.
    pub fn step_fused(
        &mut self,
        x_in: &[f32],
        cfg: &crate::ModelConfig,
        kv_cache: &mut KVCache,
        page_table: &PageTable,
        pos: usize,
        token_off: usize,
        block_id: u32,
        return_logits: bool,
    ) -> Result<Option<Vec<f32>>, CoreError> {
        let hidden = cfg.hidden_size;
        let num_layers = cfg.num_hidden_layers;
        let prefix = self.tensor_prefix.clone();
        let add_one = self.rmsnorm_weight_is_offset;

        // 1. Upload embedding.
        unsafe {
            std::ptr::copy_nonoverlapping(
                x_in.as_ptr(),
                self.x_buf.contents() as *mut f32,
                hidden,
            );
        }

        // 2. Build KV-bases index for attention kernels.
        //
        // Fixed-stride layout: [num_layers][bases_stride] u32.
        // Layer l's slot starts at byte offset l * bases_stride * 4.
        //
        // For full-attention layers:  bases_off = l * stride * 4,  count = seq
        // For sliding-window layers:  bases_off = (l * stride + start_tpos) * 4,
        //                             count = min(seq, sliding_window)
        //
        // Incremental: on consecutive steps for the same session we only write
        // num_layers new entries (one per layer for token `pos`). O(num_layers).
        // Full rebuild uses the decomposed formula:
        //   base(t, l) = base(t, 0) + l * elems_per_block_per_layer
        // reducing page-table lookups from O(seq × L) to O(seq).
        let seq        = page_table.token_count();
        let session_id = page_table.session_id();

        let need_resize = self.bases_stride < seq;
        if need_resize || self.bases_buf.is_none() {
            let new_stride = seq.next_power_of_two().max(64);
            self.bases_stride = new_stride;
            self.bases_buf = Some(self.device.new_buffer(
                (new_stride * num_layers * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            self.bases_last_seq = 0;
        }

        let bases_ptr       = self.bases_buf.as_ref().unwrap().contents() as *mut u32;
        let layout          = kv_cache.layout();
        let elems_per_layer = layout.elems_per_block_per_layer() as u32;
        let stride          = self.bases_stride;

        let incremental = !need_resize
            && self.bases_last_session == session_id
            && self.bases_last_seq + 1 == seq;

        if incremental {
            // Write only the new token's entry per layer. O(num_layers).
            let b = page_table.block_for_token(pos)
                .map_err(|e| CoreError::Backend(format!("gemma graph bases: {e}")))?;
            let o = page_table.offset_in_block(pos)
                .map_err(|e| CoreError::Backend(format!("gemma graph bases: {e}")))?;
            let base_l0 = layout.token_base_elem(b, 0, o)? as u32;
            for l in 0..num_layers {
                unsafe {
                    *bases_ptr.add(l * stride + pos) = base_l0 + l as u32 * elems_per_layer;
                }
            }
        } else {
            // Full rebuild: O(seq) lookups, O(seq × L) additions.
            let mut token_bases_l0: Vec<u32> = Vec::with_capacity(seq);
            for t in 0..seq {
                let b = page_table.block_for_token(t)
                    .map_err(|e| CoreError::Backend(format!("gemma graph bases l0: {e}")))?;
                let o = page_table.offset_in_block(t)
                    .map_err(|e| CoreError::Backend(format!("gemma graph bases l0: {e}")))?;
                token_bases_l0.push(layout.token_base_elem(b, 0, o)? as u32);
            }
            for l in 0..num_layers {
                let layer_add = l as u32 * elems_per_layer;
                let row_ptr = unsafe { bases_ptr.add(l * stride) };
                for t in 0..seq {
                    unsafe { *row_ptr.add(t) = token_bases_l0[t] + layer_add; }
                }
            }
        }

        self.bases_last_session = session_id;
        self.bases_last_seq = seq;

        let bases_ref = self.bases_buf.as_ref().unwrap();

        // 3. Initiate Command Buffer.
        let cb = self.queue.new_command_buffer();

        // 4. One encoder per layer — ensures memory ordering between layers.
        for layer in 0..num_layers {
            let enc = cb.new_compute_command_encoder();
            let spec = self.layer_specs[layer].clone();
            let n_heads     = spec.n_heads;
            let n_kv_heads  = spec.n_kv_heads;
            let head_dim    = spec.head_dim;
            let kv_head_dim = spec.kv_head_dim;
            let q_dim       = spec.q_dim;
            let kv_dim      = spec.kv_dim;
            let ffn_dim     = spec.ffn_dim;
            let is_full     = self.is_full_attention_layer(layer);
            let start_tpos  = if is_full { 0 } else { seq.saturating_sub(self.sliding_window) };
            let attn_count  = (seq - start_tpos) as u32;
            let rope_theta  = if is_full { cfg.rope_theta } else { self.rope_theta_sliding };

            let pref = format!("{prefix}model.layers.{layer}");
            // byte offset into bases_buf for this layer's slot (sliding-window aware)
            let bases_off   = ((layer * stride + start_tpos) * 4) as u64;

            // 4a. Input RMSNorm: x → x_norm
            {
                let w = self.get_w(&format!("{pref}.input_layernorm.weight"));
                self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w, &self.x_norm_buf, hidden, cfg.rms_norm_eps, add_one);
            }

            // 4b. QKV projections.
            {
                let wq = self.get_w(&format!("{pref}.self_attn.q_proj.weight"));
                let wk = self.get_w(&format!("{pref}.self_attn.k_proj.weight"));
                let wv = self.get_w(&format!("{pref}.self_attn.v_proj.weight"));
                let sq = self.try_get_w(&format!("{pref}.self_attn.q_proj.weight.qscale"))
                    .or_else(|| self.try_get_w(&format!("{pref}.self_attn.q_proj.qscale")));
                let sk = self.try_get_w(&format!("{pref}.self_attn.k_proj.weight.qscale"))
                    .or_else(|| self.try_get_w(&format!("{pref}.self_attn.k_proj.qscale")));
                let sv = self.try_get_w(&format!("{pref}.self_attn.v_proj.weight.qscale"))
                    .or_else(|| self.try_get_w(&format!("{pref}.self_attn.v_proj.qscale")));
                if let (Some(sq), Some(sk), Some(sv)) = (sq, sk, sv) {
                    self.ops.encode_qkv_i8(enc, wq, sq, wk, sk, wv, sv,
                        &self.x_norm_buf, &self.q_buf, &self.k_buf, &self.v_buf,
                        q_dim, kv_dim, hidden);
                } else {
                    self.ops.encode_qkv_f16(enc, wq, wk, wv,
                        &self.x_norm_buf, &self.q_buf, &self.k_buf, &self.v_buf,
                        q_dim, kv_dim, hidden);
                }
            }

            // 4c. Per-head Q RMSNorm (in-place, weight shared across all Q heads).
            {
                let w_qn = self.get_w(&format!("{pref}.self_attn.q_norm.weight"));
                for h in 0..n_heads {
                    let byte_off = (h * head_dim * 4) as u64;
                    self.ops.encode_rms_norm_f16w_at(
                        enc,
                        &self.q_buf, byte_off,
                        w_qn, 0,
                        &self.q_buf, byte_off,
                        head_dim, cfg.rms_norm_eps, add_one,
                    );
                }
            }

            // 4d. Per-head K RMSNorm (in-place).
            {
                let w_kn = self.get_w(&format!("{pref}.self_attn.k_norm.weight"));
                for h in 0..n_kv_heads {
                    let byte_off = (h * kv_head_dim * 4) as u64;
                    self.ops.encode_rms_norm_f16w_at(
                        enc,
                        &self.k_buf, byte_off,
                        w_kn, 0,
                        &self.k_buf, byte_off,
                        kv_head_dim, cfg.rms_norm_eps, add_one,
                    );
                }
            }

            // 4e. rotate_half RoPE on Q and K.
            self.ops.encode_rope_half_f32(enc, &self.q_buf, n_heads,    head_dim,    head_dim,    pos, rope_theta);
            self.ops.encode_rope_half_f32(enc, &self.k_buf, n_kv_heads, kv_head_dim, kv_head_dim, pos, rope_theta);

            // 4f. Write current token K/V to paged cache.
            {
                let kv_store = kv_cache.storage().as_any()
                    .downcast_ref::<cellm_cache::kvcache::MetalKvStorage>()
                    .expect("GemmaGraphState requires MetalKvStorage backend");
                let target_base = kv_cache.layout().token_base_elem(block_id, layer, token_off)?;
                kv_store.encode_write_token_f32(enc, target_base, &self.k_buf, &self.v_buf, kv_dim);

                // 4g. GQA attention (sliding-window aware via attn_count and bases_off).
                kv_store.encode_attention(
                    enc,
                    bases_ref,
                    bases_off,
                    &self.q_buf,
                    &self.attn_out_buf,
                    attn_count,
                    n_heads    as u32,
                    n_kv_heads as u32,
                    head_dim   as u32,
                    None, // scale = 1/sqrt(head_dim) by default
                    None, // soft_cap: Gemma3 does not use it
                );
            }

            // 4h. o_proj: attn_out → mlp_in_buf  (temporary storage for the attn branch).
            {
                let wo = self.get_w(&format!("{pref}.self_attn.o_proj.weight"));
                let so = self.try_get_w(&format!("{pref}.self_attn.o_proj.weight.qscale"))
                    .or_else(|| self.try_get_w(&format!("{pref}.self_attn.o_proj.qscale")));
                if let Some(so) = so {
                    self.ops.encode_mv_i8(enc, wo, so, &self.attn_out_buf, &self.mlp_in_buf, hidden, q_dim);
                } else {
                    self.ops.encode_mv_f16(enc, wo, &self.attn_out_buf, &self.mlp_in_buf, hidden, q_dim);
                }
            }

            // 4i. Post-attention RMSNorm on attn branch: mlp_in → x_norm.
            {
                let w_pa = self.get_w(&format!("{pref}.post_attention_layernorm.weight"));
                self.ops.encode_rms_norm_f16w(enc, &self.mlp_in_buf, w_pa, &self.x_norm_buf, hidden, cfg.rms_norm_eps, add_one);
            }

            // 4j. Residual add: x += post_attn_normed.
            self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.x_norm_buf, hidden);

            // 4k. Pre-FFN RMSNorm: x → mlp_in  (mlp_in now holds the FFN input).
            {
                let w_pf = self.get_w(&format!("{pref}.pre_feedforward_layernorm.weight"));
                self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w_pf, &self.mlp_in_buf, hidden, cfg.rms_norm_eps, add_one);
            }

            // 4l. Gate + Up projections.
            {
                let wg = self.get_w(&format!("{pref}.mlp.gate_proj.weight"));
                let wu = self.get_w(&format!("{pref}.mlp.up_proj.weight"));
                let sg = self.try_get_w(&format!("{pref}.mlp.gate_proj.weight.qscale"))
                    .or_else(|| self.try_get_w(&format!("{pref}.mlp.gate_proj.qscale")));
                let su = self.try_get_w(&format!("{pref}.mlp.up_proj.weight.qscale"))
                    .or_else(|| self.try_get_w(&format!("{pref}.mlp.up_proj.qscale")));
                if let (Some(sg), Some(su)) = (sg, su) {
                    self.ops.encode_mv_i8(enc, wg, sg, &self.mlp_in_buf, &self.gate_buf, ffn_dim, hidden);
                    self.ops.encode_mv_i8(enc, wu, su, &self.mlp_in_buf, &self.up_buf,   ffn_dim, hidden);
                } else {
                    self.ops.encode_mv_f16(enc, wg, &self.mlp_in_buf, &self.gate_buf, ffn_dim, hidden);
                    self.ops.encode_mv_f16(enc, wu, &self.mlp_in_buf, &self.up_buf,   ffn_dim, hidden);
                }
            }

            // 4m. GELU activation: gate = gelu(gate) * up.
            self.ops.encode_gelu_tanh_mul_f32_inplace(enc, &self.gate_buf, &self.up_buf, ffn_dim);

            // 4n. Down projection.
            {
                let wd = self.get_w(&format!("{pref}.mlp.down_proj.weight"));
                let sd = self.try_get_w(&format!("{pref}.mlp.down_proj.weight.qscale"))
                    .or_else(|| self.try_get_w(&format!("{pref}.mlp.down_proj.qscale")));
                if let Some(sd) = sd {
                    self.ops.encode_mv_i8(enc, wd, sd, &self.gate_buf, &self.down_buf, hidden, ffn_dim);
                } else {
                    self.ops.encode_mv_f16(enc, wd, &self.gate_buf, &self.down_buf, hidden, ffn_dim);
                }
            }

            // 4o. Post-FFN RMSNorm on MLP branch: down → x_norm.
            {
                let w_pff = self.get_w(&format!("{pref}.post_feedforward_layernorm.weight"));
                self.ops.encode_rms_norm_f16w(enc, &self.down_buf, w_pff, &self.x_norm_buf, hidden, cfg.rms_norm_eps, add_one);
            }

            // 4p. Residual add: x += post_ffn_normed.
            self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.x_norm_buf, hidden);
            enc.end_encoding();
        }

        // 5. Final norm + logits in a separate encoder.
        let enc = cb.new_compute_command_encoder();
        {
            let w_norm = self.get_w(&format!("{prefix}model.norm.weight"));
            self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w_norm, &self.x_norm_buf, hidden, cfg.rms_norm_eps, add_one);
        }

        // 6. LM-head logits (optional).
        if return_logits {
            let lm_head_key  = format!("{prefix}lm_head.weight");
            let embed_key    = format!("{prefix}model.embed_tokens.weight");
            let lm_name = if self.weights.contains_key(&lm_head_key)         { lm_head_key }
                     else if self.weights.contains_key("lm_head.weight")     { "lm_head.weight".to_string() }
                     else if self.weights.contains_key(&embed_key)           { embed_key }
                     else                                                     { "model.embed_tokens.weight".to_string() };
            let wl = self.weights.get(&lm_name)
                .expect("GemmaGraph: lm_head / embed_tokens weight not found");
            let sl = self.try_get_w(&format!("{lm_name}.qscale"));
            if let Some(sl) = sl {
                self.ops.encode_mv_i8(enc, wl, sl, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, hidden);
            } else {
                self.ops.encode_mv_f16(enc, wl, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, hidden);
            }
        }

        // 7. GPU sync point.
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        if !return_logits {
            return Ok(None);
        }

        // 8. Read back logits and check for divergence.
        let mut logits = vec![0.0f32; cfg.vocab_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.logits_buf.contents() as *const f32,
                logits.as_mut_ptr(),
                cfg.vocab_size,
            );
        }
        let nan_count = logits.iter().filter(|v| v.is_nan()).count();
        let inf_count = logits.iter().filter(|v| v.is_infinite()).count();
        if nan_count > 0 || inf_count > 0 {
            return Err(CoreError::Backend(format!(
                "GemmaGraphState: divergence at pos {pos} (NaN={nan_count} Inf={inf_count})"
            )));
        }

        Ok(Some(logits))
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct GemmaGraphState;
