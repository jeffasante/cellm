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

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct LlamaGraphState {
    pub device: Device,
    pub queue: CommandQueue,
    
    // Core Ops
    pub ops: MetalOps,

    // Pre-allocated Model Weights
    pub weights: HashMap<String, Buffer>,

    // Persistent Activations (Zero-Copy sequence dispatch)
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
    pub bases_buf: Option<Buffer>,
    pub bases_capacity: usize,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl LlamaGraphState {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        vocab_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, CoreError> {
        let ops = MetalOps::create().map_err(|e| CoreError::Backend(format!("{:?}", e)))?;
        let head_dim = hidden_size / num_heads;
        
        let device = ops.device.clone();
        let queue = ops.queue.clone();

        let make_buf = |len_f32: usize| {
            device.new_buffer(
                (len_f32 * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        // Activations are mapped specifically for f32 operations between kernel execution passes
        Ok(Self {
            device: device.clone(),
            queue,
            ops,
            weights: HashMap::new(),
            x_buf: make_buf(hidden_size),
            x_norm_buf: make_buf(hidden_size),
            q_buf: make_buf(hidden_size),
            k_buf: make_buf(num_kv_heads * head_dim),
            v_buf: make_buf(num_kv_heads * head_dim),
            attn_out_buf: make_buf(hidden_size),
            mlp_in_buf: make_buf(hidden_size),
            gate_buf: make_buf(intermediate_size),
            up_buf: make_buf(intermediate_size),
            down_buf: make_buf(hidden_size),
            logits_buf: make_buf(vocab_size),
            bases_buf: None,
            bases_capacity: 0,
        })
    }

    pub fn preload_weight_f16(&mut self, name: String, bytes: &[u8]) {
        let buf = self.device.new_buffer_with_data(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // eprintln!("LlamaGraphState: preloaded {} ({} bytes)", name, bytes.len());
        self.weights.insert(name, buf);
    }

    fn get_weight(&self, name: &str, alt: Option<&str>) -> &Buffer {
        match self.weights.get(name).or_else(|| alt.and_then(|a| self.weights.get(a))) {
            Some(w) => w,
            None => {
                panic!("Weight not found: {} (alt: {:?}). Available: {:?}", name, alt, self.weights.keys().take(10).collect::<Vec<_>>());
            }
        }
    }

    pub fn step_fused(
        &mut self,
        x_in: &[f32],
        cfg: &crate::ModelConfig,
        prefix: &str,
        kv_cache: &mut KVCache,
        page_table: &PageTable,
        pos: usize,
        token_off: usize,
        block_id: u32,
    ) -> Result<Vec<f32>, CoreError> {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden / n_heads;
        let kv_dim = n_kv_heads * head_dim;

        // 1. Setup execution state, send X embeddings to Metal
        unsafe {
            let ptr = self.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(x_in.as_ptr(), ptr, hidden);
        }

        // 2. Map the token bases once on CPU for the entire graph to use
        let cv = kv_cache.view_mut();
        let seq = page_table.token_count();
        let num_layers = cfg.num_hidden_layers;
        let total_bases = seq * num_layers;
        
        if self.bases_capacity < total_bases {
            let new_cap = total_bases.max(4096 * num_layers);
            self.bases_capacity = new_cap;
            self.bases_buf = Some(self.device.new_buffer(
                (new_cap * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        }

        let mut bases_u32 = Vec::with_capacity(total_bases);
        for layer in 0..num_layers {
            for tpos in 0..seq {
                let b = page_table.block_for_token(tpos).map_err(|e| {
                    CoreError::Backend(format!("graph layout failed: {e}"))
                })?;
                let o = page_table.offset_in_block(tpos).map_err(|e| {
                    CoreError::Backend(format!("graph offset failed: {e}"))
                })?;
                bases_u32.push(cv.layout.token_base_elem(b, layer, o)? as u32);
            }
        }

        if let Some(bb) = &self.bases_buf {
            unsafe {
                let ptr = bb.contents() as *mut std::ffi::c_void as *mut u32;
                std::ptr::copy_nonoverlapping(bases_u32.as_ptr(), ptr, total_bases);
            }
        }
        
        let bases_ref = self.bases_buf.as_ref().unwrap();

        // 3. Initiate Command Buffer!
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();

        // 4. Fully fuse all execution layers into a single pipeline pass loop
        for layer in 0..num_layers {
            let w_in_norm = self.get_weight(&format!("model.layers.{layer}.input_layernorm.weight"), None);
            
            // DEBUG: check x_buf before layer 0
            if layer == 0 {
                // We'll copy x_buf back after the commit to check it
            }

            self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w_in_norm, &self.x_norm_buf, hidden, cfg.rms_norm_eps, false);

            let w_q = self.get_weight(&format!("model.layers.{layer}.self_attn.q_proj.weight"), None);
            let w_k = self.get_weight(&format!("model.layers.{layer}.self_attn.k_proj.weight"), None);
            let w_v = self.get_weight(&format!("model.layers.{layer}.self_attn.v_proj.weight"), None);
            // Fused QKV projection: one kernel launch instead of three.
            self.ops.encode_qkv_f16(
                enc,
                w_q,
                w_k,
                w_v,
                &self.x_norm_buf,
                &self.q_buf,
                &self.k_buf,
                &self.v_buf,
                hidden,
                kv_dim,
                hidden,
            );

            // MISSING ROPE!!!
            self.ops.encode_rope_adj_f32(enc, &self.q_buf, n_heads, head_dim, pos, cfg.rope_theta);
            self.ops.encode_rope_adj_f32(enc, &self.k_buf, n_kv_heads, head_dim, pos, cfg.rope_theta);

            let kv_store = cv.storage.as_any().downcast_ref::<cellm_cache::kvcache::MetalKvStorage>()
                .expect("fused graph requires MetalKvStorage");
            
            let target_base = cv.layout.token_base_elem(block_id, layer, token_off)?;
            kv_store.encode_write_token_f32(enc, target_base, &self.k_buf, &self.v_buf, kv_dim);
            
            let bases_offset = (layer * seq * 4) as u64;
            kv_store.encode_attention(enc, bases_ref, bases_offset, &self.q_buf, &self.attn_out_buf, seq as u32, n_heads as u32, n_kv_heads as u32, head_dim as u32, None);

            let w_o = self.get_weight(&format!("model.layers.{layer}.self_attn.o_proj.weight"), None);
            self.ops.encode_mv_f16(enc, w_o, &self.attn_out_buf, &self.mlp_in_buf, hidden, hidden);

            self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.mlp_in_buf, hidden);

            let w_post = self.get_weight(&format!("model.layers.{layer}.post_attention_layernorm.weight"), None);
            self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w_post, &self.x_norm_buf, hidden, cfg.rms_norm_eps, false);

            let w_gate = self.get_weight(&format!("model.layers.{layer}.mlp.gate_proj.weight"), None);
            let w_up = self.get_weight(&format!("model.layers.{layer}.mlp.up_proj.weight"), None);
            self.ops.encode_mv_f16(enc, w_gate, &self.x_norm_buf, &self.gate_buf, cfg.intermediate_size, hidden);
            self.ops.encode_mv_f16(enc, w_up, &self.x_norm_buf, &self.up_buf, cfg.intermediate_size, hidden);

            self.ops.encode_silu_mul_f32_inplace(enc, &self.gate_buf, &self.up_buf, cfg.intermediate_size);

            let w_down = self.get_weight(&format!("model.layers.{layer}.mlp.down_proj.weight"), None);
            self.ops.encode_mv_f16(enc, w_down, &self.gate_buf, &self.down_buf, hidden, cfg.intermediate_size);

            self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.down_buf, hidden);
        }

        let w_norm = self.get_weight(&format!("{prefix}model.norm.weight"), Some("model.norm.weight"));
        self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w_norm, &self.x_norm_buf, hidden, cfg.rms_norm_eps, false);

        let lm_head_name = format!("{prefix}lm_head.weight");
        let embed_name = format!("{prefix}model.embed_tokens.weight");
        let w_lm = self.weights.get(&lm_head_name)
            .or_else(|| self.weights.get("lm_head.weight"))
            .or_else(|| self.weights.get(&embed_name))
            .or_else(|| self.weights.get("model.embed_tokens.weight"))
            .expect("lm_head or embed_tokens not found in graph weights");

        self.ops.encode_mv_f16(enc, w_lm, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, hidden);

        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed(); // 0 syncs across 24 layers until absolutely complete!

        let mut logits = vec![0.0f32; cfg.vocab_size];
        let mut x_check = vec![0.0f32; hidden];
        let mut x_norm_check = vec![0.0f32; hidden];

        unsafe {
            let ptr = self.logits_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, logits.as_mut_ptr(), cfg.vocab_size);
            
            let x_ptr = self.x_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(x_ptr, x_check.as_mut_ptr(), hidden);

            let xn_ptr = self.x_norm_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(xn_ptr, x_norm_check.as_mut_ptr(), hidden);
        }

        let mut nan_count = 0;
        let mut inf_count = 0;
        for &v in &logits {
            if v.is_nan() { nan_count += 1; }
            else if v.is_infinite() { inf_count += 1; }
        }
        
        if nan_count > 0 || inf_count > 0 {
            eprintln!("LlamaGraphState: step_fused divergence at pos {}", pos);
            eprintln!("  NaNs: {}, Infs: {}", nan_count, inf_count);
            eprintln!("  x_buf[0..4]: {:?}", &x_check[..4]);
            eprintln!("  x_norm_buf[0..4]: {:?}", &x_norm_check[..4]);
            
            let x_nan = x_check.iter().any(|v| v.is_nan());
            let xn_nan = x_norm_check.iter().any(|v| v.is_nan());
            eprintln!("  x has NaN: {}, x_norm has NaN: {}", x_nan, xn_nan);
        }

        Ok(logits)
    }
}
