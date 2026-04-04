use std::path::Path;

use bytemuck::cast_slice;
use cellm_cache::{KVCache, PageTable};
use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::rms_norm_f32;
use cellm_kernels::MetalOps;
use cellm_kernels::metal::MetalMatmul;
use cellm_kernels::MetalKernels;
use half::f16;

use crate::{CellmFile, ModelConfig};
use serde_json::Value;

const GEMMA_METAL_LINEAR_MAX_ELEMS: usize = 262_144;

pub struct GemmaRunner {
    file: CellmFile,
    cfg: ModelConfig,
    max_layers: usize,
    eos_token_id: Option<u32>,
    tensor_prefix: String,
    head_dim: usize,
    kv_head_dim: usize,
    rmsnorm_weight_is_offset: bool,
    is_gemma3_text: bool,
    sliding_window: usize,
    sliding_window_pattern: usize,
    rope_theta_sliding: f32,
    linear_backend: GemmaLinearBackend,
    metal_strict: bool,
    /// Present when a Metal backend is active; drives rms_norm / rope / logits on GPU.
    metal_ops: Option<MetalOps>,
}

enum GemmaLinearBackend {
    Cpu,
    Metal { ctx: MetalMatmul },
}

impl GemmaRunner {
    pub fn load(path: &Path) -> Result<Self, CoreError> {
        let file = CellmFile::load(path)?;
        let h = file.header.clone();

        let cfg = ModelConfig {
            vocab_size: h.vocab_size,
            hidden_size: h.hidden_dim,
            num_hidden_layers: h.num_layers,
            num_attention_heads: h.num_heads,
            num_key_value_heads: h.num_kv_heads,
            intermediate_size: h.intermediate_size,
            rms_norm_eps: h.rms_norm_eps,
            rope_theta: h.rope_theta,
        };

        let tensor_prefix = detect_gemma_prefix(&file)?;
        let (head_dim, kv_head_dim) = infer_gemma_head_dims(
            &file,
            &tensor_prefix,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        )?;
        let rmsnorm_weight_is_offset = true;
        let source_model_type = match &h.source_text_config {
            Some(Value::Object(obj)) => match obj.get("model_type") {
                Some(Value::String(mt)) if !mt.is_empty() => Some(mt.as_str()),
                _ => None,
            },
            _ => None,
        };
        let is_gemma3_text = h.model_type.starts_with("gemma3")
            || source_model_type.is_some_and(|mt| mt.starts_with("gemma3"));
        // Gemma3 uses mixed attention: sliding layers use local RoPE base 10k, while
        // periodic full-attention layers use the global RoPE theta from config.
        let sliding_window = if is_gemma3_text { 512 } else { usize::MAX };
        let sliding_window_pattern = if is_gemma3_text { 6 } else { usize::MAX };
        let rope_theta_sliding = if is_gemma3_text { 10_000.0 } else { cfg.rope_theta };

        Ok(Self {
            file,
            cfg: cfg.clone(),
            max_layers: cfg.num_hidden_layers,
            eos_token_id: h.eos_token_id,
            tensor_prefix,
            head_dim,
            kv_head_dim,
            rmsnorm_weight_is_offset,
            is_gemma3_text,
            sliding_window,
            sliding_window_pattern,
            rope_theta_sliding,
            linear_backend: GemmaLinearBackend::Cpu,
            metal_strict: false,
            metal_ops: None,
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
    
    pub fn is_gemma3_text(&self) -> bool {
        self.is_gemma3_text
    }

    pub fn enable_metal_linear_backend(&mut self) -> bool {
        match (MetalKernels::create_matmul(), MetalOps::create()) {
            (Ok(ctx), Ok(ops)) => {
                self.linear_backend = GemmaLinearBackend::Metal { ctx };
                self.metal_ops = Some(ops);
                self.metal_strict = false;
                true
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("gemma: failed to enable metal linear backend: {e}");
                self.linear_backend = GemmaLinearBackend::Cpu;
                self.metal_ops = None;
                self.metal_strict = false;
                false
            }
        }
    }

    pub fn enable_metal_full_backend(&mut self) -> bool {
        match (MetalKernels::create_matmul(), MetalOps::create()) {
            (Ok(ctx), Ok(ops)) => {
                self.linear_backend = GemmaLinearBackend::Metal { ctx };
                self.metal_ops = Some(ops);
                self.metal_strict = true;
                true
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("gemma: failed to enable full metal backend: {e}");
                self.linear_backend = GemmaLinearBackend::Cpu;
                self.metal_ops = None;
                self.metal_strict = false;
                false
            }
        }
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

    pub fn step_topk_from_hidden(
        &mut self,
        x0: &[f32],
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>, CoreError> {
        let cfg = self.cfg.clone();
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = self.head_dim;
        let kv_head_dim = self.kv_head_dim;
        if n_heads * head_dim == 0 || n_kv_heads * kv_head_dim == 0 {
            return Err(CoreError::Backend(
                "gemma: invalid attention head geometry".into(),
            ));
        }
        if head_dim != kv_head_dim {
            return Err(CoreError::Backend(format!(
                "gemma: head_dim mismatch q={} kv={} (mixed dims not supported yet)",
                head_dim, kv_head_dim
            )));
        }
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * kv_head_dim;

        // Ensure pagetable covers this token position.
        if pos == page_table.token_count() {
            page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                CoreError::Backend(format!("gemma step: page_table append_token failed: {e}"))
            })?;
        } else if pos > page_table.token_count() {
            return Err(CoreError::Backend(format!(
                "gemma step: non-contiguous pos {pos} (token_count={})",
                page_table.token_count()
            )));
        }

        let block_id = page_table.block_for_token(pos).map_err(|e| {
            CoreError::Backend(format!("gemma step: page_table block_for_token failed: {e}"))
        })?;
        let token_off = page_table.offset_in_block(pos).map_err(|e| {
            CoreError::Backend(format!("gemma step: page_table offset_in_block failed: {e}"))
        })?;

        if x0.len() != hidden {
            return Err(CoreError::Backend(format!(
                "gemma step_from_hidden: hidden len mismatch {} != {}",
                x0.len(),
                hidden
            )));
        }
        let mut x = x0.to_vec();

        // Per-layer scratch.
        let mut attn_norm_w = vec![0.0f32; hidden];
        let mut x_norm = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; q_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut attn_proj = vec![0.0f32; hidden];

        let mut q_norm_w = vec![0.0f32; head_dim];
        let mut k_norm_w = vec![0.0f32; kv_head_dim];
        let mut post_attn_norm_w = vec![0.0f32; hidden];
        let mut pre_ffn_norm_w = vec![0.0f32; hidden];
        let mut post_ffn_norm_w = vec![0.0f32; hidden];
        let mut mlp_in = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; cfg.intermediate_size];
        let mut up = vec![0.0f32; cfg.intermediate_size];
        let mut ffn_out = vec![0.0f32; cfg.intermediate_size];
        let mut down = vec![0.0f32; hidden];

        let mut gather_bases: Vec<usize> = Vec::new();

        for layer in 0..self.max_layers {
            let is_full_attention_layer = if self.is_gemma3_text {
                // Gemma3 pattern: 5 sliding layers, then 1 full-attention layer.
                self.sliding_window_pattern != 0 && (layer + 1) % self.sliding_window_pattern == 0
            } else {
                true
            };
            let layer_rope_theta = if self.is_gemma3_text && !is_full_attention_layer {
                self.rope_theta_sliding
            } else {
                cfg.rope_theta
            };

            // ── Attention input norm (always CPU for now, Metal for rope/logits) ──
            {
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.input_layernorm.weight"),
                    &mut attn_norm_w,
                )?;
                rms_norm_f32(&x, &attn_norm_w, cfg.rms_norm_eps, &mut x_norm);
            }

            let use_metal_norm = self.metal_ops.is_some();
            // Gemma uses rotate_half-style RoPE; keep RoPE on CPU for Gemma3 to
            // preserve correctness until Metal path matches this convention.
            let use_metal_rope = self.metal_ops.is_some();

            // QKV projections (HF weights are [out, in]).
            self.linear_f16_out_in(
                &x_norm,
                &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                q_dim,
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

            // ── Gemma per-head Q/K RMSNorm before RoPE ────────────────────────
            if use_metal_norm {
                let qw = self.tensor_f16(
                    &format!("model.layers.{layer}.self_attn.q_norm.weight"))?.to_vec();
                let kw = self.tensor_f16(
                    &format!("model.layers.{layer}.self_attn.k_norm.weight"))?.to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                let ops = self.metal_ops.as_mut().unwrap();
                for hidx in 0..n_heads {
                    let seg = &mut q[hidx * head_dim..(hidx + 1) * head_dim];
                    let inp = seg.to_vec();
                    ops.rms_norm_f16w(&inp, &qw, cfg.rms_norm_eps, add_one, seg)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                for hidx in 0..n_kv_heads {
                    let seg = &mut k[hidx * kv_head_dim..(hidx + 1) * kv_head_dim];
                    let inp = seg.to_vec();
                    ops.rms_norm_f16w(&inp, &kw, cfg.rms_norm_eps, add_one, seg)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
            } else {
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.self_attn.q_norm.weight"),
                    &mut q_norm_w,
                )?;
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.self_attn.k_norm.weight"),
                    &mut k_norm_w,
                )?;
                for hidx in 0..n_heads {
                    let start = hidx * head_dim;
                    let end = start + head_dim;
                    self.rms_norm_inplace_segment(&mut q[start..end], &q_norm_w, cfg.rms_norm_eps);
                }
                for hidx in 0..n_kv_heads {
                    let start = hidx * kv_head_dim;
                    let end = start + kv_head_dim;
                    self.rms_norm_inplace_segment(&mut k[start..end], &k_norm_w, cfg.rms_norm_eps);
                }
            }

            // ── RoPE ──────────────
            if use_metal_rope {
                let ops = self.metal_ops.as_mut().unwrap();
                ops.rope_adj_f32(&mut q, n_heads, head_dim, pos, layer_rope_theta)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                ops.rope_adj_f32(&mut k, n_kv_heads, kv_head_dim, pos, layer_rope_theta)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                rope_inplace_rotate_half_f32(&mut q, n_heads, head_dim, pos, layer_rope_theta);
                rope_inplace_rotate_half_f32(&mut k, n_kv_heads, kv_head_dim, pos, layer_rope_theta);
            }

            // Write new token K/V into paged cache.
            {
                let mut cv = kv_cache.view_mut();
                cv.write_token(block_id, layer, token_off, &k, &v)?;
            }

            // Gather historical K/V and run attention for this token.
            let seq = page_table.token_count();
            let cr = kv_cache.view();
            gather_bases.clear();
            let start_tpos = if self.is_gemma3_text && !is_full_attention_layer {
                seq.saturating_sub(self.sliding_window)
            } else {
                0
            };
            let gather_len = seq.saturating_sub(start_tpos);
            gather_bases.reserve(gather_len);
            for tpos in start_tpos..seq {
                let b = page_table.block_for_token(tpos).map_err(|e| {
                    CoreError::Backend(format!("gemma: block_for_token failed: {e}"))
                })?;
                let o = page_table.offset_in_block(tpos).map_err(|e| {
                    CoreError::Backend(format!("gemma: offset_in_block failed: {e}"))
                })?;
                gather_bases.push(cr.layout.token_base_elem(b, layer, o)?);
            }
            cr.attention_single_token_gqa_from_bases(
                &gather_bases,
                &q,
                n_heads,
                n_kv_heads,
                head_dim,
                &mut attn_out,
            )?;

            // o_proj: hidden <- hidden
            self.linear_f16_out_in(
                &attn_out,
                &format!("model.layers.{layer}.self_attn.o_proj.weight"),
                hidden,
                q_dim,
                &mut attn_proj,
            )?;

            let x_residual = x.clone();

            // ── Post-attention norm on attn branch, then residual add
            if use_metal_norm {
                let w = self.tensor_f16(
                    &format!("model.layers.{layer}.post_attention_layernorm.weight"))?.to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                let mut x_out = vec![0.0f32; hidden];
                self.metal_ops.as_mut().unwrap()
                    .rms_norm_f16w(&attn_proj, &w, cfg.rms_norm_eps, add_one, &mut x_out)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                for i in 0..hidden { x[i] = x_out[i] + x_residual[i]; }
            } else {
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                    &mut post_attn_norm_w,
                )?;
                rms_norm_f32(&attn_proj, &post_attn_norm_w, cfg.rms_norm_eps, &mut x_norm);
                for i in 0..hidden { x[i] = x_norm[i] + x_residual[i]; }
            }

            // ── Pre-FFN norm ───────
            if use_metal_norm {
                let w = self.tensor_f16(
                    &format!("model.layers.{layer}.pre_feedforward_layernorm.weight"))?.to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                self.metal_ops.as_mut().unwrap()
                    .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut mlp_in)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.pre_feedforward_layernorm.weight"),
                    &mut pre_ffn_norm_w,
                )?;
                rms_norm_f32(&x, &pre_ffn_norm_w, cfg.rms_norm_eps, &mut mlp_in);
            }

            self.linear_f16_out_in(
                &mlp_in,
                &format!("model.layers.{layer}.mlp.gate_proj.weight"),
                cfg.intermediate_size,
                hidden,
                &mut gate,
            )?;
            self.linear_f16_out_in(
                &mlp_in,
                &format!("model.layers.{layer}.mlp.up_proj.weight"),
                cfg.intermediate_size,
                hidden,
                &mut up,
            )?;
            for i in 0..gate.len() {
                ffn_out[i] = gelu_tanh_f32(gate[i]) * up[i];
            }
            self.linear_f16_out_in(
                &ffn_out,
                &format!("model.layers.{layer}.mlp.down_proj.weight"),
                hidden,
                cfg.intermediate_size,
                &mut down,
            )?;
            let x_residual = x.clone();

            // ── Post-FFN norm on MLP branch, then residual add ──────
            if use_metal_norm {
                let wffn = self.tensor_f16(
                    &format!("model.layers.{layer}.post_feedforward_layernorm.weight"))?.to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                let mut x_out = vec![0.0f32; hidden];
                self.metal_ops.as_mut().unwrap()
                    .rms_norm_f16w(&down, &wffn, cfg.rms_norm_eps, add_one, &mut x_out)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                for i in 0..hidden { x[i] = x_out[i] + x_residual[i]; }
            } else {
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.post_feedforward_layernorm.weight"),
                    &mut post_ffn_norm_w,
                )?;
                rms_norm_f32(&down, &post_ffn_norm_w, cfg.rms_norm_eps, &mut x_norm);
                for i in 0..hidden { x[i] = x_norm[i] + x_residual[i]; }
            }
        }

        // Final norm.
        let use_metal_norm = self.metal_ops.is_some();
        let mut x_final = vec![0.0f32; hidden];
        if use_metal_norm {
            let w = self.tensor_f16("model.norm.weight")?.to_vec();
            let add_one = self.rmsnorm_weight_is_offset;
            self.metal_ops.as_mut().unwrap()
                .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_final)
                .map_err(|e| CoreError::Backend(e.to_string()))?;
        } else {
            let mut norm_w = vec![0.0f32; hidden];
            self.rmsnorm_weight("model.norm.weight", &mut norm_w)?;
            rms_norm_f32(&x, &norm_w, cfg.rms_norm_eps, &mut x_final);
        }

        // Logits via tied embeddings / lm_head.
        let vocab = cfg.vocab_size;
        let k = top_k.max(1).min(vocab);

        let use_metal_logits = self.metal_ops.is_some();

        let lm_head_name = self.resolve_name("lm_head.weight");
        let maybe_lm_head = lm_head_name
            .as_ref()
            .ok()
            .and_then(|n| self.file.tensor_index(n).map(|_| n.clone()));
        let lm_src_name = maybe_lm_head
            .as_deref()
            .unwrap_or("model.embed_tokens.weight");
        let lm_src_resolved = self.resolve_name(lm_src_name)?;
        let lm_meta = self
            .tensor_meta_by_exact_name(&lm_src_resolved)
            .ok_or_else(|| CoreError::Backend(format!("unknown tensor {}", lm_src_resolved)))?;
        let lm_dtype = lm_meta.dtype.clone();

        // Compute all vocab logits on GPU when Metal is active, otherwise CPU.
        let all_logits: Vec<f32>;
        if use_metal_logits {
            let mut buf = vec![0.0f32; vocab];
            match lm_dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16_by_exact_name(&lm_src_resolved)?.to_vec();
                    self.metal_ops.as_mut().unwrap()
                        .logits_f16(&x_final, &w, vocab, hidden, &lm_src_resolved, &mut buf)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                "i8" => {
                    let w = self.tensor_i8_by_exact_name(&lm_src_resolved)?.to_vec();
                    let s = self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?.to_vec();
                    self.metal_ops.as_mut().unwrap()
                        .logits_i8(&x_final, &w, &s, vocab, hidden, &lm_src_resolved, &mut buf)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                other => return Err(CoreError::Backend(format!(
                    "unsupported lm dtype {other} for {lm_src_resolved}"))),
            }
            all_logits = buf;
        } else {
            // CPU path.
            let lm_src_f16 = if lm_dtype == "f16" {
                Some(self.tensor_f16_by_exact_name(&lm_src_resolved)?)
            } else { None };
            let lm_src_i8 = if lm_dtype == "i8" {
                Some(self.tensor_i8_by_exact_name(&lm_src_resolved)?)
            } else { None };
            let lm_src_scales = if lm_dtype == "i8" {
                Some(self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?)
            } else { None };
            let mut buf = vec![0.0f32; vocab];
            for vid in 0..vocab {
                let mut dot = 0.0f32;
                if let Some(wf16) = lm_src_f16 {
                    let row = &wf16[vid * hidden..(vid + 1) * hidden];
                    for i in 0..hidden { dot += x_final[i] * f16::from_bits(row[i]).to_f32(); }
                } else if let (Some(wi8), Some(scales)) = (lm_src_i8, lm_src_scales) {
                    let row = &wi8[vid * hidden..(vid + 1) * hidden];
                    let scale = f16::from_bits(scales[vid]).to_f32();
                    for i in 0..hidden { dot += x_final[i] * (row[i] as f32) * scale; }
                } else {
                    return Err(CoreError::Backend(format!(
                        "unsupported lm dtype {lm_dtype} for {lm_src_resolved}")));
                }
                buf[vid] = dot;
            }
            all_logits = buf;
        }

        // Top-k selection from flat logits (always on CPU – tiny work).
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut min_idx = 0usize;
        let mut min_val = f32::INFINITY;
        for (vid, &dot) in all_logits.iter().enumerate() {
            if top.len() < k {
                top.push((vid as u32, dot));
                if dot < min_val { min_val = dot; min_idx = top.len() - 1; }
            } else if dot > min_val {
                top[min_idx] = (vid as u32, dot);
                min_val = top[0].1; min_idx = 0;
                for (i, &(_, s)) in top.iter().enumerate().skip(1) {
                    if s < min_val { min_val = s; min_idx = i; }
                }
            }
        }
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(top)
    }

    fn tensor_f16(&self, name: &str) -> Result<&[u16], CoreError> {
        let resolved = self.resolve_name(name)?;
        self.tensor_f16_by_exact_name(&resolved)
    }

    fn tensor_f16_by_exact_name(&self, resolved: &str) -> Result<&[u16], CoreError> {
        let bytes = self.file.tensor_bytes(resolved)?;
        if bytes.len() % 2 != 0 {
            return Err(CoreError::Backend(format!("tensor {resolved} nbytes not even")));
        }
        Ok(cast_slice(bytes))
    }

    fn tensor_i8_by_exact_name(&self, resolved: &str) -> Result<&[i8], CoreError> {
        let bytes = self.file.tensor_bytes(resolved)?;
        Ok(cast_slice(bytes))
    }

    fn tensor_meta_by_exact_name(&self, resolved: &str) -> Option<&crate::CellmTensorIndex> {
        self.file.tensor_index(resolved)
    }

    fn embed_token(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        let hidden = out.len();
        let embed = self.tensor_f16("model.embed_tokens.weight")?;
        let vocab = self.cfg.vocab_size;
        let t = (token as usize) % vocab;
        let row = &embed[t * hidden..(t + 1) * hidden];
        let embed_scale = (hidden as f32).sqrt();
        for i in 0..hidden {
            out[i] = f16::from_bits(row[i]).to_f32() * embed_scale;
        }
        Ok(())
    }

    fn rmsnorm_weight(&self, name: &str, out: &mut [f32]) -> Result<(), CoreError> {
        let w = self.tensor_f16(name)?;
        if w.len() != out.len() {
            return Err(CoreError::Backend(format!(
                "rmsnorm weight {name} len mismatch: {} vs {}",
                w.len(),
                out.len()
            )));
        }
        for i in 0..out.len() {
            let base = f16::from_bits(w[i]).to_f32();
            out[i] = if self.rmsnorm_weight_is_offset {
                base + 1.0
            } else {
                base
            };
        }
        Ok(())
    }

    fn linear_f16_out_in(
        &mut self,
        x: &[f32],
        weight_name: &str,
        out_dim: usize,
        in_dim: usize,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        if x.len() != in_dim || out.len() != out_dim {
            return Err(CoreError::Backend(format!(
                "linear dims mismatch for {weight_name}: x={} out={} expected in={in_dim} out={out_dim}",
                x.len(),
                out.len()
            )));
        }

        let resolved = self.resolve_name(weight_name)?;
        let meta = self
            .tensor_meta_by_exact_name(&resolved)
            .ok_or_else(|| CoreError::Backend(format!("unknown tensor {resolved}")))?;
        let shape = meta.shape.clone();
        if shape.len() != 2 {
            return Err(CoreError::Backend(format!(
                "weight {weight_name} expected 2D, got {:?}",
                shape
            )));
        }
        // HF linear weight: [out, in]
        if shape[0] != out_dim || shape[1] != in_dim {
            return Err(CoreError::Backend(format!(
                "weight {weight_name} shape mismatch: {:?} expected [{out_dim},{in_dim}]",
                shape
            )));
        }
        let dtype = meta.dtype.clone();

        // Fast path: use MetalOps matrix-vector kernels with internal GPU-side caching.
        if self.metal_ops.is_some() {
            match dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16_by_exact_name(&resolved)?;
                    let w_ptr = w.as_ptr();
                    let w_len = w.len();
                    let w = unsafe { std::slice::from_raw_parts(w_ptr, w_len) };
                    self.metal_ops
                        .as_mut()
                        .unwrap()
                        .logits_f16(x, w, out_dim, in_dim, &resolved, out)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                    return Ok(());
                }
                "i8" => {
                    let w = self.tensor_i8_by_exact_name(&resolved)?;
                    let s = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                    let w_ptr = w.as_ptr();
                    let w_len = w.len();
                    let s_ptr = s.as_ptr();
                    let s_len = s.len();
                    let w = unsafe { std::slice::from_raw_parts(w_ptr, w_len) };
                    let s = unsafe { std::slice::from_raw_parts(s_ptr, s_len) };
                    self.metal_ops
                        .as_mut()
                        .unwrap()
                        .logits_i8(x, w, s, out_dim, in_dim, &resolved, out)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                    return Ok(());
                }
                _ => {}
            }
        }

        if let GemmaLinearBackend::Metal { ctx } = &self.linear_backend {
            let max_cols = if in_dim == 0 {
                1
            } else {
                (GEMMA_METAL_LINEAR_MAX_ELEMS / in_dim).max(1)
            };
            let chunk_cols = max_cols.min(out_dim.max(1));
            let mut weight_t_chunk = vec![0.0f32; in_dim * chunk_cols];
            let mut out_chunk = vec![0.0f32; chunk_cols];
            let mut metal_ok = true;

            match dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16_by_exact_name(&resolved)?;
                    if w.len() != out_dim * in_dim {
                        return Err(CoreError::Backend(format!(
                            "weight {weight_name} len mismatch: {} expected {}",
                            w.len(),
                            out_dim * in_dim
                        )));
                    }
                    let mut row_start = 0usize;
                    while row_start < out_dim {
                        let cols_n = (out_dim - row_start).min(chunk_cols);
                        for i in 0..in_dim {
                            for c in 0..cols_n {
                                let row_idx = row_start + c;
                                weight_t_chunk[i * cols_n + c] =
                                    f16::from_bits(w[row_idx * in_dim + i]).to_f32();
                            }
                        }
                        let out_slice = &mut out_chunk[..cols_n];
                        if ctx
                            .matmul_row_major_f32(
                                x,
                                1,
                                in_dim,
                                &weight_t_chunk[..in_dim * cols_n],
                                cols_n,
                                out_slice,
                            )
                            .is_err()
                        {
                            metal_ok = false;
                            break;
                        }
                        out[row_start..row_start + cols_n].copy_from_slice(out_slice);
                        row_start += cols_n;
                    }
                }
                "i8" => {
                    let w = self.tensor_i8_by_exact_name(&resolved)?;
                    let scales = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                    if w.len() != out_dim * in_dim || scales.len() != out_dim {
                        return Err(CoreError::Backend(format!(
                            "weight {weight_name} i8/qscale len mismatch: w={} scales={} expected w={} scales={}",
                            w.len(),
                            scales.len(),
                            out_dim * in_dim,
                            out_dim
                        )));
                    }
                    let mut row_start = 0usize;
                    while row_start < out_dim {
                        let cols_n = (out_dim - row_start).min(chunk_cols);
                        for i in 0..in_dim {
                            for c in 0..cols_n {
                                let row_idx = row_start + c;
                                let scale = f16::from_bits(scales[row_idx]).to_f32();
                                weight_t_chunk[i * cols_n + c] =
                                    (w[row_idx * in_dim + i] as f32) * scale;
                            }
                        }
                        let out_slice = &mut out_chunk[..cols_n];
                        if ctx
                            .matmul_row_major_f32(
                                x,
                                1,
                                in_dim,
                                &weight_t_chunk[..in_dim * cols_n],
                                cols_n,
                                out_slice,
                            )
                            .is_err()
                        {
                            metal_ok = false;
                            break;
                        }
                        out[row_start..row_start + cols_n].copy_from_slice(out_slice);
                        row_start += cols_n;
                    }
                }
                _ => {
                    metal_ok = false;
                }
            }

            if metal_ok {
                return Ok(());
            }
            if self.metal_strict {
                return Err(CoreError::Backend(format!(
                    "gemma full-metal: linear kernel failed for {weight_name}; CPU fallback disabled"
                )));
            }
        }

        match dtype.as_str() {
            "f16" => {
                let w = self.tensor_f16_by_exact_name(&resolved)?;
                if w.len() != out_dim * in_dim {
                    return Err(CoreError::Backend(format!(
                        "weight {weight_name} len mismatch: {} expected {}",
                        w.len(),
                        out_dim * in_dim
                    )));
                }
                for j in 0..out_dim {
                    let row = &w[j * in_dim..(j + 1) * in_dim];
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += x[i] * f16::from_bits(row[i]).to_f32();
                    }
                    out[j] = acc;
                }
            }
            "i8" => {
                let w = self.tensor_i8_by_exact_name(&resolved)?;
                if w.len() != out_dim * in_dim {
                    return Err(CoreError::Backend(format!(
                        "weight {weight_name} len mismatch: {} expected {}",
                        w.len(),
                        out_dim * in_dim
                    )));
                }
                let scales = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                if scales.len() != out_dim {
                    return Err(CoreError::Backend(format!(
                        "weight {weight_name} qscale len mismatch: {} expected {}",
                        scales.len(),
                        out_dim
                    )));
                }
                for j in 0..out_dim {
                    let row = &w[j * in_dim..(j + 1) * in_dim];
                    let scale = f16::from_bits(scales[j]).to_f32();
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += x[i] * ((row[i] as f32) * scale);
                    }
                    out[j] = acc;
                }
            }
            other => {
                return Err(CoreError::Backend(format!(
                    "unsupported weight dtype for {weight_name}: {other}"
                )));
            }
        }
        Ok(())
    }

    fn resolve_name(&self, name: &str) -> Result<String, CoreError> {
        if self.file.tensor_index(name).is_some() {
            return Ok(name.to_string());
        }
        if !self.tensor_prefix.is_empty() {
            let prefixed = format!("{}{}", self.tensor_prefix, name);
            if self.file.tensor_index(&prefixed).is_some() {
                return Ok(prefixed);
            }
        }
        if let Some(suffix) = name.strip_prefix("model.") {
            let text_model = format!("model.text_model.{suffix}");
            if self.file.tensor_index(&text_model).is_some() {
                return Ok(text_model);
            }
        }
        Err(CoreError::Backend(format!("unknown tensor {name}")))
    }

    fn rms_norm_inplace_segment(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        let mut sumsq = 0.0f32;
        for &v in x.iter() {
            sumsq += v * v;
        }
        let inv_rms = 1.0f32 / (sumsq / (x.len() as f32) + eps).sqrt();
        for i in 0..x.len() {
            x[i] = x[i] * inv_rms * weight[i];
        }
    }
}

fn gelu_tanh_f32(x: f32) -> f32 {
    let k = 0.797_884_6f32;
    let c = 0.044_715f32;
    0.5f32 * x * (1.0f32 + (k * (x + c * x * x * x)).tanh())
}

fn rope_inplace_rotate_half_f32(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = pos as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = x[base + i];
            let x1 = x[base + half + i];
            x[base + i] = x0 * cos - x1 * sin;
            x[base + half + i] = x1 * cos + x0 * sin;
        }
    }
}

fn detect_gemma_prefix(file: &CellmFile) -> Result<String, CoreError> {
    for prefix in ["", "language_model."] {
        let embed = format!("{prefix}model.embed_tokens.weight");
        let norm = format!("{prefix}model.norm.weight");
        if file.tensor_index(&embed).is_some() && file.tensor_index(&norm).is_some() {
            return Ok(prefix.to_string());
        }
    }
    if file
        .tensor_index("model.text_model.embed_tokens.weight")
        .is_some()
        && file.tensor_index("model.text_model.norm.weight").is_some()
    {
        return Ok(String::new());
    }
    Err(CoreError::Backend(
        "missing required gemma tensors: model.embed_tokens.weight/model.norm.weight".into(),
    ))
}

fn infer_gemma_head_dims(
    file: &CellmFile,
    prefix: &str,
    n_heads: usize,
    n_kv_heads: usize,
) -> Result<(usize, usize), CoreError> {
    let q_name = format!("{prefix}model.layers.0.self_attn.q_proj.weight");
    let k_name = format!("{prefix}model.layers.0.self_attn.k_proj.weight");
    let q_meta = file
        .tensor_index(&q_name)
        .ok_or_else(|| CoreError::Backend(format!("gemma: missing tensor {q_name}")))?;
    let k_meta = file
        .tensor_index(&k_name)
        .ok_or_else(|| CoreError::Backend(format!("gemma: missing tensor {k_name}")))?;
    if q_meta.shape.len() != 2 || k_meta.shape.len() != 2 {
        return Err(CoreError::Backend(
            "gemma: expected 2D q_proj/k_proj weights".into(),
        ));
    }
    let q_out = q_meta.shape[0];
    let k_out = k_meta.shape[0];
    if n_heads == 0 || n_kv_heads == 0 {
        return Err(CoreError::Backend(
            "gemma: num_attention_heads/num_key_value_heads must be > 0".into(),
        ));
    }
    if q_out % n_heads != 0 {
        return Err(CoreError::Backend(format!(
            "gemma: q_proj out_dim {q_out} not divisible by n_heads={n_heads}"
        )));
    }
    if k_out % n_kv_heads != 0 {
        return Err(CoreError::Backend(format!(
            "gemma: k_proj out_dim {k_out} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    Ok((q_out / n_heads, k_out / n_kv_heads))
}
