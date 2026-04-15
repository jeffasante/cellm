use std::path::Path;

use bytemuck::cast_slice;
use cellm_cache::{KVCache, PageTable};
use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::{rms_norm_f32, rope_interleaved_inplace_f32, rope_non_interleaved_inplace_f32};
use cellm_kernels::metal::MetalMatmul;
use cellm_kernels::{MetalKernels, MetalOps};
use half::f16;

use crate::{CellmFile, ModelConfig};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::llama_graph::LlamaGraphState;

pub struct LlamaRunner {
    file: CellmFile,
    cfg: ModelConfig,
    max_layers: usize,
    eos_token_id: Option<u32>,
    tensor_prefix: String,
    linear_backend: LlamaLinearBackend,
    metal_ops: Option<MetalOps>,
    metal_strict: bool,
    use_metal_mv: bool,
    use_metal_norm: bool,
    use_metal_rope: bool,
    rope_interleaved: bool,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    graph_state: Option<LlamaGraphState>,
}

enum LlamaLinearBackend {
    Cpu,
    Metal { ctx: MetalMatmul },
}

impl LlamaRunner {
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

        let tensor_prefix = detect_llama_prefix(&file)?;

        Ok(Self {
            file,
            cfg: cfg.clone(),
            max_layers: cfg.num_hidden_layers,
            eos_token_id: h.eos_token_id,
            tensor_prefix,
            linear_backend: LlamaLinearBackend::Cpu,
            metal_ops: None,
            metal_strict: false,
            use_metal_mv: std::env::var("CELLM_LLAMA_USE_MV")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            // For small Llama stacks, per-token Metal norm/RoPE dispatch overhead can
            // outweigh math cost; keep these off by default and let users opt in.
            use_metal_norm: std::env::var("CELLM_LLAMA_USE_METAL_NORM")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            use_metal_rope: std::env::var("CELLM_LLAMA_USE_METAL_ROPE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            // Some HF Llama-family checkpoints (for example SmolLM2) use
            // non-interleaved rotary embedding layout (rotate_half).
            // Keep current interleaved default for backwards compatibility.
            rope_interleaved: std::env::var("CELLM_LLAMA_ROPE_INTERLEAVED")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(true),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            graph_state: None,
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

    pub fn enable_metal_linear_backend(&mut self) -> bool {
        match MetalKernels::create_matmul() {
            Ok(ctx) => {
                self.linear_backend = LlamaLinearBackend::Metal { ctx };
                self.metal_strict = false;
                true
            }
            Err(e) => {
                eprintln!("llama: failed to enable metal linear backend: {e}");
                self.linear_backend = LlamaLinearBackend::Cpu;
                self.metal_strict = false;
                false
            }
        }
    }

    pub fn enable_metal_full_backend(&mut self) -> bool {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            // The fused graph path can regress latency for short prompts and is still
            // experimental. Keep it opt-in so default Metal runs stay fast/stable.
            let graph_enabled = std::env::var("CELLM_LLAMA_ENABLE_GRAPH")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if graph_enabled {
                let gs_res = LlamaGraphState::new(
                    self.cfg.hidden_size,
                    self.cfg.num_attention_heads,
                    self.cfg.num_key_value_heads,
                    self.cfg.vocab_size,
                    self.cfg.intermediate_size,
                );
                match gs_res {
                    Ok(mut gs) => {
                        println!("llama: detected tensor prefix: '{}'", self.tensor_prefix);
                        println!("llama: preloading weights into metal graph...");
                        for (name, data) in self.file.all_tensors() {
                            gs.preload_weight_f16(name.to_string(), data);
                        }
                        self.graph_state = Some(gs);
                        // Use the existing MetalKernels factory
                        if let Ok(ctx) = cellm_kernels::metal::MetalKernels::create_matmul() {
                            self.linear_backend = LlamaLinearBackend::Metal { ctx };
                        }
                        if let Ok(mo) = MetalOps::create() {
                            self.metal_ops = Some(mo);
                        }
                        self.metal_strict = true;
                        return true;
                    }
                    Err(e) => {
                        eprintln!("llama: failed to enable metal graph backend: {e}");
                    }
                }
            }
        }

        let mk_res = MetalKernels::create_matmul();
        let mo_res = MetalOps::create();
        match (mk_res, mo_res) {
            (Ok(ctx), Ok(mo)) => {
                self.linear_backend = LlamaLinearBackend::Metal { ctx };
                self.metal_ops = Some(mo);
                self.metal_strict = true;
                true
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("llama: failed to enable full metal backend: {e}");
                self.linear_backend = LlamaLinearBackend::Cpu;
                self.metal_ops = None;
                self.metal_strict = false;
                false
            }
        }
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
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let mut disable_graph = false;
            if let Some(gs) = &mut self.graph_state {
                // Ensure pagetable covers this token position.
                if pos == page_table.token_count() {
                    page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                        CoreError::Backend(format!("llama step: page_table append_token failed: {e}"))
                    })?;
                }
                let block_id = page_table.block_for_token(pos).map_err(|e| {
                    CoreError::Backend(format!("llama step: page_table block_for_token failed: {e}"))
                })?;
                let token_off = page_table.offset_in_block(pos).map_err(|e| {
                    CoreError::Backend(format!("llama step: page_table offset_in_block failed: {e}"))
                })?;

                if let Ok(logits) = gs.step_fused(x0, &self.cfg, &self.tensor_prefix, kv_cache, page_table, pos, token_off, block_id as u32) {
                    let has_non_finite = logits.iter().any(|v| !v.is_finite());
                    if has_non_finite {
                        eprintln!("llama fused graph: non-finite logits detected; disabling fused graph and continuing with non-fused Metal path");
                        disable_graph = true;
                    } else {
                        return self.topk_from_logits(&logits, top_k);
                    }
                }
            }
            if disable_graph {
                self.graph_state = None;
            }
        }
        let cfg = self.cfg.clone();
        let hidden = cfg.hidden_size;
        // ... (rest of the function continues below)
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden / n_heads;
        if head_dim * n_heads != hidden {
            return Err(CoreError::Backend(
                "llama: hidden_size must be divisible by num_attention_heads".into(),
            ));
        }
        let kv_dim = n_kv_heads * head_dim;

        // Ensure pagetable covers this token position.
        if pos == page_table.token_count() {
            page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                CoreError::Backend(format!("llama step: page_table append_token failed: {e}"))
            })?;
        } else if pos > page_table.token_count() {
            return Err(CoreError::Backend(format!(
                "llama step: non-contiguous pos {pos} (token_count={})",
                page_table.token_count()
            )));
        }

        let block_id = page_table.block_for_token(pos).map_err(|e| {
            CoreError::Backend(format!("llama step: page_table block_for_token failed: {e}"))
        })?;
        let token_off = page_table.offset_in_block(pos).map_err(|e| {
            CoreError::Backend(format!("llama step: page_table offset_in_block failed: {e}"))
        })?;

        if x0.len() != hidden {
            return Err(CoreError::Backend(format!(
                "llama step_from_hidden: hidden len mismatch {} != {}",
                x0.len(),
                hidden
            )));
        }
        let mut x = x0.to_vec();

        // Per-layer scratch.
        let mut attn_norm_w = vec![0.0f32; hidden];
        let mut x_norm = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; hidden];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; hidden];
        let mut attn_proj = vec![0.0f32; hidden];

        let mut post_norm_w = vec![0.0f32; hidden];
        let mut mlp_in = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; cfg.intermediate_size];
        let mut up = vec![0.0f32; cfg.intermediate_size];
        let mut down = vec![0.0f32; hidden];

        let mut gather_bases: Vec<usize> = Vec::new();

        for layer in 0..self.max_layers {
            let use_metal_norm = self.metal_ops.is_some() && self.use_metal_norm;
            let use_metal_rope = self.metal_ops.is_some() && self.use_metal_rope && self.rope_interleaved;

            // Attention input norm.
            if use_metal_norm {
                let w = self.tensor_f16(&format!("model.layers.{layer}.input_layernorm.weight"))?;
                let w_ptr = w.as_ptr();
                let w_len = w.len();
                let w = unsafe { std::slice::from_raw_parts(w_ptr, w_len) };
                let ck = format!("llama.layer.{layer}.attn_norm");
                self.metal_ops.as_ref().unwrap()
                    .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, false, &ck, &mut x_norm)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.input_layernorm.weight"),
                    &mut attn_norm_w,
                )?;
                rms_norm_f32(&x, &attn_norm_w, cfg.rms_norm_eps, &mut x_norm);
            }

            // QKV projections (HF weights are [out, in]).
            let q_name = format!("model.layers.{layer}.self_attn.q_proj.weight");
            let k_name = format!("model.layers.{layer}.self_attn.k_proj.weight");
            let v_name = format!("model.layers.{layer}.self_attn.v_proj.weight");
            let fused_qkv = self.linear_qkv_f16_out_in(
                &x_norm,
                &q_name,
                hidden,
                &k_name,
                kv_dim,
                &v_name,
                kv_dim,
                hidden,
                &mut q,
                &mut k,
                &mut v,
            )?;
            if !fused_qkv {
                self.linear_f16_out_in(&x_norm, &q_name, hidden, hidden, &mut q)?;
                self.linear_f16_out_in(&x_norm, &k_name, kv_dim, hidden, &mut k)?;
                self.linear_f16_out_in(&x_norm, &v_name, kv_dim, hidden, &mut v)?;
            }

            if use_metal_rope {
                let ops = self.metal_ops.as_ref().unwrap();
                ops.rope_adj_f32(&mut q, n_heads, head_dim, pos, cfg.rope_theta)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                ops.rope_adj_f32(&mut k, n_kv_heads, head_dim, pos, cfg.rope_theta)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else if self.rope_interleaved {
                rope_interleaved_inplace_f32(&mut q, n_heads, head_dim, pos, cfg.rope_theta);
                rope_interleaved_inplace_f32(&mut k, n_kv_heads, head_dim, pos, cfg.rope_theta);
            } else {
                rope_non_interleaved_inplace_f32(&mut q, n_heads, head_dim, head_dim, pos, cfg.rope_theta);
                rope_non_interleaved_inplace_f32(&mut k, n_kv_heads, head_dim, head_dim, pos, cfg.rope_theta);
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
            gather_bases.reserve(seq);
            for tpos in 0..seq {
                let b = page_table.block_for_token(tpos).map_err(|e| {
                    CoreError::Backend(format!("llama: block_for_token failed: {e}"))
                })?;
                let o = page_table.offset_in_block(tpos).map_err(|e| {
                    CoreError::Backend(format!("llama: offset_in_block failed: {e}"))
                })?;
                gather_bases.push(cr.layout.token_base_elem(b, layer, o)?);
            }
            cr.attention_single_token_gqa_from_bases(
                &gather_bases,
                &q,
                n_heads,
                n_kv_heads,
                head_dim, None,
                &mut attn_out,
            )?;

            // o_proj: hidden <- hidden
            self.linear_f16_out_in(
                &attn_out,
                &format!("model.layers.{layer}.self_attn.o_proj.weight"),
                hidden,
                hidden,
                &mut attn_proj,
            )?;

            for i in 0..hidden {
                x[i] += attn_proj[i];
            }

            // Post-attn norm.
            if use_metal_norm {
                let w = self.tensor_f16(&format!("model.layers.{layer}.post_attention_layernorm.weight"))?;
                let w_ptr = w.as_ptr();
                let w_len = w.len();
                let w = unsafe { std::slice::from_raw_parts(w_ptr, w_len) };
                let ck = format!("llama.layer.{layer}.mlp_norm");
                self.metal_ops.as_ref().unwrap()
                    .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, false, &ck, &mut mlp_in)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                    &mut post_norm_w,
                )?;
                rms_norm_f32(&x, &post_norm_w, cfg.rms_norm_eps, &mut mlp_in);
            }

            // MLP: gate_proj + up_proj -> silu(gate)*up -> down_proj
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

            // silu(gate) in-place: x * sigmoid(x)
            for g in gate.iter_mut() {
                let s = 1.0 / (1.0 + (-*g).exp());
                *g = *g * s;
            }
            for i in 0..gate.len() {
                gate[i] *= up[i];
            }

            self.linear_f16_out_in(
                &gate,
                &format!("model.layers.{layer}.mlp.down_proj.weight"),
                hidden,
                cfg.intermediate_size,
                &mut down,
            )?;
            for i in 0..hidden {
                x[i] += down[i];
            }
        }

        // Final norm.
        let use_metal_norm = self.metal_ops.is_some() && self.use_metal_norm;
        let use_metal_logits = self.metal_ops.is_some();
        let mut x_final = vec![0.0f32; hidden];
        if use_metal_norm {
            let w = self.tensor_f16("model.norm.weight")?;
            let w_ptr = w.as_ptr();
            let w_len = w.len();
            let w = unsafe { std::slice::from_raw_parts(w_ptr, w_len) };
            self.metal_ops.as_ref().unwrap()
                .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, false, "llama.norm", &mut x_final)
                .map_err(|e| CoreError::Backend(e.to_string()))?;
        } else {
            let mut norm_w = vec![0.0f32; hidden];
            self.rmsnorm_weight("model.norm.weight", &mut norm_w)?;
            rms_norm_f32(&x, &norm_w, cfg.rms_norm_eps, &mut x_final);
        }

        // Logits via tied embeddings: logits[v] = dot(x_final, embed[v])
        let vocab = cfg.vocab_size;
        let k = top_k.max(1).min(vocab);
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut min_idx = 0usize;
        let mut min_val = f32::INFINITY;

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
        if use_metal_logits {
            let mut buf = vec![0.0f32; vocab];
            match lm_meta.dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16_by_exact_name(&lm_src_resolved)?;
                    let w_ptr = w.as_ptr();
                    let w_len = w.len();
                    let w = unsafe { std::slice::from_raw_parts(w_ptr, w_len) };
                    self.metal_ops.as_ref().unwrap()
                        .logits_f16(&x_final, &w, vocab, hidden, &lm_src_resolved, &mut buf)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                "i8" => {
                    let w = self.tensor_i8_by_exact_name(&lm_src_resolved)?;
                    let s = self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?;
                    let w_ptr = w.as_ptr();
                    let w_len = w.len();
                    let s_ptr = s.as_ptr();
                    let s_len = s.len();
                    let w = unsafe { std::slice::from_raw_parts(w_ptr, w_len) };
                    let s = unsafe { std::slice::from_raw_parts(s_ptr, s_len) };
                    self.metal_ops.as_ref().unwrap()
                        .logits_i8(&x_final, &w, &s, vocab, hidden, &lm_src_resolved, &mut buf)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                other => return Err(CoreError::Backend(format!("unsupported lm dtype {other} for {lm_src_resolved}"))),
            }
            sanitize_logits_non_finite(&mut buf, "llama metal logits");
            return self.topk_from_logits(&buf, top_k);
        }

        let lm_src_f16 = if lm_meta.dtype == "f16" {
            Some(self.tensor_f16_by_exact_name(&lm_src_resolved)?)
        } else {
            None
        };
        let lm_src_i8 = if lm_meta.dtype == "i8" {
            Some(self.tensor_i8_by_exact_name(&lm_src_resolved)?)
        } else {
            None
        };
        let lm_i8_scales = if lm_meta.dtype == "i8" {
            Some(self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?)
        } else {
            None
        };

        for vid in 0..vocab {
            let mut dot = if let Some(w) = lm_src_f16 {
                let row = &w[vid * hidden..(vid + 1) * hidden];
                let mut acc = 0.0f32;
                for i in 0..hidden {
                    acc += x_final[i] * f16::from_bits(row[i]).to_f32();
                }
                acc
            } else if let Some(w) = lm_src_i8 {
                let scales = lm_i8_scales.unwrap();
                let row = &w[vid * hidden..(vid + 1) * hidden];
                let scale = f16::from_bits(scales[vid]).to_f32();
                let mut acc = 0.0f32;
                for i in 0..hidden {
                    acc += x_final[i] * ((row[i] as f32) * scale);
                }
                acc
            } else {
                return Err(CoreError::Backend(format!(
                    "unsupported lm dtype {}",
                    lm_meta.dtype
                )));
            };
            if !dot.is_finite() {
                dot = f32::NEG_INFINITY;
            }

            if top.len() < k {
                top.push((vid as u32, dot));
                if dot < min_val {
                    min_val = dot;
                    min_idx = top.len() - 1;
                }
            } else if dot > min_val {
                top[min_idx] = (vid as u32, dot);
                min_val = top[0].1;
                min_idx = 0;
                for (i, &(_, s)) in top.iter().enumerate().skip(1) {
                    if s < min_val {
                        min_val = s;
                        min_idx = i;
                    }
                }
            }
        }
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
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
        for i in 0..hidden {
            out[i] = f16::from_bits(row[i]).to_f32();
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
            out[i] = f16::from_bits(w[i]).to_f32();
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

        // Optional path: use MetalOps matrix-vector kernels with internal GPU-side cache.
        // Disabled by default for Llama because small-model prefill can regress.
        let need_metal_mv = dtype == "i8" || self.use_metal_mv;
        if need_metal_mv && self.metal_ops.is_some() {
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

        if let LlamaLinearBackend::Metal { ctx } = &self.linear_backend {
            let max_cols = if in_dim == 0 {
                1
            } else {
                (262_144 / in_dim).max(1)
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
                _ => {
                    metal_ok = false;
                }
            }

            if metal_ok {
                return Ok(());
            }
            if self.metal_strict {
                return Err(CoreError::Backend(format!(
                    "llama full-metal: linear kernel failed for {weight_name}; CPU fallback disabled"
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

    fn linear_qkv_f16_out_in(
        &mut self,
        x: &[f32],
        q_weight_name: &str,
        q_out_dim: usize,
        k_weight_name: &str,
        k_out_dim: usize,
        v_weight_name: &str,
        v_out_dim: usize,
        in_dim: usize,
        q_out: &mut [f32],
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<bool, CoreError> {
        if self.metal_ops.is_none() {
            return Ok(false);
        }
        if x.len() != in_dim || q_out.len() != q_out_dim || k_out.len() != k_out_dim || v_out.len() != v_out_dim {
            return Err(CoreError::Backend(format!(
                "linear qkv dims mismatch: x={} q={} k={} v={} expected in={} q_out={} k_out={} v_out={}",
                x.len(), q_out.len(), k_out.len(), v_out.len(), in_dim, q_out_dim, k_out_dim, v_out_dim
            )));
        }

        let q_resolved = self.resolve_name(q_weight_name)?;
        let k_resolved = self.resolve_name(k_weight_name)?;
        let v_resolved = self.resolve_name(v_weight_name)?;

        let q_meta = self.tensor_meta_by_exact_name(&q_resolved).ok_or_else(|| {
            CoreError::Backend(format!("unknown tensor {q_resolved}"))
        })?;
        let k_meta = self.tensor_meta_by_exact_name(&k_resolved).ok_or_else(|| {
            CoreError::Backend(format!("unknown tensor {k_resolved}"))
        })?;
        let v_meta = self.tensor_meta_by_exact_name(&v_resolved).ok_or_else(|| {
            CoreError::Backend(format!("unknown tensor {v_resolved}"))
        })?;

        if q_meta.shape.len() != 2 || k_meta.shape.len() != 2 || v_meta.shape.len() != 2 {
            return Err(CoreError::Backend(format!(
                "qkv fused expects 2D weights, got q={:?} k={:?} v={:?}",
                q_meta.shape, k_meta.shape, v_meta.shape
            )));
        }
        if q_meta.shape != [q_out_dim, in_dim]
            || k_meta.shape != [k_out_dim, in_dim]
            || v_meta.shape != [v_out_dim, in_dim]
        {
            return Err(CoreError::Backend(format!(
                "qkv fused shape mismatch: q={:?} k={:?} v={:?} expected q=[{},{}] k=[{},{}] v=[{},{}]",
                q_meta.shape, k_meta.shape, v_meta.shape, q_out_dim, in_dim, k_out_dim, in_dim, v_out_dim, in_dim
            )));
        }
        if q_meta.dtype != "f16" || k_meta.dtype != "f16" || v_meta.dtype != "f16" {
            return Ok(false);
        }

        let wq = self.tensor_f16_by_exact_name(&q_resolved)?;
        let wk = self.tensor_f16_by_exact_name(&k_resolved)?;
        let wv = self.tensor_f16_by_exact_name(&v_resolved)?;
        let wq = unsafe { std::slice::from_raw_parts(wq.as_ptr(), wq.len()) };
        let wk = unsafe { std::slice::from_raw_parts(wk.as_ptr(), wk.len()) };
        let wv = unsafe { std::slice::from_raw_parts(wv.as_ptr(), wv.len()) };

        self.metal_ops
            .as_mut()
            .unwrap()
            .logits_qkv_f16(
                x,
                wq,
                wk,
                wv,
                q_out_dim,
                k_out_dim,
                v_out_dim,
                in_dim,
                &q_resolved,
                &k_resolved,
                &v_resolved,
                q_out,
                k_out,
                v_out,
            )
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(true)
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

    pub fn topk_from_logits(&self, logits: &[f32], top_k: usize) -> Result<Vec<(u32, f32)>, CoreError> {
        let vocab = logits.len();
        let k = top_k.max(1).min(vocab);
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut min_idx = 0usize;
        let mut min_val = f32::INFINITY;

        for vid in 0..vocab {
            let dot = logits[vid];
            if top.len() < k {
                top.push((vid as u32, dot));
                if dot < min_val {
                    min_val = dot;
                    min_idx = top.len() - 1;
                }
            } else if dot > min_val {
                top[min_idx] = (vid as u32, dot);
                min_val = top[0].1;
                min_idx = 0;
                for (i, &(_, s)) in top.iter().enumerate().skip(1) {
                    if s < min_val {
                        min_val = s;
                        min_idx = i;
                    }
                }
            }
        }
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(top)
    }
}

fn sanitize_logits_non_finite(logits: &mut [f32], tag: &str) {
    let mut found = false;
    for v in logits.iter_mut() {
        if !v.is_finite() {
            *v = f32::NEG_INFINITY;
            found = true;
        }
    }
    if found {
        eprintln!("{tag}: detected non-finite logits; clamped to -inf");
    }
}

fn detect_llama_prefix(file: &CellmFile) -> Result<String, CoreError> {
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
        "missing required llama tensors: model.embed_tokens.weight/model.norm.weight".into(),
    ))
}
