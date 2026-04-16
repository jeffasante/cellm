use std::collections::HashMap;
use std::path::Path;

use bytemuck::cast_slice;
use cellm_cache::{KVCache, PageTable};
use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::rms_norm_f32;
use cellm_kernels::metal::MetalMatmul;
use cellm_kernels::{MetalKernels, MetalOps};
use half::f16;
use serde_json::Value;

use crate::{CellmFile, ModelConfig};

const QWEN_METAL_LINEAR_MAX_ELEMS: usize = 262_144;

/// Minimal text-only runner for Qwen3.5 checkpoints that contain both
/// `linear_attention` and `full_attention` layers.
///
/// Current behavior:
/// - Runs full-attention blocks when `self_attn.*` weights exist for a layer.
/// - Skips the attention part for other layers (but still runs the MLP).
/// - Uses tied embeddings for logits.
///
/// This is a correctness-first baseline intended to unblock bring-up.
pub struct QwenRunner {
    file: CellmFile,
    cfg: ModelConfig,
    max_layers: usize,
    eos_token_id: Option<u32>,
    layer_kinds: Vec<LayerKind>,
    linear_spec: Option<LinearAttnSpec>,
    partial_rotary_factor: f32,
    rmsnorm_weight_is_offset: bool,
    sessions: HashMap<u64, QwenSessionState>,
    disable_linear_attention: bool,
    disable_full_attention: bool,
    disable_full_attn_gate: bool,
    debug_pos: Option<usize>,
    linear_backend: QwenLinearBackend,
    metal_strict: bool,
    metal_ops: Option<MetalOps>,
}

enum QwenLinearBackend {
    Cpu,
    Metal { ctx: MetalMatmul },
}

impl QwenRunner {
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

        if !has_tensor_file(&file, "language_model.model.embed_tokens.weight")
            && !has_tensor_file(&file, "model.embed_tokens.weight")
        {
            return Err(CoreError::Backend(
                "missing tensor: language_model.model.embed_tokens.weight or model.embed_tokens.weight"
                    .into(),
            ));
        }
        if !has_tensor_file(&file, "language_model.model.norm.weight")
            && !has_tensor_file(&file, "model.norm.weight")
        {
            return Err(CoreError::Backend(
                "missing tensor: language_model.model.norm.weight or model.norm.weight".into(),
            ));
        }

        let (layer_kinds, linear_spec, partial_rotary_factor) =
            infer_qwen_layer_kinds_and_linear_spec(&file)?;

        Ok(Self {
            file,
            cfg: cfg.clone(),
            max_layers: cfg.num_hidden_layers,
            eos_token_id: h.eos_token_id,
            layer_kinds,
            linear_spec,
            partial_rotary_factor,
            rmsnorm_weight_is_offset: qwen_rmsnorm_is_offset(&h),
            sessions: HashMap::new(),
            disable_linear_attention: std::env::var("CELLM_QWEN_DISABLE_LINEAR")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            disable_full_attention: std::env::var("CELLM_QWEN_DISABLE_FULL")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            disable_full_attn_gate: std::env::var("CELLM_QWEN_DISABLE_FULL_GATE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            debug_pos: std::env::var("CELLM_QWEN_DEBUG_POS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok()),
            linear_backend: QwenLinearBackend::Cpu,
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

    fn top_k_logits(&self, logits: &[f32], top_k: usize) -> Vec<(u32, f32)> {
        let vocab = self.cfg.vocab_size;
        let k = top_k.max(1).min(vocab);
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut min_idx = 0usize;
        let mut min_val = f32::INFINITY;

        for vid in 0..vocab {
            let val = logits[vid];
            if top.len() < k {
                top.push((vid as u32, val));
                if val < min_val {
                    min_val = val;
                    min_idx = top.len() - 1;
                }
            } else if val > min_val {
                top[min_idx] = (vid as u32, val);
                min_val = top[0].1;
                min_idx = 0;
                for (i, &(_, v)) in top.iter().enumerate().skip(1) {
                    if v < min_val {
                        min_val = v;
                        min_idx = i;
                    }
                }
            }
        }
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        top
    }

    pub fn enable_metal_linear_backend(&mut self) -> bool {
        match MetalKernels::create_matmul() {
            Ok(ctx) => {
                self.linear_backend = QwenLinearBackend::Metal { ctx };
                self.metal_strict = false;
                true
            }
            Err(e) => {
                eprintln!("qwen: failed to enable metal linear backend: {e}");
                self.linear_backend = QwenLinearBackend::Cpu;
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

    pub fn debug_partial_rotary_factor(&self) -> f32 {
        self.partial_rotary_factor
    }

    pub fn debug_rmsnorm_weight_is_offset(&self) -> bool {
        self.rmsnorm_weight_is_offset
    }

    pub fn enable_metal_full_backend(&mut self) -> bool {
        let mk_res = MetalKernels::create_matmul();
        let mo_res = MetalOps::create();
        match (mk_res, mo_res) {
            (Ok(ctx), Ok(mo)) => {
                self.linear_backend = QwenLinearBackend::Metal { ctx };
                self.metal_strict = true;
                self.metal_ops = Some(mo);
                true
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("qwen: failed to enable full metal backend: {e}");
                self.linear_backend = QwenLinearBackend::Cpu;
                self.metal_strict = false;
                self.metal_ops = None;
                false
            }
        }
    }

    pub fn cancel_session(&mut self, session_id: u64) {
        self.sessions.remove(&session_id);
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
        let logits = self.step_inner(&x, pos, page_table, kv_cache, true)?;
        Ok(self.top_k_logits(&logits, top_k))
    }

    pub fn prefill(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
    ) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            let mut x = vec![0.0f32; hidden];
            self.embed_token(tok, &mut x)?;
            self.step_inner(&x, pos, page_table, kv_cache, false)?;
        }
        Ok(())
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
        let head_dim = hidden / n_heads;
        let rotary_dim = ((head_dim as f32) * self.partial_rotary_factor) as usize;

        // Lazy session state init.
        let session_id = page_table.session_id();
        let sess = self.sessions
            .entry(session_id)
            .or_insert_with(|| QwenSessionState {
                linear: vec![],
                graph_state: None,
            });

        if false && sess.graph_state.is_none() && self.metal_ops.is_some() && self.metal_strict {
            let mut gs = QwenGraphState::new(
                hidden,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.vocab_size,
                cfg.intermediate_size,
                self.metal_ops.clone().unwrap(),
            )?;
            for (name, bytes) in self.file.all_tensors() {
                gs.preload_weight(name.clone(), bytes);
            }
            sess.graph_state = Some(gs);
        }

        // Ensure pagetable covers this token position.
        if pos == page_table.token_count() {
            page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                CoreError::Backend(format!("qwen step: page_table append_token failed: {e}"))
            })?;
        } else if pos > page_table.token_count() {
            return Err(CoreError::Backend(format!(
                "qwen step: non-contiguous pos {pos} (token_count={})",
                page_table.token_count()
            )));
        }

        let block_id = page_table.block_for_token(pos).map_err(|e| {
            CoreError::Backend(format!("qwen step: page_table block_for_token failed: {e}"))
        })? as u32;
        let token_off = page_table.offset_in_block(pos).map_err(|e| {
            CoreError::Backend(format!("qwen step: page_table offset_in_block failed: {e}"))
        })?;

        // x = embedding(token)
        let mut x = x0.to_vec();

        let mut x_norm = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; hidden];
        let mut k = vec![0.0f32; n_kv_heads * head_dim];
        let mut v = vec![0.0f32; n_kv_heads * head_dim];
        let mut attn_out = vec![0.0f32; hidden];
        let mut attn_proj = vec![0.0f32; hidden];
        let mut mlp_in = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; cfg.intermediate_size];
        let mut up = vec![0.0f32; cfg.intermediate_size];
        let mut down = vec![0.0f32; hidden];
        let mut gather_bases = Vec::new();

        let prefix = if has_tensor_file(&self.file, "language_model.model.embed_tokens.weight") {
            "language_model."
        } else {
            ""
        };

        if false {
            if let Some(graph_state) = &mut sess.graph_state {
                let res = graph_state.step_fused(
                    &x,
                    &cfg,
                    prefix,
                    kv_cache,
                    page_table,
                    pos,
                    token_off,
                    block_id,
                    self.partial_rotary_factor,
                    self.rmsnorm_weight_is_offset,
                    return_logits,
                )?;
                return Ok(res.unwrap_or_else(Vec::new));
            }
        }

        let use_metal_norm = self.metal_ops.is_some();
        let use_metal_rope = self.metal_ops.is_some() && !self.disable_full_attention;

        for layer in 0..self.max_layers {
            // Input norm
            if use_metal_norm {
                let name = format!("{prefix}model.layers.{layer}.input_layernorm.weight");
                let w = self.tensor_f16(&name)?.to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                self.metal_ops.as_ref().unwrap()
                    .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &name, &mut x_norm)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                let mut in_norm_w = vec![0.0f32; hidden];
                self.rmsnorm_weight(&format!("{prefix}model.layers.{layer}.input_layernorm.weight"), &mut in_norm_w)?;
                if self.rmsnorm_weight_is_offset {
                    add_one_inplace(&mut in_norm_w);
                }
                rms_norm_f32(&x, &in_norm_w, cfg.rms_norm_eps, &mut x_norm);
            }

            match self.layer_kinds[layer] {
                LayerKind::FullAttention => {
                    let (q_name, k_name, v_name) = self.resolve_qkv_names(layer, prefix);
                    let fused_qkv = self.linear_qkv_f16_out_in(&x_norm, &q_name, hidden, &k_name, n_kv_heads * head_dim, &v_name, n_kv_heads * head_dim, hidden, &mut q, &mut k, &mut v)?;
                    if !fused_qkv {
                        self.linear_f16_out_in(&x_norm, &q_name, hidden, hidden, &mut q)?;
                        self.linear_f16_out_in(&x_norm, &k_name, n_kv_heads * head_dim, hidden, &mut k)?;
                        self.linear_f16_out_in(&x_norm, &v_name, n_kv_heads * head_dim, hidden, &mut v)?;
                    }
                    self.add_projection_bias_if_present(&q_name, &mut q)?;
                    self.add_projection_bias_if_present(&k_name, &mut k)?;
                    self.add_projection_bias_if_present(&v_name, &mut v)?;

                    if rotary_dim > 0 {
                        if use_metal_rope {
                            let ops = self.metal_ops.as_ref().unwrap();
                            ops.rope_half_f32(&mut q, n_heads, head_dim, rotary_dim, pos, cfg.rope_theta).map_err(|e| CoreError::Backend(e.to_string()))?;
                            ops.rope_half_f32(&mut k, n_kv_heads, head_dim, rotary_dim, pos, cfg.rope_theta).map_err(|e| CoreError::Backend(e.to_string()))?;
                        } else {
                            rope_inplace_f32_partial(&mut q, n_heads, head_dim, rotary_dim, pos, cfg.rope_theta);
                            rope_inplace_f32_partial(&mut k, n_kv_heads, head_dim, rotary_dim, pos, cfg.rope_theta);
                        }
                    }

                    {
                        let mut cv = kv_cache.view_mut();
                        cv.write_token(block_id, layer, token_off, &k, &v)?;
                    }

                    let seq = page_table.token_count();
                    let cr = kv_cache.view();
                    gather_bases.clear();
                    for tpos in 0..seq {
                        let b = page_table.block_for_token(tpos).map_err(|e| CoreError::Backend(e.to_string()))?;
                        let o = page_table.offset_in_block(tpos).map_err(|e| CoreError::Backend(e.to_string()))?;
                        gather_bases.push(cr.layout.token_base_elem(b, layer, o)?);
                    }
                    cr.attention_single_token_gqa_from_bases(&gather_bases, &q, n_heads, n_kv_heads, head_dim, None, &mut attn_out)?;

                    self.linear_f16_out_in(&attn_out, &format!("{prefix}model.layers.{layer}.self_attn.o_proj.weight"), hidden, hidden, &mut attn_proj)?;
                    self.add_projection_bias_if_present(&format!("{prefix}model.layers.{layer}.self_attn.o_proj.weight"), &mut attn_proj)?;

                    for i in 0..hidden {
                        x[i] += attn_proj[i];
                    }
                }
                LayerKind::LinearAttention => {
                }
            }

            // Post-attn norm
            if use_metal_norm {
                let name = format!("{prefix}model.layers.{layer}.post_attention_layernorm.weight");
                let w = self.tensor_f16(&name)?.to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                self.metal_ops.as_ref().unwrap()
                    .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &name, &mut mlp_in)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                let mut post_norm_w = vec![0.0f32; hidden];
                self.rmsnorm_weight(&format!("{prefix}model.layers.{layer}.post_attention_layernorm.weight"), &mut post_norm_w)?;
                if self.rmsnorm_weight_is_offset {
                    add_one_inplace(&mut post_norm_w);
                }
                rms_norm_f32(&x, &post_norm_w, cfg.rms_norm_eps, &mut mlp_in);
            }

            // MLP
            self.linear_f16_out_in(&mlp_in, &format!("{prefix}model.layers.{layer}.mlp.gate_proj.weight"), cfg.intermediate_size, hidden, &mut gate)?;
            self.linear_f16_out_in(&mlp_in, &format!("{prefix}model.layers.{layer}.mlp.up_proj.weight"), cfg.intermediate_size, hidden, &mut up)?;

            use rayon::prelude::*;
            gate.par_iter_mut().for_each(|g| {
                let s = 1.0 / (1.0 + (-*g).exp());
                *g = *g * s;
            });
            for i in 0..gate.len() {
                gate[i] *= up[i];
            }

            self.linear_f16_out_in(&gate, &format!("{prefix}model.layers.{layer}.mlp.down_proj.weight"), hidden, cfg.intermediate_size, &mut down)?;

            for i in 0..hidden {
                x[i] += down[i];
            }
        }

        // Final norm
        let mut x_final = vec![0.0f32; hidden];
        if use_metal_norm {
            let name = format!("{prefix}model.norm.weight");
            let w = self.tensor_f16(&name)?.to_vec();
            let add_one = self.rmsnorm_weight_is_offset;
            self.metal_ops.as_ref().unwrap()
                .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &name, &mut x_final)
                .map_err(|e| CoreError::Backend(e.to_string()))?;
        } else {
            let mut norm_w = vec![0.0f32; hidden];
            self.rmsnorm_weight(&format!("{prefix}model.norm.weight"), &mut norm_w)?;
            if self.rmsnorm_weight_is_offset {
                add_one_inplace(&mut norm_w);
            }
            rms_norm_f32(&x, &norm_w, cfg.rms_norm_eps, &mut x_final);
        }

        if !return_logits {
            return Ok(vec![]);
        }

        // Logits
        let embed_name = self.resolve_tensor_name("language_model.model.embed_tokens.weight");
        let lm_head_name = if has_tensor_file(&self.file, "lm_head.weight") {
            Some(self.resolve_tensor_name("lm_head.weight"))
        } else {
            None
        };
        let weight_name = lm_head_name.as_deref().unwrap_or(&embed_name);
        let vocab = cfg.vocab_size;

        let use_metal_logits = self.metal_ops.is_some();
        let mut logits = vec![0.0f32; vocab];
        if use_metal_logits {
            let dtype = self.file.tensor_index(weight_name).unwrap().dtype.clone();
            match dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16(weight_name)?.to_vec();
                    self.metal_ops.as_ref().unwrap().logits_f16(&x_final, &w, vocab, hidden, weight_name, &mut logits).map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                "i8" => {
                    let w = self.tensor_i8(weight_name)?.to_vec();
                    let s = self.tensor_f16(&format!("{weight_name}.qscale"))?.to_vec();
                    self.metal_ops.as_ref().unwrap().logits_i8(&x_final, &w, &s, vocab, hidden, weight_name, &mut logits).map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                _ => return Err(CoreError::Backend("unsupported lm_head dtype".into())),
            }
        } else {
            for vid in 0..vocab {
                logits[vid] = self.dot_row(weight_name, vid, hidden, &x_final)?;
            }
        }

        Ok(logits)
    }

    fn embed_token(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        let prefix = if has_tensor_file(&self.file, "language_model.model.embed_tokens.weight") {
            "language_model."
        } else {
            ""
        };
        let weight_name = format!("{prefix}model.embed_tokens.weight");
        let hidden = out.len();
        let resolved = self.resolve_tensor_name(&weight_name);
        let dtype = self
            .file
            .tensor_index(&resolved)
            .ok_or_else(|| CoreError::Backend(format!("unknown tensor {weight_name}")))?
            .dtype
            .clone();
        let vocab = self.cfg.vocab_size;
        let t = (token as usize) % vocab;
        match dtype.as_str() {
            "f16" => {
                let embed = self.tensor_f16(&weight_name)?;
                let row = &embed[t * hidden..(t + 1) * hidden];
                for i in 0..hidden {
                    out[i] = f16::from_bits(row[i]).to_f32();
                }
            }
            "i8" => {
                let embed = self.tensor_i8(&weight_name)?;
                let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                let scale = f16::from_bits(scales[t]).to_f32();
                let row = &embed[t * hidden..(t + 1) * hidden];
                for i in 0..hidden {
                    out[i] = (row[i] as f32) * scale;
                }
            }
            "i4" => {
                let embed = self.tensor_u8(&weight_name)?;
                let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                let scale = f16::from_bits(scales[t]).to_f32();
                let row_stride = hidden.div_ceil(2);
                let row = &embed[t * row_stride..(t + 1) * row_stride];
                for i in 0..hidden {
                    out[i] = unpack_i4(row, i) * scale;
                }
            }
            other => {
                return Err(CoreError::Backend(format!(
                    "unsupported embed dtype for {weight_name}: {other}"
                )));
            }
        }
        Ok(())
    }

    fn resolve_qkv_names(&self, layer: usize, prefix: &str) -> (String, String, String) {
        let q_name = format!("{prefix}model.layers.{layer}.self_attn.q_proj.weight");
        let k_name = format!("{prefix}model.layers.{layer}.self_attn.k_proj.weight");
        let v_name = format!("{prefix}model.layers.{layer}.self_attn.v_proj.weight");
        (q_name, k_name, v_name)
    }

    fn tensor_f16(&self, name: &str) -> Result<&[u16], CoreError> {
        let resolved = self.resolve_tensor_name(name);
        let bytes = self.file.tensor_bytes(&resolved)?;
        if bytes.len() % 2 != 0 {
            return Err(CoreError::Backend(format!(
                "tensor {name} (resolved={resolved}) nbytes not even"
            )));
        }
        Ok(cast_slice(bytes))
    }

    fn tensor_i8(&self, name: &str) -> Result<&[i8], CoreError> {
        let resolved = self.resolve_tensor_name(name);
        let bytes = self.file.tensor_bytes(&resolved)?;
        Ok(cast_slice(bytes))
    }

    fn tensor_u8(&self, name: &str) -> Result<&[u8], CoreError> {
        let resolved = self.resolve_tensor_name(name);
        self.file.tensor_bytes(&resolved)
    }

    fn tensor_shape(&self, name: &str) -> Result<Vec<usize>, CoreError> {
        let resolved = self.resolve_tensor_name(name);
        let t = self
            .file
            .tensor_index(&resolved)
            .ok_or_else(|| CoreError::Backend(format!("unknown tensor {name} (resolved={resolved})")))?;
        Ok(t.shape.clone())
    }

    fn resolve_tensor_name(&self, name: &str) -> String {
        resolve_tensor_name_file(&self.file, name)
    }

    fn dot_row(
        &self,
        weight_name: &str,
        row_idx: usize,
        in_dim: usize,
        x: &[f32],
    ) -> Result<f32, CoreError> {
        let shape = self.tensor_shape(weight_name)?;
        if shape.len() != 2 || shape[1] != in_dim {
            return Err(CoreError::Backend(format!(
                "weight {weight_name} shape mismatch: {:?} expected [*,{in_dim}]",
                shape
            )));
        }
        if row_idx >= shape[0] {
            return Err(CoreError::Backend(format!(
                "weight {weight_name} row index out of range: {row_idx} >= {}",
                shape[0]
            )));
        }
        let resolved = self.resolve_tensor_name(weight_name);
        let dtype = self
            .file
            .tensor_index(&resolved)
            .ok_or_else(|| CoreError::Backend(format!("unknown tensor {weight_name}")))?
            .dtype
            .clone();
        match dtype.as_str() {
            "f16" => {
                let w = self.tensor_f16(weight_name)?;
                let row = &w[row_idx * in_dim..(row_idx + 1) * in_dim];
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    acc += x[i] * f16::from_bits(row[i]).to_f32();
                }
                Ok(acc)
            }
            "i8" => {
                let w = self.tensor_i8(weight_name)?;
                let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                let scale = f16::from_bits(scales[row_idx]).to_f32();
                let row = &w[row_idx * in_dim..(row_idx + 1) * in_dim];
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    acc += x[i] * (row[i] as f32) * scale;
                }
                Ok(acc)
            }
            "i4" => {
                let w = self.tensor_u8(weight_name)?;
                let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                let scale = f16::from_bits(scales[row_idx]).to_f32();
                let row_stride = in_dim.div_ceil(2);
                let row = &w[row_idx * row_stride..(row_idx + 1) * row_stride];
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    acc += x[i] * unpack_i4(row, i) * scale;
                }
                Ok(acc)
            }
            other => {
                return Err(CoreError::Backend(format!(
                    "unsupported embed dtype for {weight_name}: {other}"
                )));
            }
        }
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

        let shape = self.tensor_shape(weight_name)?;
        let resolved = self.resolve_tensor_name(weight_name);
        let dtype = self
            .file
            .tensor_index(&resolved)
            .ok_or_else(|| CoreError::Backend(format!("unknown tensor {weight_name}")))?
            .dtype
            .clone();
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

        // Fast path: use MetalOps matrix-vector kernels with internal GPU-side cache.
        if self.metal_ops.is_some() {
            match dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16(weight_name)?;
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
                    let w = self.tensor_i8(weight_name)?;
                    let s = self.tensor_f16(&format!("{weight_name}.qscale"))?;
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


        // Fallback: CPU loop via optimized kernels.
        match dtype.as_str() {
            "f16" => {
                let w = self.tensor_f16(weight_name)?;
                cellm_kernels::cpu_kernels::matmul_f16_f32(&w, out_dim, in_dim, x, out);
            }
            "i8" => {
                let w = self.tensor_i8(weight_name)?;
                let s = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                cellm_kernels::cpu_kernels::matmul_i8_f32(&w, s, out_dim, in_dim, x, out);
            }
            "i4" => {
                let w = self.tensor_u8(weight_name)?;
                let s = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                cellm_kernels::cpu_kernels::matmul_i4_f32(&w, s, out_dim, in_dim, x, out);
            }
            _ => {
                return Err(CoreError::Backend(format!(
                    "qwen: CPU linear fallback unsupported dtype {dtype} for {weight_name}"
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
                "qwen qkv fused dims mismatch: x={} q={} k={} v={} expected in={} q_out={} k_out={} v_out={}",
                x.len(), q_out.len(), k_out.len(), v_out.len(), in_dim, q_out_dim, k_out_dim, v_out_dim
            )));
        }

        let q_resolved = self.resolve_tensor_name(q_weight_name);
        let k_resolved = self.resolve_tensor_name(k_weight_name);
        let v_resolved = self.resolve_tensor_name(v_weight_name);

        let q_meta = self.file.tensor_index(&q_resolved).ok_or_else(|| {
            CoreError::Backend(format!("unknown tensor {q_resolved}"))
        })?;
        let k_meta = self.file.tensor_index(&k_resolved).ok_or_else(|| {
            CoreError::Backend(format!("unknown tensor {k_resolved}"))
        })?;
        let v_meta = self.file.tensor_index(&v_resolved).ok_or_else(|| {
            CoreError::Backend(format!("unknown tensor {v_resolved}"))
        })?;

        if q_meta.shape.len() != 2 || k_meta.shape.len() != 2 || v_meta.shape.len() != 2 {
            return Err(CoreError::Backend(format!(
                "qwen qkv fused expects 2D weights, got q={:?} k={:?} v={:?}",
                q_meta.shape, k_meta.shape, v_meta.shape
            )));
        }
        if q_meta.shape != [q_out_dim, in_dim]
            || k_meta.shape != [k_out_dim, in_dim]
            || v_meta.shape != [v_out_dim, in_dim]
        {
            return Err(CoreError::Backend(format!(
                "qwen qkv fused shape mismatch: q={:?} k={:?} v={:?} expected q=[{},{}] k=[{},{}] v=[{},{}]",
                q_meta.shape, k_meta.shape, v_meta.shape, q_out_dim, in_dim, k_out_dim, in_dim, v_out_dim, in_dim
            )));
        }

        if q_meta.dtype != "f16" || k_meta.dtype != "f16" || v_meta.dtype != "f16" {
            return Ok(false);
        }

        let wq = self.tensor_f16(q_weight_name)?;
        let wk = self.tensor_f16(k_weight_name)?;
        let wv = self.tensor_f16(v_weight_name)?;
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

    fn add_projection_bias_if_present(
        &self,
        weight_name: &str,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        let Some(stem) = weight_name.strip_suffix(".weight") else {
            return Ok(());
        };
        let bias_name = format!("{stem}.bias");
        if !has_tensor_file(&self.file, &bias_name) {
            return Ok(());
        }
        let b = self.tensor_f16(&bias_name)?;
        if b.len() != out.len() {
            return Ok(());
        }
        for (o, &bv) in out.iter_mut().zip(b.iter()) {
            *o += f16::from_bits(bv).to_f32();
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn linear_attn_step(
    file: &CellmFile,
    layer: usize,
    hidden: usize,
    x_norm: &[f32],
    state: &mut LinearLayerState,
    spec: &LinearAttnSpec,
    eps: f32,
    qkv: &mut [f32],
    z: &mut [f32],
    a: &mut [f32],
    b: &mut [f32],
    mixed_qkv: &mut [f32],
    conv_out: &mut [f32],
    q_rep: &mut [f32],
    k_rep: &mut [f32],
    v: &mut [f32],
    core_out: &mut [f32],
    normed: &mut [f32],
    tmp_kv: &mut [f32],
    tmp_delta: &mut [f32],
    out_proj: &mut [f32],
) -> Result<(), CoreError> {
    let key_dim = spec.num_k_heads * spec.head_k_dim;
    let value_dim = spec.num_v_heads * spec.head_v_dim;
    let conv_dim = key_dim * 2 + value_dim;
    if qkv.len() != conv_dim
        || mixed_qkv.len() != conv_dim
        || conv_out.len() != conv_dim
        || q_rep.len() != spec.num_v_heads * spec.head_k_dim
        || k_rep.len() != spec.num_v_heads * spec.head_k_dim
        || v.len() != value_dim
        || z.len() != value_dim
        || core_out.len() != value_dim
        || normed.len() != value_dim
        || out_proj.len() != hidden
        || tmp_kv.len() != spec.head_v_dim
        || tmp_delta.len() != spec.head_v_dim
    {
        return Err(CoreError::Backend(
            "qwen linear_attn: scratch size mismatch".into(),
        ));
    }

    linear_f16_out_in_file(file, x_norm, &format!("language_model.model.layers.{layer}.linear_attn.in_proj_qkv.weight"), conv_dim, hidden, qkv)?;
    linear_f16_out_in_file(file, x_norm, &format!("language_model.model.layers.{layer}.linear_attn.in_proj_z.weight"), value_dim, hidden, z)?;
    linear_f16_out_in_file(file, x_norm, &format!("language_model.model.layers.{layer}.linear_attn.in_proj_a.weight"), spec.num_v_heads, hidden, a)?;
    linear_f16_out_in_file(file, x_norm, &format!("language_model.model.layers.{layer}.linear_attn.in_proj_b.weight"), spec.num_v_heads, hidden, b)?;

    if spec.num_k_heads == 0 { return Err(CoreError::Backend("qwen linear_attn: num_k_heads=0".into())); }
    let ratio = spec.num_v_heads / spec.num_k_heads;

    mixed_qkv.copy_from_slice(qkv);

    let conv_w = tensor_f16_file(file, &format!("language_model.model.layers.{layer}.linear_attn.conv1d.weight"))?;
    let kernel_size = spec.conv_kernel_size;
    causal_conv1d_update_silu(mixed_qkv, &mut state.conv, conv_w, kernel_size, conv_out);

    let q_base = &conv_out[..key_dim];
    let k_base = &conv_out[key_dim..2 * key_dim];
    v.copy_from_slice(&conv_out[2 * key_dim..]);

    if ratio == 1 {
        q_rep.copy_from_slice(q_base);
        k_rep.copy_from_slice(k_base);
    } else {
        for kh in 0..spec.num_k_heads {
            let qh = &q_base[kh * spec.head_k_dim..(kh + 1) * spec.head_k_dim];
            let khs = &k_base[kh * spec.head_k_dim..(kh + 1) * spec.head_k_dim];
            for r in 0..ratio {
                let vh = kh * ratio + r;
                q_rep[vh * spec.head_k_dim..(vh + 1) * spec.head_k_dim].copy_from_slice(qh);
                k_rep[vh * spec.head_k_dim..(vh + 1) * spec.head_k_dim].copy_from_slice(khs);
            }
        }
    }

    if spec.use_qk_l2norm_in_kernel {
        for h in 0..spec.num_v_heads {
            l2norm_inplace(&mut q_rep[h * spec.head_k_dim..(h + 1) * spec.head_k_dim], 1e-6);
            l2norm_inplace(&mut k_rep[h * spec.head_k_dim..(h + 1) * spec.head_k_dim], 1e-6);
        }
    }
    let scale = 1.0 / (spec.head_k_dim as f32).sqrt();
    for v_val in q_rep.iter_mut() { *v_val *= scale; }

    let a_log = tensor_f16_file(file, &format!("language_model.model.layers.{layer}.linear_attn.A_log"))?;
    let dt_bias = tensor_f16_file(file, &format!("language_model.model.layers.{layer}.linear_attn.dt_bias"))?;

    core_out.fill(0.0);
    for h in 0..spec.num_v_heads {
        let beta = sigmoid_f32(b[h]);
        let g = -f16::from_bits(a_log[h]).to_f32().exp() * softplus_f32(a[h] + f16::from_bits(dt_bias[h]).to_f32());
        let g_exp = g.exp();
        let qh = &q_rep[h * spec.head_k_dim..(h + 1) * spec.head_k_dim];
        let kh = &k_rep[h * spec.head_k_dim..(h + 1) * spec.head_k_dim];
        let vh = &v[h * spec.head_v_dim..(h + 1) * spec.head_v_dim];
        let st_base = h * spec.head_k_dim * spec.head_v_dim;
        let st = &mut state.recurrent[st_base..st_base + spec.head_k_dim * spec.head_v_dim];
        for s in st.iter_mut() { *s *= g_exp; }
        tmp_kv.fill(0.0);
        for i in 0..spec.head_k_dim {
            let ki = kh[i];
            if ki != 0.0 {
                let row = &st[i * spec.head_v_dim..(i + 1) * spec.head_v_dim];
                for j in 0..spec.head_v_dim { tmp_kv[j] += row[j] * ki; }
            }
        }
        for j in 0..spec.head_v_dim { tmp_delta[j] = (vh[j] - tmp_kv[j]) * beta; }
        for i in 0..spec.head_k_dim {
            let ki = kh[i];
            if ki != 0.0 {
                let row = &mut st[i * spec.head_v_dim..(i + 1) * spec.head_v_dim];
                for j in 0..spec.head_v_dim { row[j] += ki * tmp_delta[j]; }
            }
        }
        let out_dst = &mut core_out[h * spec.head_v_dim..(h + 1) * spec.head_v_dim];
        for i in 0..spec.head_k_dim {
            let qi = qh[i];
            if qi != 0.0 {
                let row = &st[i * spec.head_v_dim..(i + 1) * spec.head_v_dim];
                for j in 0..spec.head_v_dim { out_dst[j] += row[j] * qi; }
            }
        }
    }

    let norm_w = tensor_f16_file(file, &format!("language_model.model.layers.{layer}.linear_attn.norm.weight"))?;
    for h in 0..spec.num_v_heads {
        let src = &core_out[h * spec.head_v_dim..(h + 1) * spec.head_v_dim];
        let gate = &z[h * spec.head_v_dim..(h + 1) * spec.head_v_dim];
        let mut var = 0.0f32;
        for &v_val in src { var += v_val * v_val; }
        var /= spec.head_v_dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        let dst_seg = &mut normed[h * spec.head_v_dim..(h + 1) * spec.head_v_dim];
        for j in 0..spec.head_v_dim {
            let wj = f16::from_bits(norm_w[j]).to_f32();
            dst_seg[j] = src[j] * inv * wj * silu_f32(gate[j]);
        }
    }
    linear_f16_out_in_file(file, normed, &format!("language_model.model.layers.{layer}.linear_attn.out_proj.weight"), hidden, value_dim, out_proj)?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Clone)]
struct LinearAttnSpec {
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_kernel_size: usize,
    use_qk_l2norm_in_kernel: bool,
}

#[derive(Debug)]
struct LinearLayerState {
    conv: Vec<f32>,
    recurrent: Vec<f32>,
}

#[derive(Debug)]
struct QwenSessionState {
    linear: Vec<Option<LinearLayerState>>,
    graph_state: Option<QwenGraphState>,
}

fn infer_qwen_layer_kinds_and_linear_spec(
    file: &CellmFile,
) -> Result<(Vec<LayerKind>, Option<LinearAttnSpec>, f32), CoreError> {
    let n_layers = file.header.num_layers;
    let mut kinds: Vec<LayerKind> = Vec::with_capacity(n_layers);
    let mut partial_rotary_factor = 1.0f32;
    let mut spec: Option<LinearAttnSpec> = None;

    if let Some(Value::Object(obj)) = &file.header.source_text_config {
        if let Some(Value::Object(rope_params)) = obj.get("rope_parameters") {
            if let Some(v) = rope_params.get("partial_rotary_factor").and_then(|v| v.as_f64()) {
                partial_rotary_factor = (v as f32).clamp(0.0, 1.0);
            }
        }
        if let Some(Value::Array(layer_types)) = obj.get("layer_types") {
            for v in layer_types {
                let s = v.as_str().unwrap_or("");
                kinds.push(match s {
                    "linear_attention" => LayerKind::LinearAttention,
                    _ => LayerKind::FullAttention,
                });
            }
        }
        let get_usize = |k: &str| obj.get(k).and_then(|v| v.as_u64()).map(|x| x as usize);
        if let (Some(num_k), Some(num_v), Some(hk), Some(hv), Some(kw)) = (
            get_usize("linear_num_key_heads"), get_usize("linear_num_value_heads"),
            get_usize("linear_key_head_dim"), get_usize("linear_value_head_dim"),
            get_usize("linear_conv_kernel_dim")
        ) {
            spec = Some(LinearAttnSpec { num_k_heads: num_k, num_v_heads: num_v, head_k_dim: hk, head_v_dim: hv, conv_kernel_size: kw, use_qk_l2norm_in_kernel: true });
        }
    }

    if kinds.len() != n_layers {
        kinds.clear();
        for layer in 0..n_layers {
            let has_linear = has_tensor_file(file, &format!("language_model.model.layers.{layer}.linear_attn.in_proj_qkv.weight"));
            kinds.push(if has_linear { LayerKind::LinearAttention } else { LayerKind::FullAttention });
        }
    }
    Ok((kinds, spec, partial_rotary_factor))
}

fn qwen_rmsnorm_is_offset(header: &crate::cellm_file::CellmHeader) -> bool {
    let check = |s: &str| s.starts_with("qwen3_5");
    header.model_type.starts_with("qwen3_5") || header.source_model_type.as_deref().is_some_and(check)
}

fn add_one_inplace(x: &mut [f32]) { for v in x { *v += 1.0; } }
fn sigmoid_f32(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn silu_f32(x: f32) -> f32 { x * sigmoid_f32(x) }
fn softplus_f32(x: f32) -> f32 { if x > 20.0 { x } else if x < -20.0 { x.exp() } else { (1.0 + x.exp()).ln() } }
fn l2norm_inplace(x: &mut [f32], eps: f32) {
    let mut s = 0.0f32;
    for &v in x.iter() { s += v * v; }
    let inv = 1.0 / (s + eps).sqrt();
    for v in x.iter_mut() { *v *= inv; }
}
fn l2norm_value(x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &v in x { s += v * v; }
    s.sqrt()
}

fn rope_inplace_f32_partial(x: &mut [f32], n_heads: usize, head_dim: usize, rotary_dim: usize, pos: usize, theta: f32) {
    let half = rotary_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f32) / rotary_dim as f32);
            let angle = pos as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = x[base + i]; let x1 = x[base + half + i];
            x[base + i] = x0 * cos - x1 * sin;
            x[base + half + i] = x1 * cos + x0 * sin;
        }
    }
}

fn causal_conv1d_update_silu(x: &[f32], state: &mut [f32], weight_f16: &[u16], kernel_size: usize, out: &mut [f32]) {
    let ch = x.len();
    for c in 0..ch {
        let st_base = c * kernel_size;
        let w_base = c * kernel_size;
        let mut acc = 0.0f32;
        for t in 0..kernel_size - 1 {
            acc += f16::from_bits(weight_f16[w_base + t]).to_f32() * state[st_base + t + 1];
        }
        acc += f16::from_bits(weight_f16[w_base + kernel_size - 1]).to_f32() * x[c];
        out[c] = silu_f32(acc);
        for t in 0..kernel_size - 1 { state[st_base + t] = state[st_base + t + 1]; }
        state[st_base + kernel_size - 1] = x[c];
    }
}

fn resolve_tensor_name_file(file: &CellmFile, name: &str) -> String {
    if file.tensor_index(name).is_some() { return name.to_string(); }
    
    let stem = if let Some(rest) = name.strip_prefix("language_model.model.") { rest }
               else if let Some(rest) = name.strip_prefix("model.language_model.") { rest }
               else if let Some(rest) = name.strip_prefix("model.") { rest }
               else { name };
    
    let candidates = [
        format!("model.{stem}"),
        format!("language_model.model.{stem}"),
        format!("model.language_model.{stem}"),
    ];
    for c in candidates {
        if file.tensor_index(&c).is_some() { return c; }
    }
    name.to_string()
}

fn has_tensor_file(file: &CellmFile, name: &str) -> bool { file.tensor_index(&resolve_tensor_name_file(file, name)).is_some() }

fn tensor_index_file<'a>(file: &'a CellmFile, name: &str) -> Option<&'a crate::cellm_file::CellmTensorIndex> {
    file.tensor_index(&resolve_tensor_name_file(file, name))
}

fn tensor_f16_file<'a>(file: &'a CellmFile, name: &str) -> Result<&'a [u16], CoreError> {
    let b = file.tensor_bytes(&resolve_tensor_name_file(file, name))?;
    Ok(cast_slice(b))
}

fn tensor_i8_file<'a>(file: &'a CellmFile, name: &str) -> Result<&'a [i8], CoreError> {
    let b = file.tensor_bytes(&resolve_tensor_name_file(file, name))?;
    Ok(cast_slice(b))
}

fn tensor_u8_file<'a>(file: &'a CellmFile, name: &str) -> Result<&'a [u8], CoreError> {
    file.tensor_bytes(&resolve_tensor_name_file(file, name))
}

fn tensor_shape_file(file: &CellmFile, name: &str) -> Result<Vec<usize>, CoreError> {
    Ok(tensor_index_file(file, name).ok_or_else(|| CoreError::Backend(format!("unknown tensor {name}")))?.shape.clone())
}

fn linear_f16_out_in_file(file: &CellmFile, x: &[f32], name: &str, out_dim: usize, in_dim: usize, out: &mut [f32]) -> Result<(), CoreError> {
    let w = tensor_f16_file(file, name)?;
    for j in 0..out_dim {
        let row = &w[j * in_dim..(j+1)*in_dim];
        let mut acc = 0.0f32;
        for i in 0..in_dim { acc += x[i] * f16::from_bits(row[i]).to_f32(); }
        out[j] = acc;
    }
    Ok(())
}

fn unpack_i4(packed: &[u8], idx: usize) -> f32 {
    let b = packed[idx / 2];
    let n = if idx % 2 == 0 { b & 0xf } else { b >> 4 };
    (n as i8 - 8) as f32
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct QwenGraphState {
    pub device: metal::Device,
    pub queue: metal::CommandQueue,
    pub ops: MetalOps,
    pub weights: HashMap<String, metal::Buffer>,
    pub x_buf: metal::Buffer,
    pub x_norm_buf: metal::Buffer,
    pub q_buf: metal::Buffer,
    pub k_buf: metal::Buffer,
    pub v_buf: metal::Buffer,
    pub attn_out_buf: metal::Buffer,
    pub mlp_in_buf: metal::Buffer,
    pub gate_buf: metal::Buffer,
    pub up_buf: metal::Buffer,
    pub down_buf: metal::Buffer,
    pub logits_buf: metal::Buffer,
    pub bases_buf: Option<metal::Buffer>,
    pub bases_capacity: usize,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl QwenGraphState {
    pub fn new(hidden: usize, heads: usize, kv_heads: usize, vocab: usize, inter: usize, ops: MetalOps) -> Result<Self, CoreError> {
        let dev = ops.device.clone();
        let make = |l: usize| dev.new_buffer((l * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let q_buf = make(hidden);
        let k_buf = make(kv_heads * (hidden/heads));
        let v_buf = make(kv_heads * (hidden/heads));
        let x_buf = make(hidden);
        let x_norm_buf = make(hidden);
        let attn_out_buf = make(hidden);
        let mlp_in_buf = make(hidden);
        let gate_buf = make(inter);
        let up_buf = make(inter);
        let down_buf = make(hidden);
        let logits_buf = make(vocab);

        Ok(Self {
            device: dev, queue: ops.queue.clone(), ops, weights: HashMap::new(),
            x_buf, x_norm_buf, q_buf, k_buf, v_buf,
            attn_out_buf, mlp_in_buf, gate_buf, up_buf, down_buf,
            logits_buf, bases_buf: None, bases_capacity: 0       })
    }
    pub fn preload_weight(&mut self, name: String, bytes: &[u8]) {
        let b = self.device.new_buffer_with_data(bytes.as_ptr() as *const _, bytes.len() as u64, metal::MTLResourceOptions::StorageModeShared);
        self.weights.insert(name, b);
    }
    fn get_w(&self, name: &str) -> &metal::Buffer {
        match self.try_get_w(name) {
            Some(b) => b,
            None => {
                let keys: Vec<_> = self.weights.keys().collect();
                panic!("weight {name} not found. Available keys (up to 5): {:?}", &keys[0..5.min(keys.len())]);
            }
        }
    }
    fn try_get_w(&self, name: &str) -> Option<&metal::Buffer> {
        if let Some(b) = self.weights.get(name) { return Some(b); }
        let stem = if let Some(rest) = name.strip_prefix("language_model.model.") { rest }
                   else if let Some(rest) = name.strip_prefix("model.language_model.") { rest }
                   else if let Some(rest) = name.strip_prefix("model.") { rest }
                   else { name };
        let candidates = [
            format!("model.{stem}"),
            format!("language_model.model.{stem}"),
            format!("model.language_model.{stem}"),
        ];
        for c in candidates {
            if let Some(b) = self.weights.get(&c) { return Some(b); }
        }
        None
    }
    pub fn step_fused(&mut self, x_in: &[f32], cfg: &ModelConfig, prefix: &str, kv_cache: &mut KVCache, page_table: &PageTable, pos: usize, off: usize, bid: u32, rotary: f32, offset: bool, logits: bool) -> Result<Option<Vec<f32>>, CoreError> {
        let h = cfg.hidden_size; let nh = cfg.num_attention_heads; let nkv = cfg.num_key_value_heads; let hd = h/nh;
        unsafe { std::ptr::copy_nonoverlapping(x_in.as_ptr(), self.x_buf.contents() as *mut f32, h); }
        let cv = kv_cache.view_mut();
        let total = pos + 1;
        if self.bases_capacity < total * cfg.num_hidden_layers {
            self.bases_capacity = total * cfg.num_hidden_layers;
            self.bases_buf = Some(self.device.new_buffer((self.bases_capacity * 4) as u64, metal::MTLResourceOptions::StorageModeShared));
        }
        let mut bases = Vec::with_capacity(total * cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            for t in 0..total {
                let b = page_table.block_for_token(t).map_err(|e| CoreError::Backend(e.to_string()))?;
                let o = page_table.offset_in_block(t).map_err(|e| CoreError::Backend(e.to_string()))?;
                bases.push(cv.layout.token_base_elem(b, l, o)? as u32);
            }
        }
        if let Some(bb) = &self.bases_buf { unsafe { std::ptr::copy_nonoverlapping(bases.as_ptr(), bb.contents() as *mut u32, bases.len()); } }
        let bb = self.bases_buf.as_ref().unwrap();
        let cb = self.queue.new_command_buffer(); let enc = cb.new_compute_command_encoder();
        for l in 0..cfg.num_hidden_layers {
            let pref = format!("{prefix}model.layers.{l}");
            self.ops.encode_rms_norm_f16w(enc, &self.x_buf, self.get_w(&format!("{pref}.input_layernorm.weight")), &self.x_norm_buf, h, cfg.rms_norm_eps, offset);
            let (wq, wk, wv) = (self.get_w(&format!("{pref}.self_attn.q_proj.weight")), self.get_w(&format!("{pref}.self_attn.k_proj.weight")), self.get_w(&format!("{pref}.self_attn.v_proj.weight")));
            let (bq, bk, bv) = (self.try_get_w(&format!("{pref}.self_attn.q_proj.bias")), self.try_get_w(&format!("{pref}.self_attn.k_proj.bias")), self.try_get_w(&format!("{pref}.self_attn.v_proj.bias")));
            let (sq, sk, sv) = (self.try_get_w(&format!("{pref}.self_attn.q_proj.qscale")), self.try_get_w(&format!("{pref}.self_attn.k_proj.qscale")), self.try_get_w(&format!("{pref}.self_attn.v_proj.qscale")));
            if let (Some(q), Some(k), Some(v)) = (sq, sk, sv) { self.ops.encode_qkv_i8_bias(enc, wq, q, wk, k, wv, v, &self.x_norm_buf, bq, bk, bv, &self.q_buf, &self.k_buf, &self.v_buf, h, nkv * hd, h); }
            else { self.ops.encode_qkv_f16_bias(enc, wq, wk, wv, &self.x_norm_buf, bq, bk, bv, &self.q_buf, &self.k_buf, &self.v_buf, h, nkv * hd, h); }
            self.ops.encode_rope_half_f32(enc, &self.q_buf, nh, hd, (hd as f32 * rotary) as usize, pos, cfg.rope_theta);
            self.ops.encode_rope_half_f32(enc, &self.k_buf, nkv, hd, (hd as f32 * rotary) as usize, pos, cfg.rope_theta);
            let km = kv_cache.storage().as_any().downcast_ref::<cellm_cache::kvcache::MetalKvStorage>().unwrap();
            km.encode_write_token_f32(enc, kv_cache.layout().token_base_elem(bid, l, off)?, &self.k_buf, &self.v_buf, nkv * hd);
            km.encode_attention(enc, bb, (l * total * 4) as u64, &self.q_buf, &self.attn_out_buf, total as u32, nh as u32, nkv as u32, hd as u32, None);
            let wo = self.get_w(&format!("{pref}.self_attn.o_proj.weight")); let bo = self.try_get_w(&format!("{pref}.self_attn.o_proj.bias"));
            if let Some(so) = self.try_get_w(&format!("{pref}.self_attn.o_proj.qscale")) { self.ops.encode_mv_i8_bias(enc, wo, so, &self.attn_out_buf, bo, &self.mlp_in_buf, h, h); }
            else { self.ops.encode_mv_f16_bias(enc, wo, &self.attn_out_buf, bo, &self.mlp_in_buf, h, h); }
            self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.mlp_in_buf, h);
            self.ops.encode_rms_norm_f16w(enc, &self.x_buf, self.get_w(&format!("{pref}.post_attention_layernorm.weight")), &self.x_norm_buf, h, cfg.rms_norm_eps, offset);
            let (wg, wu) = (self.get_w(&format!("{pref}.mlp.gate_proj.weight")), self.get_w(&format!("{pref}.mlp.up_proj.weight")));
            let (bg, bu) = (self.try_get_w(&format!("{pref}.mlp.gate_proj.bias")), self.try_get_w(&format!("{pref}.mlp.up_proj.bias")));
            let (sg, su) = (self.try_get_w(&format!("{pref}.mlp.gate_proj.qscale")), self.try_get_w(&format!("{pref}.mlp.up_proj.qscale")));
            if let (Some(g), Some(u)) = (sg, su) { self.ops.encode_mv_i8_bias(enc, wg, g, &self.x_norm_buf, bg, &self.gate_buf, cfg.intermediate_size, h); self.ops.encode_mv_i8_bias(enc, wu, u, &self.x_norm_buf, bu, &self.up_buf, cfg.intermediate_size, h); }
            else { self.ops.encode_mv_f16_bias(enc, wg, &self.x_norm_buf, bg, &self.gate_buf, cfg.intermediate_size, h); self.ops.encode_mv_f16_bias(enc, wu, &self.x_norm_buf, bu, &self.up_buf, cfg.intermediate_size, h); }
            self.ops.encode_silu_mul_f32_inplace(enc, &self.gate_buf, &self.up_buf, cfg.intermediate_size);
            let wd = self.get_w(&format!("{pref}.mlp.down_proj.weight")); let bd = self.try_get_w(&format!("{pref}.mlp.down_proj.bias"));
            if let Some(sd) = self.try_get_w(&format!("{pref}.mlp.down_proj.qscale")) { self.ops.encode_mv_i8_bias(enc, wd, sd, &self.gate_buf, bd, &self.down_buf, h, cfg.intermediate_size); }
            else { self.ops.encode_mv_f16_bias(enc, wd, &self.gate_buf, bd, &self.down_buf, h, cfg.intermediate_size); }
            self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.down_buf, h);
        }
        self.ops.encode_rms_norm_f16w(enc, &self.x_buf, self.get_w(&format!("{prefix}model.norm.weight")), &self.x_norm_buf, h, cfg.rms_norm_eps, offset);
        if logits {
            let (wl, sl) = if let Some(wl) = self.try_get_w(&format!("{prefix}lm_head.weight")) {
                (wl, self.try_get_w(&format!("{prefix}lm_head.weight.qscale")))
            } else {
                let name = format!("{prefix}model.embed_tokens.weight");
                (self.get_w(&name), self.try_get_w(&format!("{name}.qscale")))
            };

            if let Some(s) = sl {
                self.ops.encode_mv_i8(enc, wl, s, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, h);
            } else {
                self.ops.encode_mv_f16(enc, wl, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, h);
            }
        }
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        if !logits { return Ok(None); }
        let mut out = vec![0.0f32; cfg.vocab_size];
        unsafe { std::ptr::copy_nonoverlapping(self.logits_buf.contents() as *const f32, out.as_mut_ptr(), cfg.vocab_size); }
        
        // Final NaN check to avoid panic in sort
        if out[0].is_nan() {
            return Err(CoreError::Backend("Metal output NaN".into()));
        }
        
        Ok(Some(out))
    }
}
#[cfg(any(target_os = "macos", target_os = "ios"))]
impl std::fmt::Debug for QwenGraphState { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_struct("QwenGraphState").finish() } }
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct QwenGraphState;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl std::fmt::Debug for QwenGraphState { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_struct("QwenGraphState (No Metal)").finish() } }
