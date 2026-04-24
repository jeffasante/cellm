use std::collections::HashMap;
use std::sync::Mutex;
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
use metal::Buffer;

const QWEN_METAL_LINEAR_MAX_ELEMS: usize = 262_144;

/// Minimal text-only runner for Qwen3.5 checkpoints that contain both
/// `linear_attention` and `full_attention` layers.
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
    graph_state: Option<QwenGraphState>,
    tensor_prefix: String,
}

enum QwenLinearBackend {
    Cpu,
    Metal { ctx: MetalMatmul },
}

impl QwenRunner {
    pub fn load(path: &Path) -> Result<Self, CoreError> {
        let file = CellmFile::load(path)?;
        let h = file.header.clone();

        // Try to infer head_dim from k_proj tensor shape if not explicitly stored
        let head_dim = h.head_dim.unwrap_or_else(|| {
            // Infer from k_proj weight shape: [num_kv_heads * head_dim, hidden_dim]
            for t in &h.tensors {
                if t.name.contains("self_attn.k_proj.weight") && t.shape.len() == 2 {
                    let kv_dim = t.shape[0];
                    let kv_heads = h.num_kv_heads.max(1);
                    if kv_dim % kv_heads == 0 {
                        let inferred = kv_dim / kv_heads;
                        eprintln!("DEBUG: Inferred head_dim={} from k_proj shape {:?}, kv_heads={}", inferred, t.shape, kv_heads);
                        return inferred;
                    }
                }
            }
            let default = h.hidden_dim / h.num_heads;
            eprintln!("DEBUG: Using default head_dim={} (hidden/num_heads)", default);
            default
        });

        let cfg = ModelConfig {
            vocab_size: h.vocab_size,
            hidden_size: h.hidden_dim,
            num_hidden_layers: h.num_layers,
            num_attention_heads: h.num_heads,
            num_key_value_heads: h.num_kv_heads,
            head_dim,
            intermediate_size: h.intermediate_size,
            rms_norm_eps: h.rms_norm_eps,
            rope_theta: h.rope_theta,
        };
        eprintln!("DEBUG: Model config: hidden={}, heads={}, head_dim={}, attn_dim={}",
                  cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim,
                  cfg.num_attention_heads * cfg.head_dim);

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
            graph_state: None,
            tensor_prefix: "".to_string(),
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
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
                self.metal_strict = false;
                self.metal_ops = Some(mo.clone());
                
                let mut gs = QwenGraphState::new(
                    self.cfg.hidden_size,
                    self.cfg.num_attention_heads,
                    self.cfg.num_key_value_heads,
                    self.cfg.head_dim,
                    self.cfg.vocab_size,
                    self.cfg.intermediate_size,
                    mo,
                ).expect("failed to create graph state");
                for (name, bytes) in self.file.all_tensors() {
                    if let Some(t) = self.file.tensor_index(name) {
                         gs.tensor_dtypes.insert(name.clone(), t.dtype.clone());
                    }
                    if name.ends_with(".bias") {
                        // Bias tensors are stored as f16 but the GPU shaders read them as float32.
                        // Convert f16 -> f32 before uploading.
                        let f16_data: &[u16] = bytemuck::cast_slice(bytes);
                        let f32_data: Vec<f32> = f16_data.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
                        let f32_bytes = bytemuck::cast_slice::<f32, u8>(&f32_data);
                        gs.preload_weight(name.clone(), f32_bytes);
                    } else {
                        gs.preload_weight(name.clone(), bytes);
                    }
                }
                self.graph_state = Some(gs);
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
        let mut x_all = vec![0.0f32; tokens.len() * self.cfg.hidden_size];
        for (i, &tok) in tokens.iter().enumerate() {
            self.embed_token(tok, &mut x_all[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size])?;
            if start_pos + i == page_table.token_count() {
                page_table.append_token(kv_cache.allocator_mut()).map_err(|e| CoreError::Backend(format!("qwen prefill append_token failed: {e}")))?;
            }
        }

        if self.metal_ops.is_some() && self.metal_strict {
            if let Some(graph_state) = &mut self.graph_state {
                let prefix = if has_tensor_file(&self.file, "language_model.model.embed_tokens.weight") { "language_model." } else { "" };
                graph_state.prefill_fused(&x_all, &self.cfg, prefix, kv_cache, page_table, start_pos, self.partial_rotary_factor, self.rmsnorm_weight_is_offset)?;
                return Ok(());
            }
        }

        for i in 0..tokens.len() {
            let pos = start_pos + i;
            let x = &x_all[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size];
            self.step_inner(x, pos, page_table, kv_cache, false)?;
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
        let head_dim = cfg.head_dim;
        let attn_dim = n_heads * head_dim;
        let rotary_dim = ((head_dim as f32) * self.partial_rotary_factor) as usize;

        // Ensure pagetable covers this token position.
        if pos == page_table.token_count() {
            page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                CoreError::Backend(format!("qwen step: page_table append_token failed: {e}"))
            })?;
        }

        let block_id = page_table.block_for_token(pos).map_err(|e| {
            CoreError::Backend(format!("qwen step: page_table block_for_token failed: {e}"))
        })? as u32;
        let token_off = page_table.offset_in_block(pos).map_err(|e| {
            CoreError::Backend(format!("qwen step: page_table offset_in_block failed: {e}"))
        })?;

        let prefix = if has_tensor_file(&self.file, "language_model.model.embed_tokens.weight") { "language_model." } else { "" };

        if self.metal_ops.is_some() && self.metal_strict {
            if let Some(graph_state) = &mut self.graph_state {
                let res = graph_state.step_fused(
                    &x0,
                    &cfg,
                    prefix,
                    kv_cache,
                    page_table,
                    pos,
                    token_off,
                    block_id as u32,
                    self.partial_rotary_factor,
                    self.rmsnorm_weight_is_offset,
                    return_logits,
                )?;
                return Ok(res.unwrap_or_else(Vec::new));
            }
        }

        let mut x = x0.to_vec();

        let mut x_norm = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; attn_dim];
        let mut k = vec![0.0f32; n_kv_heads * head_dim];
        let mut v = vec![0.0f32; n_kv_heads * head_dim];
        let mut attn_out = vec![0.0f32; attn_dim];
        let mut attn_proj = vec![0.0f32; hidden];
        let mut mlp_in = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; cfg.intermediate_size];
        let mut up = vec![0.0f32; cfg.intermediate_size];
        let mut down = vec![0.0f32; hidden];

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
                if self.rmsnorm_weight_is_offset { add_one_inplace(&mut in_norm_w); }
                rms_norm_f32(&x, &in_norm_w, cfg.rms_norm_eps, &mut x_norm);
            }


            match self.layer_kinds[layer] {
                LayerKind::FullAttention => {
                    let (q_name, k_name, v_name) = self.resolve_qkv_names(layer, prefix);
                    let fused_qkv = self.linear_qkv_f16_out_in(&x_norm, &q_name, attn_dim, &k_name, n_kv_heads * head_dim, &v_name, n_kv_heads * head_dim, hidden, &mut q, &mut k, &mut v)?;
                    if !fused_qkv {
                        self.linear_f16_out_in(&x_norm, &q_name, attn_dim, hidden, &mut q)?;
                        self.linear_f16_out_in(&x_norm, &k_name, n_kv_heads * head_dim, hidden, &mut k)?;
                        self.linear_f16_out_in(&x_norm, &v_name, n_kv_heads * head_dim, hidden, &mut v)?;
                    }
                    self.add_projection_bias_if_present(&q_name, &mut q)?;
                    self.add_projection_bias_if_present(&k_name, &mut k)?;
                    self.add_projection_bias_if_present(&v_name, &mut v)?;

                    // QK Norm (Head-tied or Per-head)
                    let q_norm_w_name = format!("{prefix}model.layers.{layer}.self_attn.q_norm.weight");
                    let k_norm_w_name = format!("{prefix}model.layers.{layer}.self_attn.k_norm.weight");
                    let has_qn = has_tensor_file(&self.file, &q_norm_w_name);
                    let has_kn = has_tensor_file(&self.file, &k_norm_w_name);
                    if has_qn || has_kn {
                        if has_qn {
                            let q_norm_shape = self.tensor_shape(&q_norm_w_name)?;
                            let q_norm_len = q_norm_shape.iter().product::<usize>();
                            let mut q_norm_w = vec![0.0f32; q_norm_len];
                            self.rmsnorm_weight(&q_norm_w_name, &mut q_norm_w)?;
                            
                            for hidx in 0..n_heads {
                                let start = hidx * head_dim;
                                let end = start + head_dim;
                                let in_slice = q[start..end].to_vec();
                                let w_slice = if q_norm_len >= end { &q_norm_w[start..end] } else { &q_norm_w[..head_dim] };
                                rms_norm_f32(&in_slice, w_slice, cfg.rms_norm_eps, &mut q[start..end]);
                            }
                        }
                        if has_kn {
                            let k_norm_shape = self.tensor_shape(&k_norm_w_name)?;
                            let k_norm_len = k_norm_shape.iter().product::<usize>();
                            let mut k_norm_w = vec![0.0f32; k_norm_len];
                            self.rmsnorm_weight(&k_norm_w_name, &mut k_norm_w)?;

                            for hidx in 0..n_kv_heads {
                                let start = hidx * head_dim;
                                let end = start + head_dim;
                                let in_slice = k[start..end].to_vec();
                                let w_slice = if k_norm_len >= end { &k_norm_w[start..end] } else { &k_norm_w[..head_dim] };
                                rms_norm_f32(&in_slice, w_slice, cfg.rms_norm_eps, &mut k[start..end]);
                            }
                        }
                    }

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
                    let mut gather_bases = Vec::with_capacity(seq);
                    for tpos in 0..seq {
                        let b = page_table.block_for_token(tpos).map_err(|e| CoreError::Backend(e.to_string()))?;
                        let o = page_table.offset_in_block(tpos).map_err(|e| CoreError::Backend(e.to_string()))?;
                        gather_bases.push(cr.layout.token_base_elem(b, layer, o)?);
                    }
                    cr.attention_single_token_gqa_from_bases(
                        &gather_bases, 
                        &q, 
                        n_heads, 
                        n_kv_heads, 
                        head_dim, 
                        None, // attn_scale
                        None, // soft_cap
                        &mut attn_out
                    )?;

                    self.linear_f16_out_in(&attn_out, &format!("{prefix}model.layers.{layer}.self_attn.o_proj.weight"), hidden, attn_dim, &mut attn_proj)?;
                    self.add_projection_bias_if_present(&format!("{prefix}model.layers.{layer}.self_attn.o_proj.weight"), &mut attn_proj)?;

                    for i in 0..hidden { x[i] += attn_proj[i]; }
                }
                LayerKind::LinearAttention => {}
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
                if self.rmsnorm_weight_is_offset { add_one_inplace(&mut post_norm_w); }
                rms_norm_f32(&x, &post_norm_w, cfg.rms_norm_eps, &mut mlp_in);
            }

            // MLP
            self.linear_f16_out_in(&mlp_in, &format!("{prefix}model.layers.{layer}.mlp.gate_proj.weight"), cfg.intermediate_size, hidden, &mut gate)?;
            self.linear_f16_out_in(&mlp_in, &format!("{prefix}model.layers.{layer}.mlp.up_proj.weight"), cfg.intermediate_size, hidden, &mut up)?;

            use rayon::prelude::*;
            gate.par_iter_mut().enumerate().for_each(|(i, g)| {
                let s = 1.0 / (1.0 + (-*g).exp());
                *g = (*g * s) * up[i];
            });

            self.linear_f16_out_in(&gate, &format!("{prefix}model.layers.{layer}.mlp.down_proj.weight"), hidden, cfg.intermediate_size, &mut down)?;
            for i in 0..hidden { x[i] += down[i]; }

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
            if self.rmsnorm_weight_is_offset { add_one_inplace(&mut norm_w); }
            rms_norm_f32(&x, &norm_w, cfg.rms_norm_eps, &mut x_final);
        }

        if !return_logits { return Ok(vec![]); }

        // Logits
        let embed_name = self.resolve_tensor_name("language_model.model.embed_tokens.weight");
        let lm_head_name = if has_tensor_file(&self.file, "lm_head.weight") { Some(self.resolve_tensor_name("lm_head.weight")) } else { None };
        let weight_name = lm_head_name.as_deref().unwrap_or(&embed_name);
        let vocab = cfg.vocab_size;

        println!("DEBUG: vocab={}, hidden={}, weight_name={}, tensor_len={}", vocab, hidden, weight_name, self.tensor_f16(weight_name)?.len());
        let mut logits = vec![0.0f32; vocab];
        let dtype = self.file.tensor_index(weight_name).unwrap().dtype.clone();
        
        match dtype.as_str() {
            "f16" => {
                let w = self.tensor_f16(weight_name)?.to_vec();
                if let Some(ops) = &mut self.metal_ops {
                    ops.logits_f16(&x_final, &w, vocab, hidden, weight_name, &mut logits).map_err(|e| CoreError::Backend(e.to_string()))?;
                } else {
                    cellm_kernels::cpu_kernels::matmul_f16_f32(&w, vocab, hidden, &x_final, &mut logits);
                }
            }
            "i8" => {
                let w = self.tensor_i8(weight_name)?.to_vec();
                let s = self.tensor_f16(&format!("{weight_name}.qscale"))?.to_vec();
                if let Some(ops) = &mut self.metal_ops {
                    ops.logits_i8(&x_final, &w, &s, vocab, hidden, weight_name, &mut logits).map_err(|e| CoreError::Backend(e.to_string()))?;
                } else {
                    cellm_kernels::cpu_kernels::matmul_i8_f32(&w, &s, vocab, hidden, &x_final, &mut logits);
                }
            }
            "i4" => {
                let w = self.tensor_u8(weight_name)?.to_vec();
                let s = self.tensor_f16(&format!("{weight_name}.qscale"))?.to_vec();
                let gs = if s.len() > vocab { hidden / (s.len() / vocab) } else { hidden };
                if let Some(ops) = &mut self.metal_ops {
                    ops.logits_i4(&x_final, &w, &s, vocab, hidden, gs, weight_name, &mut logits).map_err(|e| CoreError::Backend(e.to_string()))?;
                } else {
                    cellm_kernels::cpu_kernels::matmul_i4_f32(&w, &s, vocab, hidden, gs, &x_final, &mut logits);
                }
            }
            "q1_0_g128" => {
                let w = self.file.tensor_bytes(&weight_name)?.to_vec();
                if let Some(ops) = &mut self.metal_ops {
                    ops.logits_q1(&x_final, &w, vocab, hidden, weight_name, &mut logits).map_err(|e| CoreError::Backend(e.to_string()))?;
                } else {
                    for vid in 0..vocab { logits[vid] = self.dot_row(weight_name, vid, hidden, &x_final)?; }
                }
            }
            _ => return Err(CoreError::Backend(format!("unsupported lm_head dtype: {dtype}").into())),
        }

        Ok(logits)
    }

    fn embed_token(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        let prefix = if has_tensor_file(&self.file, "language_model.model.embed_tokens.weight") { "language_model." } else { "" };
        let weight_name = format!("{prefix}model.embed_tokens.weight");
        let hidden = out.len();
        let resolved = self.resolve_tensor_name(&weight_name);
        let t_meta = self.file.tensor_index(&resolved).ok_or_else(|| CoreError::Backend(format!("unknown tensor {weight_name}")))?;
        let vocab = self.cfg.vocab_size;
        let t = (token as usize) % vocab;
        match t_meta.dtype.as_str() {
            "f16" => {
                let embed = self.tensor_f16(&weight_name)?;
                let row = &embed[t * hidden..(t + 1) * hidden];
                for i in 0..hidden { out[i] = f16::from_bits(row[i]).to_f32(); }
            }
            "i8" => {
                let embed = self.tensor_i8(&weight_name)?;
                let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                let scale = f16::from_bits(scales[t]).to_f32();
                let row = &embed[t * hidden..(t + 1) * hidden];
                for i in 0..hidden { out[i] = (row[i] as f32) * scale; }
            }
            "q1_0_g128" => {
                let embed = self.file.tensor_bytes(&resolved)?;
                let row_stride = (hidden / 128) * 18;
                let row = &embed[t * row_stride..(t + 1) * row_stride];
                for i in 0..(hidden / 128) {
                    let block = &row[i * 18..(i + 1) * 18];
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let bits = &block[2..18];
                    for b in 0..16 {
                        let bb = bits[b];
                        for j in 0..8 {
                            let bit_val = if (bb & (1 << j)) != 0 { d } else { -d };
                            out[i * 128 + b * 8 + j] = bit_val;
                        }
                    }
                }
            }
            "i4" => {
                let embed = self.tensor_u8(&weight_name)?;
                let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                let scale = f16::from_bits(scales[t]).to_f32();
                let row_stride = hidden.div_ceil(2);
                let row = &embed[t * row_stride..(t + 1) * row_stride];
                for i in 0..hidden {
                    let byte = row[i / 2];
                    let n = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    let q = (n as i8) - 8;
                    out[i] = (q as f32) * scale;
                }
            }
            other => return Err(CoreError::Backend(format!("unsupported embed dtype for {weight_name}: {other}"))),
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
        if bytes.len() % 2 != 0 { return Err(CoreError::Backend(format!("tensor {name} nbytes not even"))); }
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
        let t = self.file.tensor_index(&resolved).ok_or_else(|| CoreError::Backend(format!("unknown tensor {name}")))?;
        Ok(t.shape.clone())
    }

    fn resolve_tensor_name(&self, name: &str) -> String { resolve_tensor_name_file(&self.file, name) }

    fn dot_row(&self, weight_name: &str, row_idx: usize, in_dim: usize, x: &[f32]) -> Result<f32, CoreError> {
        let _shape = self.tensor_shape(weight_name)?;
        let resolved = self.resolve_tensor_name(weight_name);
        let dtype = self.file.tensor_index(&resolved).ok_or_else(|| CoreError::Backend(format!("unknown tensor {weight_name}")))?.dtype.clone();
        match dtype.as_str() {
            "f16" => {
                let w = self.tensor_f16(weight_name)?;
                let row = &w[row_idx * in_dim..(row_idx + 1) * in_dim];
                let mut acc = 0.0f32;
                for i in 0..in_dim { acc += x[i] * f16::from_bits(row[i]).to_f32(); }
                Ok(acc)
            }
            "i8" => {
                let w = self.tensor_i8(weight_name)?;
                let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                let scale = f16::from_bits(scales[row_idx]).to_f32();
                let row = &w[row_idx * in_dim..(row_idx + 1) * in_dim];
                let mut acc = 0.0f32;
                for i in 0..in_dim { acc += x[i] * (row[i] as f32) * scale; }
                Ok(acc)
            }
            "i4" => {
                let bytes = self.file.tensor_bytes(&resolved)?;
                let row_stride = (in_dim + 1) / 2;
                let row = &bytes[row_idx * row_stride..(row_idx + 1) * row_stride];
                let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                
                let out_dim = self.tensor_shape(weight_name)?[0];
                let n_scales = scales.len();
                let scales_per_row = n_scales / out_dim;
                let group_size = if scales_per_row > 0 { in_dim / scales_per_row } else { in_dim };
                let row_scales = &scales[row_idx * scales_per_row..(row_idx + 1) * scales_per_row];

                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    let byte = row[i / 2];
                    let n = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    let q = (n as i8) - 8;
                    let scale = f16::from_bits(row_scales[i / group_size]).to_f32();
                    acc += (q as f32) * scale * x[i];
                }
                Ok(acc)
            }
            "q1_0_g128" => {
                let bytes = self.file.tensor_bytes(&resolved)?;
                let row_stride = (in_dim / 128) * 18;
                let row = &bytes[row_idx * row_stride..(row_idx + 1) * row_stride];
                let mut acc = 0.0f32;
                for i in 0..(in_dim / 128) {
                    let block = &row[i * 18..(i + 1) * 18];
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let bits = &block[2..18];
                    for b in 0..16 {
                        let bb = bits[b];
                        for j in 0..8 {
                            let bit_val = if (bb & (1 << j)) != 0 { d } else { -d };
                            acc += bit_val * x[i * 128 + b * 8 + j];
                        }
                    }
                }
                Ok(acc)
            }
            _ => Err(CoreError::Backend(format!("unsupported dot_row dtype for {weight_name}: {dtype}"))),
        }
    }

    fn rmsnorm_weight(&self, name: &str, out: &mut [f32]) -> Result<(), CoreError> {
        let w = self.tensor_f16(name)?;
        for i in 0..out.len() { out[i] = f16::from_bits(w[i]).to_f32(); }
        Ok(())
    }

    fn linear_f16_out_in(&mut self, x: &[f32], weight_name: &str, out_dim: usize, in_dim: usize, out: &mut [f32]) -> Result<(), CoreError> {
        let resolved = self.resolve_tensor_name(weight_name);
        let dtype = self.file.tensor_index(&resolved).unwrap().dtype.clone();
        if self.metal_ops.is_some() {
            match dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16(weight_name)?;
                    let w_ptr = w.as_ptr();
                    let w_len = w.len();
                    self.metal_ops.as_mut().unwrap().logits_f16(x, unsafe { std::slice::from_raw_parts(w_ptr, w_len) }, out_dim, in_dim, &resolved, out).map_err(|e| CoreError::Backend(e.to_string()))?;
                    return Ok(());
                }
                "i8" => {
                    let w = self.tensor_i8(weight_name)?;
                    let s = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                    let w_ptr = w.as_ptr();
                    let w_len = w.len();
                    let s_ptr = s.as_ptr();
                    let s_len = s.len();
                    self.metal_ops.as_mut().unwrap().logits_i8(x, unsafe { std::slice::from_raw_parts(w_ptr, w_len) }, unsafe { std::slice::from_raw_parts(s_ptr, s_len) }, out_dim, in_dim, &resolved, out).map_err(|e| CoreError::Backend(e.to_string()))?;
                    return Ok(());
                }
                "i4" => {
                    let w = self.tensor_u8(weight_name)?.to_vec();
                    let s = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                    let gs = if s.len() > out_dim { in_dim / (s.len() / out_dim) } else { in_dim };
                    let s_vec = s.to_vec();
                    self.metal_ops.as_mut().unwrap().logits_i4(x, &w, &s_vec, out_dim, in_dim, gs, &resolved, out).map_err(|e| CoreError::Backend(e.to_string()))?;
                    return Ok(());
                }
                "q1_0_g128" => {
                    let w = self.file.tensor_bytes(&resolved)?;
                    self.metal_ops.as_mut().unwrap().logits_q1(x, w, out_dim, in_dim, &resolved, out).map_err(|e| CoreError::Backend(e.to_string()))?;
                    return Ok(());
                }
                _ => {}
            }
        }
        match dtype.as_str() {
            "f16" => cellm_kernels::cpu_kernels::matmul_f16_f32(self.tensor_f16(weight_name)?, out_dim, in_dim, x, out),
            "i8" => cellm_kernels::cpu_kernels::matmul_i8_f32(self.tensor_i8(weight_name)?, self.tensor_f16(&format!("{weight_name}.qscale"))?, out_dim, in_dim, x, out),
            "i4" => {
                let s = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                let gs = if s.len() > out_dim { in_dim / (s.len() / out_dim) } else { in_dim };

                if self.metal_ops.is_some() {
                    let w = self.tensor_u8(weight_name)?.to_vec();
                    let s_vec = s.to_vec();
                    self.metal_ops.as_mut().unwrap().logits_i4(x, &w, &s_vec, out_dim, in_dim, gs, &resolved, out).map_err(|e| CoreError::Backend(e.to_string()))?;
                } else {
                    cellm_kernels::cpu_kernels::matmul_i4_f32(self.tensor_u8(weight_name)?, s, out_dim, in_dim, gs, x, out);
                }
            }
            "q1_0_g128" => {
                for r in 0..out_dim {
                    out[r] = self.dot_row(weight_name, r, in_dim, x)?;
                }
            }
            _ => return Err(CoreError::Backend(format!("linear fallback unsupported dtype {dtype}"))),
        }
        Ok(())
    }

    fn linear_qkv_f16_out_in(&mut self, x: &[f32], qn: &str, qd: usize, kn: &str, kd: usize, vn: &str, vd: usize, ind: usize, qo: &mut [f32], ko: &mut [f32], vo: &mut [f32]) -> Result<bool, CoreError> {
        if self.metal_ops.is_none() { return Ok(false); }
        let qr = self.resolve_tensor_name(qn);
        let kr = self.resolve_tensor_name(kn);
        let vr = self.resolve_tensor_name(vn);
        if self.file.tensor_index(&qr).unwrap().dtype != "f16" { return Ok(false); }
        let wq = self.tensor_f16(qn)?;
        let wk = self.tensor_f16(kn)?;
        let wv = self.tensor_f16(vn)?;
        let wq_ptr = wq.as_ptr(); let wq_len = wq.len();
        let wk_ptr = wk.as_ptr(); let wk_len = wk.len();
        let wv_ptr = wv.as_ptr(); let wv_len = wv.len();
        
        self.metal_ops.as_mut().unwrap().logits_qkv_f16(x, 
            unsafe { std::slice::from_raw_parts(wq_ptr, wq_len) }, 
            unsafe { std::slice::from_raw_parts(wk_ptr, wk_len) }, 
            unsafe { std::slice::from_raw_parts(wv_ptr, wv_len) }, 
            qd, kd, vd, ind, &qr, &kr, &vr, qo, ko, vo).map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(true)
    }

    fn add_projection_bias_if_present(&self, weight_name: &str, out: &mut [f32]) -> Result<(), CoreError> {
        let Some(stem) = weight_name.strip_suffix(".weight") else { return Ok(()); };
        let b_name = format!("{stem}.bias");
        if !has_tensor_file(&self.file, &b_name) { return Ok(()); }
        let b = self.tensor_f16(&b_name)?;
        for (o, &bv) in out.iter_mut().zip(b.iter()) { *o += f16::from_bits(bv).to_f32(); }
        Ok(())
    }
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
    pub q_norm_buf: metal::Buffer,
    pub k_norm_buf: metal::Buffer,
    pub tensor_dtypes: HashMap<String, String>,
    pub weight_cache: Mutex<HashMap<String, Option<metal::Buffer>>>,
    pub bases_buf: Option<metal::Buffer>,
    pub bases_capacity: usize,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl QwenGraphState {
    pub fn new(hidden: usize, heads: usize, kv_heads: usize, head_dim: usize, vocab: usize, inter: usize, ops: MetalOps) -> Result<Self, CoreError> {
        let dev = ops.device.clone();
        let attn_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let q_buf = dev.new_buffer((attn_dim * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let k_buf = dev.new_buffer((kv_dim * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let v_buf = dev.new_buffer((kv_dim * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let x_buf = dev.new_buffer((hidden * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let x_norm_buf = dev.new_buffer((hidden * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let attn_out_buf = dev.new_buffer((attn_dim * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let mlp_in_buf = dev.new_buffer((hidden * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let gate_buf = dev.new_buffer((inter * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let up_buf = dev.new_buffer((inter * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let down_buf = dev.new_buffer((hidden * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let logits_buf = dev.new_buffer((vocab * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let q_norm_buf = dev.new_buffer((attn_dim * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let k_norm_buf = dev.new_buffer((kv_dim * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

        Ok(Self {
            device: dev, queue: ops.queue.clone(), ops, weights: HashMap::new(),
            weight_cache: Mutex::new(HashMap::new()),
            x_buf, x_norm_buf, q_buf, k_buf, v_buf,
            attn_out_buf, mlp_in_buf, gate_buf, up_buf, down_buf,
            logits_buf, q_norm_buf, k_norm_buf, 
            tensor_dtypes: HashMap::new(),
            bases_buf: None, bases_capacity: 0       })
    }

    fn get_w_cached(&self, name: &str) -> &metal::Buffer {
        let mut cache = self.weight_cache.lock().unwrap();
        if let Some(Some(b)) = cache.get(name) { return unsafe { std::mem::transmute::<&metal::Buffer, &metal::Buffer>(b) }; }
        let b = self.try_get_w(name).cloned();
        cache.insert(name.to_string(), b.clone());
        if let Some(_) = b {
             return self.weights.get(&self.resolve_tensor_name(name)).expect("Weight disappeared");
        }
        panic!("Weight not found: {}", name);
    }

    fn try_get_w_cached(&self, name: &str) -> Option<&metal::Buffer> {
        let mut cache = self.weight_cache.lock().unwrap();
        if let Some(res) = cache.get(name) { return unsafe { std::mem::transmute::<Option<&metal::Buffer>, Option<&metal::Buffer>>(res.as_ref()) }; }
        let b = self.try_get_w(name).cloned();
        cache.insert(name.to_string(), b);
        self.try_get_w(name)
    }
    fn resolve_tensor_name(&self, name: &str) -> String {
        if self.weights.contains_key(name) { return name.to_string(); }
        // Try stripping known prefixes
        let stem = name.strip_prefix("language_model.model.").unwrap_or(
            name.strip_prefix("model.language_model.").unwrap_or(
                name.strip_prefix("language_model.").unwrap_or(
                    name.strip_prefix("model.").unwrap_or(name)
                )
            )
        );
        let candidates = [
            format!("{stem}"),
            format!("model.{stem}"),
            format!("language_model.model.{stem}"),
            format!("model.language_model.{stem}"),
            format!("language_model.{stem}"),
        ];
        for c in candidates { if self.weights.contains_key(&c) { return c; } }
        name.to_string()
    }

    pub fn preload_weight(&mut self, name: String, bytes: &[u8]) {
        self.weights.insert(name, self.device.new_buffer_with_data(bytes.as_ptr() as *const _, bytes.len() as u64, metal::MTLResourceOptions::StorageModeShared));
    }
    
    fn try_get_w(&self, name: &str) -> Option<&metal::Buffer> { self.weights.get(&self.resolve_tensor_name(name)) }

    fn get_dtype(&self, name: &str) -> String {
        self.tensor_dtypes.get(&self.resolve_tensor_name(name)).cloned().unwrap_or_default()
    }


    fn check_nan(&self, buf: &metal::BufferRef, n: usize, msg: &str) {
        let mut data = vec![0.0f32; n];
        unsafe { std::ptr::copy_nonoverlapping(buf.contents() as *const f32, data.as_mut_ptr(), n); }
        if data.iter().any(|v| v.is_nan()) {
            println!("CRITICAL: NaNs detected in {}!", msg);
        }
    }

    pub fn step_fused(&mut self, x_in: &[f32], cfg: &ModelConfig, prefix: &str, kv_cache: &mut KVCache, page_table: &PageTable, pos: usize, off: usize, bid: u32, rotary: f32, offset: bool, logits: bool) -> Result<Option<Vec<f32>>, CoreError> {
        let h = cfg.hidden_size;
        unsafe { std::ptr::copy_nonoverlapping(x_in.as_ptr(), self.x_buf.contents() as *mut f32, h); }
        let total = pos + 1;
        if self.bases_capacity < total * cfg.num_hidden_layers {
            self.bases_capacity = total * cfg.num_hidden_layers;
            self.bases_buf = Some(self.device.new_buffer((self.bases_capacity * 4) as u64, metal::MTLResourceOptions::StorageModeShared));
        }
        let mut bases = Vec::with_capacity(total * cfg.num_hidden_layers);
        let cv = kv_cache.view();
        for l in 0..cfg.num_hidden_layers {
            for t in 0..total {
                let b = page_table.block_for_token(t).map_err(|e| CoreError::Backend(e.to_string()))?;
                let o = page_table.offset_in_block(t).map_err(|e| CoreError::Backend(e.to_string()))?;
                bases.push(cv.layout.token_base_elem(b, l, o)? as u32);
            }
        }
        if let Some(bb) = &self.bases_buf { unsafe { std::ptr::copy_nonoverlapping(bases.as_ptr(), bb.contents() as *mut u32, bases.len()); } }
        let bb = self.bases_buf.as_ref().unwrap();

        let cb = self.queue.new_command_buffer();
        for l in 0..cfg.num_hidden_layers {
            self.encode_single_layer_efficient(cb, l, cfg, prefix, kv_cache, total, total, off, bid, rotary, offset, bb);
        }
        
        let enc_final = cb.new_compute_command_encoder();
        self.ops.encode_rms_norm_f16w(enc_final, &self.x_buf, self.get_w_cached(&format!("{prefix}model.norm.weight")), &self.x_norm_buf, h, cfg.rms_norm_eps, offset);
        enc_final.end_encoding();

        if logits {
            let enc_logits = cb.new_compute_command_encoder();
            let lm_head_name = format!("{prefix}lm_head.weight");
            let embed_name = format!("{prefix}model.embed_tokens.weight");
            let (target_name, wl, sl) = if self.tensor_dtypes.contains_key(&lm_head_name) {
                (lm_head_name.clone(), self.get_w_cached(&lm_head_name), self.try_get_w_cached(&format!("{lm_head_name}.qscale")))
            } else {
                (embed_name.clone(), self.get_w_cached(&embed_name), self.try_get_w_cached(&format!("{embed_name}.qscale")))
            };
            
            if let Some(s) = sl { self.ops.encode_mv_i8(enc_logits, wl, s, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, h); }
            else if self.get_dtype(&target_name) == "q1_0_g128" { self.ops.encode_mv_q1(enc_logits, wl, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, h); }
            else { self.ops.encode_mv_f16(enc_logits, wl, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, h); }
            enc_logits.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();

        if !logits { return Ok(None); }
        let mut out = vec![0.0f32; cfg.vocab_size];
        unsafe { std::ptr::copy_nonoverlapping(self.logits_buf.contents() as *const f32, out.as_mut_ptr(), cfg.vocab_size); }
        Ok(Some(out))
    }

    pub fn prefill_fused(&mut self, x_all: &[f32], cfg: &ModelConfig, prefix: &str, kv_cache: &mut KVCache, page_table: &PageTable, start_pos: usize, rotary: f32, offset: bool) -> Result<(), CoreError> {
        let h = cfg.hidden_size; let num_tokens = x_all.len() / h; if num_tokens == 0 { return Ok(()); }
        let max_total = start_pos + num_tokens;
        let mut all_bases = Vec::with_capacity(max_total * cfg.num_hidden_layers);
        let cv = kv_cache.view();
        for l in 0..cfg.num_hidden_layers {
            for t in 0..max_total {
                let b = page_table.block_for_token(t).unwrap();
                let o = page_table.offset_in_block(t).unwrap();
                all_bases.push(cv.layout.token_base_elem(b as u32, l, o).unwrap() as u32);
            }
        }
        let bases_buf = self.device.new_buffer_with_data(all_bases.as_ptr() as *const _, (all_bases.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        // Upload all token embeddings to GPU once so CBs can blit without CPU involvement.
        let x_all_buf = self.device.new_buffer_with_data(
            x_all.as_ptr() as *const _,
            (x_all.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        // Encode all per-token CBs without waiting between them. The Metal queue is FIFO
        // so the GPU executes them serially and correctly. The CPU can encode CB[i+1]
        // while the GPU is still executing CB[i], eliminating 151 round-trip stalls.
        let mut last_cb = None;
        for i in 0..num_tokens {
            let cb = self.queue.new_command_buffer();
            let pos = start_pos + i;
            let bid = page_table.block_for_token(pos).map_err(|e| CoreError::Backend(e.to_string()))?;
            let off = page_table.offset_in_block(pos).map_err(|e| CoreError::Backend(e.to_string()))?;
            // GPU-side blit: copy embedding i from x_all_buf into x_buf (no CPU write, no sync needed).
            let blit = cb.new_blit_command_encoder();
            blit.copy_from_buffer(&x_all_buf, (i * h * 4) as u64, &self.x_buf, 0, (h * 4) as u64);
            blit.end_encoding();
            for l in 0..cfg.num_hidden_layers {
                self.encode_single_layer_efficient(cb, l, cfg, prefix, kv_cache, start_pos + i + 1, max_total, off, bid as u32, rotary, offset, &bases_buf);
            }
            cb.commit();
            last_cb = Some(cb);
        }
        // Single sync point after all tokens are submitted.
        if let Some(cb) = last_cb { cb.wait_until_completed(); }
        self.queue.new_command_buffer().commit(); // sync
        Ok(())
    }

    fn encode_single_layer_efficient(&self, cb: &metal::CommandBufferRef, l: usize, cfg: &ModelConfig, prefix: &str, kv_cache: &mut KVCache, total_tokens_now: usize, stride: usize, off: usize, bid: u32, rotary: f32, offset: bool, bases_buf: &Buffer) {
        let h = cfg.hidden_size; let nh = cfg.num_attention_heads; let nkv = cfg.num_key_value_heads; let hd = cfg.head_dim;
        let pref = format!("{prefix}model.layers.{l}");
        let enc = cb.new_compute_command_encoder();
        self.ops.encode_rms_norm_f16w(enc, &self.x_buf, self.get_w_cached(&format!("{pref}.input_layernorm.weight")), &self.x_norm_buf, h, cfg.rms_norm_eps, offset);

        let wq = self.get_w_cached(&format!("{pref}.self_attn.q_proj.weight"));
        let wk = self.get_w_cached(&format!("{pref}.self_attn.k_proj.weight"));
        let wv = self.get_w_cached(&format!("{pref}.self_attn.v_proj.weight"));
        let bq = self.try_get_w_cached(&format!("{pref}.self_attn.q_proj.bias"));
        let bk = self.try_get_w_cached(&format!("{pref}.self_attn.k_proj.bias"));
        let bv = self.try_get_w_cached(&format!("{pref}.self_attn.v_proj.bias"));
        let (qs, ks, vs) = (self.try_get_w_cached(&format!("{pref}.self_attn.q_proj.weight.qscale")), self.try_get_w_cached(&format!("{pref}.self_attn.k_proj.weight.qscale")), self.try_get_w_cached(&format!("{pref}.self_attn.v_proj.weight.qscale")));
        let q_dtype = self.get_dtype(&format!("{pref}.self_attn.q_proj.weight"));
        
        if q_dtype == "q1_0_g128" {
            self.ops.encode_mv_q1(enc, wq, &self.x_norm_buf, &self.q_buf, nh * hd, h);
            self.ops.encode_mv_q1(enc, wk, &self.x_norm_buf, &self.k_buf, nkv * hd, h);
            self.ops.encode_mv_q1(enc, wv, &self.x_norm_buf, &self.v_buf, nkv * hd, h);
            if let Some(b) = bq { self.ops.encode_add_f32_inplace(enc, &self.q_buf, b, nh * hd); }
            if let Some(b) = bk { self.ops.encode_add_f32_inplace(enc, &self.k_buf, b, nkv * hd); }
            if let Some(b) = bv { self.ops.encode_add_f32_inplace(enc, &self.v_buf, b, nkv * hd); }
        } else if q_dtype == "i8" && qs.is_some() && ks.is_some() && vs.is_some() { 
            self.ops.encode_qkv_i8_bias(enc, wq, qs.unwrap(), wk, ks.unwrap(), wv, vs.unwrap(), &self.x_norm_buf, bq, bk, bv, &self.q_buf, &self.k_buf, &self.v_buf, nh * hd, nkv * hd, h); 
        } else { 
            // Force fallback for i4 to individual linear_f16_out_in (which now supports Metal i4)
            enc.end_encoding();
            return;
        }

        // QK Norm (Head-tied or Per-head)
        let q_norm_w = self.try_get_w_cached(&format!("{pref}.self_attn.q_norm.weight"));
        let k_norm_w = self.try_get_w_cached(&format!("{pref}.self_attn.k_norm.weight"));
        if let Some(qw) = q_norm_w {
            let qw_len = qw.length() / 2; // f16
            for hidx in 0..nh {
                let w_off = if qw_len >= (nh * hd) as u64 { (hidx * hd * 2) as u64 } else { 0 };
                self.ops.encode_rms_norm_f16w_at(enc, &self.q_buf, (hidx * hd * 4) as u64, qw, w_off, &self.q_buf, (hidx * hd * 4) as u64, hd, cfg.rms_norm_eps, offset);
            }
        }
        if let Some(kw) = k_norm_w {
            let kw_len = kw.length() / 2; // f16
            for hidx in 0..nkv {
                let w_off = if kw_len >= (nkv * hd) as u64 { (hidx * hd * 2) as u64 } else { 0 };
                self.ops.encode_rms_norm_f16w_at(enc, &self.k_buf, (hidx * hd * 4) as u64, kw, w_off, &self.k_buf, (hidx * hd * 4) as u64, hd, cfg.rms_norm_eps, offset);
            }
        }

        let km = kv_cache.storage().as_any().downcast_ref::<cellm_cache::kvcache::MetalKvStorage>().unwrap();
        self.ops.encode_rope_half_f32(enc, &self.q_buf, nh, hd, (hd as f32 * rotary) as usize, total_tokens_now - 1, cfg.rope_theta);
        self.ops.encode_rope_half_f32(enc, &self.k_buf, nkv, hd, (hd as f32 * rotary) as usize, total_tokens_now - 1, cfg.rope_theta);
        km.encode_write_token_f32(enc, kv_cache.layout().token_base_elem(bid, l, off).unwrap(), &self.k_buf, &self.v_buf, nkv * hd);

        km.encode_attention(
            enc, 
            bases_buf, 
            (l * stride * 4) as u64, 
            &self.q_buf, 
            &self.attn_out_buf, 
            total_tokens_now as u32, 
            nh as u32, 
            nkv as u32, 
            hd as u32, 
            None, // attn_scale
            None  // soft_cap
        );

        let wo = self.get_w_cached(&format!("{pref}.self_attn.o_proj.weight"));
        let bo = self.try_get_w_cached(&format!("{pref}.self_attn.o_proj.bias"));
        let so = self.try_get_w_cached(&format!("{pref}.self_attn.o_proj.weight.qscale")).or_else(|| self.try_get_w_cached(&format!("{pref}.self_attn.o_proj.qscale")));
        if self.get_dtype(&format!("{pref}.self_attn.o_proj.weight")) == "q1_0_g128" {
            self.ops.encode_mv_q1(enc, wo, &self.attn_out_buf, &self.mlp_in_buf, h, h);
            if let Some(b) = bo { self.ops.encode_add_f32_inplace(enc, &self.mlp_in_buf, b, h); }
        } else if let Some(s) = so { 
            self.ops.encode_mv_i8_bias(enc, wo, s, &self.attn_out_buf, bo, &self.mlp_in_buf, h, h); 
        } else { 
            self.ops.encode_mv_f16_bias(enc, wo, &self.attn_out_buf, bo, &self.mlp_in_buf, h, h); 
        }
        self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.mlp_in_buf, h);
        
        self.ops.encode_rms_norm_f16w(enc, &self.x_buf, self.get_w_cached(&format!("{pref}.post_attention_layernorm.weight")), &self.x_norm_buf, h, cfg.rms_norm_eps, offset);

        let wg = self.get_w_cached(&format!("{pref}.mlp.gate_proj.weight"));
        let wu = self.get_w_cached(&format!("{pref}.mlp.up_proj.weight"));
        let bg = self.try_get_w_cached(&format!("{pref}.mlp.gate_proj.bias"));
        let bu = self.try_get_w_cached(&format!("{pref}.mlp.up_proj.bias"));
        let sg = self.try_get_w_cached(&format!("{pref}.mlp.gate_proj.weight.qscale")).or_else(|| self.try_get_w_cached(&format!("{pref}.mlp.gate_proj.qscale")));
        let su = self.try_get_w_cached(&format!("{pref}.mlp.up_proj.weight.qscale")).or_else(|| self.try_get_w_cached(&format!("{pref}.mlp.up_proj.qscale")));

        if self.get_dtype(&format!("{pref}.mlp.gate_proj.weight")) == "q1_0_g128" {
            self.ops.encode_mv_q1(enc, wg, &self.x_norm_buf, &self.gate_buf, cfg.intermediate_size, h);
            self.ops.encode_mv_q1(enc, wu, &self.x_norm_buf, &self.up_buf, cfg.intermediate_size, h);
            if let Some(b) = bg { self.ops.encode_add_f32_inplace(enc, &self.gate_buf, b, cfg.intermediate_size); }
            if let Some(b) = bu { self.ops.encode_add_f32_inplace(enc, &self.up_buf, b, cfg.intermediate_size); }
        } else if let (Some(g), Some(u)) = (sg, su) { 
            self.ops.encode_mv_i8_bias(enc, wg, g, &self.x_norm_buf, bg, &self.gate_buf, cfg.intermediate_size, h); 
            self.ops.encode_mv_i8_bias(enc, wu, u, &self.x_norm_buf, bu, &self.up_buf, cfg.intermediate_size, h); 
        } else { 
            self.ops.encode_mv_f16_bias(enc, wg, &self.x_norm_buf, bg, &self.gate_buf, cfg.intermediate_size, h); 
            self.ops.encode_mv_f16_bias(enc, wu, &self.x_norm_buf, bu, &self.up_buf, cfg.intermediate_size, h); 
        }
        self.ops.encode_silu_mul_f32_inplace(enc, &self.gate_buf, &self.up_buf, cfg.intermediate_size);
        
        // Final Down Proj
        let wd = self.get_w_cached(&format!("{pref}.mlp.down_proj.weight"));
        let bd = self.try_get_w_cached(&format!("{pref}.mlp.down_proj.bias"));
        let sd = self.try_get_w_cached(&format!("{pref}.mlp.down_proj.weight.qscale")).or_else(|| self.try_get_w_cached(&format!("{pref}.mlp.down_proj.qscale")));
        if self.get_dtype(&format!("{pref}.mlp.down_proj.weight")) == "q1_0_g128" {
            self.ops.encode_mv_q1(enc, wd, &self.gate_buf, &self.down_buf, h, cfg.intermediate_size);
            if let Some(b) = bd { self.ops.encode_add_f32_inplace(enc, &self.down_buf, b, h); }
        } else if let Some(s) = sd { 
            self.ops.encode_mv_i8_bias(enc, wd, s, &self.gate_buf, bd, &self.down_buf, h, cfg.intermediate_size); 
        } else { 
            self.ops.encode_mv_f16_bias(enc, wd, &self.gate_buf, bd, &self.down_buf, h, cfg.intermediate_size); 
        }
        
        
        self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.down_buf, h);
        
        enc.end_encoding();
    }
}
#[cfg(any(target_os = "macos", target_os = "ios"))]
impl std::fmt::Debug for QwenGraphState { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_struct("QwenGraphState").finish() } }
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct QwenGraphState;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl std::fmt::Debug for QwenGraphState { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_struct("QwenGraphState (No Metal)").finish() } }

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

fn qwen_rmsnorm_is_offset(header: &crate::CellmHeader) -> bool {
    let check = |s: &str| s.contains("gemma");
    header.model_type.contains("gemma") || header.source_model_type.as_deref().is_some_and(check)
}

fn rms_norm_inplace_segment(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 { return; }
    let mut mean_sq = 0.0f32;
    for &v in x.iter() { mean_sq += v * v; }
    mean_sq /= n as f32;
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
    for i in 0..n { x[i] = x[i] * inv_rms * weight[i]; }
}

fn add_one_inplace(x: &mut [f32]) { for v in x { *v += 1.0; } }

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

fn resolve_tensor_name_file(file: &CellmFile, name: &str) -> String {
    if file.tensor_index(name).is_some() { return name.to_string(); }
    let stem = if let Some(rest) = name.strip_prefix("language_model.model.") { rest }
               else if let Some(rest) = name.strip_prefix("model.language_model.") { rest }
               else if let Some(rest) = name.strip_prefix("model.") { rest }
               else { name };
    for c in [format!("model.{stem}"), format!("language_model.model.{stem}"), format!("model.language_model.{stem}")] {
        if file.tensor_index(&c).is_some() { return c; }
    }
    name.to_string()
}

fn has_tensor_file(file: &CellmFile, name: &str) -> bool { file.tensor_index(&resolve_tensor_name_file(file, name)).is_some() }

fn unpack_i4(packed: &[u8], idx: usize) -> f32 {
    let b = packed[idx / 2];
    let n = if idx % 2 == 0 { b & 0xf } else { b >> 4 };
    (n as i8 - 8) as f32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerKind { FullAttention, LinearAttention }

#[derive(Debug, Clone)]
struct LinearAttnSpec { num_k_heads: usize, num_v_heads: usize, head_k_dim: usize, head_v_dim: usize, conv_kernel_size: usize, use_qk_l2norm_in_kernel: bool }

#[derive(Debug)]
struct LinearLayerState { conv: Vec<f32>, recurrent: Vec<f32> }

struct QwenSessionState { linear: Vec<Option<LinearLayerState>> }
