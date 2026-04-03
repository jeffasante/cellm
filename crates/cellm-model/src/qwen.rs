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
        let cfg = self.cfg.clone();
        let hidden = cfg.hidden_size;
        let n_kv_heads = cfg.num_key_value_heads;
        let session_id = page_table.session_id();

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
        })?;
        let token_off = page_table.offset_in_block(pos).map_err(|e| {
            CoreError::Backend(format!("qwen step: page_table offset_in_block failed: {e}"))
        })?;

        // Lazy session state init (for linear-attention layers).
        if let Some(spec) = &self.linear_spec {
            self.sessions
                .entry(session_id)
                .or_insert_with(|| QwenSessionState::new(self.cfg.num_hidden_layers, &self.layer_kinds, spec));
        }

        // x = embedding(token)
        let mut x = vec![0.0f32; hidden];
        self.embed_token(token, &mut x)?;

        // Per-layer scratch.
        let mut in_norm_w = vec![0.0f32; hidden];
        let mut x_norm = vec![0.0f32; hidden];

        let mut post_norm_w = vec![0.0f32; hidden];
        let mut mlp_in = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; cfg.intermediate_size];
        let mut up = vec![0.0f32; cfg.intermediate_size];
        let mut down = vec![0.0f32; hidden];

        // Attention buffers (sized per layer when needed).
        let mut q_raw: Vec<f32> = Vec::new();
        let mut q: Vec<f32> = Vec::new();
        let mut q_gate: Vec<f32> = Vec::new();
        let mut k: Vec<f32> = Vec::new();
        let mut v: Vec<f32> = Vec::new();
        let mut attn_out: Vec<f32> = Vec::new();
        let mut attn_proj: Vec<f32> = vec![0.0f32; hidden];
        let mut q_head_norm_w: Vec<f32> = Vec::new();
        let mut k_head_norm_w: Vec<f32> = Vec::new();

        // Linear-attention (Gated DeltaNet) scratch (allocated lazily if used).
        let mut lin_qkv: Vec<f32> = Vec::new();
        let mut lin_z: Vec<f32> = Vec::new();
        let mut lin_a: Vec<f32> = Vec::new();
        let mut lin_b: Vec<f32> = Vec::new();
        let mut lin_mixed_qkv: Vec<f32> = Vec::new();
        let mut lin_conv_out: Vec<f32> = Vec::new();
        let mut lin_q_rep: Vec<f32> = Vec::new();
        let mut lin_k_rep: Vec<f32> = Vec::new();
        let mut lin_v: Vec<f32> = Vec::new();
        let mut lin_core_out: Vec<f32> = Vec::new();
        let mut lin_normed: Vec<f32> = Vec::new();
        let mut lin_tmp_kv: Vec<f32> = Vec::new();
        let mut lin_tmp_delta: Vec<f32> = Vec::new();

        let mut gather_bases: Vec<usize> = Vec::new();

        for layer in 0..self.max_layers {
            let use_metal_norm = self.metal_ops.is_some();
            let use_metal_rope = self.metal_ops.is_some();

            let layer_in_norm = l2norm_value(&x);
            // Norm before attention / linear-attn.
            if use_metal_norm {
                let w = self.tensor_f16(&format!("language_model.model.layers.{layer}.input_layernorm.weight"))?.to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                self.metal_ops.as_mut().unwrap()
                    .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_norm)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                self.rmsnorm_weight(
                    &format!("language_model.model.layers.{layer}.input_layernorm.weight"),
                    &mut in_norm_w,
                )?;
                if self.rmsnorm_weight_is_offset {
                    add_one_inplace(&mut in_norm_w);
                }
                rms_norm_f32(&x, &in_norm_w, cfg.rms_norm_eps, &mut x_norm);
            }

            match self.layer_kinds.get(layer).copied().unwrap_or(LayerKind::FullAttention) {
                LayerKind::FullAttention => {
                if self.disable_full_attention {
                    continue;
                }
                // Read shapes to infer head geometry.
                let k_shape = self.tensor_shape(&format!(
                    "language_model.model.layers.{layer}.self_attn.k_proj.weight"
                ))?;
                let o_shape = self.tensor_shape(&format!(
                    "language_model.model.layers.{layer}.self_attn.o_proj.weight"
                ))?;
                if k_shape.len() != 2 || o_shape.len() != 2 {
                    return Err(CoreError::Backend(format!(
                        "qwen: expected 2D k/o proj weights at layer {layer}"
                    )));
                }
                let kv_dim = k_shape[0];
                if kv_dim % n_kv_heads != 0 {
                    return Err(CoreError::Backend(format!(
                        "qwen: kv_dim {kv_dim} not divisible by n_kv_heads={n_kv_heads} at layer {layer}"
                    )));
                }
                let head_dim = kv_dim / n_kv_heads;
                let attn_in = o_shape[1];
                if attn_in % head_dim != 0 {
                    return Err(CoreError::Backend(format!(
                        "qwen: o_proj in_dim {attn_in} not divisible by head_dim={head_dim} at layer {layer}"
                    )));
                }
                let n_heads = attn_in / head_dim;

                // Projections.
                let q_shape = self.tensor_shape(&format!(
                    "language_model.model.layers.{layer}.self_attn.q_proj.weight"
                ))?;
                if q_shape.len() != 2 || q_shape[1] != hidden {
                    return Err(CoreError::Backend(format!(
                        "qwen: unexpected q_proj.weight shape {:?} at layer {layer}",
                        q_shape
                    )));
                }
                q_raw.resize(q_shape[0], 0.0);
                self.linear_f16_out_in(
                    &x_norm,
                    &format!("language_model.model.layers.{layer}.self_attn.q_proj.weight"),
                    q_raw.len(),
                    hidden,
                    &mut q_raw,
                )?;

                k.resize(kv_dim, 0.0);
                v.resize(kv_dim, 0.0);
                self.linear_f16_out_in(
                    &x_norm,
                    &format!("language_model.model.layers.{layer}.self_attn.k_proj.weight"),
                    kv_dim,
                    hidden,
                    &mut k,
                )?;
                self.linear_f16_out_in(
                    &x_norm,
                    &format!("language_model.model.layers.{layer}.self_attn.v_proj.weight"),
                    kv_dim,
                    hidden,
                    &mut v,
                )?;

                // q and optional gate.
                let q_main_dim = n_heads * head_dim;
                if q_raw.len() == q_main_dim * 2 {
                    q.resize(q_main_dim, 0.0);
                    q_gate.resize(q_main_dim, 0.0);
                    // Qwen3.5 full-attn uses per-head packed q_proj output:
                    // [head0_q, head0_gate, head1_q, head1_gate, ...].
                    for h in 0..n_heads {
                        let src = h * (2 * head_dim);
                        let dst = h * head_dim;
                        q[dst..dst + head_dim].copy_from_slice(&q_raw[src..src + head_dim]);
                        q_gate[dst..dst + head_dim]
                            .copy_from_slice(&q_raw[src + head_dim..src + 2 * head_dim]);
                    }
                } else if q_raw.len() == q_main_dim {
                    q.clear();
                    q.extend_from_slice(&q_raw);
                    q_gate.clear();
                } else {
                    return Err(CoreError::Backend(format!(
                        "qwen: unexpected q_proj out_dim {} at layer {layer} (expected {q_main_dim} or {})",
                        q_raw.len(),
                        q_main_dim * 2
                    )));
                }

                // Qwen-style per-head Q/K RMSNorm (present on full-attention layers).
                if use_metal_norm {
                    let qw = self.tensor_f16(&format!("language_model.model.layers.{layer}.self_attn.q_norm.weight"))?.to_vec();
                    let kw = self.tensor_f16(&format!("language_model.model.layers.{layer}.self_attn.k_norm.weight"))?.to_vec();
                    let add_one = self.rmsnorm_weight_is_offset;
                    let ops = self.metal_ops.as_mut().unwrap();
                    for h in 0..n_heads {
                        let seg = &mut q[h * head_dim..(h + 1) * head_dim];
                        let inp = seg.to_vec();
                        ops.rms_norm_f16w(&inp, &qw, cfg.rms_norm_eps, add_one, seg)
                            .map_err(|e| CoreError::Backend(e.to_string()))?;
                    }
                    for h in 0..n_kv_heads {
                        let seg = &mut k[h * head_dim..(h + 1) * head_dim];
                        let inp = seg.to_vec();
                        ops.rms_norm_f16w(&inp, &kw, cfg.rms_norm_eps, add_one, seg)
                            .map_err(|e| CoreError::Backend(e.to_string()))?;
                    }
                } else {
                    q_head_norm_w.resize(head_dim, 0.0);
                    k_head_norm_w.resize(head_dim, 0.0);
                    self.rmsnorm_weight(
                        &format!("language_model.model.layers.{layer}.self_attn.q_norm.weight"),
                        &mut q_head_norm_w,
                    )?;
                    self.rmsnorm_weight(
                        &format!("language_model.model.layers.{layer}.self_attn.k_norm.weight"),
                        &mut k_head_norm_w,
                    )?;
                    if self.rmsnorm_weight_is_offset {
                        add_one_inplace(&mut q_head_norm_w);
                        add_one_inplace(&mut k_head_norm_w);
                    }
                    for h in 0..n_heads {
                        rms_norm_inplace_f32(
                            &mut q[h * head_dim..(h + 1) * head_dim],
                            &q_head_norm_w,
                            cfg.rms_norm_eps,
                        );
                    }
                    for h in 0..n_kv_heads {
                        rms_norm_inplace_f32(
                            &mut k[h * head_dim..(h + 1) * head_dim],
                            &k_head_norm_w,
                            cfg.rms_norm_eps,
                        );
                    }
                }

                let mut rotary_dim = ((head_dim as f32) * self.partial_rotary_factor) as usize;
                rotary_dim = rotary_dim.min(head_dim);
                if rotary_dim % 2 != 0 {
                    rotary_dim -= 1;
                }
                if rotary_dim > 0 {
                    if use_metal_rope {
                        let ops = self.metal_ops.as_mut().unwrap();
                        ops.rope_half_f32(&mut q, n_heads, head_dim, rotary_dim, pos, cfg.rope_theta)
                            .map_err(|e| CoreError::Backend(e.to_string()))?;
                        ops.rope_half_f32(&mut k, n_kv_heads, head_dim, rotary_dim, pos, cfg.rope_theta)
                            .map_err(|e| CoreError::Backend(e.to_string()))?;
                    } else {
                        rope_inplace_f32_partial(&mut q, n_heads, head_dim, rotary_dim, pos, cfg.rope_theta);
                        rope_inplace_f32_partial(
                            &mut k,
                            n_kv_heads,
                            head_dim,
                            rotary_dim,
                            pos,
                            cfg.rope_theta,
                        );
                    }
                }

                // Write new token K/V into paged cache.
                {
                    let mut cv = kv_cache.view_mut();
                    cv.write_token(block_id, layer, token_off, &k, &v)?;
                }

                // Gather historical token bases and run attention for this token.
                let seq = page_table.token_count();
                let cr = kv_cache.view();
                gather_bases.clear();
                gather_bases.reserve(seq);
                for tpos in 0..seq {
                    let b = page_table.block_for_token(tpos).map_err(|e| {
                        CoreError::Backend(format!("qwen: block_for_token failed: {e}"))
                    })?;
                    let o = page_table.offset_in_block(tpos).map_err(|e| {
                        CoreError::Backend(format!("qwen: offset_in_block failed: {e}"))
                    })?;
                    gather_bases.push(cr.layout.token_base_elem(b, layer, o)?);
                }
                attn_out.resize(q_main_dim, 0.0);
                cr.attention_single_token_gqa_from_bases(
                    &gather_bases,
                    &q,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    &mut attn_out,
                )?;

                // Optional output gate.
                if !q_gate.is_empty() && !self.disable_full_attn_gate {
                    for i in 0..attn_out.len() {
                        let g = 1.0 / (1.0 + (-q_gate[i]).exp());
                        attn_out[i] *= g;
                    }
                }

                // o_proj: hidden <- attn_in
                self.linear_f16_out_in(
                    &attn_out,
                    &format!("language_model.model.layers.{layer}.self_attn.o_proj.weight"),
                    hidden,
                    attn_in,
                    &mut attn_proj,
                )?;

                for i in 0..hidden {
                    x[i] += attn_proj[i];
                }
                }
                LayerKind::LinearAttention => {
                    if self.disable_linear_attention {
                        continue;
                    }
                    let Some(spec) = &self.linear_spec else {
                        return Err(CoreError::Backend(
                            "qwen: layer marked linear_attention but no linear_spec".into(),
                        ));
                    };
                    let s = self
                        .sessions
                        .get_mut(&session_id)
                        .expect("session state exists when linear_spec exists");
                    let layer_state = s.linear[layer].as_mut().ok_or_else(|| {
                        CoreError::Backend(format!(
                            "qwen: missing linear-attn state for layer {layer} (session={session_id})"
                        ))
                    })?;

                    // Allocate scratch once.
                    if lin_qkv.is_empty() {
                        let key_dim = spec.num_k_heads * spec.head_k_dim;
                        let value_dim = spec.num_v_heads * spec.head_v_dim;
                        let conv_dim = key_dim * 2 + value_dim;
                        lin_qkv.resize(conv_dim, 0.0);
                        lin_z.resize(value_dim, 0.0);
                        lin_a.resize(spec.num_v_heads, 0.0);
                        lin_b.resize(spec.num_v_heads, 0.0);
                        lin_mixed_qkv.resize(conv_dim, 0.0);
                        lin_conv_out.resize(conv_dim, 0.0);
                        lin_q_rep.resize(spec.num_v_heads * spec.head_k_dim, 0.0);
                        lin_k_rep.resize(spec.num_v_heads * spec.head_k_dim, 0.0);
                        lin_v.resize(value_dim, 0.0);
                        lin_core_out.resize(value_dim, 0.0);
                        lin_normed.resize(value_dim, 0.0);
                        lin_tmp_kv.resize(spec.head_v_dim, 0.0);
                        lin_tmp_delta.resize(spec.head_v_dim, 0.0);
                    }

                    linear_attn_step(
                        &self.file,
                        layer,
                        hidden,
                        &x_norm,
                        layer_state,
                        spec,
                        cfg.rms_norm_eps,
                        &mut lin_qkv,
                        &mut lin_z,
                        &mut lin_a,
                        &mut lin_b,
                        &mut lin_mixed_qkv,
                        &mut lin_conv_out,
                        &mut lin_q_rep,
                        &mut lin_k_rep,
                        &mut lin_v,
                        &mut lin_core_out,
                        &mut lin_normed,
                        &mut lin_tmp_kv,
                        &mut lin_tmp_delta,
                        &mut attn_proj,
                    )?;
                    for i in 0..hidden {
                        x[i] += attn_proj[i];
                    }
                }
            }
            let layer_post_mixer_norm = l2norm_value(&x);

            // Post-attn norm (run regardless; even if attention was skipped).
            if use_metal_norm {
                let w = self.tensor_f16(&format!("language_model.model.layers.{layer}.post_attention_layernorm.weight"))?.to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                self.metal_ops.as_mut().unwrap()
                    .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut mlp_in)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                self.rmsnorm_weight(
                    &format!("language_model.model.layers.{layer}.post_attention_layernorm.weight"),
                    &mut post_norm_w,
                )?;
                if self.rmsnorm_weight_is_offset {
                    add_one_inplace(&mut post_norm_w);
                }
                rms_norm_f32(&x, &post_norm_w, cfg.rms_norm_eps, &mut mlp_in);
            }

            // MLP: gate_proj + up_proj -> silu(gate)*up -> down_proj
            self.linear_f16_out_in(
                &mlp_in,
                &format!("language_model.model.layers.{layer}.mlp.gate_proj.weight"),
                cfg.intermediate_size,
                hidden,
                &mut gate,
            )?;
            self.linear_f16_out_in(
                &mlp_in,
                &format!("language_model.model.layers.{layer}.mlp.up_proj.weight"),
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
                &format!("language_model.model.layers.{layer}.mlp.down_proj.weight"),
                hidden,
                cfg.intermediate_size,
                &mut down,
            )?;
            for i in 0..hidden {
                x[i] += down[i];
            }
            let layer_out_norm = l2norm_value(&x);
            if self.debug_pos == Some(pos) {
                let kind = self.layer_kinds.get(layer).copied().unwrap_or(LayerKind::FullAttention);
                eprintln!(
                    "[qwen-debug] pos={pos} layer={layer} kind={kind:?} in={:.6} post_mixer={:.6} out={:.6}",
                    layer_in_norm, layer_post_mixer_norm, layer_out_norm
                );
            }
        }

        // Final norm.
        let use_metal_norm = self.metal_ops.is_some();
        let use_metal_logits = self.metal_ops.is_some();
        let mut x_final = vec![0.0f32; hidden];
        if use_metal_norm {
            let w = self.tensor_f16("language_model.model.norm.weight")?.to_vec();
            let add_one = self.rmsnorm_weight_is_offset;
            self.metal_ops.as_mut().unwrap()
                .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &mut x_final)
                .map_err(|e| CoreError::Backend(e.to_string()))?;
        } else {
            let mut norm_w = vec![0.0f32; hidden];
            self.rmsnorm_weight("language_model.model.norm.weight", &mut norm_w)?;
            if self.rmsnorm_weight_is_offset {
                add_one_inplace(&mut norm_w);
            }
            rms_norm_f32(&x, &norm_w, cfg.rms_norm_eps, &mut x_final);
        }
        if self.debug_pos == Some(pos) {
            eprintln!(
                "[qwen-debug] pos={pos} final pre_norm={:.6} final={:.6}",
                l2norm_value(&x),
                l2norm_value(&x_final)
            );
        }

        // Logits via lm_head when present, otherwise tied embeddings.
        let embed_name = self.resolve_tensor_name("language_model.model.embed_tokens.weight");
        let lm_head_name = if self.file.tensor_index("lm_head.weight").is_some() {
            Some(self.resolve_tensor_name("lm_head.weight"))
        } else {
            None
        };
        let vocab = cfg.vocab_size;
        let k = top_k.max(1).min(vocab);
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut min_idx = 0usize;
        let mut min_val = f32::INFINITY;

        let weight_name = if let Some(name) = lm_head_name.as_deref() {
            name
        } else {
            embed_name.as_str()
        };

        if use_metal_logits {
            let resolved = self.resolve_tensor_name(weight_name);
            let dtype = self
                .file
                .tensor_index(&resolved)
                .ok_or_else(|| CoreError::Backend(format!("unknown tensor {weight_name}")))?
                .dtype
                .clone();
            let mut buf = vec![0.0f32; vocab];
            match dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16(weight_name)?.to_vec();
                    self.metal_ops.as_mut().unwrap()
                        .logits_f16(&x_final, &w, vocab, hidden, weight_name, &mut buf)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                "i8" => {
                    let w = self.tensor_i8(weight_name)?.to_vec();
                    let s = self.tensor_f16(&format!("{weight_name}.qscale"))?.to_vec();
                    self.metal_ops.as_mut().unwrap()
                        .logits_i8(&x_final, &w, &s, vocab, hidden, weight_name, &mut buf)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                other => return Err(CoreError::Backend(format!("unsupported lm dtype {other} for {weight_name}"))),
            }
            for vid in 0..vocab {
                let dot = buf[vid];
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
        } else {
            for vid in 0..vocab {
                let dot = self.dot_row(weight_name, vid, hidden, &x_final)?;

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
        }

        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(top)
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

    fn embed_token(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        let hidden = out.len();
        let weight_name = "language_model.model.embed_tokens.weight";
        let shape = self.tensor_shape(weight_name)?;
        if shape.len() != 2 || shape[1] != hidden {
            return Err(CoreError::Backend(format!(
                "embed weight shape mismatch: {:?}, expected [vocab,{hidden}]",
                shape
            )));
        }
        let resolved = self.resolve_tensor_name(weight_name);
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
                let embed = self.tensor_f16(weight_name)?;
                let row = &embed[t * hidden..(t + 1) * hidden];
                for i in 0..hidden {
                    out[i] = f16::from_bits(row[i]).to_f32();
                }
            }
            "i8" => {
                let embed = self.tensor_i8(weight_name)?;
                let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                let scale = f16::from_bits(scales[t]).to_f32();
                let row = &embed[t * hidden..(t + 1) * hidden];
                for i in 0..hidden {
                    out[i] = (row[i] as f32) * scale;
                }
            }
            "i4" => {
                let embed = self.tensor_u8(weight_name)?;
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
            other => Err(CoreError::Backend(format!(
                "unsupported weight dtype for {weight_name}: {other}"
            ))),
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

        if let QwenLinearBackend::Metal { ctx } = &self.linear_backend {
            let chunk_cols = (QWEN_METAL_LINEAR_MAX_ELEMS / in_dim.max(1)).max(1).min(out_dim);
            let mut weight_t_chunk = vec![0.0f32; in_dim * chunk_cols];
            let mut out_chunk = vec![0.0f32; chunk_cols];
            let mut metal_ok = true;
            match dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16(weight_name)?;
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
                    let w = self.tensor_i8(weight_name)?;
                    let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
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
                "i4" => {
                    let w = self.tensor_u8(weight_name)?;
                    let scales = self.tensor_f16(&format!("{weight_name}.qscale"))?;
                    let row_stride = in_dim.div_ceil(2);
                    if w.len() != out_dim * row_stride || scales.len() != out_dim {
                        return Err(CoreError::Backend(format!(
                            "weight {weight_name} i4/qscale len mismatch: w={} scales={} expected w={} scales={}",
                            w.len(),
                            scales.len(),
                            out_dim * row_stride,
                            out_dim
                        )));
                    }
                    let mut row_start = 0usize;
                    while row_start < out_dim {
                        let cols_n = (out_dim - row_start).min(chunk_cols);
                        for i in 0..in_dim {
                            for c in 0..cols_n {
                                let row_idx = row_start + c;
                                let row = &w[row_idx * row_stride..(row_idx + 1) * row_stride];
                                let scale = f16::from_bits(scales[row_idx]).to_f32();
                                weight_t_chunk[i * cols_n + c] = unpack_i4(row, i) * scale;
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
                    "qwen full-metal: linear kernel failed for {weight_name}; CPU fallback disabled"
                )));
            }
        }

        match dtype.as_str() {
            "f16" => {
                let w = self.tensor_f16(weight_name)?;
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
                let w = self.tensor_i8(weight_name)?;
                let scales_name = format!("{weight_name}.qscale");
                let scales = self.tensor_f16(&scales_name)?;
                for j in 0..out_dim {
                    let row = &w[j * in_dim..(j + 1) * in_dim];
                    let scale = f16::from_bits(scales[j]).to_f32();
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += x[i] * (row[i] as f32) * scale;
                    }
                    out[j] = acc;
                }
            }
            "i4" => {
                let w = self.tensor_u8(weight_name)?;
                let row_stride = in_dim.div_ceil(2);
                let scales_name = format!("{weight_name}.qscale");
                let scales = self.tensor_f16(&scales_name)?;
                for j in 0..out_dim {
                    let row = &w[j * row_stride..(j + 1) * row_stride];
                    let scale = f16::from_bits(scales[j]).to_f32();
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += x[i] * unpack_i4(row, i) * scale;
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

    // 1) Projections: qkv, z, a, b
    linear_f16_out_in_file(
        file,
        x_norm,
        &format!("language_model.model.layers.{layer}.linear_attn.in_proj_qkv.weight"),
        conv_dim,
        hidden,
        qkv,
    )?;
    linear_f16_out_in_file(
        file,
        x_norm,
        &format!("language_model.model.layers.{layer}.linear_attn.in_proj_z.weight"),
        value_dim,
        hidden,
        z,
    )?;
    linear_f16_out_in_file(
        file,
        x_norm,
        &format!("language_model.model.layers.{layer}.linear_attn.in_proj_a.weight"),
        spec.num_v_heads,
        hidden,
        a,
    )?;
    linear_f16_out_in_file(
        file,
        x_norm,
        &format!("language_model.model.layers.{layer}.linear_attn.in_proj_b.weight"),
        spec.num_v_heads,
        hidden,
        b,
    )?;

    if spec.num_k_heads == 0 {
        return Err(CoreError::Backend("qwen linear_attn: num_k_heads=0".into()));
    }
    if spec.num_v_heads % spec.num_k_heads != 0 {
        return Err(CoreError::Backend(format!(
            "qwen linear_attn: num_v_heads={} not divisible by num_k_heads={}",
            spec.num_v_heads, spec.num_k_heads
        )));
    }
    let ratio = spec.num_v_heads / spec.num_k_heads;

    // 2) Mixed QKV for the causal conv.
    //
    // HF Qwen3.5 uses plain packed layout:
    //   [q_all, k_all, v_all] with sizes [key_dim, key_dim, value_dim].
    if qkv.len() != conv_dim {
        return Err(CoreError::Backend(format!(
            "qwen linear_attn: in_proj_qkv out_dim {} unexpected (expected {})",
            qkv.len(),
            conv_dim
        )));
    }
    mixed_qkv.copy_from_slice(qkv);

    // 3) Causal depthwise conv1d update + SiLU.
    let conv_w = tensor_f16_file(
        file,
        &format!("language_model.model.layers.{layer}.linear_attn.conv1d.weight"),
    )?;
    let conv_shape = tensor_shape_file(
        file,
        &format!("language_model.model.layers.{layer}.linear_attn.conv1d.weight"),
    )?;
    if conv_shape.len() != 3 || conv_shape[0] != conv_dim {
        return Err(CoreError::Backend(format!(
            "qwen linear_attn: unexpected conv1d.weight shape {:?} at layer {layer}",
            conv_shape
        )));
    }
    let kernel_size = conv_shape[1].max(conv_shape[2]);
    if kernel_size != spec.conv_kernel_size {
        return Err(CoreError::Backend(format!(
            "qwen linear_attn: conv kernel_size={kernel_size} != spec.conv_kernel_size={} at layer {layer}",
            spec.conv_kernel_size
        )));
    }
    if state.conv.len() != conv_dim * kernel_size {
        return Err(CoreError::Backend(format!(
            "qwen linear_attn: conv_state len {} unexpected (expected {})",
            state.conv.len(),
            conv_dim * kernel_size
        )));
    }
    causal_conv1d_update_silu(mixed_qkv, &mut state.conv, conv_w, kernel_size, conv_out);

    // 4) Split conv output into q/k/v and reshape.
    let q_base = &conv_out[..key_dim];
    let k_base = &conv_out[key_dim..2 * key_dim];
    v.copy_from_slice(&conv_out[2 * key_dim..]);

    // repeat query/key if num_v_heads > num_k_heads (GQA-style)
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

    // 5) Kernel normalization (l2norm) and scaling.
    if spec.use_qk_l2norm_in_kernel {
        for h in 0..spec.num_v_heads {
            l2norm_inplace(&mut q_rep[h * spec.head_k_dim..(h + 1) * spec.head_k_dim], 1e-6);
            l2norm_inplace(&mut k_rep[h * spec.head_k_dim..(h + 1) * spec.head_k_dim], 1e-6);
        }
    }
    let scale = 1.0 / (spec.head_k_dim as f32).sqrt();
    for v in q_rep.iter_mut() {
        *v *= scale;
    }
    // 6) Compute g + beta for recurrent delta rule.
    let a_log = tensor_f16_file(
        file,
        &format!("language_model.model.layers.{layer}.linear_attn.A_log"),
    )?;
    let dt_bias = tensor_f16_file(
        file,
        &format!("language_model.model.layers.{layer}.linear_attn.dt_bias"),
    )?;
    if a_log.len() != spec.num_v_heads || dt_bias.len() != spec.num_v_heads {
        return Err(CoreError::Backend(format!(
            "qwen linear_attn: A_log/dt_bias len mismatch at layer {layer} (A_log={} dt_bias={} expected {})",
            a_log.len(),
            dt_bias.len(),
            spec.num_v_heads
        )));
    }

    // 7) Recurrent gated delta rule (single step) + state update.
    if state.recurrent.len() != spec.num_v_heads * spec.head_k_dim * spec.head_v_dim {
        return Err(CoreError::Backend(format!(
            "qwen linear_attn: recurrent_state len {} unexpected (expected {})",
            state.recurrent.len(),
            spec.num_v_heads * spec.head_k_dim * spec.head_v_dim
        )));
    }
    core_out.fill(0.0);
    for h in 0..spec.num_v_heads {
        let beta = sigmoid_f32(b[h]);
        let g = -f16::from_bits(a_log[h]).to_f32().exp()
            * softplus_f32(a[h] + f16::from_bits(dt_bias[h]).to_f32());
        let g_exp = g.exp();

        let qh = &q_rep[h * spec.head_k_dim..(h + 1) * spec.head_k_dim];
        let kh = &k_rep[h * spec.head_k_dim..(h + 1) * spec.head_k_dim];
        let vh = &v[h * spec.head_v_dim..(h + 1) * spec.head_v_dim];

        let st_base = h * spec.head_k_dim * spec.head_v_dim;
        let st = &mut state.recurrent[st_base..st_base + spec.head_k_dim * spec.head_v_dim];

        for s in st.iter_mut() {
            *s *= g_exp;
        }

        tmp_kv.fill(0.0);
        for i in 0..spec.head_k_dim {
            let ki = kh[i];
            if ki == 0.0 {
                continue;
            }
            let row = &st[i * spec.head_v_dim..(i + 1) * spec.head_v_dim];
            for j in 0..spec.head_v_dim {
                tmp_kv[j] += row[j] * ki;
            }
        }

        for j in 0..spec.head_v_dim {
            tmp_delta[j] = (vh[j] - tmp_kv[j]) * beta;
        }

        for i in 0..spec.head_k_dim {
            let ki = kh[i];
            if ki == 0.0 {
                continue;
            }
            let row = &mut st[i * spec.head_v_dim..(i + 1) * spec.head_v_dim];
            for j in 0..spec.head_v_dim {
                row[j] += ki * tmp_delta[j];
            }
        }

        let out_dst = &mut core_out[h * spec.head_v_dim..(h + 1) * spec.head_v_dim];
        out_dst.fill(0.0);
        for i in 0..spec.head_k_dim {
            let qi = qh[i];
            if qi == 0.0 {
                continue;
            }
            let row = &st[i * spec.head_v_dim..(i + 1) * spec.head_v_dim];
            for j in 0..spec.head_v_dim {
                out_dst[j] += row[j] * qi;
            }
        }
    }

    // 8) Output gated RMSNorm per head.
    let norm_w = tensor_f16_file(
        file,
        &format!("language_model.model.layers.{layer}.linear_attn.norm.weight"),
    )?;
    if norm_w.len() != spec.head_v_dim {
        return Err(CoreError::Backend(format!(
            "qwen linear_attn: norm.weight len {} mismatch (expected {}) at layer {layer}",
            norm_w.len(),
            spec.head_v_dim
        )));
    }
    for h in 0..spec.num_v_heads {
        let src = &core_out[h * spec.head_v_dim..(h + 1) * spec.head_v_dim];
        let gate = &z[h * spec.head_v_dim..(h + 1) * spec.head_v_dim];

        let mut var = 0.0f32;
        for &v in src {
            var += v * v;
        }
        var /= spec.head_v_dim as f32;
        let inv = 1.0 / (var + eps).sqrt();

        let dst = &mut normed[h * spec.head_v_dim..(h + 1) * spec.head_v_dim];
        for j in 0..spec.head_v_dim {
            let wj = f16::from_bits(norm_w[j]).to_f32();
            dst[j] = src[j] * inv * wj * silu_f32(gate[j]);
        }
    }

    // 9) Output projection: hidden <- value_dim
    linear_f16_out_in_file(
        file,
        normed,
        &format!("language_model.model.layers.{layer}.linear_attn.out_proj.weight"),
        hidden,
        value_dim,
        out_proj,
    )?;

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
}

impl QwenSessionState {
    fn new(num_layers: usize, kinds: &[LayerKind], spec: &LinearAttnSpec) -> Self {
        let key_dim = spec.num_k_heads * spec.head_k_dim;
        let value_dim = spec.num_v_heads * spec.head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        // Match `torch_causal_conv1d_update`: the cached conv state keeps `kernel_size`
        // values per channel (not kernel_size-1).
        let state_len = spec.conv_kernel_size;
        let mut linear = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let kind = kinds.get(i).copied().unwrap_or(LayerKind::FullAttention);
            if kind == LayerKind::LinearAttention {
                linear.push(Some(LinearLayerState {
                    conv: vec![0.0; conv_dim * state_len],
                    recurrent: vec![0.0; spec.num_v_heads * spec.head_k_dim * spec.head_v_dim],
                }));
            } else {
                linear.push(None);
            }
        }
        Self { linear }
    }
}

fn infer_qwen_layer_kinds_and_linear_spec(
    file: &CellmFile,
) -> Result<(Vec<LayerKind>, Option<LinearAttnSpec>, f32), CoreError> {
    let n_layers = file.header.num_layers;
    let mut kinds: Vec<LayerKind> = Vec::with_capacity(n_layers);
    let mut partial_rotary_factor = 1.0f32;

    // Prefer config-derived layer_types if present.
    let mut spec: Option<LinearAttnSpec> = None;
    if let Some(Value::Object(obj)) = &file.header.source_text_config {
        if let Some(Value::Object(rope_params)) = obj.get("rope_parameters") {
            if let Some(v) = rope_params.get("partial_rotary_factor").and_then(|v| v.as_f64()) {
                partial_rotary_factor = (v as f32).clamp(0.0, 1.0);
            }
        }

        if let Some(Value::Array(layer_types)) = obj.get("layer_types") {
            for (i, v) in layer_types.iter().enumerate() {
                let s = v
                    .as_str()
                    .ok_or_else(|| CoreError::Backend("qwen: layer_types not strings".into()))?;
                kinds.push(match s {
                    "full_attention" => LayerKind::FullAttention,
                    "linear_attention" => LayerKind::LinearAttention,
                    other => {
                        return Err(CoreError::Backend(format!(
                            "qwen: unknown layer_types[{i}]={other}"
                        )))
                    }
                });
            }
        }

        // Linear-attn spec (optional; can infer from tensors if missing).
        let get_usize = |k: &str| -> Option<usize> { obj.get(k).and_then(|v| v.as_u64()).map(|x| x as usize) };
        if let (Some(num_k_heads), Some(num_v_heads), Some(head_k_dim), Some(head_v_dim), Some(conv_kernel_size)) = (
            get_usize("linear_num_key_heads"),
            get_usize("linear_num_value_heads"),
            get_usize("linear_key_head_dim"),
            get_usize("linear_value_head_dim"),
            get_usize("linear_conv_kernel_dim"),
        ) {
            spec = Some(LinearAttnSpec {
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_kernel_size,
                use_qk_l2norm_in_kernel: true,
            });
        }
    }

    // Fallback: infer kinds from tensors if layer_types missing/invalid.
    if kinds.len() != n_layers {
        kinds.clear();
        for layer in 0..n_layers {
            let has_self = has_tensor_file(
                file,
                &format!("language_model.model.layers.{layer}.self_attn.q_proj.weight"),
            );
            let has_linear = has_tensor_file(
                file,
                &format!("language_model.model.layers.{layer}.linear_attn.in_proj_qkv.weight"),
            );
            kinds.push(match (has_self, has_linear) {
                (true, _) => LayerKind::FullAttention,
                (false, true) => LayerKind::LinearAttention,
                (false, false) => LayerKind::FullAttention,
            });
        }
    }

    // If any linear layers exist but we couldn't parse spec, infer a minimal spec from tensor shapes.
    if spec.is_none() && kinds.iter().any(|k| *k == LayerKind::LinearAttention) {
        let mut layer0 = None;
        for (i, k) in kinds.iter().enumerate() {
            if *k == LayerKind::LinearAttention {
                layer0 = Some(i);
                break;
            }
        }
        let layer = layer0.ok_or_else(|| CoreError::Backend("qwen: no linear layers".into()))?;
        let qkv = tensor_index_file(
            file,
            &format!("language_model.model.layers.{layer}.linear_attn.in_proj_qkv.weight"),
        )
            .ok_or_else(|| CoreError::Backend("qwen: missing linear in_proj_qkv".into()))?;
        let z = tensor_index_file(
            file,
            &format!("language_model.model.layers.{layer}.linear_attn.in_proj_z.weight"),
        )
            .ok_or_else(|| CoreError::Backend("qwen: missing linear in_proj_z".into()))?;
        let a = tensor_index_file(
            file,
            &format!("language_model.model.layers.{layer}.linear_attn.in_proj_a.weight"),
        )
            .ok_or_else(|| CoreError::Backend("qwen: missing linear in_proj_a".into()))?;
        let norm = tensor_index_file(
            file,
            &format!("language_model.model.layers.{layer}.linear_attn.norm.weight"),
        )
            .ok_or_else(|| CoreError::Backend("qwen: missing linear norm.weight".into()))?;
        let conv = tensor_index_file(
            file,
            &format!("language_model.model.layers.{layer}.linear_attn.conv1d.weight"),
        )
            .ok_or_else(|| CoreError::Backend("qwen: missing linear conv1d.weight".into()))?;

        let conv_dim = qkv.shape[0];
        let value_dim = z.shape[0];
        let num_v_heads = a.shape[0];
        let head_v_dim = norm.shape[0];
        if num_v_heads == 0 || head_v_dim == 0 {
            return Err(CoreError::Backend("qwen: invalid inferred linear spec".into()));
        }
        if value_dim != num_v_heads * head_v_dim {
            return Err(CoreError::Backend(format!(
                "qwen: inferred value_dim {value_dim} != num_v_heads*head_v_dim {}",
                num_v_heads * head_v_dim
            )));
        }
        let key_dim2 = (conv_dim - value_dim) / 2;
        let head_k_dim = head_v_dim;
        let num_k_heads = key_dim2 / head_k_dim;
        let kernel = conv.shape.iter().skip(1).copied().max().unwrap_or(1);

        spec = Some(LinearAttnSpec {
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size: kernel,
            use_qk_l2norm_in_kernel: true,
        });
    }

    Ok((kinds, spec, partial_rotary_factor))
}

fn qwen_rmsnorm_is_offset(header: &crate::cellm_file::CellmHeader) -> bool {
    if header.model_type.starts_with("qwen3_5") {
        return true;
    }
    if header
        .source_model_type
        .as_deref()
        .is_some_and(|m| m.starts_with("qwen3_5"))
    {
        return true;
    }
    if let Some(Value::Object(obj)) = &header.source_text_config {
        if obj
            .get("model_type")
            .and_then(|v| v.as_str())
            .is_some_and(|m| m.starts_with("qwen3_5"))
        {
            return true;
        }
    }
    false
}

fn add_one_inplace(x: &mut [f32]) {
    for v in x {
        *v += 1.0;
    }
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
}

fn softplus_f32(x: f32) -> f32 {
    // Stable softplus.
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn l2norm_inplace(x: &mut [f32], eps: f32) {
    let mut s = 0.0f32;
    for &v in x.iter() {
        s += v * v;
    }
    let inv = 1.0 / (s + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn l2norm_value(x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    s.sqrt()
}

fn rms_norm_inplace_f32(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let mut var = 0.0f32;
    for &v in x.iter() {
        var += v * v;
    }
    var /= x.len().max(1) as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        x[i] = x[i] * inv * weight[i];
    }
}

fn rope_inplace_f32_partial(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    pos: usize,
    theta: f32,
) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert!(rotary_dim <= head_dim);
    debug_assert!(rotary_dim % 2 == 0);

    let half = rotary_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f32) / rotary_dim as f32);
            let angle = pos as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = x[base + i];
            let x1 = x[base + half + i];
            x[base + i] = x0 * cos - x1 * sin;
            x[base + half + i] = x1 * cos + x0 * sin;
        }
    }
}

fn causal_conv1d_update_silu(
    x: &[f32],          // [channels]
    state: &mut [f32],  // [channels, kernel_size] (time-ordered oldest->newest)
    weight_f16: &[u16], // stored as [channels, *, *] but each channel has `kernel_size` contiguous weights
    kernel_size: usize,
    out: &mut [f32],
) {
    let channels = x.len();
    let state_len = kernel_size;
    debug_assert_eq!(state.len(), channels * state_len);
    debug_assert_eq!(out.len(), channels);

    for c in 0..channels {
        let st_base = c * state_len;
        let w_base = c * kernel_size;

        let mut acc = 0.0f32;
        // Match `torch_causal_conv1d_update` which concatenates `[state, x]`, runs a
        // conv1d, then takes the last `seq_len` outputs. For `seq_len=1`, this
        // corresponds to applying the kernel over `[state[1..], x]`.
        for t in 0..kernel_size.saturating_sub(1) {
            acc += f16::from_bits(weight_f16[w_base + t]).to_f32() * state[st_base + t + 1];
        }
        acc += f16::from_bits(weight_f16[w_base + kernel_size - 1]).to_f32() * x[c];

        out[c] = silu_f32(acc);

        // Shift state and append current x.
        if state_len > 0 {
            for t in 0..(state_len - 1) {
                state[st_base + t] = state[st_base + t + 1];
            }
            state[st_base + state_len - 1] = x[c];
        }
    }
}

fn resolve_tensor_name_file(file: &CellmFile, name: &str) -> String {
    if file.tensor_index(name).is_some() {
        return name.to_string();
    }
    if let Some(rest) = name.strip_prefix("language_model.model.") {
        let candidates = [
            format!("model.{rest}"),
            format!("model.language_model.{rest}"),
        ];
        for candidate in candidates {
            if file.tensor_index(&candidate).is_some() {
                return candidate;
            }
        }
    }
    if let Some(rest) = name.strip_prefix("model.") {
        let candidates = [
            format!("language_model.model.{rest}"),
            format!("model.language_model.{rest}"),
        ];
        for candidate in candidates {
            if file.tensor_index(&candidate).is_some() {
                return candidate;
            }
        }
    }
    if let Some(rest) = name.strip_prefix("model.language_model.") {
        let candidates = [
            format!("language_model.model.{rest}"),
            format!("model.{rest}"),
        ];
        for candidate in candidates {
            if file.tensor_index(&candidate).is_some() {
                return candidate;
            }
        }
    }
    name.to_string()
}

fn has_tensor_file(file: &CellmFile, name: &str) -> bool {
    let resolved = resolve_tensor_name_file(file, name);
    file.tensor_index(&resolved).is_some()
}

fn tensor_index_file<'a>(
    file: &'a CellmFile,
    name: &str,
) -> Option<&'a crate::cellm_file::CellmTensorIndex> {
    let resolved = resolve_tensor_name_file(file, name);
    file.tensor_index(&resolved)
}

fn tensor_f16_file<'a>(file: &'a CellmFile, name: &str) -> Result<&'a [u16], CoreError> {
    let resolved = resolve_tensor_name_file(file, name);
    let bytes = file.tensor_bytes(&resolved)?;
    if bytes.len() % 2 != 0 {
        return Err(CoreError::Backend(format!("tensor {name} nbytes not even")));
    }
    Ok(cast_slice(bytes))
}

fn tensor_i8_file<'a>(file: &'a CellmFile, name: &str) -> Result<&'a [i8], CoreError> {
    let resolved = resolve_tensor_name_file(file, name);
    let bytes = file.tensor_bytes(&resolved)?;
    Ok(cast_slice(bytes))
}

fn tensor_u8_file<'a>(file: &'a CellmFile, name: &str) -> Result<&'a [u8], CoreError> {
    let resolved = resolve_tensor_name_file(file, name);
    file.tensor_bytes(&resolved)
}

fn tensor_shape_file(file: &CellmFile, name: &str) -> Result<Vec<usize>, CoreError> {
    let resolved = resolve_tensor_name_file(file, name);
    let t = file
        .tensor_index(&resolved)
        .ok_or_else(|| CoreError::Backend(format!("unknown tensor {name}")))?;
    Ok(t.shape.clone())
}

fn linear_f16_out_in_file(
    file: &CellmFile,
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

    let resolved = resolve_tensor_name_file(file, weight_name);
    let shape = tensor_shape_file(file, weight_name)?;
    let meta = file
        .tensor_index(&resolved)
        .ok_or_else(|| CoreError::Backend(format!("unknown tensor {weight_name}")))?;
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
    match meta.dtype.as_str() {
        "f16" => {
            let w = tensor_f16_file(file, weight_name)?;
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
            let w = tensor_i8_file(file, weight_name)?;
            if w.len() != out_dim * in_dim {
                return Err(CoreError::Backend(format!(
                    "weight {weight_name} len mismatch: {} expected {}",
                    w.len(),
                    out_dim * in_dim
                )));
            }
            let scales_name = format!("{weight_name}.qscale");
            let scales = tensor_f16_file(file, &scales_name)?;
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
                    acc += x[i] * (row[i] as f32) * scale;
                }
                out[j] = acc;
            }
        }
        "i4" => {
            let w = tensor_u8_file(file, weight_name)?;
            let row_stride = in_dim.div_ceil(2);
            if w.len() != out_dim * row_stride {
                return Err(CoreError::Backend(format!(
                    "weight {weight_name} len mismatch: {} expected {}",
                    w.len(),
                    out_dim * row_stride
                )));
            }
            let scales_name = format!("{weight_name}.qscale");
            let scales = tensor_f16_file(file, &scales_name)?;
            if scales.len() != out_dim {
                return Err(CoreError::Backend(format!(
                    "weight {weight_name} qscale len mismatch: {} expected {}",
                    scales.len(),
                    out_dim
                )));
            }
            for j in 0..out_dim {
                let row = &w[j * row_stride..(j + 1) * row_stride];
                let scale = f16::from_bits(scales[j]).to_f32();
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    acc += x[i] * unpack_i4(row, i) * scale;
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

fn unpack_i4(packed_row: &[u8], idx: usize) -> f32 {
    let byte = packed_row[idx / 2];
    let nibble = if idx % 2 == 0 {
        byte & 0x0f
    } else {
        (byte >> 4) & 0x0f
    };
    (nibble as i8 - 8) as f32
}
