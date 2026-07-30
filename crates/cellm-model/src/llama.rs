// Author: Jeffrey Asante (https://jeffasante.github.io/)
use std::collections::HashMap;
use std::path::Path;

use bytemuck::cast_slice;
use cellm_cache::{KVCache, PageTable};
use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::{
    rms_norm_f32,
    rope_interleaved_inplace_f32,
    rope_interleaved_inplace_f32_with_freqs,
    rope_non_interleaved_inplace_f32,
    rope_non_interleaved_inplace_f32_with_freqs,
};
use cellm_kernels::metal::MetalMatmul;
use cellm_kernels::{MetalKernels, MetalOps};
use half::f16;
#[cfg(feature = "webgpu")]
use wgpu::Buffer;

use crate::{CellmFile, ModelConfig};

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        Order: i32, TransA: i32, TransB: i32,
        M: i32, N: i32, K: i32,
        alpha: f32, A: *const f32, lda: i32,
        B: *const f32, ldb: i32,
        beta: f32, C: *mut f32, ldc: i32,
    );
}
#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::llama_graph::LlamaGraphState;

pub struct LlamaRunner {
    file: CellmFile,
    cfg: ModelConfig,
    max_layers: usize,
    pub eos_token_id: Option<u32>,
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
    /// Cached dequantized f32 weights for cblas_sgemm on macOS/iOS
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    f32_weight_cache: HashMap<String, Vec<f32>>,
    /// Cached pre-dequantized affine i4 weights (f32) for cblas_sgemm
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    affine_f32_weight_cache: HashMap<String, Vec<f32>>,
    #[cfg(feature = "webgpu")]
    pub webgpu_weight_cache: HashMap<String, Buffer>,
    // Pre-allocated scratch buffers for step_inner (avoids O(hidden) alloc per call)
    buf_attn_norm_w: Vec<f32>,
    buf_x_norm: Vec<f32>,
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_attn_out: Vec<f32>,
    buf_attn_proj: Vec<f32>,
    buf_post_norm_w: Vec<f32>,
    buf_mlp_in: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_down: Vec<f32>,
    buf_gather_bases: Vec<usize>,
    cached_inv_freqs: Vec<f32>,
    /// Pre-cached f32 norm weights (attn + ffn + final) to avoid f16→f32 conversion every step
    cached_norm_weights: HashMap<String, Vec<f32>>,
    /// Pre-built layer tensor names to avoid format!() every step
    cached_layer_names: Vec<LayerNamesCached>,
}

#[derive(Clone)]
struct LayerNamesCached {
    attn_norm: String,
    q_proj: String,
    k_proj: String,
    v_proj: String,
    o_proj: String,
    ffn_norm: String,
    gate_proj: String,
    up_proj: String,
    down_proj: String,
}

enum LlamaLinearBackend {
    Cpu,
    Metal { ctx: MetalMatmul },
    #[cfg(feature = "webgpu")]
    WebGpu { ctx: cellm_kernels::webgpu::WebGpuBackend },
}

impl LlamaRunner {
    pub fn load(path: &Path) -> Result<Self, CoreError> {
        let file = CellmFile::load(path)?;
        Self::from_file(file)
    }

    pub fn from_file(file: CellmFile) -> Result<Self, CoreError> {
        let h = file.header.clone();

        let cfg = ModelConfig {
            vocab_size: h.vocab_size,
            hidden_size: h.hidden_dim,
            num_hidden_layers: h.num_layers,
            num_attention_heads: h.num_heads,
            num_key_value_heads: h.num_kv_heads,
            head_dim: h.head_dim.unwrap_or(h.hidden_dim / h.num_heads),
            intermediate_size: h.intermediate_size,
            rms_norm_eps: h.rms_norm_eps,
            rope_theta: h.rope_theta,
            rope_scaling_type: h.rope_scaling_type.clone(),
            rope_scaling_factor: h.rope_scaling_factor,
            rope_scaling_original_max_position_embeddings: h.rope_scaling_original_max_position_embeddings,
            rope_scaling_low_freq_factor: h.rope_scaling_low_freq_factor,
            rope_scaling_high_freq_factor: h.rope_scaling_high_freq_factor,
            attention_softcap: 0.0,
            ..ModelConfig::default()
        };

        let tensor_prefix = detect_llama_prefix(&file)?;

        let hidden = cfg.hidden_size;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;

        // Precompute RoPE inv_freqs with optional llama3 scaling
        let inv_freqs = compute_inv_freqs(
            cfg.head_dim,
            cfg.rope_theta,
            &cfg.rope_scaling_type,
            cfg.rope_scaling_factor,
            cfg.rope_scaling_original_max_position_embeddings,
            cfg.rope_scaling_low_freq_factor,
            cfg.rope_scaling_high_freq_factor,
        );

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
            // Prefer model metadata; retain the environment override for older files.
            rope_interleaved: h.rope_interleaved.unwrap_or_else(|| {
                std::env::var("CELLM_LLAMA_ROPE_INTERLEAVED")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false)
            }),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            graph_state: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            f32_weight_cache: HashMap::new(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            affine_f32_weight_cache: HashMap::new(),
            #[cfg(feature = "webgpu")]
            webgpu_weight_cache: HashMap::new(),
            // Pre-allocated scratch buffers for step_inner (avoids O(hidden) alloc per call)
            buf_attn_norm_w: vec![0.0f32; hidden],
            buf_x_norm: vec![0.0f32; hidden],
            buf_q: vec![0.0f32; hidden],
            buf_k: vec![0.0f32; kv_dim],
            buf_v: vec![0.0f32; kv_dim],
            buf_attn_out: vec![0.0f32; hidden],
            buf_attn_proj: vec![0.0f32; hidden],
            buf_post_norm_w: vec![0.0f32; hidden],
            buf_mlp_in: vec![0.0f32; hidden],
            buf_gate: vec![0.0f32; inter],
            buf_up: vec![0.0f32; inter],
            buf_down: vec![0.0f32; hidden],
            buf_gather_bases: Vec::with_capacity(256),
            cached_inv_freqs: inv_freqs,
            cached_norm_weights: HashMap::new(),
            cached_layer_names: Vec::new(),
        })
    }

    /// Pre-cache layer tensor names and norm weights to eliminate per-step allocations.
    /// Called lazily on first step_inner invocation.
    fn ensure_caches(&mut self) -> Result<(), CoreError> {
        if !self.cached_layer_names.is_empty() {
            return Ok(());
        }
        let prefix = &self.tensor_prefix;
        let max_layers = self.max_layers;
        self.cached_layer_names = (0..max_layers).map(|l| {
            let base = if prefix.is_empty() {
                format!("model.layers.{l}")
            } else {
                format!("{prefix}model.layers.{l}")
            };
            LayerNamesCached {
                attn_norm: format!("{base}.input_layernorm.weight"),
                q_proj: format!("{base}.self_attn.q_proj.weight"),
                k_proj: format!("{base}.self_attn.k_proj.weight"),
                v_proj: format!("{base}.self_attn.v_proj.weight"),
                o_proj: format!("{base}.self_attn.o_proj.weight"),
                ffn_norm: format!("{base}.post_attention_layernorm.weight"),
                gate_proj: format!("{base}.mlp.gate_proj.weight"),
                up_proj: format!("{base}.mlp.up_proj.weight"),
                down_proj: format!("{base}.mlp.down_proj.weight"),
            }
        }).collect();

        // Pre-cache all norm weights as f32 to avoid f16→f32 conversion every step
        let hidden = self.cfg.hidden_size;
        for l in 0..max_layers {
            let ln = &self.cached_layer_names[l];
            for name in [&ln.attn_norm, &ln.ffn_norm] {
                if !self.cached_norm_weights.contains_key(name.as_str()) {
                    if let Ok(w_f16) = self.tensor_f16(name) {
                        let w_f32: Vec<f32> = w_f16.iter().map(|&v| f16::from_bits(v).to_f32()).collect();
                        self.cached_norm_weights.insert(name.clone(), w_f32);
                    }
                }
            }
        }
        // Cache final norm weight
        let final_norm_name = if self.tensor_prefix.is_empty() {
            "model.norm.weight".to_string()
        } else {
            format!("{}model.norm.weight", self.tensor_prefix)
        };
        if !self.cached_norm_weights.contains_key(final_norm_name.as_str()) {
            if let Ok(w_f16) = self.tensor_f16(&final_norm_name) {
                let w_f32: Vec<f32> = w_f16.iter().map(|&v| f16::from_bits(v).to_f32()).collect();
                self.cached_norm_weights.insert(final_norm_name, w_f32);
            }
        }
        Ok(())
    }

    pub fn file(&self) -> &CellmFile {
        &self.file
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

    #[cfg(feature = "webgpu")]
    pub fn enable_webgpu_backend(&mut self, ctx: cellm_kernels::webgpu::WebGpuBackend) {
        self.linear_backend = LlamaLinearBackend::WebGpu { ctx };
    }

    #[cfg(feature = "webgpu")]
    pub fn take_webgpu_backend(&mut self) -> Option<cellm_kernels::webgpu::WebGpuBackend> {
        let mut old = LlamaLinearBackend::Cpu;
        std::mem::swap(&mut self.linear_backend, &mut old);
        if let LlamaLinearBackend::WebGpu { ctx } = old {
            Some(ctx)
        } else {
            self.linear_backend = old;
            None
        }
    }

    pub fn enable_metal_full_backend(&mut self) -> bool {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            // Fused graph supports f16/f32/bf16. i8 kernels exist (encode_mv_i8)
            // but have a known numerical issue. Set CELLM_LLAMA_ENABLE_GRAPH_I8=1
            // to enable experimental i8 fused graph support.
            // i8/i4 weights dequantized to f16 during preload below
            let graph_enabled = std::env::var("CELLM_LLAMA_ENABLE_GRAPH")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(true);
            if graph_enabled {
                let gs_res = LlamaGraphState::new(
                    self.cfg.hidden_size,
                    self.cfg.num_attention_heads,
                    self.cfg.num_key_value_heads,
                    self.cfg.vocab_size,
                    self.cfg.intermediate_size,
                    self.rope_interleaved,
                );
                match gs_res {
                    Ok(mut gs) => {
                        println!("llama: detected tensor prefix: '{}'", self.tensor_prefix);
                        println!("llama: preloading weights into metal graph...");
                        let dtype_map: std::collections::HashMap<&str, &str> = self.file.header.tensors.iter()
                            .map(|t| (t.name.as_str(), t.dtype.as_str())).collect();
                        let mut scale_seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
                        for t in &self.file.header.tensors {
                            if t.name.ends_with(".qscale") {
                                if let Some(base) = t.name.strip_suffix(".qscale") { scale_seen.insert(base); }
                            }
                        }
                        for (name, data) in self.file.all_tensors() {
                            if name.ends_with(".qscale") { continue; }
                            let dtype = dtype_map.get(name.as_str()).copied().unwrap_or("");
                            if (dtype == "i8" || dtype == "i4") && scale_seen.contains(name.as_str()) {
                                let qscale_name = format!("{name}.qscale");
                                if let Ok(sbytes) = self.file.tensor_bytes(&qscale_name) {
                                    let scales: &[u16] = bytemuck::cast_slice(sbytes);
                                    let rows = scales.len();
                                    let cols = if rows > 0 { data.len() / rows } else { 0 };
                                    let mut f16_data: Vec<u16> = vec![0u16; data.len()];
                                    for r in 0..rows {
                                        let scale = half::f16::from_bits(scales[r]).to_f32();
                                        let off = r * cols;
                                        if dtype == "i8" {
                                            for c in 0..cols {
                                                let v = (data[off + c] as i8) as f32 * scale;
                                                f16_data[off + c] = half::f16::from_f32(v).to_bits();
                                            }
                                        } else {
                                            for c in 0..cols {
                                                let b = data[off + c / 2];
                                                let n = if c % 2 == 0 { b & 0xF } else { b >> 4 };
                                                let v = (n as i8 - 8) as f32 * scale;
                                                f16_data[off + c] = half::f16::from_f32(v).to_bits();
                                            }
                                        }
                                    }
                                    gs.preload_weight_f16(name.to_string(), bytemuck::cast_slice(&f16_data));
                                    continue;
                                }
                            }
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

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn reserve_metal_sequence_capacity(&mut self, max_seq: usize) {
        if let Some(gs) = &mut self.graph_state {
            gs.reserve_sequence_capacity(max_seq, self.cfg.num_hidden_layers);
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub fn reserve_metal_sequence_capacity(&mut self, _max_seq: usize) {
        // no-op on non-Apple platforms
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
        let logits = self.step_inner(x0, pos, page_table, kv_cache, true)?;
        self.topk_from_logits(&logits, top_k)
    }

    pub fn step_from_hidden(
        &mut self,
        x0: &[f32],
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
    ) -> Result<(), CoreError> {
        let _ = self.step_inner(x0, pos, page_table, kv_cache, false)?;
        Ok(())
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
                    CoreError::Backend(format!("llama prefill: page_table append_token failed: {e}"))
                })?;
            }
            let mut x = vec![0.0f32; self.cfg.hidden_size];
            self.embed_token(tok, &mut x)?;
            self.step_inner(&x, pos, page_table, kv_cache, false)?;
            }
            Ok(())
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
                        CoreError::Backend(format!("llama prefill_topk: page_table append_token failed: {e}"))
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

        /// Batched prefill: embed all tokens on CPU, then dispatch the entire
    /// sequence in a single Metal command buffer.  This eliminates the
    /// per-token `cb.commit/wait` overhead that makes Llama prefill ~10x
    /// slower than it should be on Apple Silicon.
    pub async fn prefill_fused(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        return_logits: bool,
    ) -> Result<Option<Vec<f32>>, CoreError> {
        let h = self.cfg.hidden_size;
        let mut x_all = vec![0.0f32; tokens.len() * h];
        for (i, &tok) in tokens.iter().enumerate() {
            self.embed_token(tok, &mut x_all[i * h..(i + 1) * h])?;
        }
        self.prefill_fused_hidden(&x_all, start_pos, page_table, kv_cache, return_logits).await
    }

    /// Like `prefill_fused` but the caller has already embedded the tokens
    /// (e.g. VLM image features mixed with text embeddings).
    pub async fn prefill_fused_hidden(
        &mut self,
        x_all: &[f32],
        start_pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        return_logits: bool,
    ) -> Result<Option<Vec<f32>>, CoreError> {
        // Ensure pagetable covers all tokens
        let num_tokens = x_all.len() / self.cfg.hidden_size;
        for i in 0..num_tokens {
            let pos = start_pos + i;
            if pos == page_table.token_count() {
                page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                    CoreError::Backend(format!("llama prefill_fused: page_table append_token failed: {e}"))
                })?;
            }
        }

        // Use batched GPU prefill when available.
        #[cfg(feature = "webgpu")]
        {
            let mut gpu_last_x = None;
            {
                let LlamaRunner {
                    file,
                    cfg,
                    max_layers,
                    tensor_prefix,
                    linear_backend,
                    webgpu_weight_cache,
                    rope_interleaved,
                    ..
                } = &mut *self;

                if let LlamaLinearBackend::WebGpu { ctx } = linear_backend {
                    let hidden = cfg.hidden_size;
                    let n_heads = cfg.num_attention_heads;
                    let n_kv_heads = cfg.num_key_value_heads;
                    let head_dim = cfg.head_dim;
                    let kv_dim = n_kv_heads * head_dim;
                    let num_tokens = x_all.len() / hidden;

                    let mut x = x_all.to_vec();
                    let mut x_norm = vec![0.0f32; x.len()];
                    let mut qkv = vec![0.0f32; num_tokens * (hidden + 2 * kv_dim)];
                    let mut attn_out = vec![0.0f32; num_tokens * hidden];
                    let mut gate_up = vec![0.0f32; num_tokens * 2 * cfg.intermediate_size];
                    let mut gather_bases = Vec::with_capacity(num_tokens + start_pos);
                    let rms_norm_eps = cfg.rms_norm_eps;
                    let rope_theta = cfg.rope_theta;
                    let rope_interleaved = *rope_interleaved;
                    let inter = cfg.intermediate_size;

                    for layer in 0..*max_layers {
                        // Attn Norm
                        let ln_w_buf = {
                            let name = format!("l{layer}.attn_norm");
                            if !webgpu_weight_cache.contains_key(&name) {
                                let tensor_name = if tensor_prefix.is_empty() { format!("model.layers.{layer}.input_layernorm.weight") } else { format!("{tensor_prefix}.model.layers.{layer}.input_layernorm.weight") };
                                let bytes = file.tensor_bytes(&tensor_name)?;
                                let w: &[f32] = bytemuck::cast_slice(bytes);
                                webgpu_weight_cache.insert(name.clone(), ctx.upload_f32(w));
                            }
                            webgpu_weight_cache.get(&name).unwrap().clone()
                        };
                        ctx.rms_norm_batch(&ln_w_buf, num_tokens as u32, hidden as u32, rms_norm_eps, &x, &mut x_norm).await;

                        // QKV Projections
                        let q_w_buf = {
                            let name = format!("l{layer}.q");
                            if !webgpu_weight_cache.contains_key(&name) {
                                let tensor_name = if tensor_prefix.is_empty() { format!("model.layers.{layer}.self_attn.q_proj.weight") } else { format!("{tensor_prefix}.model.layers.{layer}.self_attn.q_proj.weight") };
                                let bytes = file.tensor_bytes(&tensor_name)?;
                                let w: &[u16] = bytemuck::cast_slice(bytes);
                                webgpu_weight_cache.insert(name.clone(), ctx.upload_f16(w));
                            }
                            webgpu_weight_cache.get(&name).unwrap().clone()
                        };
                        let k_w_buf = {
                            let name = format!("l{layer}.k");
                            if !webgpu_weight_cache.contains_key(&name) {
                                let tensor_name = if tensor_prefix.is_empty() { format!("model.layers.{layer}.self_attn.k_proj.weight") } else { format!("{tensor_prefix}.model.layers.{layer}.self_attn.k_proj.weight") };
                                let bytes = file.tensor_bytes(&tensor_name)?;
                                let w: &[u16] = bytemuck::cast_slice(bytes);
                                webgpu_weight_cache.insert(name.clone(), ctx.upload_f16(w));
                            }
                            webgpu_weight_cache.get(&name).unwrap().clone()
                        };
                        let v_w_buf = {
                            let name = format!("l{layer}.v");
                            if !webgpu_weight_cache.contains_key(&name) {
                                let tensor_name = if tensor_prefix.is_empty() { format!("model.layers.{layer}.self_attn.v_proj.weight") } else { format!("{tensor_prefix}.model.layers.{layer}.self_attn.v_proj.weight") };
                                let bytes = file.tensor_bytes(&tensor_name)?;
                                let w: &[u16] = bytemuck::cast_slice(bytes);
                                webgpu_weight_cache.insert(name.clone(), ctx.upload_f16(w));
                            }
                            webgpu_weight_cache.get(&name).unwrap().clone()
                        };

                        let (q_out, rest) = qkv.split_at_mut(num_tokens * hidden);
                        let (k_out, v_out) = rest.split_at_mut(num_tokens * kv_dim);

                        ctx.matmul_batch_f16(&q_w_buf, num_tokens as u32, hidden as u32, hidden as u32, &x_norm, q_out).await;
                        ctx.matmul_batch_f16(&k_w_buf, num_tokens as u32, kv_dim as u32, hidden as u32, &x_norm, k_out).await;
                        ctx.matmul_batch_f16(&v_w_buf, num_tokens as u32, kv_dim as u32, hidden as u32, &x_norm, v_out).await;

                        // RoPE + KV Cache + Attention (CPU fallback for now)
                        for i in 0..num_tokens {
                            let pos = start_pos + i;
                            let q_i = &mut q_out[i * hidden..(i + 1) * hidden];
                            let k_i = &mut k_out[i * kv_dim..(i + 1) * kv_dim];
                            let v_i = &mut v_out[i * kv_dim..(i + 1) * kv_dim];

                            if rope_interleaved {
                                rope_interleaved_inplace_f32(q_i, n_heads, head_dim, pos, rope_theta);
                                rope_interleaved_inplace_f32(k_i, n_kv_heads, head_dim, pos, rope_theta);
                            } else {
                                rope_non_interleaved_inplace_f32(q_i, n_heads, head_dim, head_dim, pos, rope_theta);
                                rope_non_interleaved_inplace_f32(k_i, n_kv_heads, head_dim, head_dim, pos, rope_theta);
                            }

                            let block_id = page_table.block_for_token(pos).map_err(|e| CoreError::Backend(e.to_string()))?;
                            let token_off = page_table.offset_in_block(pos).map_err(|e| CoreError::Backend(e.to_string()))?;
                            kv_cache.view_mut().write_token(block_id, layer, token_off, k_i, v_i).map_err(|e| CoreError::Backend(e.to_string()))?;
                        }

                        // Full Attention (CPU)
                        {
                            let cr = kv_cache.view();
                            for i in 0..num_tokens {
                                let pos = start_pos + i;
                                let q_i = &q_out[i * hidden..(i + 1) * hidden];
                                let out_i = &mut attn_out[i * hidden..(i + 1) * hidden];

                                let seq = pos + 1;
                                gather_bases.clear();
                                for tpos in 0..seq {
                                    let b = page_table.block_for_token(tpos).map_err(|e| CoreError::Backend(e.to_string()))?;
                                    let o = page_table.offset_in_block(tpos).map_err(|e| CoreError::Backend(e.to_string()))?;
                                    gather_bases.push(cr.layout.token_base_elem(b, layer, o).map_err(|e| CoreError::Backend(e.to_string()))?);
                                }
                                cr.attention_single_token_gqa_from_bases(
                                    &gather_bases,
                                    q_i,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    None,
                                    None,
                                    out_i,
                                ).map_err(|e| CoreError::Backend(e.to_string()))?;
                            }
                        }

                        // O Proj
                        let o_w_buf = {
                            let name = format!("l{layer}.o");
                            if !webgpu_weight_cache.contains_key(&name) {
                                let tensor_name = if tensor_prefix.is_empty() { format!("model.layers.{layer}.self_attn.o_proj.weight") } else { format!("{tensor_prefix}.model.layers.{layer}.self_attn.o_proj.weight") };
                                let bytes = file.tensor_bytes(&tensor_name)?;
                                let w: &[u16] = bytemuck::cast_slice(bytes);
                                webgpu_weight_cache.insert(name.clone(), ctx.upload_f16(w));
                            }
                            webgpu_weight_cache.get(&name).unwrap().clone()
                        };
                        ctx.matmul_batch_f16(&o_w_buf, num_tokens as u32, hidden as u32, hidden as u32, &attn_out, &mut x_norm).await;
                        for i in 0..x.len() { x[i] += x_norm[i]; }

                        // FFN Norm
                        let ffn_ln_w_buf = {
                            let name = format!("l{layer}.ffn_norm");
                            if !webgpu_weight_cache.contains_key(&name) {
                                let tensor_name = if tensor_prefix.is_empty() { format!("model.layers.{layer}.post_attention_layernorm.weight") } else { format!("{tensor_prefix}.model.layers.{layer}.post_attention_layernorm.weight") };
                                let bytes = file.tensor_bytes(&tensor_name)?;
                                let w: &[f32] = bytemuck::cast_slice(bytes);
                                webgpu_weight_cache.insert(name.clone(), ctx.upload_f32(w));
                            }
                            webgpu_weight_cache.get(&name).unwrap().clone()
                        };
                        ctx.rms_norm_batch(&ffn_ln_w_buf, num_tokens as u32, hidden as u32, rms_norm_eps, &x, &mut x_norm).await;

                        // Gate + Up Projs
                        let gate_w_buf = {
                            let name = format!("l{layer}.gate");
                            if !webgpu_weight_cache.contains_key(&name) {
                                let tensor_name = if tensor_prefix.is_empty() { format!("model.layers.{layer}.mlp.gate_proj.weight") } else { format!("{tensor_prefix}.model.layers.{layer}.mlp.gate_proj.weight") };
                                let bytes = file.tensor_bytes(&tensor_name)?;
                                let w: &[u16] = bytemuck::cast_slice(bytes);
                                webgpu_weight_cache.insert(name.clone(), ctx.upload_f16(w));
                            }
                            webgpu_weight_cache.get(&name).unwrap().clone()
                        };
                        let up_w_buf = {
                            let name = format!("l{layer}.up");
                            if !webgpu_weight_cache.contains_key(&name) {
                                let tensor_name = if tensor_prefix.is_empty() { format!("model.layers.{layer}.mlp.up_proj.weight") } else { format!("{tensor_prefix}.model.layers.{layer}.mlp.up_proj.weight") };
                                let bytes = file.tensor_bytes(&tensor_name)?;
                                let w: &[u16] = bytemuck::cast_slice(bytes);
                                webgpu_weight_cache.insert(name.clone(), ctx.upload_f16(w));
                            }
                            webgpu_weight_cache.get(&name).unwrap().clone()
                        };
                        let (gate, up) = gate_up.split_at_mut(num_tokens * inter);
                        ctx.matmul_batch_f16(&gate_w_buf, num_tokens as u32, inter as u32, hidden as u32, &x_norm, gate).await;
                        ctx.matmul_batch_f16(&up_w_buf, num_tokens as u32, inter as u32, hidden as u32, &x_norm, up).await;
                        ctx.silu_mul(gate, up).await;

                        // Down Proj
                        let down_w_buf = {
                            let name = format!("l{layer}.down");
                            if !webgpu_weight_cache.contains_key(&name) {
                                let tensor_name = if tensor_prefix.is_empty() { format!("model.layers.{layer}.mlp.down_proj.weight") } else { format!("{tensor_prefix}.model.layers.{layer}.mlp.down_proj.weight") };
                                let bytes = file.tensor_bytes(&tensor_name)?;
                                let w: &[u16] = bytemuck::cast_slice(bytes);
                                webgpu_weight_cache.insert(name.clone(), ctx.upload_f16(w));
                            }
                            webgpu_weight_cache.get(&name).unwrap().clone()
                        };
                        ctx.matmul_batch_f16(&down_w_buf, num_tokens as u32, hidden as u32, inter as u32, gate, &mut x_norm).await;
                        for i in 0..x.len() { x[i] += x_norm[i]; }
                    }

                    if return_logits {
                        gpu_last_x = Some(x[(num_tokens - 1) * hidden..].to_vec());
                    }
                }
            }
            if let Some(last_x) = gpu_last_x {
                return Ok(Some(self.compute_logits(&last_x)?));
            }
            if let LlamaLinearBackend::WebGpu { .. } = &self.linear_backend {
                return Ok(None);
            }
        }

        // Metal path (fused graph)
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            if let Some(gs) = &mut self.graph_state {
                if kv_cache.encoding() != cellm_cache::KvEncodingKind::TurboQuant {
                    return gs.prefill_fused(
                        x_all, &self.cfg, &self.tensor_prefix,
                        kv_cache, page_table, start_pos, return_logits,
                    );
                }
            }
        }

        // Fallback to per-token path (non-Metal/non-WebGPU or TurboQuant KV cache).
        let hidden = self.cfg.hidden_size;
        let num_tokens = x_all.len() / hidden;
        let mut last_logits = None;
        for i in 0..num_tokens {
            let pos = start_pos + i;
            let mut x = vec![0.0f32; hidden];
            x.copy_from_slice(&x_all[i * hidden..(i + 1) * hidden]);
            let is_last = i == num_tokens - 1;
            let logits = self.step_inner(&x, pos, page_table, kv_cache, return_logits && is_last)?;
            if is_last && return_logits {
                last_logits = Some(logits);
            }
        }
        Ok(last_logits)
    }

    fn step_inner(
        &mut self,
        x0: &[f32],
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        return_logits: bool,
    ) -> Result<Vec<f32>, CoreError> {
        // Ensure caches are populated (lazy init on first call)
        self.ensure_caches()?;

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let mut disable_graph = false;
            if let Some(gs) = &mut self.graph_state {
                if kv_cache.encoding() == cellm_cache::KvEncodingKind::TurboQuant {
                    // Fallback for TurboQuant
                } else {
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

                    match gs.step_fused(x0, &self.cfg, &self.tensor_prefix, kv_cache, page_table, pos, token_off, block_id as u32, return_logits) {
                        Ok(maybe_logits) => {
                            if let Some(logits) = maybe_logits {
                                let has_non_finite = logits.iter().any(|v| !v.is_finite());
                                if has_non_finite {
                                    eprintln!("llama fused graph: non-finite logits detected; disabling fused graph and continuing with non-fused Metal path");
                                    disable_graph = true;
                                } else {
                                    return Ok(logits);
                                }
                            } else {
                                return Ok(vec![]);
                            }
                        }
                        Err(e) => {
                            eprintln!("llama fused graph: step_fused failed at pos {pos}: {e}; falling back to CPU");
                            disable_graph = true;
                        }
                    }
                }
            }
            if disable_graph {
                self.graph_state = None;
            }
        }

        // Use references to config fields to avoid cloning
        let hidden = self.cfg.hidden_size;
        let n_heads = self.cfg.num_attention_heads;
        let n_kv_heads = self.cfg.num_key_value_heads;
        let head_dim = self.cfg.head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = self.cfg.intermediate_size;
        let eps = self.cfg.rms_norm_eps;
        let rope_theta = self.cfg.rope_theta;

        if head_dim * n_heads != hidden {
            return Err(CoreError::Backend(
                "llama: hidden_size must be divisible by num_attention_heads".into(),
            ));
        }

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

        // Copy x0 into a working buffer
        let mut x = vec![0.0f32; hidden];
        x.copy_from_slice(x0);

        // Per-layer scratch buffers
        let debug_timing = std::env::var("CELLM_STEP_TIMING").is_ok();
        let mut t_linear_ns = 0u64;
        let mut t_attn_ns = 0u64;
        let mut t_norm_ns = 0u64;
        let mut t_rope_ns = 0u64;
        #[cfg(not(target_arch = "wasm32"))]
        let t_step_start = std::time::Instant::now();

        let mut attn_norm_w = vec![0.0f32; hidden];
        let mut x_norm = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; hidden];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; hidden];
        let mut attn_proj = vec![0.0f32; hidden];
        let mut post_norm_w = vec![0.0f32; hidden];
        let mut mlp_in = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        let mut down = vec![0.0f32; hidden];
        let mut gather_bases: Vec<usize> = Vec::with_capacity(256);

        // Clone cached layer names to avoid borrow conflicts with mutable self calls
        let layer_names = self.cached_layer_names.clone();

        for layer in 0..self.max_layers {
            let use_metal_norm = self.metal_ops.is_some() && self.use_metal_norm;
            let use_metal_rope = self.metal_ops.is_some() && self.use_metal_rope && self.rope_interleaved;

            // Attention input norm.
            #[cfg(not(target_arch = "wasm32"))]
            let t0 = std::time::Instant::now();
            let ln = &layer_names[layer];
            if use_metal_norm {
                let w = self.tensor_f16(&ln.attn_norm)?;
                let w_ptr = w.as_ptr();
                let w_len = w.len();
                let w = unsafe { std::slice::from_raw_parts(w_ptr, w_len) };
                let ck = format!("llama.layer.{layer}.attn_norm");
                self.metal_ops.as_ref().unwrap()
                    .rms_norm_f16w(&x, &w, eps, false, &ck, &mut x_norm)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                // Use cached norm weights
                if let Some(cached_w) = self.cached_norm_weights.get(&ln.attn_norm) {
                    attn_norm_w.copy_from_slice(cached_w);
                } else {
                    self.rmsnorm_weight(&ln.attn_norm, &mut attn_norm_w)?;
                }
                rms_norm_f32(&x, &attn_norm_w, eps, &mut x_norm);
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                t_norm_ns += t0.elapsed().as_nanos() as u64;
            }

            // QKV projections (HF weights are [out, in]).
            #[cfg(not(target_arch = "wasm32"))]
            let t0 = std::time::Instant::now();
            let fused_qkv = self.linear_qkv_f16_out_in(
                &x_norm,
                &ln.q_proj,
                hidden,
                &ln.k_proj,
                kv_dim,
                &ln.v_proj,
                kv_dim,
                hidden,
                &mut q,
                &mut k,
                &mut v,
            )?;
            if !fused_qkv {
                self.linear_f16_out_in(&x_norm, &ln.q_proj, hidden, hidden, &mut q)?;
                self.linear_f16_out_in(&x_norm, &ln.k_proj, kv_dim, hidden, &mut k)?;
                self.linear_f16_out_in(&x_norm, &ln.v_proj, kv_dim, hidden, &mut v)?;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                t_linear_ns += t0.elapsed().as_nanos() as u64;
            }

            #[cfg(not(target_arch = "wasm32"))]
            let t0 = std::time::Instant::now();
            if use_metal_rope {
                let ops = self.metal_ops.as_ref().unwrap();
                ops.rope_adj_f32(&mut q, n_heads, head_dim, pos, rope_theta)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                ops.rope_adj_f32(&mut k, n_kv_heads, head_dim, pos, rope_theta)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else if !self.cached_inv_freqs.is_empty() {
                // Use precomputed scaled inv_freqs (for llama3-style RoPE scaling)
                if self.rope_interleaved {
                    rope_interleaved_inplace_f32_with_freqs(&mut q, n_heads, head_dim, pos, &self.cached_inv_freqs);
                    rope_interleaved_inplace_f32_with_freqs(&mut k, n_kv_heads, head_dim, pos, &self.cached_inv_freqs);
                } else {
                    rope_non_interleaved_inplace_f32_with_freqs(&mut q, n_heads, head_dim, head_dim, pos, &self.cached_inv_freqs);
                    rope_non_interleaved_inplace_f32_with_freqs(&mut k, n_kv_heads, head_dim, head_dim, pos, &self.cached_inv_freqs);
                }
            } else if self.rope_interleaved {
                rope_interleaved_inplace_f32(&mut q, n_heads, head_dim, pos, rope_theta);
                rope_interleaved_inplace_f32(&mut k, n_kv_heads, head_dim, pos, rope_theta);
            } else {
                rope_non_interleaved_inplace_f32(&mut q, n_heads, head_dim, head_dim, pos, rope_theta);
                rope_non_interleaved_inplace_f32(&mut k, n_kv_heads, head_dim, head_dim, pos, rope_theta);
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                t_rope_ns += t0.elapsed().as_nanos() as u64;
            }

            // Write new token K/V into paged cache.
            {
                let mut cv = kv_cache.view_mut();
                cv.write_token(block_id, layer, token_off, &k, &v)?;
            }

            // Gather historical K/V and run attention for this token.
            #[cfg(not(target_arch = "wasm32"))]
            let t0 = std::time::Instant::now();
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
                head_dim,
                None, // attn_scale
                None, // soft_cap
                &mut attn_out,
            )?;
            #[cfg(not(target_arch = "wasm32"))]
            {
                t_attn_ns += t0.elapsed().as_nanos() as u64;
            }

            // o_proj: hidden <- hidden
            #[cfg(not(target_arch = "wasm32"))]
            let t0 = std::time::Instant::now();
            self.linear_f16_out_in(
                &attn_out,
                &ln.o_proj,
                hidden,
                hidden,
                &mut attn_proj,
            )?;
            #[cfg(not(target_arch = "wasm32"))]
            {
                t_linear_ns += t0.elapsed().as_nanos() as u64;
            }

            for i in 0..hidden {
                x[i] += attn_proj[i];
            }

            let ffn_norm_name = &ln.ffn_norm;
            let gate_name = &ln.gate_proj;
            let up_name = &ln.up_proj;
            let down_name = &ln.down_proj;

            let mut ffn_done = false;
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                let has_metal = self.metal_ops.is_some();
                if has_metal {
                    // Check if all weights are f16
                    let mut all_f16 = false;
                    let gate_res = self.resolve_name(gate_name);
                    if let Ok(res_name) = gate_res {
                        if let Some(meta) = self.tensor_meta_by_exact_name(&res_name) {
                            if meta.dtype == "f16" {
                                all_f16 = true;
                            }
                        }
                    }

                    if all_f16 {
                        let norm_w_data: Vec<u16>;
                        let w1b_data: Vec<u16>;
                        let w3b_data: Vec<u16>;
                        let w2b_data: Vec<u16>;

                        if let (Ok(nw), Ok(w1), Ok(w3), Ok(w2)) = (
                            self.tensor_f16(ffn_norm_name),
                            self.tensor_f16(gate_name),
                            self.tensor_f16(up_name),
                            self.tensor_f16(down_name),
                        ) {
                            norm_w_data = nw.to_vec();
                            w1b_data = w1.to_vec();
                            w3b_data = w3.to_vec();
                            w2b_data = w2.to_vec();

                            let ops = self.metal_ops.as_ref().unwrap();

                            if ops.ensure_named_buf("ffn_x", hidden).is_ok()
                                && ops.ensure_named_buf("ffn_norm_out", hidden).is_ok()
                                && ops.ensure_named_buf("ffn_gate", inter).is_ok()
                                && ops.ensure_named_buf("ffn_up", inter).is_ok()
                                && ops.ensure_named_buf("ffn_down", hidden).is_ok()
                            {
                                if let (Ok(norm_wb), Ok(w1b), Ok(w3b), Ok(w2b)) = (
                                    ops.ensure_tensor_cached(ffn_norm_name, &norm_w_data),
                                    ops.ensure_tensor_cached(gate_name, &w1b_data),
                                    ops.ensure_tensor_cached(up_name, &w3b_data),
                                    ops.ensure_tensor_cached(down_name, &w2b_data),
                                ) {
                                    if ops.write_named_buf("ffn_x", &x).is_ok() {
                                        if let (Ok(xb), Ok(nb), Ok(gb), Ok(ub), Ok(db)) = (
                                            ops.get_named_buf("ffn_x"),
                                            ops.get_named_buf("ffn_norm_out"),
                                            ops.get_named_buf("ffn_gate"),
                                            ops.get_named_buf("ffn_up"),
                                            ops.get_named_buf("ffn_down"),
                                        ) {
                                            let batch_res = ops.run_batch(|enc| {
                                                ops.encode_rms_norm_f16w(enc, &xb, &norm_wb, &nb, hidden, eps, false);
                                                ops.encode_mv_f16_bias(enc, &w1b, &nb, None, &gb, inter, hidden);
                                                ops.encode_mv_f16_bias(enc, &w3b, &nb, None, &ub, inter, hidden);
                                                ops.encode_silu_mul_f32_inplace(enc, &gb, &ub, inter);
                                                ops.encode_mv_f16_bias(enc, &w2b, &gb, None, &db, hidden, inter);
                                                ops.encode_add_f32_inplace(enc, &xb, &db, hidden);
                                                Ok(())
                                            });

                                            if batch_res.is_ok() {
                                                if ops.read_named_buf("ffn_x", &mut x).is_ok() {
                                                    ffn_done = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !ffn_done {
                // CPU fallback or non-f16 path
                // Post-attn norm.
                #[cfg(not(target_arch = "wasm32"))]
                let t0 = std::time::Instant::now();
                if use_metal_norm {
                    let w = self.tensor_f16(ffn_norm_name)?;
                    let w_ptr = w.as_ptr();
                    let w_len = w.len();
                    let w = unsafe { std::slice::from_raw_parts(w_ptr, w_len) };
                    let ck = format!("llama.layer.{layer}.mlp_norm");
                    self.metal_ops.as_ref().unwrap()
                        .rms_norm_f16w(&x, &w, eps, false, &ck, &mut mlp_in)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                } else {
                    // Use cached norm weights
                    if let Some(cached_w) = self.cached_norm_weights.get(ffn_norm_name) {
                        post_norm_w.copy_from_slice(cached_w);
                    } else {
                        self.rmsnorm_weight(ffn_norm_name, &mut post_norm_w)?;
                    }
                    rms_norm_f32(&x, &post_norm_w, eps, &mut mlp_in);
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    t_norm_ns += t0.elapsed().as_nanos() as u64;
                }

                // MLP: gate_proj + up_proj -> silu(gate)*up -> down_proj
                #[cfg(not(target_arch = "wasm32"))]
                let t0 = std::time::Instant::now();
                self.linear_f16_out_in(&mlp_in, gate_name, inter, hidden, &mut gate)?;
                self.linear_f16_out_in(&mlp_in, up_name, inter, hidden, &mut up)?;

                // silu(gate) in-place: x * sigmoid(x)
                for g in gate.iter_mut() {
                    let s = 1.0 / (1.0 + (-*g).exp());
                    *g = *g * s;
                }
                for i in 0..gate.len() {
                    gate[i] *= up[i];
                }

                self.linear_f16_out_in(&gate, down_name, hidden, inter, &mut down)?;
                #[cfg(not(target_arch = "wasm32"))]
                {
                    t_linear_ns += t0.elapsed().as_nanos() as u64;
                }
                for i in 0..hidden {
                    x[i] += down[i];
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        if debug_timing {
            let total = t_step_start.elapsed().as_nanos() as u64;
            eprintln!(
                "STEP_TIMING pos={} linear={:.2}ms attn={:.2}ms norm={:.2}ms rope={:.2}ms other={:.2}ms total={:.2}ms",
                pos,
                t_linear_ns as f64 / 1e6,
                t_attn_ns as f64 / 1e6,
                t_norm_ns as f64 / 1e6,
                t_rope_ns as f64 / 1e6,
                (total.saturating_sub(t_linear_ns + t_attn_ns + t_norm_ns + t_rope_ns)) as f64 / 1e6,
                total as f64 / 1e6,
            );
        }
        if !return_logits {
            return Ok(vec![]);
        }
        self.compute_logits(&x)
    }

    fn compute_logits(&self, x: &[f32]) -> Result<Vec<f32>, CoreError> {
        let hidden = self.cfg.hidden_size;
        let cfg = &self.cfg;

        // Final norm.
        let use_metal_norm = self.metal_ops.is_some() && self.use_metal_norm;
        let mut x_final = vec![0.0f32; hidden];
        if use_metal_norm {
            let w = self.tensor_f16("model.norm.weight")?;
            let w_ptr = w.as_ptr();
            let w_len = w.len();
            let w = unsafe { std::slice::from_raw_parts(w_ptr, w_len) };
            self.metal_ops.as_ref().unwrap()
                .rms_norm_f16w(x, &w, cfg.rms_norm_eps, false, "llama.norm", &mut x_final)
                .map_err(|e| CoreError::Backend(e.to_string()))?;
        } else {
            let mut norm_w = vec![0.0f32; hidden];
            self.rmsnorm_weight("model.norm.weight", &mut norm_w)?;
            rms_norm_f32(x, &norm_w, cfg.rms_norm_eps, &mut x_final);
        }

        // Logits via tied embeddings: logits[v] = dot(x_final, embed[v])
        let vocab = cfg.vocab_size;
        let mut buf = vec![0.0f32; vocab];

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

        let lm_dtype_raw = lm_meta.dtype.clone();
        let lm_dtype = lm_dtype_raw.trim().to_ascii_lowercase();
        let use_metal_logits = self.metal_ops.is_some()
            && lm_dtype != "i4" && lm_dtype != "i2";

        if use_metal_logits {
            match lm_dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16_by_exact_name(&lm_src_resolved)?;
                    self.metal_ops.as_ref().unwrap()
                        .logits_f16(&x_final, &w, vocab, hidden, &lm_src_resolved, &mut buf)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                "i8" => {
                    let w = self.tensor_i8_by_exact_name(&lm_src_resolved)?;
                    let s = self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?;
                    self.metal_ops.as_ref().unwrap()
                        .logits_i8(&x_final, &w, &s, vocab, hidden, &lm_src_resolved, &mut buf)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                "i4" | "i2" => {}
                other => return Err(CoreError::Backend(format!("unsupported lm dtype {other} for {lm_src_resolved}"))),
            }
            if lm_dtype == "f16" || lm_dtype == "i8" {
                sanitize_logits_non_finite(&mut buf, "llama metal logits");
                return Ok(buf);
            }
        }

        let lm_src_f16 = if lm_dtype == "f16" {
            Some(self.tensor_f16_by_exact_name(&lm_src_resolved)?)
        } else {
            None
        };
        let lm_src_i8 = if lm_dtype == "i8" {
            Some(self.tensor_i8_by_exact_name(&lm_src_resolved)?)
        } else {
            None
        };
        let lm_i8_scales = if lm_dtype == "i8" {
            Some(self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?)
        } else {
            None
        };
        let lm_src_i4 = if lm_dtype == "i4" {
            Some(self.tensor_u8_by_exact_name(&lm_src_resolved)?)
        } else {
            None
        };
        let lm_i4_scales = if lm_dtype == "i4" {
            Some(self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?)
        } else {
            None
        };

        if let Some(weights) = lm_src_f16 {
            cellm_kernels::cpu_kernels::matmul_f16_f32(
                weights,
                vocab,
                hidden,
                &x_final,
                &mut buf,
            );
            sanitize_logits_non_finite(&mut buf, "llama CPU f16 logits");
            return Ok(buf);
        }

        for vid in 0..vocab {
            let mut dot = if let Some(w) = lm_src_f16 {
                let row = &w[vid * hidden..(vid + 1) * hidden];
                let mut acc = 0.0f32;
                for i in 0..hidden {
                    acc += x_final[i] * f16::from_bits(row[i]).to_f32();
                }
                acc
            } else if let Some(w) = lm_src_i8 {
                let scales = lm_i8_scales.as_ref().unwrap();
                let row = &w[vid * hidden..(vid + 1) * hidden];
                let scale = f16::from_bits(scales[vid]).to_f32();
                let mut acc = 0.0f32;
                for i in 0..hidden {
                    acc += x_final[i] * ((row[i] as f32) * scale);
                }
                acc
            } else if let (Some(w), Some(scales)) = (&lm_src_i4, &lm_i4_scales) {
                let row_stride = hidden.div_ceil(2);
                let row = &w[vid * row_stride..(vid + 1) * row_stride];
                let scale = f16::from_bits(scales[vid]).to_f32();
                // Unpack i4 values: each byte holds 2 values
                let mut acc = 0.0f32;
                for i in 0..hidden {
                    let packed = row[i / 2];
                    let val = if i % 2 == 0 {
                        ((packed & 0x0F) as i8 - 8) as f32
                    } else {
                        ((packed >> 4) as i8 - 8) as f32
                    };
                    acc += x_final[i] * val * scale;
                }
                acc
            } else {
                return Err(CoreError::Backend(format!(
                    "unsupported lm dtype {}",
                    lm_dtype_raw
                )));
            };
            if !dot.is_finite() {
                dot = f32::NEG_INFINITY;
            }
            buf[vid] = dot;
        }
        Ok(buf)
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

    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, CoreError> {
        let resolved = self.resolve_name(name)?;
        let bytes = self.file.tensor_bytes(&resolved)?;
        let dtype = self.tensor_meta_by_exact_name(&resolved).map(|t| t.dtype.as_str()).unwrap_or("f16");
        if dtype == "f32" {
            Ok(cast_slice(bytes).to_vec())
        } else if dtype == "f16" {
            let u: &[u16] = cast_slice(bytes);
            Ok(u.iter().map(|&v| f16::from_bits(v).to_f32()).collect())
        } else {
            Err(CoreError::Backend(format!("tensor {name} unsupported dtype {dtype} for f32 conversion")))
        }
    }

    fn tensor_i8_by_exact_name(&self, resolved: &str) -> Result<&[i8], CoreError> {
        let bytes = self.file.tensor_bytes(resolved)?;
        Ok(cast_slice(bytes))
    }

    fn tensor_u8_by_exact_name(&self, resolved: &str) -> Result<&[u8], CoreError> {
        let bytes = self.file.tensor_bytes(resolved)?;
        Ok(bytes)
    }

    fn tensor_meta_by_exact_name(&self, resolved: &str) -> Option<&crate::CellmTensorIndex> {
        self.file.tensor_index(resolved)
    }

    fn embed_token(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        let hidden = out.len();
        let name = "model.embed_tokens.weight";
        let resolved = self.resolve_name(name)?;
        let meta = self.tensor_meta_by_exact_name(&resolved).ok_or_else(|| {
            CoreError::Backend(format!("unknown tensor {resolved}"))
        })?;
        let vocab = self.cfg.vocab_size;
        let t = (token as usize) % vocab;

        match meta.dtype.as_str() {
            "f16" => {
                let embed = self.tensor_f16_by_exact_name(&resolved)?;
                let row = &embed[t * hidden..(t + 1) * hidden];
                for i in 0..hidden {
                    out[i] = f16::from_bits(row[i]).to_f32();
                }
            }
            "i8" => {
                let embed = self.tensor_i8_by_exact_name(&resolved)?;
                let scales = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                let scale = f16::from_bits(scales[t]).to_f32();
                let row = &embed[t * hidden..(t + 1) * hidden];
                for i in 0..hidden {
                    out[i] = (row[i] as f32) * scale;
                }
            }
            "i4" => {
                let embed = self.tensor_u8_by_exact_name(&resolved)?;
                let scales = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                let scale = f16::from_bits(scales[t]).to_f32();
                let row_stride = hidden.div_ceil(2);
                let row = &embed[t * row_stride..(t + 1) * row_stride];
                for i in 0..hidden {
                    let packed = row[i / 2];
                    let val = if i % 2 == 0 {
                        ((packed & 0x0F) as i8 - 8) as f32
                    } else {
                        ((packed >> 4) as i8 - 8) as f32
                    };
                    out[i] = val * scale;
                }
            }
            other => {
                return Err(CoreError::Backend(format!(
                    "unsupported embed dtype for {name}: {other}"
                )));
            }
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
                "i4" => {
                    let w = self.tensor_u8_by_exact_name(&resolved)?;
                    let s = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                    let w_ptr = w.as_ptr();
                    let w_len = w.len();
                    let s_ptr = s.as_ptr();
                    let s_len = s.len();
                    let w = unsafe { std::slice::from_raw_parts(w_ptr, w_len) };
                    let s = unsafe { std::slice::from_raw_parts(s_ptr as *const u16, s_len) };
                    self.metal_ops
                        .as_mut()
                        .unwrap()
                        .logits_i4(x, w, s, out_dim, in_dim, in_dim, &resolved, out)
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
                // On macOS/iOS, use cblas_sgemm with cached f32 weights for optimal throughput
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                {
                    let cache_key = resolved.clone();
                    if !self.f32_weight_cache.contains_key(&cache_key) {
                        let mut w_f32 = vec![0.0f32; out_dim * in_dim];
                        for i in 0..w.len() {
                            w_f32[i] = f16::from_bits(w[i]).to_f32();
                        }
                        self.f32_weight_cache.insert(cache_key.clone(), w_f32);
                    }
                    let w_f32 = self.f32_weight_cache.get(&cache_key).unwrap();
                    // out = x (1 x in_dim) * W^T (in_dim x out_dim)
                    // cblas_sgemm: C = alpha * A * B + beta * C
                    // A = x [1 x in_dim], B = W^T [in_dim x out_dim] (= W [out_dim x in_dim] transposed)
                    // We use CblasTrans on B: B is stored row-major as [out_dim, in_dim]
                    unsafe {
                        cblas_sgemm(
                            101, // CblasRowMajor
                            111, // CblasNoTrans  (A = x, 1 x K)
                            112, // CblasTrans    (B = W, N x K -> K x N after transpose)
                            1,                    // M = 1 (single token)
                            out_dim as i32,       // N = out_dim
                            in_dim as i32,        // K = in_dim
                            1.0,                  // alpha
                            x.as_ptr(),           // A
                            in_dim as i32,        // lda
                            w_f32.as_ptr(),       // B (stored as [out_dim, in_dim])
                            in_dim as i32,        // ldb
                            0.0,                  // beta
                            out.as_mut_ptr(),     // C
                            out_dim as i32,       // ldc
                        );
                    }
                    // skip the non-macOS fallback
                }
                #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                cellm_kernels::cpu_kernels::matmul_f16_f32(w, out_dim, in_dim, x, out);
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
                let s = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                if s.len() != out_dim {
                    return Err(CoreError::Backend(format!(
                        "weight {weight_name} qscale len mismatch: {} expected {}",
                        s.len(),
                        out_dim
                    )));
                }
                cellm_kernels::cpu_kernels::matmul_i8_f32(w, s, out_dim, in_dim, x, out);
            }
            "i4" => {
                let resolved = self.resolve_name(weight_name)?;
                let w = self.tensor_u8_by_exact_name(&resolved)?;
                let s = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                cellm_kernels::cpu_kernels::matmul_i4_f32(w, s, out_dim, in_dim, in_dim, x, out);
            }
            "u32" => {
                // BASE/MLX-style unsigned affine Q4 with per-group f32
                // scales and biases. The packed bytes are consumed directly;
                // their u32 dtype records the established affine-i4 convention.
                let w = self.tensor_u8_by_exact_name(&resolved)?;
                let base = resolved.strip_suffix(".weight").unwrap_or(&resolved);
                let scales = self.tensor_f32(&format!("{base}.scales"))?;
                let biases = self.tensor_f32(&format!("{base}.biases"))?;
                let groups_per_row = scales.len() / out_dim;
                if groups_per_row == 0 || scales.len() != biases.len() {
                    return Err(CoreError::Backend(format!(
                        "invalid affine-i4 parameters for {weight_name}"
                    )));
                }
                let group_size = in_dim.div_ceil(groups_per_row);
                cellm_kernels::cpu_kernels::matmul_affine_i4_f32(
                    w, &scales, &biases, out_dim, in_dim, group_size, x, out,
                );
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
        // The checkpoint uses the "model.text_model" namespace.
        // Returning an empty prefix lets the weight‑lookup fallback prepend the correct name.
        return Ok(String::new());
    }
    Err(CoreError::Backend(
        "missing required llama tensors: model.embed_tokens.weight/model.norm.weight".into(),
    ))
}

/// Precompute RoPE inverse frequencies with optional llama3-style scaling.
/// Returns empty vec if no scaling is needed.
pub fn compute_inv_freqs(
    head_dim: usize,
    rope_theta: f32,
    rope_scaling_type: &Option<String>,
    rope_scaling_factor: Option<f32>,
    original_max_position_embeddings: Option<usize>,
    low_freq_factor: Option<f32>,
    high_freq_factor: Option<f32>,
) -> Vec<f32> {
    let needs_scaling = rope_scaling_type
        .as_deref()
        .map(|t| t == "llama3")
        .unwrap_or(false);

    if !needs_scaling {
        return Vec::new();
    }

    let factor = rope_scaling_factor.unwrap_or(1.0);
    let original_max = original_max_position_embeddings.unwrap_or(8192) as f32;
    let low_freq = low_freq_factor.unwrap_or(1.0);
    let high_freq = high_freq_factor.unwrap_or(4.0);
    let low_freq_wavelen = original_max / low_freq;
    let high_freq_wavelen = original_max / high_freq;

    let half = head_dim / 2;
    let mut inv_freqs = Vec::with_capacity(half);
    for i in 0..half {
        let inv_freq = rope_theta.powf(-(2.0 * i as f32) / head_dim as f32);
        let wavelen = std::f32::consts::TAU / inv_freq;
        let scaled = if wavelen < high_freq_wavelen {
            // High frequency: don't scale
            inv_freq
        } else if wavelen > low_freq_wavelen {
            // Low frequency: scale by factor
            inv_freq / factor
        } else {
            // Transition: smooth interpolation
            let smooth = (original_max / wavelen - low_freq) / (high_freq - low_freq);
            (1.0 - smooth) * inv_freq / factor + smooth * inv_freq
        };
        inv_freqs.push(scaled);
    }
    inv_freqs
}
