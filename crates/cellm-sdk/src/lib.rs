// Author: Jeffrey Asante (https://jeffasante.github.io/)
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
// use std::process::Command; // Removed as LiteRtProxy (which used it) is gone.

use cellm_cache::{KVCache, KvEncodingKind, KvStorageKind, PageTable};
use cellm_core::KvCacheLayout;
use cellm_model::{gemma::GemmaRunner, llama::LlamaRunner, qwen::QwenRunner, lfm::LfmRunner, deepseek_v4::DeepSeekV4Runner, CellmFile, ModelConfig};
use cellm_scheduler::{BatchDetector, BatchSessionInfo, PolicyExecutor, RoundRobinScheduler, SchedulingPolicy, Session as SchedSession, SessionState, ThermalLevel, ThermalPolicy};
use serde_json::Value;

#[cfg(not(target_arch = "wasm32"))]
type StatsInstant = std::time::Instant;
#[cfg(target_arch = "wasm32")]
type StatsInstant = ();

#[cfg(not(target_arch = "wasm32"))]
fn stats_instant_now() -> StatsInstant {
    std::time::Instant::now()
}

#[cfg(target_arch = "wasm32")]
fn stats_instant_now() -> StatsInstant {}

#[cfg(not(target_arch = "wasm32"))]
fn stats_elapsed_secs(snapshot: &StatsInstant) -> Option<f64> {
    Some(snapshot.elapsed().as_secs_f64())
}

#[cfg(target_arch = "wasm32")]
fn stats_elapsed_secs(_: &StatsInstant) -> Option<f64> {
    None
}

pub type SessionId = u64;

pub mod embed_ffi;
pub mod ffi;
pub mod vlm;
#[cfg(target_os = "android")]
pub mod jni;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub tokens_per_block: usize,
    pub total_blocks: usize,
    pub top_k: usize,
    pub temperature: f64,
    pub repeat_penalty: f64,
    pub repeat_window: usize,
    pub seed: u64,
    pub backend: BackendKind,
    pub kv_encoding: KvEncodingKind,
    pub turboq_int8_dot: bool,
    pub turboq_qjl_corr: bool,
    /// Scheduling policy: Fair (round-robin), LatencyFirst, or ThroughputFirst.
    pub scheduling_policy: SchedulingPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Metal,
    WebGpu,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            tokens_per_block: 16,
            total_blocks: 256,
            top_k: 40,
            temperature: 0.8,
            repeat_penalty: 1.05,
            repeat_window: 64,
            seed: 1,
            backend: BackendKind::Cpu,
            kv_encoding: KvEncodingKind::F16,
            turboq_int8_dot: true,
            turboq_qjl_corr: true,
            scheduling_policy: SchedulingPolicy::Fair,
        }
    }
}

#[derive(Debug)]
struct EngineSession {
    page_table: PageTable,
    next_pos: usize,
    last_token: Option<u32>,
    recent: Vec<u32>,
    pending_out: VecDeque<u32>,
    rng: XorShift64,
    cached_prompt: Vec<u32>,
    cached_next_pos: usize,
    cached_last_token: Option<u32>,
    cached_recent: Vec<u32>,
}

enum Runner {
    Llama(LlamaRunner),
    Gemma(GemmaRunner),
    Qwen(QwenRunner),
    Lfm(LfmRunner),
    DeepSeekV4(DeepSeekV4Runner),
}

impl Runner {
    pub fn prefill(&mut self, tokens: &[u32], start_pos: usize, pt: &mut PageTable, kv: &mut KVCache) -> anyhow::Result<()> {
        match self {
            Runner::Llama(r) => r.prefill(tokens, start_pos, pt, kv).map_err(|e| anyhow::anyhow!(e)),
            Runner::Gemma(r) => {
                // Fallback for Gemma
                for (i, &tok) in tokens.iter().enumerate() {
                    r.step_topk(tok, start_pos + i, pt, kv, 0).map_err(|e| anyhow::anyhow!(e))?;
                }
                Ok(())
            }
            Runner::Qwen(r) => {
                r.prefill(tokens, start_pos, pt, kv).map_err(|e| anyhow::anyhow!(e))
            }
            Runner::Lfm(r) => {
                r.prefill(tokens, start_pos, pt, kv).map_err(|e| anyhow::anyhow!(e))
            }
            Runner::DeepSeekV4(r) => {
                // Fallback for DeepSeekV4
                for (i, &tok) in tokens.iter().enumerate() {
                    r.step_topk(tok, start_pos + i, pt, kv, 0).map_err(|e| anyhow::anyhow!(e))?;
                }
                Ok(())
            }
        }
    }

    pub fn is_stop_token(&self, token: u32) -> bool {
        match self {
            Runner::Llama(r) => Some(token) == r.eos_token_id,
            Runner::Gemma(r) => Some(token) == r.eos_token_id,
            Runner::Qwen(r) => Some(token) == r.eos_token_id,
            Runner::Lfm(r) => Some(token) == r.eos_token_id,
            Runner::DeepSeekV4(r) => Some(token) == r.eos_token_id,
        }
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        match self {
            Runner::Llama(r) => r.eos_token_id,
            Runner::Gemma(r) => r.eos_token_id,
            Runner::Qwen(r) => r.eos_token_id,
            Runner::Lfm(r) => r.eos_token_id,
            Runner::DeepSeekV4(r) => r.eos_token_id,
        }
    }
}



/// cellm public API engine.
///
/// Early wiring that owns a shared paged KV cache while each session owns a `PageTable`.
/// Text-only for now.
pub struct Engine {
    model_path: PathBuf,
    cfg: ModelConfig,
    runner: Runner,
    bos_token_id: Option<u32>,
    backend: BackendKind,
    kv_cache: KVCache,
    sessions: HashMap<SessionId, EngineSession>,
    session_meta: HashMap<SessionId, SchedSession>,
    next_session_id: SessionId,
    rr: RoundRobinScheduler,
    policy_exec: PolicyExecutor,
    batch_detector: BatchDetector,
    thermal: ThermalPolicy,
    top_k: usize,
    temperature: f64,
    repeat_penalty: f64,
    repeat_window: usize,
    seed: u64,
    /// Total tokens generated across all sessions (lifetime).
    total_tokens_generated: u64,
    /// Timestamp of the last stats snapshot for tok/s calculation.
    last_stats_snapshot: StatsInstant,
    /// Tokens generated since last stats snapshot.
    tokens_since_snapshot: u64,
    /// Cached tokens-per-second from the most recent stats() call.
    cached_tok_per_sec: f64,
}

impl Engine {
    pub fn new(model_path: &Path, engine_cfg: EngineConfig) -> anyhow::Result<Self> {
        // Size rayon for single-stream decode before any kernel runs. Idempotent
        // and a no-op once a global pool exists, so it is safe on every create.
        // Without this, embedders reaching us through the C FFI (rather than the
        // `infer` CLI) get rayon's default all-core pool, where Apple E-cores
        // straggle on every join.
        cellm_kernels::cpu_kernels::init_decode_thread_pool();
        apply_turboquant_runtime_config(&engine_cfg);
        let selected_backend = resolve_backend(engine_cfg.backend);
        let file = CellmFile::load(model_path)?;
        let header = file.header.clone();

        let text_model_type = effective_text_model_type(&header);
        let mut runner = match text_model_type.as_str() {
            "llama" | "smollm3" => Runner::Llama(LlamaRunner::load(model_path)?),
            t if t.starts_with("gemma") => Runner::Gemma(GemmaRunner::load(model_path)?),
            t if t.starts_with("qwen") => Runner::Qwen(QwenRunner::load(model_path)?),
            t if t.starts_with("lfm") => Runner::Lfm(LfmRunner::load(model_path)?),
            "deepseek_v4" => Runner::DeepSeekV4(DeepSeekV4Runner::load(model_path)?),
            other => anyhow::bail!(
                "unsupported model_type for Engine: model_type={} effective_text_model_type={other}",
                header.model_type
            ),
        };
        if selected_backend == BackendKind::Metal {
            match &mut runner {
                // Qwen and LFM models may have partial Metal support (e.g. LinearAttention layers
                // fall back to CPU, but matmul still uses Metal). Partial support is fine — the
                // runner sets up metal_ops internally even when enable_metal_full_backend returns
                // false. Only hard-fail if the Metal backend can't be created at all.
                Runner::Qwen(r) => {
                    r.enable_metal_full_backend();
                }
                Runner::Gemma(r) => {
                    if !r.enable_metal_full_backend() {
                        anyhow::bail!("Gemma full-metal backend requested but unavailable");
                    }
                }
                Runner::Llama(r) => {
                    if !r.enable_metal_full_backend() {
                        anyhow::bail!("Llama full-metal backend requested but unavailable");
                    }
                }
                Runner::Lfm(r) => {
                    r.enable_metal_full_backend();
                }
                Runner::DeepSeekV4(r) => {
                    r.enable_metal_full_backend();
                }
            }
        }

        let cfg = match &runner {
            Runner::Llama(r) => r.config().clone(),
            Runner::Gemma(r) => r.config().clone(),
            Runner::Qwen(r) => r.config().clone(),
            Runner::Lfm(r) => r.config().clone(),
            Runner::DeepSeekV4(r) => r.config().clone(),
        };

        let head_dim = match &runner {
            Runner::Llama(_) => cfg.hidden_size / cfg.num_attention_heads,
            Runner::Gemma(_) => infer_gemma_kv_head_dim(&file)?,
            Runner::Qwen(_) => infer_qwen_kv_head_dim(&file)?,
            Runner::Lfm(_) => cfg.hidden_size / cfg.num_attention_heads,
            Runner::DeepSeekV4(_) => cfg.head_dim,
        };

        let layout = KvCacheLayout {
            total_blocks: engine_cfg.total_blocks,
            tokens_per_block: engine_cfg.tokens_per_block,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
        };
        let storage_kind = match selected_backend {
            BackendKind::Cpu => KvStorageKind::Cpu,
            BackendKind::Metal => KvStorageKind::Metal,
            BackendKind::WebGpu => KvStorageKind::Cpu, // CPU and WebGPU both use CPU KV storage for now
        };
        let kv_cache = KVCache::new_with_kind_and_encoding(layout, storage_kind, engine_cfg.kv_encoding)?;

        Ok(Self {
            model_path: model_path.to_path_buf(),
            cfg,
            runner,
            bos_token_id: header.bos_token_id,
            backend: selected_backend,
            kv_cache,
            sessions: HashMap::new(),
            session_meta: HashMap::new(),
            next_session_id: 1,
            rr: RoundRobinScheduler::new(),
            policy_exec: PolicyExecutor::new(engine_cfg.scheduling_policy),
            batch_detector: BatchDetector::new(),
            thermal: ThermalPolicy::default(),
            top_k: engine_cfg.top_k,
            temperature: engine_cfg.temperature,
            repeat_penalty: engine_cfg.repeat_penalty,
            repeat_window: engine_cfg.repeat_window,
            seed: engine_cfg.seed,
            total_tokens_generated: 0,
            last_stats_snapshot: stats_instant_now(),
            tokens_since_snapshot: 0,
            cached_tok_per_sec: 0.0,
        })
    }

    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            BackendKind::Cpu => "cpu",
            BackendKind::Metal => "metal",
            BackendKind::WebGpu => "webgpu",
        }
    }

    /// Construct an Engine from an in-memory model buffer (for WASM targets).
    ///
    /// This mirrors `new()` but uses `CellmFile::load_from_bytes` and the runners'
    /// `from_file` constructors to avoid filesystem access.
    pub fn from_bytes(model_bytes: &[u8], engine_cfg: EngineConfig) -> anyhow::Result<Self> {
        Self::from_vec(model_bytes.to_vec(), engine_cfg)
    }

    /// Construct an Engine from owned in-memory model bytes.
    ///
    /// This is the preferred WASM path because JavaScript already copies the
    /// selected file into WASM memory; keeping ownership avoids another large copy.
    pub fn from_vec(model_bytes: Vec<u8>, engine_cfg: EngineConfig) -> anyhow::Result<Self> {
        apply_turboquant_runtime_config(&engine_cfg);
        let selected_backend = resolve_backend(engine_cfg.backend);
        let file = CellmFile::load_from_vec(model_bytes)?;
        let header = file.header.clone();

        let text_model_type = effective_text_model_type(&header);

        // Compute head_dim from tensor metadata before consuming `file`.
        let head_dim = match text_model_type.as_str() {
            "llama" | "smollm3" => header.hidden_dim / header.num_heads,
            t if t.starts_with("gemma") => {
                let hd = infer_gemma_kv_head_dim(&file)?;
                hd
            }
            t if t.starts_with("qwen") => {
                let hd = infer_qwen_kv_head_dim(&file)?;
                hd
            }
            t if t.starts_with("lfm") => header.hidden_dim / header.num_heads,
            "deepseek_v4" => header.head_dim.unwrap_or(0),
            _ => header.hidden_dim / header.num_heads.max(1),
        };

        let runner = match text_model_type.as_str() {
            "llama" | "smollm3" => Runner::Llama(LlamaRunner::from_file(file)?),
            t if t.starts_with("gemma") => Runner::Gemma(GemmaRunner::from_file(file)?),
            t if t.starts_with("qwen") => Runner::Qwen(QwenRunner::from_file(file)?),
            t if t.starts_with("lfm") => Runner::Lfm(LfmRunner::from_file(file)?),
            "deepseek_v4" => Runner::DeepSeekV4(DeepSeekV4Runner::from_file(file)?),
            other => anyhow::bail!(
                "unsupported model_type for Engine: model_type={} effective_text_model_type={other}",
                header.model_type
            ),
        };

        let cfg = match &runner {
            Runner::Llama(r) => r.config().clone(),
            Runner::Gemma(r) => r.config().clone(),
            Runner::Qwen(r) => r.config().clone(),
            Runner::Lfm(r) => r.config().clone(),
            Runner::DeepSeekV4(r) => r.config().clone(),
        };

        let layout = KvCacheLayout {
            total_blocks: engine_cfg.total_blocks,
            tokens_per_block: engine_cfg.tokens_per_block,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
        };
        let storage_kind = match selected_backend {
            BackendKind::Metal => KvStorageKind::Metal,
            _ => KvStorageKind::Cpu,
        };
        let kv_cache = KVCache::new_with_kind_and_encoding(layout, storage_kind, engine_cfg.kv_encoding)?;

        Ok(Self {
            model_path: std::path::PathBuf::new(),
            cfg,
            runner,
            bos_token_id: header.bos_token_id,
            backend: selected_backend,
            kv_cache,
            sessions: HashMap::new(),
            session_meta: HashMap::new(),
            next_session_id: 1,
            rr: RoundRobinScheduler::new(),
            policy_exec: PolicyExecutor::new(engine_cfg.scheduling_policy),
            batch_detector: BatchDetector::new(),
            thermal: ThermalPolicy::default(),
            top_k: engine_cfg.top_k,
            temperature: engine_cfg.temperature,
            repeat_penalty: engine_cfg.repeat_penalty,
            repeat_window: engine_cfg.repeat_window,
            seed: engine_cfg.seed,
            total_tokens_generated: 0,
            last_stats_snapshot: stats_instant_now(),
            tokens_since_snapshot: 0,
            cached_tok_per_sec: 0.0,
        })
    }

    pub fn generate_text_litert(&mut self, _prompt: &str) -> anyhow::Result<String> {
        anyhow::bail!("LiteRT proxy text generation is not enabled in this build")
    }

    pub fn generate_multimodal_litert(
        &mut self,
        _prompt: &str,
        _image_path: Option<&Path>,
        _audio_path: Option<&Path>,
    ) -> anyhow::Result<String> {
        anyhow::bail!("LiteRT proxy multimodal generation is not enabled in this build")
    }

    pub fn is_litert_proxy(&self) -> bool {
        false
    }

    pub fn create_session(&mut self) -> SessionId {
        let id = self.next_session_id;
        self.next_session_id += 1;

        let tokens_per_block = self.kv_cache.layout().tokens_per_block;
        let page_table = PageTable::new(id, tokens_per_block).expect("valid engine config");

        self.sessions.insert(
            id,
            EngineSession {
                page_table,
                next_pos: 0,
                last_token: None,
                recent: Vec::new(),
                pending_out: VecDeque::new(),
                rng: XorShift64::seeded(self.seed_for_session(id)),
                cached_prompt: Vec::new(),
                cached_next_pos: 0,
                cached_last_token: None,
                cached_recent: Vec::new(),
            },
        );
        self.session_meta.insert(id, SchedSession::new(id));
        id
    }

    /// Submit token ids (already-tokenized) and return the next token id (greedy).
    pub fn submit_tokens(&mut self, id: SessionId, tokens: &[u32]) -> anyhow::Result<u32> {
        let (next, _cache_hit) = self.submit_tokens_cached(id, tokens)?;
        Ok(next)
    }

    /// Submit token ids and optionally reuse cached prefill state for identical prompts.
    /// Returns `(next_token, cache_hit)`.
    pub fn submit_tokens_cached(&mut self, id: SessionId, tokens: &[u32]) -> anyhow::Result<(u32, bool)> {
        let temperature = self.temperature;
        let repeat_penalty = self.repeat_penalty;
        let repeat_window = self.repeat_window;

        let cache_trace = std::env::var("CELLM_DEBUG_PREFILL_CACHE")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        let s = self
            .sessions
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("unknown session id: {id}"))?;
        let meta = self
            .session_meta
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("missing session metadata for session id: {id}"))?;
        meta.transition(SessionState::Prefill)
            .map_err(|e| anyhow::anyhow!("session transition failed: {e:?}"))?;

        if !s.cached_prompt.is_empty()
            && s.cached_prompt == tokens
            && s.cached_last_token.is_some()
            && s.page_table.token_count() >= s.cached_next_pos
        {
            if cache_trace {
                eprintln!(
                    "[cellm-sdk] prefill cache HIT sid={} prompt_tokens={} next_pos={}",
                    id,
                    tokens.len(),
                    s.cached_next_pos
                );
            }
            s.page_table
                .truncate_tokens(self.kv_cache.allocator_mut(), s.cached_next_pos)
                .map_err(|e| anyhow::anyhow!("truncate_tokens failed: {e}"))?;
            s.next_pos = s.cached_next_pos;
            s.last_token = s.cached_last_token;
            s.recent = s.cached_recent.clone();
            s.pending_out.clear();
            meta.transition(SessionState::Decoding)
                .map_err(|e| anyhow::anyhow!("session transition failed: {e:?}"))?;
            self.rr.add(id);
            self.total_tokens_generated += 1;
            self.tokens_since_snapshot += 1;
            return Ok((s.cached_last_token.unwrap_or(0), true));
        }
        if cache_trace {
            eprintln!(
                "[cellm-sdk] prefill cache MISS sid={} incoming={} cached={} cached_last_token={} token_count={} cached_next_pos={}",
                id,
                tokens.len(),
                s.cached_prompt.len(),
                s.cached_last_token.is_some(),
                s.page_table.token_count(),
                s.cached_next_pos
            );
        }

        let next;
        s.pending_out.clear();

        // A miss means the snapshot `reset_session` kept describes a different
        // prompt, so its blocks are dead weight: prefill below starts at
        // `next_pos` and appends, never reusing them. Leaving them allocated
        // leaks a prompt's worth of blocks per question until the allocator is
        // empty and prefill fails with "out of KV blocks".
        s.page_table
            .free_all(self.kv_cache.allocator_mut())
            .map_err(|e| anyhow::anyhow!("free_all failed: {e}"))?;
        s.next_pos = 0;
        s.last_token = None;
        s.recent.clear();
        s.cached_prompt.clear();
        s.cached_next_pos = 0;
        s.cached_last_token = None;
        s.cached_recent.clear();

        let (tokens_prefill, last_tok) = if tokens.len() > 1 {
            (&tokens[..tokens.len() - 1], tokens[tokens.len() - 1])
        } else {
            (&[][..], tokens[0])
        };

        if !tokens_prefill.is_empty() {
            self.runner.prefill(tokens_prefill, s.next_pos, &mut s.page_table, &mut self.kv_cache)?;
            s.recent.extend_from_slice(tokens_prefill);
            s.next_pos += tokens_prefill.len();
        }

        let pos = s.next_pos;
        let tok = last_tok;
        let mut cand = match &mut self.runner {
            Runner::Llama(r) => {
                r.step_topk(tok, pos, &mut s.page_table, &mut self.kv_cache, self.top_k)?
            }
            Runner::Gemma(r) => {
                let top_k = if r.is_gemma3_text() { self.top_k.max(8) } else { self.top_k };
                r.step_topk(tok, pos, &mut s.page_table, &mut self.kv_cache, top_k)?
            }
            Runner::Qwen(r) => {
                r.step_topk(tok, pos, &mut s.page_table, &mut self.kv_cache, self.top_k)?
            }
            Runner::Lfm(r) => {
                r.step_topk(tok, pos, &mut s.page_table, &mut self.kv_cache, self.top_k)?
            }
            Runner::DeepSeekV4(r) => {
                r.step_topk(tok, pos, &mut s.page_table, &mut self.kv_cache, self.top_k)?
            }
        };
        if let Runner::Gemma(r) = &self.runner {
            if r.is_gemma3_text() {
                apply_gemma3_stability_candidate_filter(&mut cand, &s.recent, self.backend);
            }
        }
        next = select_next_with_params(
            temperature,
            repeat_penalty,
            repeat_window,
            &cand,
            &s.recent,
            &mut s.rng,
        )?;
        s.recent.push(tok);
        s.next_pos += 1;

        s.last_token = Some(next);
        s.cached_prompt = tokens.to_vec();
        s.cached_next_pos = s.next_pos;
        s.cached_last_token = s.last_token;
        s.cached_recent = s.recent.clone();
        meta.add_prompt_tokens(tokens.len());
        meta.transition(SessionState::Decoding)
            .map_err(|e| anyhow::anyhow!("session transition failed: {e:?}"))?;
        self.rr.add(id);
        self.total_tokens_generated += 1;
        self.tokens_since_snapshot += 1;
        Ok((next, false))
    }

    /// Run a single decode step for the next scheduled session (greedy).
    pub fn step_decode(&mut self) -> anyhow::Result<Option<(SessionId, u32)>> {
        if self.thermal.should_pause_decode() || self.rr.is_empty() {
            return Ok(None);
        }

        if let Some(pair) = self.pop_pending_scheduled() {
            return Ok(Some(pair));
        }

        let burst = self.decode_burst_budget();
        if burst == 0 {
            return Ok(None);
        }
        let _produced = self.pump_decode_burst(burst)?;

        if let Some(pair) = self.pop_pending_scheduled() {
            return Ok(Some(pair));
        }
        Ok(None)
    }

    pub fn cancel_session(&mut self, id: SessionId) -> anyhow::Result<()> {
        self.rr.remove(id);
        let mut s = self
            .sessions
            .remove(&id)
            .ok_or_else(|| anyhow::anyhow!("unknown session id: {id}"))?;
        s.page_table
            .free_all(self.kv_cache.allocator_mut())
            .map_err(|e| anyhow::anyhow!("free_all failed: {e}"))?;
        if let Runner::Qwen(r) = &mut self.runner {
            r.cancel_session(id);
        }
        if let Some(meta) = self.session_meta.get_mut(&id) {
            let _ = meta.transition(SessionState::Terminal);
        }
        self.session_meta.remove(&id);
        Ok(())
    }

    /// Reset session decode state while preserving any cached prefill snapshot.
    pub fn reset_session(&mut self, id: SessionId) -> anyhow::Result<()> {
        self.rr.remove(id);
        let s = self
            .sessions
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("unknown session id: {id}"))?;
        let keep_tokens = if s.cached_last_token.is_some() && !s.cached_prompt.is_empty() {
            s.cached_next_pos
        } else {
            0
        };
        s.page_table
            .truncate_tokens(self.kv_cache.allocator_mut(), keep_tokens)
            .map_err(|e| anyhow::anyhow!("truncate_tokens failed: {e}"))?;
        if keep_tokens > 0 {
            s.next_pos = s.cached_next_pos;
            s.last_token = s.cached_last_token;
            s.recent = s.cached_recent.clone();
        } else {
            s.next_pos = 0;
            s.last_token = None;
            s.recent.clear();
        }
        s.pending_out.clear();

        let meta = self
            .session_meta
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("missing session metadata for session id: {id}"))?;
        if meta.state() == SessionState::Decoding {
            meta.transition(SessionState::Suspended)
                .map_err(|e| anyhow::anyhow!("session transition failed: {e:?}"))?;
        }
        meta.transition(SessionState::Queued)
            .map_err(|e| anyhow::anyhow!("session transition failed: {e:?}"))?;
        Ok(())
    }

    pub fn suspend_session(&mut self, id: SessionId) -> anyhow::Result<()> {
        let meta = self
            .session_meta
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("unknown session id: {id}"))?;
        meta.transition(SessionState::Suspended)
            .map_err(|e| anyhow::anyhow!("session transition failed: {e:?}"))?;
        self.rr.remove(id);
        Ok(())
    }

    pub fn resume_session(&mut self, id: SessionId) -> anyhow::Result<()> {
        let s = self
            .sessions
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("unknown session id: {id}"))?;
        let meta = self
            .session_meta
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("missing session metadata for session id: {id}"))?;

        let target = if s.last_token.is_some() {
            SessionState::Decoding
        } else {
            SessionState::Queued
        };
        meta.transition(target)
            .map_err(|e| anyhow::anyhow!("session transition failed: {e:?}"))?;
        if s.last_token.is_some() {
            self.rr.add(id);
        }
        Ok(())
    }

    pub fn set_thermal_level(&mut self, level: ThermalLevel) {
        self.thermal.set_level(level);
    }

    pub fn thermal_level(&self) -> ThermalLevel {
        self.thermal.level()
    }

    pub fn stats(&self) -> EngineStats {
        let elapsed = stats_elapsed_secs(&self.last_stats_snapshot).unwrap_or(0.0);
        let tok_per_sec = if elapsed > 0.0 {
            self.tokens_since_snapshot as f64 / elapsed
        } else {
            self.cached_tok_per_sec
        };
        EngineStats {
            active_sessions: self.sessions.len(),
            used_kv_blocks: self.kv_cache.allocator().in_use_count(),
            free_kv_blocks: self.kv_cache.allocator().free_count(),
            thermal_level: self.thermal.level(),
            total_tokens_generated: self.total_tokens_generated,
            current_tok_per_sec: tok_per_sec,
            scheduling_policy: self.policy_exec.policy(),
        }
    }

    /// Reset the tok/s measurement window (called by the consumer after reading stats).
    pub fn reset_stats_window(&mut self) {
        self.last_stats_snapshot = stats_instant_now();
        self.tokens_since_snapshot = 0;
    }

    /// Set the scheduling policy at runtime.
    pub fn set_scheduling_policy(&mut self, policy: SchedulingPolicy) {
        self.policy_exec.set_policy(policy);
    }

    pub fn scheduling_policy(&self) -> SchedulingPolicy {
        self.policy_exec.policy()
    }

    pub fn model_config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn has_session(&self, id: SessionId) -> bool {
        self.sessions.contains_key(&id)
    }

    pub fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            top_k: self.top_k,
            temperature: self.temperature,
            seed: self.seed,
            repeat_penalty: self.repeat_penalty,
            repeat_window: self.repeat_window,
        }
    }

    /// Total number of tokens generated so far across all sessions (lifetime).
    pub fn total_tokens_generated(&self) -> u64 {
        self.total_tokens_generated
    }

    /// Number of active (non-terminated) sessions.
    pub fn num_active_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Number of free KV cache blocks remaining.
    pub fn num_free_blocks(&self) -> usize {
        self.kv_cache.allocator().free_count()
    }

    /// Number of buffered (undecoded) tokens pending for a session.
    pub fn pending_tokens(&self, id: SessionId) -> usize {
        self.sessions
            .get(&id)
            .map(|s| s.pending_out.len())
            .unwrap_or(0)
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.runner.eos_token_id()
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn is_stop_token(&self, token: u32) -> bool {
        self.runner.is_stop_token(token)
    }

    fn seed_for_session(&self, id: SessionId) -> u64 {
        // A cheap derivation to make sessions independent.
        let mut x = id ^ 0x9E3779B97F4A7C15u64;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9u64);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D049BB133111EBu64);
        x ^= x >> 31;
        // mix in a stable engine-level seed
        x ^ self.seed
    }

    // sampling logic lives in `select_next_with_params` to avoid borrow issues

    fn decode_burst_budget(&self) -> usize {
        if self.rr.is_empty() || self.thermal.should_pause_decode() {
            return 0;
        }
        let active_cap = self.thermal.max_active_decode_sessions();
        if active_cap == 0 {
            return 0;
        }
        let active_sessions = self.sessions.len().max(1);
        let session_cap = if active_cap == usize::MAX {
            active_sessions
        } else {
            active_cap.min(active_sessions)
        };
        let backend_cap = match self.backend {
            BackendKind::Metal => 4usize,
            BackendKind::WebGpu => 4usize,
            BackendKind::Cpu => 2usize,
        };
        session_cap.min(backend_cap).max(1)
    }

    fn pop_pending_scheduled(&mut self) -> Option<(SessionId, u32)> {
        let n = self.sessions.len().max(1);
        for _ in 0..n {
            let id = self.rr.next()?;
            let suspended_or_terminal = self
                .session_meta
                .get(&id)
                .map(|m| matches!(m.state(), SessionState::Suspended | SessionState::Terminal))
                .unwrap_or(true);
            if suspended_or_terminal {
                continue;
            }
            if let Some(s) = self.sessions.get_mut(&id) {
                if let Some(tok) = s.pending_out.pop_front() {
                    return Some((id, tok));
                }
            }
        }
        None
    }

    fn pump_decode_burst(&mut self, budget: usize) -> anyhow::Result<usize> {
        if budget == 0 { return Ok(0); }
        match self.policy_exec.policy() {
            SchedulingPolicy::Fair => {
                let mut produced = 0usize;
        for _ in 0..budget {
            let id = self.pick_next_decodable_fair();
            let Some(id) = id else {
                break;
            };
            let Some(tok) = self.decode_one_for_session(id)? else {
                break;
            };
            if let Some(s) = self.sessions.get_mut(&id) {
                s.pending_out.push_back(tok);
                produced += 1;
            }
        }
        Ok(produced)
            }
            SchedulingPolicy::LatencyFirst => self.decode_burst_latency(budget),
            SchedulingPolicy::ThroughputFirst => self.decode_burst_throughput(budget),
        }
    }

    fn decode_burst_latency(&mut self, budget: usize) -> anyhow::Result<usize> {
        let sessions: Vec<&cellm_scheduler::Session> = self.session_meta.values().collect();
        let plan = self.policy_exec.tick(&sessions, &[]);
        let mut produced = 0usize;
        for &id in &plan.decode_ids {
            if produced >= budget { break; }
            if let Some(tok) = self.decode_one_for_session(id)? {
                if let Some(s) = self.sessions.get_mut(&id) { s.pending_out.push_back(tok); produced += 1; }
            }
        }
        Ok(produced)
    }

    fn decode_burst_throughput(&mut self, budget: usize) -> anyhow::Result<usize> {
        use std::collections::HashMap;
        let mut info: HashMap<SessionId, BatchSessionInfo> = HashMap::new();
        for (&id, s) in &self.sessions {
            let meta = self.session_meta.get(&id);
            let is_dec = meta.map(|m| matches!(m.state(), SessionState::Decoding)).unwrap_or(false);
            info.insert(id, BatchSessionInfo::new(is_dec, s.last_token.is_some(), s.next_pos));
        }
        let ids: Vec<SessionId> = self.sessions.keys().copied().collect();
        let groups = self.batch_detector.detect(&ids, &info);
        let sessions: Vec<&cellm_scheduler::Session> = self.session_meta.values().collect();
        let _ = self.policy_exec.tick(&sessions, &[]);
        let mut produced = 0usize;
        for g in &groups {
            for &id in &g.session_ids {
                if produced >= budget { break; }
                if let Some(tok) = self.decode_one_for_session(id)? {
                    if let Some(s) = self.sessions.get_mut(&id) { s.pending_out.push_back(tok); produced += 1; }
                }
            }
        }
        Ok(produced)
    }

    fn pick_next_decodable_fair(&mut self) -> Option<SessionId> {
        let n = self.sessions.len().max(1);
        for _ in 0..n {
            let id = self.rr.next()?;
            let Some(meta) = self.session_meta.get(&id) else {
                continue;
            };
            if matches!(meta.state(), SessionState::Suspended | SessionState::Terminal) {
                continue;
            }
            let Some(s) = self.sessions.get(&id) else {
                continue;
            };
            if s.last_token.is_none() {
                continue;
            }
            return Some(id);
        }
        None
    }

    fn decode_one_for_session(&mut self, id: SessionId) -> anyhow::Result<Option<u32>> {
        let temperature = self.temperature;
        let repeat_penalty = self.repeat_penalty;
        let repeat_window = self.repeat_window;
        let backend = self.backend;
        let top_k = self.top_k;

        let mut s = match self.sessions.remove(&id) {
            Some(s) => s,
            None => return Ok(None),
        };
        let mut meta = match self.session_meta.remove(&id) {
            Some(m) => m,
            None => {
                self.sessions.insert(id, s);
                return Ok(None);
            }
        };

        let out = (|| -> anyhow::Result<Option<u32>> {
            if matches!(meta.state(), SessionState::Suspended | SessionState::Terminal) {
                return Ok(None);
            }
            let Some(cur) = s.last_token else {
                return Ok(None);
            };
            let pos = s.next_pos;
            let mut cand = match &mut self.runner {
                Runner::Llama(r) => r.step_topk(cur, pos, &mut s.page_table, &mut self.kv_cache, top_k)?,
                Runner::Gemma(r) => {
                    let top_k = if r.is_gemma3_text() { top_k.max(8) } else { top_k };
                    r.step_topk(cur, pos, &mut s.page_table, &mut self.kv_cache, top_k)?
                }
                Runner::Qwen(r) => r.step_topk(cur, pos, &mut s.page_table, &mut self.kv_cache, top_k)?,
                Runner::Lfm(r) => r.step_topk(cur, pos, &mut s.page_table, &mut self.kv_cache, top_k)?,
                Runner::DeepSeekV4(r) => r.step_topk(cur, pos, &mut s.page_table, &mut self.kv_cache, top_k)?,
            };
            if let Runner::Gemma(r) = &self.runner {
                if r.is_gemma3_text() {
                    apply_gemma3_stability_candidate_filter(&mut cand, &s.recent, backend);
                }
            }
            let next = select_next_with_params(
                temperature,
                repeat_penalty,
                repeat_window,
                &cand,
                &s.recent,
                &mut s.rng,
            )?;
            s.recent.push(cur);
            s.last_token = Some(next);
            s.next_pos += 1;
            meta.add_generated_token();

            if self.runner.is_stop_token(next) {
                let _ = meta.transition(SessionState::Terminal);
            }

            Ok(Some(next))
        })();

        // Increment token counters if a token was produced.
        if let Ok(Some(_)) = &out {
            self.total_tokens_generated += 1;
            self.tokens_since_snapshot += 1;
        }

        self.sessions.insert(id, s);
        self.session_meta.insert(id, meta);
        out
    }
}

fn apply_turboquant_runtime_config(cfg: &EngineConfig) {
    if cfg.kv_encoding != KvEncodingKind::TurboQuant {
        return;
    }
    std::env::set_var(
        "CELLM_TURBOQ_INT8_DOT",
        if cfg.turboq_int8_dot { "1" } else { "0" },
    );
    std::env::set_var(
        "CELLM_TURBOQ_QJL_CORR",
        if cfg.turboq_qjl_corr { "1" } else { "0" },
    );
}

fn resolve_backend(requested: BackendKind) -> BackendKind {
    #[cfg(not(feature = "webgpu"))]
    if matches!(requested, BackendKind::WebGpu) {
        return BackendKind::Cpu;
    }
    requested
}

fn effective_text_model_type(header: &cellm_model::CellmHeader) -> String {
    if let Some(Value::Object(obj)) = &header.source_text_config {
        if let Some(Value::String(mt)) = obj.get("model_type") {
            if !mt.is_empty() {
                return mt.clone();
            }
        }
    }
    header.model_type.clone()
}

// allow_litert_proxy() was removed.

#[derive(Debug, Clone, Copy)]
struct XorShift64(u64);

impl XorShift64 {
    fn seeded(seed: u64) -> Self {
        let s = if seed == 0 { 0x1234_5678_9ABC_DEF0u64 } else { seed };
        Self(s)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        // map to [0,1)
        let v = self.next_u64() >> 40; // 24 bits
        (v as f32) / ((1u32 << 24) as f32)
    }
}

fn select_next_with_params(
    temperature: f64,
    repeat_penalty: f64,
    repeat_window: usize,
    candidates: &[(u32, f32)],
    recent: &[u32],
    rng: &mut XorShift64,
) -> anyhow::Result<u32> {
    if candidates.is_empty() {
        anyhow::bail!("no candidates");
    }

    let temperature = temperature as f32;
    let repeat_penalty = repeat_penalty as f32;

    let mut ids: Vec<u32> = Vec::with_capacity(candidates.len());
    let mut scores: Vec<f32> = Vec::with_capacity(candidates.len());
    for &(id, s) in candidates {
        ids.push(id);
        scores.push(s);
    }

    // Apply repetition penalty first (before temperature/greedy selection)
    if repeat_penalty > 1.0 && repeat_window > 0 && !recent.is_empty() {
        let start = recent.len().saturating_sub(repeat_window);
        for i in 0..scores.len() {
            if recent[start..].contains(&ids[i]) {
                scores[i] /= repeat_penalty;
            }
        }
    }

    // Greedy selection (temperature 0) - pick highest score after penalty
    if temperature <= 0.0 {
        let mut best_idx = 0;
        let mut best_score = scores[0];
        for i in 1..scores.len() {
            if scores[i] > best_score {
                best_score = scores[i];
                best_idx = i;
            }
        }
        return Ok(ids[best_idx]);
    }

    let mut max = f32::NEG_INFINITY;
    for &s in &scores {
        if s > max {
            max = s;
        }
    }
    let mut weights = Vec::with_capacity(scores.len());
    let mut sum = 0.0f32;
    for &s in &scores {
        let w = ((s - max) / temperature).exp();
        weights.push(w);
        sum += w;
    }
    if sum == 0.0 {
        return Ok(ids[0]);
    }

    let r = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for i in 0..weights.len() {
        acc += weights[i];
        if r <= acc {
            return Ok(ids[i]);
        }
    }
    Ok(*ids.last().unwrap())
}

fn apply_gemma3_stability_candidate_filter(
    cand: &mut Vec<(u32, f32)>,
    recent: &[u32],
    backend: BackendKind,
) {
    if backend != BackendKind::Metal {
        return;
    }
    // Device-specific Gemma3 metal runs can collapse onto empty/markdown/newline tokens.
    // Drop known degenerate ids when alternatives exist.
    if cand.len() > 1 {
        cand.retain(|(id, _)| *id != 106);
    }
    if cand.len() > 1 {
        cand.retain(|(id, _)| *id != 1018);
    }
    if cand.len() > 1 {
        // Markdown-heavy loops seen on-device: '#', ' #', and code-fence token.
        cand.retain(|(id, _)| !matches!(*id, 236865 | 997 | 2717));
    }
    if cand.len() > 1 {
        // Star-heavy loops seen on-device: '*', ' *', and bold marker patterns.
        cand.retain(|(id, _)| !matches!(*id, 236829 | 808 | 13513));
    }
    if cand.len() > 1 {
        // Horizontal-rule / separator loops seen on-device: '-', '--', '---', '----', '&#'.
        cand.retain(|(id, _)| !matches!(*id, 236772 | 726 | 7243 | 1040 | 21841));
    }
    if cand.len() > 1 {
        // Additional low-information loop tokens observed on-device.
        cand.retain(|(id, _)| !matches!(*id, 4368 | 1340 | 236775 | 236913 | 3056));
    }
    if cand.len() > 1 {
        // Zero-width / invisible Unicode loop tokens observed on-device.
        cand.retain(|(id, _)| !matches!(*id, 237141 | 237218 | 237243));
    }
    if cand.len() > 1 {
        // Another recurrent low-information loop token observed on-device ("Model").
        cand.retain(|(id, _)| *id != 4968);
    }

    // If we're already in a newline/markdown loop, steer away from pure formatting tokens.
    if looks_like_format_loop(recent) && cand.len() > 1 {
        cand.retain(|(id, _)| !matches!(*id, 107 | 108 | 109 | 110 | 1018 | 236865));
    }

    // If a token is already heavily repeated in the recent window, avoid selecting it again
    // when alternatives exist. This helps break low-information loops like "is ... is ...".
    if cand.len() > 1 && recent.len() >= 8 {
        let tail = &recent[recent.len().saturating_sub(16)..];
        let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for &t in tail {
            *counts.entry(t).or_insert(0) += 1;
        }
        let original = cand.clone();
        cand.retain(|(id, _)| counts.get(id).copied().unwrap_or(0) < 3);
        if cand.is_empty() {
            *cand = original;
        }
    }
}

fn looks_like_format_loop(recent: &[u32]) -> bool {
    if recent.len() < 6 {
        return false;
    }
    let tail = &recent[recent.len() - 6..];
    let format_count = tail
        .iter()
        .filter(|&&id| matches!(id, 107 | 108 | 109 | 110 | 1018 | 236865))
        .count();
    format_count >= 4
}

pub struct EngineStats {
    pub active_sessions: usize,
    pub used_kv_blocks: usize,
    pub free_kv_blocks: usize,
    pub thermal_level: ThermalLevel,
    pub total_tokens_generated: u64,
    pub current_tok_per_sec: f64,
    pub scheduling_policy: SchedulingPolicy,
}

// ---------------------------------------------------------------------------
// Thinking mode helpers
// ---------------------------------------------------------------------------

/// Wrap a prompt with thinking prefill for ChatML-style models.
///
/// This adds `<|im_start|>assistant\n<think>\n` to the prompt, which triggers
/// the model to generate thinking content before the actual response.
///
/// # Example
/// ```ignore
/// let prompt = "What is 2+2?";
/// let prompt_with_think = wrap_prompt_with_think(prompt);
/// // prompt_with_think = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n"
/// ```
pub fn wrap_prompt_with_think(prompt: &str) -> String {
    let mut s = String::with_capacity(prompt.len() + 64);
    s.push_str("<|im_start|>user\n");
    s.push_str(prompt);
    s.push_str("<|im_end|>\n<|im_start|>assistant\n<think>\n");
    s
}

/// Strip thinking blocks from generated text.
///
/// Removes all content between `<think>` and `</think>` tags (inclusive).
/// Use this when you want to hide the model's reasoning process from the user.
///
/// # Example
/// ```ignore
/// let text = "<think>\nLet me think...\n</think>\n\nThe answer is 4.";
/// let clean = strip_think_blocks(text);
/// // clean = "\n\nThe answer is 4."
/// ```
pub fn strip_think_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut in_think = false;
    while !rest.is_empty() {
        if !in_think {
            if let Some(idx) = rest.find("<think>") {
                out.push_str(&rest[..idx]);
                rest = &rest[idx + "<think>".len()..];
                in_think = true;
            } else {
                out.push_str(rest);
                break;
            }
        } else if let Some(end) = rest.find("</think>") {
            rest = &rest[end + "</think>".len()..];
            in_think = false;
        } else {
            break;
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub top_k: usize,
    pub temperature: f64,
    pub seed: u64,
    pub repeat_penalty: f64,
    pub repeat_window: usize,
}

fn infer_qwen_kv_head_dim(file: &CellmFile) -> anyhow::Result<usize> {
    let h = &file.header;
    let kv_heads = h.num_kv_heads.max(1);
    for t in &h.tensors {
        if t.name.contains(".self_attn.k_proj.weight") && t.shape.len() == 2 {
            let kv_dim = t.shape[0];
            if kv_dim % kv_heads == 0 {
                return Ok(kv_dim / kv_heads);
            }
        }
    }
    anyhow::bail!(
        "unable to infer qwen KV head_dim (no self_attn.k_proj.weight found in tensor list)"
    )
}

fn infer_gemma_kv_head_dim(file: &CellmFile) -> anyhow::Result<usize> {
    let h = &file.header;
    let kv_heads = h.num_kv_heads.max(1);
    let mut max_head_dim = 0usize;
    for t in &h.tensors {
        if t.name.contains(".self_attn.k_proj.weight") && t.shape.len() == 2 {
            let kv_dim = t.shape[0];
            if kv_dim % kv_heads == 0 {
                max_head_dim = max_head_dim.max(kv_dim / kv_heads);
            }
        }
    }
    if max_head_dim > 0 {
        Ok(max_head_dim)
    } else {
        anyhow::bail!(
            "unable to infer gemma KV head_dim (no self_attn.k_proj.weight found in tensor list)"
        )
    }
}
