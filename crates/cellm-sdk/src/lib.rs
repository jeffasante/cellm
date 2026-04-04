use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use cellm_cache::{KVCache, KvStorageKind, PageTable};
use cellm_core::KvCacheLayout;
use cellm_model::{gemma::GemmaRunner, llama::LlamaRunner, qwen::QwenRunner, CellmFile, ModelConfig};
use cellm_scheduler::{RoundRobinScheduler, Session as SchedSession, SessionState, ThermalLevel, ThermalPolicy};
use serde_json::Value;

pub type SessionId = u64;

pub mod ffi;
pub mod vlm;

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Metal,
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
        }
    }
}

#[derive(Debug)]
struct EngineSession {
    page_table: PageTable,
    next_pos: usize,
    last_token: Option<u32>,
    recent: Vec<u32>,
    rng: XorShift64,
}

enum Runner {
    Llama(LlamaRunner),
    Gemma(GemmaRunner),
    Qwen(QwenRunner),
}

/// cellm public API engine.
///
/// Early wiring that owns a shared paged KV cache while each session owns a `PageTable`.
/// Text-only for now.
pub struct Engine {
    model_path: PathBuf,
    cfg: ModelConfig,
    runner: Runner,
    backend: BackendKind,
    kv_cache: KVCache,
    sessions: HashMap<SessionId, EngineSession>,
    session_meta: HashMap<SessionId, SchedSession>,
    next_session_id: SessionId,
    rr: RoundRobinScheduler,
    thermal: ThermalPolicy,
    top_k: usize,
    temperature: f64,
    repeat_penalty: f64,
    repeat_window: usize,
    seed: u64,
}

impl Engine {
    pub fn new(model_path: &Path, engine_cfg: EngineConfig) -> Result<Self> {
        let selected_backend = resolve_backend(engine_cfg.backend);
        let file = CellmFile::load(model_path)?;
        let header = file.header.clone();

        let text_model_type = effective_text_model_type(&header);
        let mut runner = match text_model_type.as_str() {
            "llama" => Runner::Llama(LlamaRunner::load(model_path)?),
            t if t.starts_with("gemma") => Runner::Gemma(GemmaRunner::load(model_path)?),
            t if t.starts_with("qwen") => Runner::Qwen(QwenRunner::load(model_path)?),
            other => anyhow::bail!(
                "unsupported model_type for Engine: model_type={} effective_text_model_type={other}",
                header.model_type
            ),
        };
        if selected_backend == BackendKind::Metal {
            match &mut runner {
                Runner::Qwen(r) => {
                    if !r.enable_metal_full_backend() {
                        anyhow::bail!("Qwen full-metal backend requested but unavailable");
                    }
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
            }
        }

        let cfg = match &runner {
            Runner::Llama(r) => r.config().clone(),
            Runner::Gemma(r) => r.config().clone(),
            Runner::Qwen(r) => r.config().clone(),
        };

        let head_dim = match &runner {
            Runner::Llama(_) => cfg.hidden_size / cfg.num_attention_heads,
            Runner::Gemma(_) => infer_gemma_kv_head_dim(&file)?,
            Runner::Qwen(_) => infer_qwen_kv_head_dim(&file)?,
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
        };
        let kv_cache = KVCache::new_with_kind(layout, storage_kind)?;

        Ok(Self {
            model_path: model_path.to_path_buf(),
            cfg,
            runner,
            backend: selected_backend,
            kv_cache,
            sessions: HashMap::new(),
            session_meta: HashMap::new(),
            next_session_id: 1,
            rr: RoundRobinScheduler::new(),
            thermal: ThermalPolicy::default(),
            top_k: engine_cfg.top_k,
            temperature: engine_cfg.temperature,
            repeat_penalty: engine_cfg.repeat_penalty,
            repeat_window: engine_cfg.repeat_window,
            seed: engine_cfg.seed,
        })
    }

    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            BackendKind::Cpu => "cpu",
            BackendKind::Metal => "metal",
        }
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
                rng: XorShift64::seeded(self.seed_for_session(id)),
            },
        );
        self.session_meta.insert(id, SchedSession::new(id));
        id
    }

    /// Submit token ids (already-tokenized) and return the next token id (greedy).
    pub fn submit_tokens(&mut self, id: SessionId, tokens: &[u32]) -> Result<u32> {
        let temperature = self.temperature;
        let repeat_penalty = self.repeat_penalty;
        let repeat_window = self.repeat_window;

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

        let mut next = 0u32;
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = s.next_pos + i;
            let cand = match &mut self.runner {
                Runner::Llama(r) => {
                    r.step_topk(tok, pos, &mut s.page_table, &mut self.kv_cache, self.top_k)?
                }
                Runner::Gemma(r) => {
                    r.step_topk(tok, pos, &mut s.page_table, &mut self.kv_cache, self.top_k)?
                }
                Runner::Qwen(r) => {
                    r.step_topk(tok, pos, &mut s.page_table, &mut self.kv_cache, self.top_k)?
                }
            };
            next = select_next_with_params(
                temperature,
                repeat_penalty,
                repeat_window,
                &cand,
                &s.recent,
                &mut s.rng,
            )?;
            s.recent.push(tok);
        }

        s.next_pos += tokens.len();
        s.last_token = Some(next);
        meta.add_prompt_tokens(tokens.len());
        meta.transition(SessionState::Decoding)
            .map_err(|e| anyhow::anyhow!("session transition failed: {e:?}"))?;
        self.rr.add(id);
        Ok(next)
    }

    /// Run a single decode step for the next scheduled session (greedy).
    pub fn step_decode(&mut self) -> Result<Option<(SessionId, u32)>> {
        let temperature = self.temperature;
        let repeat_penalty = self.repeat_penalty;
        let repeat_window = self.repeat_window;

        if self.thermal.should_pause_decode() || self.rr.is_empty() {
            return Ok(None);
        }

        let n = self.sessions.len().max(1);
        for _ in 0..n {
            let id = match self.rr.next() {
                Some(id) => id,
                None => return Ok(None),
            };

            let s = match self.sessions.get_mut(&id) {
                Some(s) => s,
                None => continue,
            };
            let meta = match self.session_meta.get_mut(&id) {
                Some(m) => m,
                None => continue,
            };
            if meta.state() == SessionState::Suspended || meta.state() == SessionState::Terminal {
                continue;
            }

            let Some(cur) = s.last_token else {
                continue;
            };

            let pos = s.next_pos;
            let cand = match &mut self.runner {
                Runner::Llama(r) => {
                    r.step_topk(cur, pos, &mut s.page_table, &mut self.kv_cache, self.top_k)?
                }
                Runner::Gemma(r) => {
                    r.step_topk(cur, pos, &mut s.page_table, &mut self.kv_cache, self.top_k)?
                }
                Runner::Qwen(r) => {
                    r.step_topk(cur, pos, &mut s.page_table, &mut self.kv_cache, self.top_k)?
                }
            };
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
            return Ok(Some((id, next)));
        }

        Ok(None)
    }

    pub fn cancel_session(&mut self, id: SessionId) -> Result<()> {
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

    pub fn suspend_session(&mut self, id: SessionId) -> Result<()> {
        let meta = self
            .session_meta
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("unknown session id: {id}"))?;
        meta.transition(SessionState::Suspended)
            .map_err(|e| anyhow::anyhow!("session transition failed: {e:?}"))?;
        self.rr.remove(id);
        Ok(())
    }

    pub fn resume_session(&mut self, id: SessionId) -> Result<()> {
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
        EngineStats {
            active_sessions: self.sessions.len(),
            used_kv_blocks: self.kv_cache.allocator().in_use_count(),
            free_kv_blocks: self.kv_cache.allocator().free_count(),
            thermal_level: self.thermal.level(),
        }
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
}

fn resolve_backend(requested: BackendKind) -> BackendKind {
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
) -> Result<u32> {
    if candidates.is_empty() {
        anyhow::bail!("no candidates");
    }
    if temperature <= 0.0 {
        return Ok(candidates[0].0);
    }

    let temperature = temperature as f32;
    let repeat_penalty = repeat_penalty as f32;

    let mut ids: Vec<u32> = Vec::with_capacity(candidates.len());
    let mut scores: Vec<f32> = Vec::with_capacity(candidates.len());
    for &(id, s) in candidates {
        ids.push(id);
        scores.push(s);
    }

    if repeat_penalty > 1.0 && repeat_window > 0 && !recent.is_empty() {
        let start = recent.len().saturating_sub(repeat_window);
        for i in 0..scores.len() {
            if recent[start..].contains(&ids[i]) {
                scores[i] /= repeat_penalty;
            }
        }
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

pub struct EngineStats {
    pub active_sessions: usize,
    pub used_kv_blocks: usize,
    pub free_kv_blocks: usize,
    pub thermal_level: ThermalLevel,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub top_k: usize,
    pub temperature: f64,
    pub seed: u64,
    pub repeat_penalty: f64,
    pub repeat_window: usize,
}

fn infer_qwen_kv_head_dim(file: &CellmFile) -> Result<usize> {
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

fn infer_gemma_kv_head_dim(file: &CellmFile) -> Result<usize> {
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
        "unable to infer gemma KV head_dim (no self_attn.k_proj.weight found in tensor list)"
    )
}
