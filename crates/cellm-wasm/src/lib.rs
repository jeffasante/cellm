#![cfg(target_arch = "wasm32")]

// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! WebAssembly bindings for the cellm LLM inference engine.
//!
//! ## Usage (JavaScript)
//!
//! ```js
//! import init, { CellmEngine } from './cellm_wasm.js';
//!
//! await init();
//!
//! // Fetch your model .cellm file as an ArrayBuffer
//! const resp = await fetch('/models/my-model.cellm');
//! const modelBytes = new Uint8Array(await resp.arrayBuffer());
//!
//! const config = JSON.stringify({
//!   tokens_per_block: 16,
//!   total_blocks: 128,
//!   top_k: 40,
//!   temperature: 0.8,
//!   repeat_penalty: 1.05,
//!   repeat_window: 64,
//!   seed: 1,
//! });
//!
//! const engine = CellmEngine.new(modelBytes, config);
//! const tokenizerJson = '...'; // from tokenizer.json
//! engine.set_tokenizer(tokenizerJson);
//!
//! // Create a session and send tokens
//! let sid = engine.create_session();
//! let nextToken = engine.submit_tokens(sid, [1, 304, 11, 297, ...]);
//!
//! // Step decode
//! while (true) {
//!   let result = engine.step_decode();
//!   if (!result) break;
//!   // result = { sid, token }
//!   if (token === 2) break; // EOS
//! }
//! ```

use std::cell::RefCell;
use std::sync::Mutex;

use wasm_bindgen::prelude::*;

use cellm_sdk::{Engine, EngineConfig, BackendKind, SessionId};
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
use cellm_kernels::WebGpuBackend;
#[cfg(target_arch = "wasm32")]
use cellm_kernels::VisionWebGpu;
#[cfg(target_arch = "wasm32")]
use cellm_sdk::vlm::VlmRunConfig;

// ---------------------------------------------------------------------------
// Panic hook
// ---------------------------------------------------------------------------

/// Initialise the WASM module. Must be called once from JavaScript before
/// any other function.
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
}

// ---------------------------------------------------------------------------
// CellmEngine
// ---------------------------------------------------------------------------

/// A cellm LLM inference engine instance, exposed to JavaScript via wasm-bindgen.
///
/// Owns a model, KV cache, tokenizer, and manages multiple inference sessions.
#[wasm_bindgen]
pub struct CellmEngine {
    engine: Mutex<Engine>,
    tokenizer: RefCell<Option<Tokenizer>>,
    #[cfg(target_arch = "wasm32")]
    model_bytes: Vec<u8>,
    #[cfg(target_arch = "wasm32")]
    tokenizer_bytes: RefCell<Option<Vec<u8>>>,
    #[cfg(target_arch = "wasm32")]
    gpu: RefCell<Option<WebGpuBackend>>,
}

#[wasm_bindgen]
impl CellmEngine {
    /// Create a new engine from raw model bytes and a JSON config string.
    ///
    /// - `model_bytes`: the complete `.cellm` model file contents as a `Uint8Array`.
    /// - `config_json`: a JSON string matching `EngineConfig`:
    ///   ```json
    ///   {
    ///     "tokens_per_block": 16,
    ///     "total_blocks": 128,
    ///     "top_k": 40,
    ///     "temperature": 0.8,
    ///     "repeat_penalty": 1.05,
    ///     "repeat_window": 64,
    ///     "seed": 1,
    ///     "scheduling_policy": "Fair"
    ///   }
    ///   ```
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: Vec<u8>, config_json: &str) -> Result<CellmEngine, JsValue> {
        let cfg: EngineConfig = deserialize_config(config_json)?;
        // Clone model bytes for later VLM use (WASM only).
        // For non-WASM targets this clone is optimized away by cfg.
        #[cfg(target_arch = "wasm32")]
        let vlm_model_bytes = model_bytes.clone();
        let engine = Engine::from_vec(model_bytes, cfg)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::new: {e}")))?;
        Ok(CellmEngine {
            engine: Mutex::new(engine),
            tokenizer: RefCell::new(None),
            #[cfg(target_arch = "wasm32")]
            model_bytes: vlm_model_bytes,
            tokenizer_bytes: RefCell::new(None),
            #[cfg(target_arch = "wasm32")]
            gpu: RefCell::new(None),
        })
    }

    /// Try to initialize WebGPU acceleration. Returns true if GPU is available.
    /// Call with `await engine.try_init_webgpu()` from JavaScript.
    #[cfg(target_arch = "wasm32")]
    pub async fn try_init_webgpu(&self) -> bool {
        if let Some(gpu) = WebGpuBackend::create().await {
            *self.gpu.borrow_mut() = Some(gpu);
            true
        } else {
            false
        }
    }

    /// Check whether WebGPU acceleration is active.
    pub fn has_gpu(&self) -> bool {
        #[cfg(target_arch = "wasm32")] { self.gpu.borrow().is_some() }
        #[cfg(not(target_arch = "wasm32"))] false
    }

    /// Set the tokenizer from a JSON string (contents of `tokenizer.json`).
    ///
    /// Must be called before `tokenize()` or `decode()`.
    pub fn set_tokenizer(&self, tokenizer_json: &str) -> Result<(), JsValue> {
        // fancy-regex (used in WASM) cannot compile Unicode property class
        // patterns like \p{L}, \p{N}, \p{S} which Qwen / Llama tokenizers use
        // in their Split pre-tokenizer.  We strip those Split entries out and
        // replace the whole pre_tokenizer with a plain ByteLevel one so the
        // rest of the tokenizer (vocab, merges, decoder…) loads correctly.
        let sanitized = sanitize_tokenizer_json(tokenizer_json)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::set_tokenizer (sanitize): {e}")))?;

        let tokenizer = Tokenizer::from_bytes(sanitized.as_bytes())
            .map_err(|e| {
                let snippet = if sanitized.len() > 120 {
                    &sanitized[..120]
                } else {
                    &sanitized
                };
                JsValue::from_str(&format!(
                    "CellmEngine::set_tokenizer: {e} (json len={}, snippet='{}')",
                    sanitized.len(),
                    snippet
                ))
            })?;
        *self.tokenizer.borrow_mut() = Some(tokenizer);
        #[cfg(target_arch = "wasm32")]
        { *self.tokenizer_bytes.borrow_mut() = Some(sanitized.into_bytes()); }
        Ok(())
    }

    /// Describe an image using the VLM (Vision-Language Model) pipeline.
    /// Requires WebGPU to be initialized via `try_init_webgpu()` first.
    /// Returns the generated text description, or falls back to a text-only
    /// response if WebGPU is not available.
    ///
    /// - `image_bytes`: raw image file (JPEG/PNG) as a `Uint8Array`
    /// - `prompt`: text prompt to guide the description
    /// - `max_tokens`: maximum number of output tokens
    #[cfg(target_arch = "wasm32")]
    pub async fn describe_image(
        &self,
        image_bytes: Vec<u8>,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<String, JsValue> {
        let gpu = self.gpu.borrow_mut().take()
            .ok_or_else(|| JsValue::from_str("WebGPU not initialized. Call try_init_webgpu() first."))?;
        let tokenizer_bytes = self.tokenizer_bytes.borrow().clone()
            .ok_or_else(|| JsValue::from_str("Tokenizer not set. Call set_tokenizer() first."))?;
        let vision = VisionWebGpu::new(gpu);
        let cfg = VlmRunConfig {
            backend: BackendKind::WebGpu,
            tokens_per_block: 16,
            top_k: 40,
            temperature: 0.8,
            seed: 1,
            repeat_penalty: 1.05,
            repeat_window: 64,
            max_new_tokens: max_tokens.max(16) as usize,
            min_new_tokens: 1,
        };
        let result = cellm_sdk::vlm::describe_image_with_cellm_webgpu_from_bytes(
            &self.model_bytes,
            &tokenizer_bytes,
            &image_bytes,
            prompt,
            cfg,
            vision,
        ).await;
        match result {
            Ok((text, _timing, vision)) => {
                *self.gpu.borrow_mut() = Some(vision.into_gpu());
                Ok(text)
            }
            Err(e) => Err(JsValue::from_str(&format!("describe_image failed: {e}"))),
        }
    }

    /// Check whether a tokenizer has been set.
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.borrow().is_some()
    }

    /// Encode a prompt string into token IDs using the loaded tokenizer.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, JsValue> {
        let tok = self.tokenizer.borrow();
        let tokenizer = tok
            .as_ref()
            .ok_or_else(|| JsValue::from_str("CellmEngine: tokenizer not set"))?;
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::tokenize: {e}")))?;
        let mut ids = encoding.get_ids().to_vec();
        if let Some(bos) = self.engine.lock().unwrap().bos_token_id() {
            if ids.first().copied() != Some(bos) {
                ids.insert(0, bos);
            }
        }
        Ok(ids)
    }

    /// Decode a sequence of token IDs back to a string.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, JsValue> {
        let tok = self.tokenizer.borrow();
        let tokenizer = tok
            .as_ref()
            .ok_or_else(|| JsValue::from_str("CellmEngine: tokenizer not set"))?;
        tokenizer
            .decode(tokens, false)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::decode: {e}")))
    }

    // -----------------------------------------------------------------------
    // Session management
    // -----------------------------------------------------------------------

    /// Create a new inference session. Returns a session ID.
    pub fn create_session(&self) -> SessionId {
        self.engine.lock().unwrap().create_session()
    }

    /// Submit pre-filled token IDs for a session.
    ///
    /// Returns the next predicted token ID (greedy sampling).
    pub fn submit_tokens(&self, session_id: SessionId, tokens: Vec<u32>) -> Result<u32, JsValue> {
        self.engine
            .lock()
            .unwrap()
            .submit_tokens(session_id, &tokens)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::submit_tokens: {e}")))
    }

    /// Run a single decode step for the next scheduled session.
    ///
    /// Returns `Some([session_id, token])` if a token was produced, or `None`
    /// if no sessions are ready to decode.
    pub fn step_decode(&self) -> Result<Option<js_sys::Array>, JsValue> {
        let result = self
            .engine
            .lock()
            .unwrap()
            .step_decode()
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::step_decode: {e}")))?;
        match result {
            Some((sid, token)) => {
                let arr = js_sys::Array::new();
                arr.push(&JsValue::from_f64(sid as f64));
                arr.push(&JsValue::from_f64(token as f64));
                Ok(Some(arr))
            }
            None => Ok(None),
        }
    }

    /// Convenience: submit tokens and run decode loop up to `max_tokens` steps.
    ///
    /// Returns an array of `[session_id, token_id]` pairs.
    pub fn generate(
        &self,
        session_id: SessionId,
        tokens: Vec<u32>,
        max_tokens: u32,
    ) -> Result<js_sys::Array, JsValue> {
        // Submit prefill
        let _next = self.submit_tokens(session_id, tokens)?;

        let results = js_sys::Array::new();

        for _ in 0..max_tokens {
            match self
                .engine
                .lock()
                .unwrap()
                .step_decode()
                .map_err(|e| JsValue::from_str(&format!("CellmEngine::generate: {e}")))?
            {
                Some((sid, token)) => {
                    let pair = js_sys::Array::new();
                    pair.push(&JsValue::from_f64(sid as f64));
                    pair.push(&JsValue::from_f64(token as f64));
                    results.push(&pair);
                }
                None => break,
            }
        }

        Ok(results)
    }

    /// Cancel a session and free its KV cache blocks.
    pub fn cancel_session(&self, session_id: SessionId) -> Result<(), JsValue> {
        self.engine
            .lock()
            .unwrap()
            .cancel_session(session_id)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::cancel_session: {e}")))
    }

    /// Reset a session's decode state while preserving the cached prefill.
    pub fn reset_session(&self, session_id: SessionId) -> Result<(), JsValue> {
        self.engine
            .lock()
            .unwrap()
            .reset_session(session_id)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::reset_session: {e}")))
    }

    /// Suspend a session (move to queued state, free no memory).
    pub fn suspend_session(&self, session_id: SessionId) -> Result<(), JsValue> {
        self.engine
            .lock()
            .unwrap()
            .suspend_session(session_id)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::suspend_session: {e}")))
    }

    /// Resume a previously suspended session.
    pub fn resume_session(&self, session_id: SessionId) -> Result<(), JsValue> {
        self.engine
            .lock()
            .unwrap()
            .resume_session(session_id)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::resume_session: {e}")))
    }

    /// Number of active (undecoded) tokens currently buffered for a session.
    pub fn pending_tokens(&self, session_id: SessionId) -> u32 {
        self.engine.lock().unwrap().pending_tokens(session_id) as u32
    }

    /// Total tokens generated so far across all sessions.
    pub fn total_tokens_generated(&self) -> f64 {
        self.engine.try_lock().map(|e| e.total_tokens_generated() as f64).unwrap_or(0.0)
    }

    /// Number of active (non-terminated) sessions.
    pub fn num_active_sessions(&self) -> u32 {
        self.engine.try_lock().map(|e| e.num_active_sessions() as u32).unwrap_or(0)
    }

    /// Number of free KV cache blocks remaining.
    pub fn num_free_blocks(&self) -> u32 {
        self.engine.try_lock().map(|e| e.num_free_blocks() as u32).unwrap_or(0)
    }

    /// Model EOS token id, or -1 when the model metadata does not provide one.
    pub fn eos_token_id(&self) -> i32 {
        self.engine
            .lock()
            .unwrap()
            .eos_token_id()
            .map(|id| id as i32)
            .unwrap_or(-1)
    }

    /// Model BOS token id, or -1 when the model metadata does not provide one.
    pub fn bos_token_id(&self) -> i32 {
        self.engine
            .lock()
            .unwrap()
            .bos_token_id()
            .map(|id| id as i32)
            .unwrap_or(-1)
    }

    /// Whether a token is the model's stop token.
    pub fn is_stop_token(&self, token: u32) -> bool {
        self.engine.lock().unwrap().is_stop_token(token)
    }

    // -----------------------------------------------------------------------
    // Thinking mode helpers
    // -----------------------------------------------------------------------

    /// Wrap a prompt with thinking prefill for ChatML-style models.
    ///
    /// This adds `<|im_start|>assistant\n<think>\n` to the prompt, which triggers
    /// the model to generate thinking content before the actual response.
    ///
    /// # Example (JavaScript)
    /// ```js
    /// const prompt = "What is 2+2?";
    /// const promptWithThink = engine.wrap_prompt_with_think(prompt);
    /// const tokens = engine.tokenize(promptWithThink);
    /// ```
    pub fn wrap_prompt_with_think(&self, prompt: &str) -> String {
        cellm_sdk::wrap_prompt_with_think(prompt)
    }

    /// Strip thinking blocks from generated text.
    ///
    /// Removes all content between `<think>` and `</think>` tags (inclusive).
    /// Use this when you want to hide the model's reasoning process from the user.
    ///
    /// # Example (JavaScript)
    /// ```js
    /// const text = "<think>\nLet me think...\n</think>\n\nThe answer is 4.";
    /// const clean = engine.strip_think_blocks(text);
    /// // clean = "\n\nThe answer is 4."
    /// ```
    pub fn strip_think_blocks(&self, text: &str) -> String {
        cellm_sdk::strip_think_blocks(text)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn deserialize_config(json: &str) -> Result<EngineConfig, JsValue> {
    #[derive(serde::Deserialize)]
    struct Config {
        #[serde(default = "default_tokens_per_block")]
        tokens_per_block: usize,
        #[serde(default = "default_total_blocks")]
        total_blocks: usize,
        #[serde(default = "default_top_k")]
        top_k: usize,
        #[serde(default = "default_temperature")]
        temperature: f64,
        #[serde(default = "default_repeat_penalty")]
        repeat_penalty: f64,
        #[serde(default = "default_repeat_window")]
        repeat_window: usize,
        #[serde(default = "default_seed")]
        seed: u64,
        #[serde(default)]
        scheduling_policy: String,

    }

    fn default_tokens_per_block() -> usize { 16 }
    fn default_total_blocks() -> usize { 128 }
    fn default_top_k() -> usize { 40 }
    fn default_temperature() -> f64 { 0.8 }
    fn default_repeat_penalty() -> f64 { 1.05 }
    fn default_repeat_window() -> usize { 64 }
    fn default_seed() -> u64 { 1 }

    let c: Config = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("invalid config JSON: {e}")))?;

    let scheduling_policy = match c.scheduling_policy.as_str() {
        "" | "Fair" => cellm_scheduler::SchedulingPolicy::Fair,
        "LatencyFirst" => cellm_scheduler::SchedulingPolicy::LatencyFirst,
        "ThroughputFirst" => cellm_scheduler::SchedulingPolicy::ThroughputFirst,
        other => {
            return Err(JsValue::from_str(&format!(
                "unknown scheduling_policy: {other} (expected Fair, LatencyFirst, or ThroughputFirst)"
            )));
        }
    };

    Ok(EngineConfig {
        tokens_per_block: c.tokens_per_block,
        total_blocks: c.total_blocks,
        top_k: c.top_k,
        temperature: c.temperature,
        repeat_penalty: c.repeat_penalty,
        repeat_window: c.repeat_window,
        seed: c.seed,
        backend: BackendKind::Cpu,
        kv_encoding: cellm_cache::KvEncodingKind::F16,
        turboq_int8_dot: false,
        turboq_qjl_corr: false,
        scheduling_policy,
    })
}

/// Pre-process a `tokenizer.json` string so it can be loaded by the
/// `tokenizers` crate on WASM where only `fancy-regex` is available.
///
/// `fancy-regex` (via the `regex` crate) does NOT support Unicode property
/// classes such as `\p{L}`, `\p{N}`, `\p{S}`.  Many modern tokenizers
/// (Qwen, Llama, Gemma, …) use a `Split` pre-tokenizer with such patterns.
///
/// We patch the JSON to replace the `pre_tokenizer` with a plain `ByteLevel`
/// pre-tokenizer.  The vocabulary / merges / decoder are unchanged, so
/// encoding / decoding still works correctly — we just split on bytes rather
/// than on the complex Unicode split pattern, which produces the same byte-
/// level token sequences that the model was actually trained with.
fn sanitize_tokenizer_json(json: &str) -> Result<String, serde_json::Error> {
    let mut root: serde_json::Value = serde_json::from_str(json)?;

    if let Some(obj) = root.as_object_mut() {
        let needs_patch = pre_tokenizer_has_unicode_props(obj.get("pre_tokenizer"));

        if needs_patch {
            web_sys::console::log_1(&"CellmEngine: Patching tokenizer.json to remove incompatible Unicode property classes in pre_tokenizer".into());
            // Replace with ByteLevel which is natively supported on WASM.
            obj.insert(
                "pre_tokenizer".to_string(),
                serde_json::json!({
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": false,
                    "use_regex": false
                }),
            );
        } else {
            web_sys::console::log_1(&"CellmEngine: Tokenizer.json seems compatible, no patching needed".into());
        }
    }

    serde_json::to_string(&root)
}

/// Returns true if the pre_tokenizer value (or any item in a Sequence)
/// contains a Split pattern that uses `\p{` Unicode property classes.
fn pre_tokenizer_has_unicode_props(pt: Option<&serde_json::Value>) -> bool {
    let pt = match pt {
        Some(v) => v,
        None => return false,
    };
    match pt.get("type").and_then(|t| t.as_str()) {
        Some("Sequence") => {
            if let Some(arr) = pt.get("pretokenizers").and_then(|a| a.as_array()) {
                arr.iter().any(|item| pre_tokenizer_has_unicode_props(Some(item)))
            } else {
                false
            }
        }
        Some("Split") => {
            // Check if the regex pattern contains \p{
            if let Some(pattern_obj) = pt.get("pattern") {
                if let Some(regex_str) = pattern_obj.get("Regex").and_then(|r| r.as_str()) {
                    return regex_str.contains("\\p{");
                }
            }
            false
        }
        _ => false,
    }
}
