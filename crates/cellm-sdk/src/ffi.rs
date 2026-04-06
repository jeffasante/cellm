use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::sync::{Mutex, OnceLock};
use std::path::Path;
use std::slice;

use crate::{vlm::VlmRunConfig, BackendKind, Engine, EngineConfig, SessionId};
use cellm_cache::KvEncodingKind;
use cellm_kernels::MetalKernels;
use cellm_scheduler::ThermalLevel;
use serde_json::Value;
use tokenizers::Tokenizer;

thread_local! {
    static LAST_VLM_TIMINGS_MS: RefCell<Option<(f64, f64, f64, f64)>> = const { RefCell::new(None) };
    static LAST_VLM_ENCODER_LAYER_MS: RefCell<Option<Vec<f64>>> = const { RefCell::new(None) };
}

static LAST_ERROR: OnceLock<Mutex<Option<String>>> = OnceLock::new();

fn last_error_cell() -> &'static Mutex<Option<String>> {
    LAST_ERROR.get_or_init(|| Mutex::new(None))
}

fn set_last_error(msg: impl Into<String>) {
    let mut g = last_error_cell()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    *g = Some(msg.into());
}

fn take_last_error() -> Option<String> {
    let mut g = last_error_cell()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    g.take()
}

fn ensure_last_error(msg: &str) {
    let mut g = last_error_cell()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let empty = g
        .as_ref()
        .map(|s| s.trim().is_empty())
        .unwrap_or(true);
    if empty {
        *g = Some(msg.to_string());
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "panic with non-string payload".to_string()
}

fn set_last_vlm_timings_ms(patch_ms: f64, encoder_ms: f64, decode_ms: f64, total_ms: f64) {
    LAST_VLM_TIMINGS_MS.with(|t| *t.borrow_mut() = Some((patch_ms, encoder_ms, decode_ms, total_ms)));
}

fn set_last_vlm_encoder_layer_ms(layer_ms: Vec<f64>) {
    LAST_VLM_ENCODER_LAYER_MS.with(|t| *t.borrow_mut() = Some(layer_ms));
}

fn clear_last_vlm_timings_ms() {
    LAST_VLM_TIMINGS_MS.with(|t| *t.borrow_mut() = None);
    LAST_VLM_ENCODER_LAYER_MS.with(|t| *t.borrow_mut() = None);
}

fn get_last_vlm_timings_ms() -> Option<(f64, f64, f64, f64)> {
    LAST_VLM_TIMINGS_MS.with(|t| *t.borrow())
}

fn get_last_vlm_encoder_layer_ms() -> Option<Vec<f64>> {
    LAST_VLM_ENCODER_LAYER_MS.with(|t| t.borrow().clone())
}

fn cstr_to_str<'a>(s: *const c_char) -> Result<&'a str, String> {
    if s.is_null() {
        return Err("null string pointer".into());
    }
    unsafe { CStr::from_ptr(s) }
        .to_str()
        .map_err(|e| format!("invalid utf-8: {e}"))
}

fn backend_from_ffi(kind: u32) -> Result<BackendKind, String> {
    match kind {
        0 => Ok(BackendKind::Cpu),
        1 => Ok(BackendKind::Metal),
        other => Err(format!("invalid backend kind: {other} (expected 0=cpu, 1=metal)")),
    }
}

fn kv_encoding_from_ffi(kind: u32) -> Result<KvEncodingKind, String> {
    match kind {
        0 => Ok(KvEncodingKind::F16),
        1 => Ok(KvEncodingKind::TurboQuant),
        other => Err(format!(
            "invalid kv encoding kind: {other} (expected 0=f16, 1=turboquant)"
        )),
    }
}

/// Opaque engine handle for C/Swift/Kotlin.
#[allow(non_camel_case_types)]
pub type cellm_engine_t = u64;

/// Opaque tokenizer handle for C/Swift/Kotlin.
#[allow(non_camel_case_types)]
pub type cellm_tokenizer_t = u64;

struct TokenizerHandle {
    tok: Tokenizer,
    added_token_ids: HashMap<String, u32>,
}

#[no_mangle]
pub extern "C" fn cellm_last_error_message(out_buf: *mut c_char, buf_len: usize) -> usize {
    if out_buf.is_null() || buf_len == 0 {
        return 0;
    }
    let msg = take_last_error().unwrap_or_default();
    let bytes = msg.as_bytes();
    let n = bytes.len().min(buf_len.saturating_sub(1));
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf as *mut u8, n);
        *out_buf.add(n) = 0;
    }
    n
}

#[no_mangle]
pub extern "C" fn cellm_metal_smoke_test() -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        MetalKernels::smoke_test_add_f32()
            .map_err(|e| format!("metal_smoke_test failed: {e}"))
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(panic_payload) => {
            set_last_error(format!(
                "metal_smoke_test panicked: {}",
                panic_payload_to_string(panic_payload)
            ));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_tokenizer_create(tokenizer_path: *const c_char) -> cellm_tokenizer_t {
    let result = (|| {
        let tokenizer_path = cstr_to_str(tokenizer_path)?;
        let path = Path::new(tokenizer_path);
        let tok = load_tokenizer(path).map_err(|e| format!("tokenizer_create failed: {e}"))?;
        let added_token_ids = load_added_token_ids(path)?;
        let handle = TokenizerHandle { tok, added_token_ids };
        Ok::<cellm_tokenizer_t, String>(Box::into_raw(Box::new(handle)) as u64)
    })();

    match result {
        Ok(h) => h,
        Err(e) => {
            set_last_error(e);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_tokenizer_destroy(tok: cellm_tokenizer_t) {
    if tok == 0 {
        return;
    }
    unsafe {
        drop(Box::from_raw(tok as *mut TokenizerHandle));
    }
}

/// Encode a UTF-8 prompt into token ids.
///
/// If `out_tokens` is null or `max_tokens == 0`, returns the required token count (and does not write).
/// Otherwise writes up to `max_tokens` tokens and returns the number written.
#[no_mangle]
pub extern "C" fn cellm_tokenizer_encode(
    tok: cellm_tokenizer_t,
    text_utf8: *const c_char,
    out_tokens: *mut u32,
    max_tokens: usize,
) -> usize {
    let result = (|| {
        if tok == 0 {
            return Err("tokenizer_encode: null tokenizer".to_string());
        }
        let text = cstr_to_str(text_utf8)?;
        let handle = unsafe { &*(tok as *const TokenizerHandle) };
        let ids = encode_with_explicit_added_tokens(&handle.tok, &handle.added_token_ids, text)
            .map_err(|e| format!("encode failed: {e}"))?;
        if out_tokens.is_null() || max_tokens == 0 {
            return Ok::<usize, String>(ids.len());
        }
        let n = ids.len().min(max_tokens);
        unsafe {
            for i in 0..n {
                *out_tokens.add(i) = ids[i];
            }
        }
        Ok::<usize, String>(n)
    })();

    match result {
        Ok(n) => n,
        Err(e) => {
            set_last_error(e);
            0
        }
    }
}

/// Decode token ids into UTF-8.
///
/// If `out_buf` is null or `buf_len == 0`, returns the required byte length (excluding null terminator).
/// Otherwise writes up to `buf_len-1` bytes and null-terminates. Returns bytes written (excluding null).
#[no_mangle]
pub extern "C" fn cellm_tokenizer_decode(
    tok: cellm_tokenizer_t,
    tokens: *const u32,
    token_count: usize,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    let result = (|| {
        if tok == 0 {
            return Err("tokenizer_decode: null tokenizer".to_string());
        }
        if tokens.is_null() || token_count == 0 {
            return Err("tokenizer_decode: null/empty tokens".to_string());
        }

        let handle = unsafe { &*(tok as *const TokenizerHandle) };
        let ids = unsafe { std::slice::from_raw_parts(tokens, token_count) };
        let text = handle
            .tok
            .decode(ids, true)
            .map_err(|e| format!("decode failed: {e}"))?;
        let bytes = text.as_bytes();

        if out_buf.is_null() || buf_len == 0 {
            return Ok::<usize, String>(bytes.len());
        }

        let n = bytes.len().min(buf_len.saturating_sub(1));
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf as *mut u8, n);
            *out_buf.add(n) = 0;
        }
        Ok::<usize, String>(n)
    })();

    match result {
        Ok(n) => n,
        Err(e) => {
            set_last_error(e);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_engine_create(
    model_path: *const c_char,
    tokens_per_block: u32,
    total_blocks: u32,
    top_k: u32,
) -> cellm_engine_t {
    let result = (|| {
        let model_path = cstr_to_str(model_path)?;
        let cfg = EngineConfig {
            tokens_per_block: tokens_per_block as usize,
            total_blocks: total_blocks as usize,
            top_k: top_k as usize,
            temperature: 0.0,
            repeat_penalty: 1.0,
            repeat_window: 0,
            seed: 0,
            backend: BackendKind::Cpu,
            kv_encoding: KvEncodingKind::F16,
            turboq_int8_dot: true,
            turboq_qjl_corr: true,
        };
        let engine = Engine::new(Path::new(model_path), cfg)
            .map_err(|e| format!("engine_create failed: {e}"))?;
        Ok::<cellm_engine_t, String>(Box::into_raw(Box::new(engine)) as u64)
    })();

    match result {
        Ok(h) => h,
        Err(e) => {
            set_last_error(e);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_engine_destroy(engine: cellm_engine_t) {
    if engine == 0 {
        return;
    }
    unsafe {
        drop(Box::from_raw(engine as *mut Engine));
    }
}

#[no_mangle]
pub extern "C" fn cellm_engine_create_v2(
    model_path: *const c_char,
    tokens_per_block: u32,
    total_blocks: u32,
    top_k: u32,
    temperature: f32,
    repeat_penalty: f32,
    repeat_window: u32,
    seed: u64,
) -> cellm_engine_t {
    let result = (|| {
        let model_path = cstr_to_str(model_path)?;
        let cfg = EngineConfig {
            tokens_per_block: tokens_per_block as usize,
            total_blocks: total_blocks as usize,
            top_k: top_k as usize,
            temperature: temperature as f64,
            repeat_penalty: repeat_penalty as f64,
            repeat_window: repeat_window as usize,
            seed,
            backend: BackendKind::Cpu,
            kv_encoding: KvEncodingKind::F16,
            turboq_int8_dot: true,
            turboq_qjl_corr: true,
        };
        let engine = Engine::new(Path::new(model_path), cfg)
            .map_err(|e| format!("engine_create_v2 failed: {e}"))?;
        Ok::<cellm_engine_t, String>(Box::into_raw(Box::new(engine)) as u64)
    })();

    match result {
        Ok(h) => h,
        Err(e) => {
            set_last_error(e);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_engine_create_v3(
    model_path: *const c_char,
    tokens_per_block: u32,
    total_blocks: u32,
    top_k: u32,
    temperature: f32,
    repeat_penalty: f32,
    repeat_window: u32,
    seed: u64,
    backend: u32, // 0=cpu, 1=metal
) -> cellm_engine_t {
    ensure_last_error("engine_create_v3 failed: no detail");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (|| {
            let model_path = cstr_to_str(model_path)?;
            let backend = backend_from_ffi(backend)?;
            let cfg = EngineConfig {
                tokens_per_block: tokens_per_block as usize,
                total_blocks: total_blocks as usize,
                top_k: top_k as usize,
                temperature: temperature as f64,
                repeat_penalty: repeat_penalty as f64,
                repeat_window: repeat_window as usize,
                seed,
                backend,
                kv_encoding: KvEncodingKind::F16,
                turboq_int8_dot: true,
                turboq_qjl_corr: true,
            };
            let engine = Engine::new(Path::new(model_path), cfg)
                .map_err(|e| format!("engine_create_v3 failed: {e}"))?;
            Ok::<cellm_engine_t, String>(Box::into_raw(Box::new(engine)) as u64)
        })()
    }));

    match result {
        Ok(Ok(h)) => h,
        Ok(Err(e)) => {
            set_last_error(e);
            ensure_last_error("engine_create_v3 failed: empty error");
            0
        }
        Err(panic_payload) => {
            set_last_error(format!(
                "engine_create_v3 panicked: {}",
                panic_payload_to_string(panic_payload)
            ));
            ensure_last_error("engine_create_v3 panicked without payload detail");
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_engine_create_v4(
    model_path: *const c_char,
    tokens_per_block: u32,
    total_blocks: u32,
    top_k: u32,
    temperature: f32,
    repeat_penalty: f32,
    repeat_window: u32,
    seed: u64,
    backend: u32,     // 0=cpu, 1=metal
    kv_encoding: u32, // 0=f16, 1=turboquant
    turboq_int8_dot: u32,
    turboq_qjl_corr: u32,
) -> cellm_engine_t {
    ensure_last_error("engine_create_v4 failed: no detail");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (|| {
            let model_path = cstr_to_str(model_path)?;
            let backend = backend_from_ffi(backend)?;
            let kv_encoding = kv_encoding_from_ffi(kv_encoding)?;
            let cfg = EngineConfig {
                tokens_per_block: tokens_per_block as usize,
                total_blocks: total_blocks as usize,
                top_k: top_k as usize,
                temperature: temperature as f64,
                repeat_penalty: repeat_penalty as f64,
                repeat_window: repeat_window as usize,
                seed,
                backend,
                kv_encoding,
                turboq_int8_dot: turboq_int8_dot != 0,
                turboq_qjl_corr: turboq_qjl_corr != 0,
            };
            let engine = Engine::new(Path::new(model_path), cfg)
                .map_err(|e| format!("engine_create_v4 failed: {e}"))?;
            Ok::<cellm_engine_t, String>(Box::into_raw(Box::new(engine)) as u64)
        })()
    }));

    match result {
        Ok(Ok(h)) => h,
        Ok(Err(e)) => {
            set_last_error(e);
            ensure_last_error("engine_create_v4 failed: empty error");
            0
        }
        Err(panic_payload) => {
            set_last_error(format!(
                "engine_create_v4 panicked: {}",
                panic_payload_to_string(panic_payload)
            ));
            ensure_last_error("engine_create_v4 panicked without payload detail");
            0
        }
    }
}

fn load_tokenizer(path: &std::path::Path) -> Result<Tokenizer, String> {
    match Tokenizer::from_file(path) {
        Ok(t) => Ok(t),
        Err(e) => {
            let normalized = try_normalize_tokenizer_json(path)?;
            if let Some(p) = normalized {
                return Tokenizer::from_file(&p)
                    .map_err(|e| format!("load tokenizer failed (normalized {:?}): {e}", p));
            }
            Err(format!("load tokenizer failed: {e}"))
        }
    }
}

fn load_added_token_ids(path: &std::path::Path) -> Result<HashMap<String, u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read tokenizer failed: {e}"))?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| format!("parse tokenizer failed: {e}"))?;
    let mut out = HashMap::new();
    if let Some(arr) = v.get("added_tokens").and_then(|x| x.as_array()) {
        for item in arr {
            let Some(content) = item.get("content").and_then(|x| x.as_str()) else {
                continue;
            };
            let Some(id) = item.get("id").and_then(|x| x.as_u64()) else {
                continue;
            };
            out.insert(content.to_string(), id as u32);
        }
    }
    Ok(out)
}

fn encode_with_explicit_added_tokens(
    tok: &Tokenizer,
    added_token_ids: &HashMap<String, u32>,
    text: &str,
) -> Result<Vec<u32>, String> {
    if added_token_ids.is_empty() {
        let enc = tok
            .encode(text, false)
            .map_err(|e| format!("tokenize failed: {e}"))?;
        return Ok(enc.get_ids().to_vec());
    }

    let mut specials: Vec<(&str, u32)> = added_token_ids
        .iter()
        .map(|(token, &id)| (token.as_str(), id))
        .collect();
    specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut out: Vec<u32> = Vec::new();
    let mut i = 0usize;
    let mut chunk_start = 0usize;
    let bytes = text.as_bytes();

    while i < bytes.len() {
        let rest = &text[i..];
        let mut matched: Option<(&str, u32)> = None;
        for &(token, id) in &specials {
            if rest.starts_with(token) {
                matched = Some((token, id));
                break;
            }
        }

        if let Some((token, id)) = matched {
            if chunk_start < i {
                let chunk = &text[chunk_start..i];
                let enc = tok
                    .encode(chunk, false)
                    .map_err(|e| format!("tokenize failed: {e}"))?;
                out.extend_from_slice(enc.get_ids());
            }
            out.push(id);
            i += token.len();
            chunk_start = i;
        } else {
            let ch = rest
                .chars()
                .next()
                .ok_or_else(|| "tokenize failed: invalid utf-8 boundary".to_string())?;
            i += ch.len_utf8();
        }
    }

    if chunk_start < text.len() {
        let chunk = &text[chunk_start..];
        let enc = tok
            .encode(chunk, false)
            .map_err(|e| format!("tokenize failed: {e}"))?;
        out.extend_from_slice(enc.get_ids());
    }

    Ok(out)
}

fn try_normalize_tokenizer_json(path: &std::path::Path) -> Result<Option<std::path::PathBuf>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read tokenizer failed: {e}"))?;
    let mut v: Value = serde_json::from_slice(&bytes).map_err(|e| format!("parse tokenizer failed: {e}"))?;

    let merges = v
        .get_mut("model")
        .and_then(|m| m.get_mut("merges"))
        .and_then(|m| m.as_array_mut());
    let Some(merges) = merges else {
        return Ok(None);
    };
    if merges.is_empty() {
        return Ok(None);
    }

    let looks_like_pairs = merges[0]
        .as_array()
        .is_some_and(|a| a.len() == 2 && a[0].is_string() && a[1].is_string());
    if !looks_like_pairs {
        return Ok(None);
    }

    let mut out: Vec<Value> = Vec::with_capacity(merges.len());
    for m in merges.iter() {
        let Some(pair) = m.as_array() else {
            return Ok(None);
        };
        if pair.len() != 2 {
            return Ok(None);
        }
        let Some(a) = pair[0].as_str() else {
            return Ok(None);
        };
        let Some(b) = pair[1].as_str() else {
            return Ok(None);
        };
        out.push(Value::String(format!("{a} {b}")));
    }
    *merges = out;

    let tmp = std::env::temp_dir().join(format!(
        "cellm_tokenizer_normalized_{}.json",
        std::process::id()
    ));
    std::fs::write(&tmp, serde_json::to_vec(&v).map_err(|e| format!("json serialize failed: {e}"))?)
        .map_err(|e| format!("write normalized tokenizer failed: {e}"))?;
    Ok(Some(tmp))
}

#[no_mangle]
pub extern "C" fn cellm_session_create(engine: cellm_engine_t) -> SessionId {
    if engine == 0 {
        set_last_error("session_create: null engine".to_string());
        return 0;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let e = unsafe { &mut *(engine as *mut Engine) };
        e.create_session()
    }));
    match result {
        Ok(sid) => sid,
        Err(panic_payload) => {
            set_last_error(format!(
                "session_create panicked: {}",
                panic_payload_to_string(panic_payload)
            ));
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_session_cancel(engine: cellm_engine_t, session: SessionId) -> i32 {
    if engine == 0 {
        set_last_error("session_cancel: null engine".to_string());
        return -1;
    }
    let e = unsafe { &mut *(engine as *mut Engine) };
    match e.cancel_session(session) {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(format!("session_cancel failed: {err}"));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_session_suspend(engine: cellm_engine_t, session: SessionId) -> i32 {
    if engine == 0 {
        set_last_error("session_suspend: null engine".to_string());
        return -1;
    }
    let e = unsafe { &mut *(engine as *mut Engine) };
    match e.suspend_session(session) {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(format!("session_suspend failed: {err}"));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_session_resume(engine: cellm_engine_t, session: SessionId) -> i32 {
    if engine == 0 {
        set_last_error("session_resume: null engine".to_string());
        return -1;
    }
    let e = unsafe { &mut *(engine as *mut Engine) };
    match e.resume_session(session) {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(format!("session_resume failed: {err}"));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_engine_set_thermal_level(engine: cellm_engine_t, level: u32) -> i32 {
    if engine == 0 {
        set_last_error("set_thermal_level: null engine".to_string());
        return -1;
    }
    let thermal = match level {
        0 => ThermalLevel::Nominal,
        1 => ThermalLevel::Elevated,
        2 => ThermalLevel::Critical,
        3 => ThermalLevel::Emergency,
        _ => {
            set_last_error(format!("set_thermal_level: invalid level {level} (expected 0..3)"));
            return -1;
        }
    };

    let e = unsafe { &mut *(engine as *mut Engine) };
    e.set_thermal_level(thermal);
    0
}

#[no_mangle]
pub extern "C" fn cellm_submit_tokens(
    engine: cellm_engine_t,
    session: SessionId,
    tokens: *const u32,
    token_count: usize,
    out_next_token: *mut u32,
) -> i32 {
    if engine == 0 {
        set_last_error("submit_tokens: null engine".to_string());
        return -1;
    }
    if tokens.is_null() || token_count == 0 {
        set_last_error("submit_tokens: null/empty tokens".to_string());
        return -1;
    }
    if out_next_token.is_null() {
        set_last_error("submit_tokens: null out_next_token".to_string());
        return -1;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let e = unsafe { &mut *(engine as *mut Engine) };
        let slice = unsafe { std::slice::from_raw_parts(tokens, token_count) };
        e.submit_tokens(session, slice)
    }));
    match result {
        Ok(Ok(next)) => {
            unsafe {
                *out_next_token = next;
            }
            0
        }
        Ok(Err(err)) => {
            set_last_error(format!("submit_tokens failed: {err}"));
            -1
        }
        Err(panic_payload) => {
            set_last_error(format!(
                "submit_tokens panicked: {}",
                panic_payload_to_string(panic_payload)
            ));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_step_decode(
    engine: cellm_engine_t,
    out_session: *mut SessionId,
    out_token: *mut u32,
) -> i32 {
    if engine == 0 {
        set_last_error("step_decode: null engine".to_string());
        return -1;
    }
    if out_session.is_null() || out_token.is_null() {
        set_last_error("step_decode: null outputs".to_string());
        return -1;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let e = unsafe { &mut *(engine as *mut Engine) };
        e.step_decode()
    }));
    match result {
        Ok(Ok(Some((sid, tok)))) => {
            unsafe {
                *out_session = sid;
                *out_token = tok;
            }
            1
        }
        Ok(Ok(None)) => 0,
        Ok(Err(err)) => {
            set_last_error(format!("step_decode failed: {err}"));
            -1
        }
        Err(panic_payload) => {
            set_last_error(format!(
                "step_decode panicked: {}",
                panic_payload_to_string(panic_payload)
            ));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_engine_kv_stats(
    engine: cellm_engine_t,
    out_used_blocks: *mut u32,
    out_free_blocks: *mut u32,
) -> i32 {
    if engine == 0 {
        set_last_error("kv_stats: null engine".to_string());
        return -1;
    }
    if out_used_blocks.is_null() || out_free_blocks.is_null() {
        set_last_error("kv_stats: null outputs".to_string());
        return -1;
    }

    let e = unsafe { &mut *(engine as *mut Engine) };
    let st = e.stats();
    unsafe {
        *out_used_blocks = st.used_kv_blocks as u32;
        *out_free_blocks = st.free_kv_blocks as u32;
    }
    0
}

#[no_mangle]
pub extern "C" fn cellm_engine_backend_name(
    engine: cellm_engine_t,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if engine == 0 {
        set_last_error("backend_name: null engine".to_string());
        return 0;
    }
    if out_buf.is_null() || buf_len == 0 {
        set_last_error("backend_name: null/empty output buffer".to_string());
        return 0;
    }
    let e = unsafe { &*(engine as *const Engine) };
    let msg = e.backend_name();
    let bytes = msg.as_bytes();
    let n = bytes.len().min(buf_len.saturating_sub(1));
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf as *mut u8, n);
        *out_buf.add(n) = 0;
    }
    n
}

#[no_mangle]
pub extern "C" fn cellm_vlm_describe_image(
    engine: cellm_engine_t,
    session: SessionId,
    image_bytes: *const u8,
    image_len: usize,
    prompt_utf8: *const c_char,
    out_buf: *mut c_char,
    buf_len: usize,
) -> i32 {
    let result: Result<(), String> = (|| {
        clear_last_vlm_timings_ms();
        if engine == 0 {
            return Err("vlm_describe_image: null engine".to_string());
        }
        if session == 0 {
            return Err("vlm_describe_image: null session".to_string());
        }
        if image_bytes.is_null() || image_len == 0 {
            return Err("vlm_describe_image: null/empty image bytes".to_string());
        }
        let prompt = cstr_to_str(prompt_utf8)?;
        let img = unsafe { slice::from_raw_parts(image_bytes, image_len) };
        let e = unsafe { &mut *(engine as *mut Engine) };
        if !e.has_session(session) {
            return Err(format!("vlm_describe_image: unknown session id {session}"));
        }
        let sampling = e.sampling_params();
        let (text, timing) = crate::vlm::describe_image_with_cellm_timed(
            e.model_path(),
            img,
            prompt,
            VlmRunConfig {
                backend: e.backend(),
                tokens_per_block: 16,
                top_k: sampling.top_k.max(1),
                temperature: sampling.temperature as f32,
                seed: sampling.seed,
                repeat_penalty: sampling.repeat_penalty as f32,
                repeat_window: sampling.repeat_window,
                max_new_tokens: 128,
                min_new_tokens: 1,
            },
        )
        .map_err(|err| format!("vlm_describe_image failed: {err}"))?;
        set_last_vlm_timings_ms(
            timing.patch_ms,
            timing.encoder_ms,
            timing.decode_ms,
            timing.total_ms,
        );
        set_last_vlm_encoder_layer_ms(timing.encoder_layer_ms);

        if !out_buf.is_null() && buf_len > 0 {
            let bytes = text.as_bytes();
            let n = bytes.len().min(buf_len.saturating_sub(1));
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf as *mut u8, n);
                *out_buf.add(n) = 0;
            }
        }
        Ok(())
    })();

    match result {
        Ok(_) => 0,
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn cellm_vlm_last_timings_ms(
    out_patch_ms: *mut f64,
    out_encoder_ms: *mut f64,
    out_decode_ms: *mut f64,
    out_total_ms: *mut f64,
) -> i32 {
    if out_patch_ms.is_null()
        || out_encoder_ms.is_null()
        || out_decode_ms.is_null()
        || out_total_ms.is_null()
    {
        set_last_error("vlm_last_timings_ms: null output pointer".to_string());
        return -1;
    }

    let Some((patch_ms, encoder_ms, decode_ms, total_ms)) = get_last_vlm_timings_ms() else {
        set_last_error("vlm_last_timings_ms: no timings available yet".to_string());
        return -1;
    };

    unsafe {
        *out_patch_ms = patch_ms;
        *out_encoder_ms = encoder_ms;
        *out_decode_ms = decode_ms;
        *out_total_ms = total_ms;
    }
    0
}

#[no_mangle]
pub extern "C" fn cellm_vlm_last_encoder_layer_count() -> u32 {
    get_last_vlm_encoder_layer_ms()
        .map(|v| v.len() as u32)
        .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn cellm_vlm_last_encoder_layer_time_ms(
    layer_index: u32,
    out_ms: *mut f64,
) -> i32 {
    if out_ms.is_null() {
        set_last_error("vlm_last_encoder_layer_time_ms: null output pointer".to_string());
        return -1;
    }
    let Some(layer_ms) = get_last_vlm_encoder_layer_ms() else {
        set_last_error("vlm_last_encoder_layer_time_ms: no timings available yet".to_string());
        return -1;
    };
    let idx = layer_index as usize;
    if idx >= layer_ms.len() {
        set_last_error(format!(
            "vlm_last_encoder_layer_time_ms: index {} out of range 0..{}",
            idx,
            layer_ms.len().saturating_sub(1)
        ));
        return -1;
    }
    unsafe {
        *out_ms = layer_ms[idx];
    }
    0
}
