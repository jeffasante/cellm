use std::cell::RefCell;
use std::ffi::{c_char, CStr};
use std::path::Path;
use std::slice;

use crate::{BackendKind, Engine, EngineConfig, SessionId};
use serde_json::Value;
use tokenizers::Tokenizer;

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn set_last_error(msg: impl Into<String>) {
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(msg.into()));
}

fn take_last_error() -> Option<String> {
    LAST_ERROR.with(|e| e.borrow_mut().take())
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

/// Opaque engine handle for C/Swift/Kotlin.
#[allow(non_camel_case_types)]
pub type cellm_engine_t = u64;

/// Opaque tokenizer handle for C/Swift/Kotlin.
#[allow(non_camel_case_types)]
pub type cellm_tokenizer_t = u64;

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
pub extern "C" fn cellm_tokenizer_create(tokenizer_path: *const c_char) -> cellm_tokenizer_t {
    let result = (|| {
        let tokenizer_path = cstr_to_str(tokenizer_path)?;
        let path = Path::new(tokenizer_path);
        let tok = load_tokenizer(path).map_err(|e| format!("tokenizer_create failed: {e}"))?;
        Ok::<cellm_tokenizer_t, String>(Box::into_raw(Box::new(tok)) as u64)
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
        drop(Box::from_raw(tok as *mut Tokenizer));
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
        let t = unsafe { &*(tok as *const Tokenizer) };
        let enc = t.encode(text, true).map_err(|e| format!("encode failed: {e}"))?;
        let ids = enc.get_ids();
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

        let t = unsafe { &*(tok as *const Tokenizer) };
        let ids = unsafe { std::slice::from_raw_parts(tokens, token_count) };
        let text = t
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
    let result = (|| {
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
        };
        let engine = Engine::new(Path::new(model_path), cfg)
            .map_err(|e| format!("engine_create_v3 failed: {e}"))?;
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
    let e = unsafe { &mut *(engine as *mut Engine) };
    e.create_session()
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

    let e = unsafe { &mut *(engine as *mut Engine) };
    let slice = unsafe { std::slice::from_raw_parts(tokens, token_count) };
    match e.submit_tokens(session, slice) {
        Ok(next) => {
            unsafe {
                *out_next_token = next;
            }
            0
        }
        Err(err) => {
            set_last_error(format!("submit_tokens failed: {err}"));
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
    let e = unsafe { &mut *(engine as *mut Engine) };
    match e.step_decode() {
        Ok(Some((sid, tok))) => {
            unsafe {
                *out_session = sid;
                *out_token = tok;
            }
            1
        }
        Ok(None) => 0,
        Err(err) => {
            set_last_error(format!("step_decode failed: {err}"));
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
        if engine == 0 {
            return Err("vlm_describe_image: null engine".to_string());
        }
        if session == 0 {
            return Err("vlm_describe_image: null session".to_string());
        }
        if image_bytes.is_null() || image_len == 0 {
            return Err("vlm_describe_image: null/empty image bytes".to_string());
        }
        let _prompt = cstr_to_str(prompt_utf8)?;
        let _img = unsafe { slice::from_raw_parts(image_bytes, image_len) };

        // Not implemented: there is currently no vision encoder / multimodal runner in the Rust core.
        // This stub exists so iOS/macOS apps can wire up the image pick + FFI path now.
        //
        // When VLM is implemented, this function should:
        // - decode the image into pixels
        // - run the vision encoder
        // - pack multimodal tokens/embeddings
        // - run the text decoder to produce an answer
        let _ = out_buf;
        let _ = buf_len;
        Err("VLM not implemented yet (vision encoder + multimodal runner missing)".to_string())
    })();

    match result {
        Ok(_) => 0,
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}
