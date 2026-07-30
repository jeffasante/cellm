// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! C FFI for text embeddings.
//!
//! Deliberately separate from the generation engine: an embedder has no KV
//! cache, no sampler and no session lifecycle, so reusing `cellm_engine_t`
//! would mean threading a mode flag through every call. A distinct opaque
//! handle keeps both surfaces honest.
//!
//! Typical use from Swift/Kotlin:
//! ```c
//! cellm_embedder_t e = cellm_embedder_create(model_path, tokenizer_path);
//! int32_t dim = cellm_embedder_dim(e);
//! float *v = malloc(dim * sizeof(float));
//! cellm_embed_text(e, "hello world", CELLM_EMBED_DOCUMENT, v, dim);
//! cellm_embedder_destroy(e);
//! ```

use std::ffi::c_char;
use std::path::Path;
use std::sync::Mutex;

use cellm_model::lfm::LfmRunner;
use cellm_model::lfm_encoder::{DOCUMENT_PREFIX, QUERY_PREFIX};
use tokenizers::Tokenizer;

use crate::ffi::{cstr_to_str, load_tokenizer, set_last_error};

/// Opaque embedder handle for C/Swift/Kotlin.
#[allow(non_camel_case_types)]
pub type cellm_embedder_t = u64;

/// Embed the text as-is, with no instruction prefix.
pub const CELLM_EMBED_NONE: u32 = 0;
/// Prefix with `"query: "` — use for search queries.
pub const CELLM_EMBED_QUERY: u32 = 1;
/// Prefix with `"document: "` — use for indexed content.
pub const CELLM_EMBED_DOCUMENT: u32 = 2;

struct EmbedderHandle {
    runner: Mutex<LfmRunner>,
    tokenizer: Tokenizer,
    dim: usize,
}

fn prefix_for(kind: u32) -> Result<&'static str, String> {
    match kind {
        CELLM_EMBED_NONE => Ok(""),
        CELLM_EMBED_QUERY => Ok(QUERY_PREFIX),
        CELLM_EMBED_DOCUMENT => Ok(DOCUMENT_PREFIX),
        other => Err(format!(
            "invalid embed prefix kind: {other} (expected 0=none, 1=query, 2=document)"
        )),
    }
}

/// Load an embedding model. Returns 0 on failure; call
/// `cellm_last_error_message` for details.
///
/// The model must be a bidirectional LFM2 checkpoint; a generative model is
/// rejected here rather than silently producing causal (and therefore wrong)
/// embeddings.
#[no_mangle]
pub extern "C" fn cellm_embedder_create(
    model_path: *const c_char,
    tokenizer_path: *const c_char,
) -> cellm_embedder_t {
    let result = (|| {
        let model_path = cstr_to_str(model_path)?;
        let tokenizer_path = cstr_to_str(tokenizer_path)?;

        let runner = LfmRunner::load(Path::new(model_path))
            .map_err(|e| format!("embedder_create: failed to load model: {e}"))?;
        if !runner.is_bidirectional() {
            return Err(format!(
                "embedder_create: '{model_path}' is not a bidirectional embedding model"
            ));
        }
        let dim = runner.hidden_size();

        // Reuse the generation path's loader: it normalizes tokenizer JSON that
        // the `tokenizers` crate refuses outright.
        let tokenizer = load_tokenizer(Path::new(tokenizer_path))
            .map_err(|e| format!("embedder_create: failed to load tokenizer: {e}"))?;

        let handle = EmbedderHandle {
            runner: Mutex::new(runner),
            tokenizer,
            dim,
        };
        Ok::<cellm_embedder_t, String>(Box::into_raw(Box::new(handle)) as u64)
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
pub extern "C" fn cellm_embedder_destroy(embedder: cellm_embedder_t) {
    if embedder == 0 {
        return;
    }
    unsafe {
        drop(Box::from_raw(embedder as *mut EmbedderHandle));
    }
}

/// Embedding dimensionality, or -1 on a null handle.
#[no_mangle]
pub extern "C" fn cellm_embedder_dim(embedder: cellm_embedder_t) -> i32 {
    if embedder == 0 {
        set_last_error("embedder_dim: null embedder");
        return -1;
    }
    let handle = unsafe { &*(embedder as *const EmbedderHandle) };
    handle.dim as i32
}

/// Maximum number of tokens the encoder will consider. Longer inputs are
/// truncated, so callers that care should chunk before calling.
#[no_mangle]
pub extern "C" fn cellm_embedder_max_tokens(embedder: cellm_embedder_t) -> i32 {
    if embedder == 0 {
        set_last_error("embedder_max_tokens: null embedder");
        return -1;
    }
    let handle = unsafe { &*(embedder as *const EmbedderHandle) };
    let runner = handle
        .runner
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    runner.max_encode_len() as i32
}

/// Embed one string into `out_vec` (which must hold at least
/// `cellm_embedder_dim` floats). Returns 0 on success, -1 on failure.
///
/// The result is L2-normalized, so cosine similarity is a plain dot product.
#[no_mangle]
pub extern "C" fn cellm_embed_text(
    embedder: cellm_embedder_t,
    text_utf8: *const c_char,
    prefix_kind: u32,
    out_vec: *mut f32,
    out_capacity: usize,
) -> i32 {
    let result = (|| {
        if embedder == 0 {
            return Err("embed_text: null embedder".to_string());
        }
        if out_vec.is_null() {
            return Err("embed_text: null output buffer".to_string());
        }
        let text = cstr_to_str(text_utf8)?;
        let prefix = prefix_for(prefix_kind)?;
        let handle = unsafe { &*(embedder as *const EmbedderHandle) };
        if out_capacity < handle.dim {
            return Err(format!(
                "embed_text: output capacity {} < embedding dim {}",
                out_capacity, handle.dim
            ));
        }

        let prefixed = format!("{prefix}{text}");
        let encoding = handle
            .tokenizer
            .encode(prefixed.as_str(), true)
            .map_err(|e| format!("embed_text: tokenize failed: {e}"))?;
        let tokens = encoding.get_ids();
        if tokens.is_empty() {
            return Err("embed_text: text produced no tokens".to_string());
        }

        let mut runner = handle
            .runner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let vec = runner
            .embed_sequence(tokens)
            .map_err(|e| format!("embed_text: encode failed: {e}"))?;
        drop(runner);

        if vec.len() != handle.dim {
            return Err(format!(
                "embed_text: encoder returned {} dims, expected {}",
                vec.len(),
                handle.dim
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(vec.as_ptr(), out_vec, handle.dim);
        }
        Ok::<(), String>(())
    })();

    match result {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}

/// Embed several strings in one call, writing `count * dim` floats into
/// `out_vecs` row-major. Returns the number of texts embedded, or -1 on
/// failure.
///
/// Exists because per-string FFI round trips dominate when a host app indexes
/// a whole document corpus at once.
///
/// # Safety
/// `texts` must point to `count` valid, null-terminated UTF-8 pointers.
#[no_mangle]
pub unsafe extern "C" fn cellm_embed_texts(
    embedder: cellm_embedder_t,
    texts: *const *const c_char,
    count: usize,
    prefix_kind: u32,
    out_vecs: *mut f32,
    out_capacity: usize,
) -> i32 {
    let result = (|| {
        if embedder == 0 {
            return Err("embed_texts: null embedder".to_string());
        }
        if texts.is_null() || out_vecs.is_null() {
            return Err("embed_texts: null pointer".to_string());
        }
        if count == 0 {
            return Ok::<usize, String>(0);
        }
        let prefix = prefix_for(prefix_kind)?;
        let handle = unsafe { &*(embedder as *const EmbedderHandle) };
        let needed = count
            .checked_mul(handle.dim)
            .ok_or_else(|| "embed_texts: output size overflow".to_string())?;
        if out_capacity < needed {
            return Err(format!(
                "embed_texts: output capacity {out_capacity} < required {needed}"
            ));
        }

        let ptrs = unsafe { std::slice::from_raw_parts(texts, count) };
        let mut runner = handle
            .runner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        for (i, &p) in ptrs.iter().enumerate() {
            let text = cstr_to_str(p)
                .map_err(|e| format!("embed_texts: text {i}: {e}"))?;
            let prefixed = format!("{prefix}{text}");
            let encoding = handle
                .tokenizer
                .encode(prefixed.as_str(), true)
                .map_err(|e| format!("embed_texts: text {i}: tokenize failed: {e}"))?;
            let tokens = encoding.get_ids();
            if tokens.is_empty() {
                return Err(format!("embed_texts: text {i} produced no tokens"));
            }
            let vec = runner
                .embed_sequence(tokens)
                .map_err(|e| format!("embed_texts: text {i}: encode failed: {e}"))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    vec.as_ptr(),
                    out_vecs.add(i * handle.dim),
                    handle.dim,
                );
            }
        }
        Ok::<usize, String>(count)
    })();

    match result {
        Ok(n) => n as i32,
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}
