// Exercises the embedder through the C FFI surface exactly as Swift will:
// raw pointers, capacity checks, and the query/document prefix split.
use std::ffi::CString;

use cellm_sdk::embed_ffi::*;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn last_error() -> String {
    let mut buf = vec![0i8; 1024];
    let n = cellm_sdk::ffi::cellm_last_error_message(buf.as_mut_ptr(), buf.len());
    String::from_utf8_lossy(&buf[..n].iter().map(|&b| b as u8).collect::<Vec<u8>>()).into_owned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "models/LFM2.5-Embedding-350M-int8.cellm".to_string());
    let tok = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "models/hf/LFM2.5-Embedding-350M/tokenizer.json".to_string());

    let c_model = CString::new(model.clone()).unwrap();
    let c_tok = CString::new(tok).unwrap();
    let e = cellm_embedder_create(c_model.as_ptr(), c_tok.as_ptr());
    assert!(e != 0, "embedder_create failed: {}", last_error());

    let dim = cellm_embedder_dim(e);
    println!("dim = {dim}, max_tokens = {}", cellm_embedder_max_tokens(e));
    assert!(dim > 0);
    let dim = dim as usize;

    let query = "how do I reset my password?";
    let docs = [
        "To change your password, open Settings and choose Account Security.",
        "The mitochondrion is the powerhouse of the cell.",
        "Paris is the capital and most populous city of France.",
    ];

    let mut qv = vec![0.0f32; dim];
    let c_q = CString::new(query).unwrap();
    let rc = cellm_embed_text(e, c_q.as_ptr(), CELLM_EMBED_QUERY, qv.as_mut_ptr(), dim);
    assert_eq!(rc, 0, "embed_text failed");

    let c_docs: Vec<CString> = docs.iter().map(|d| CString::new(*d).unwrap()).collect();
    let ptrs: Vec<*const std::ffi::c_char> = c_docs.iter().map(|c| c.as_ptr()).collect();
    let mut dv = vec![0.0f32; dim * docs.len()];
    let n = unsafe {
        cellm_embed_texts(
            e,
            ptrs.as_ptr(),
            ptrs.len(),
            CELLM_EMBED_DOCUMENT,
            dv.as_mut_ptr(),
            dv.len(),
        )
    };
    assert_eq!(n, docs.len() as i32, "embed_texts failed");

    println!("\nquery: {query:?}");
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, d) in docs.iter().enumerate() {
        let s = dot(&qv, &dv[i * dim..(i + 1) * dim]);
        println!("  {s:+.4}  {d}");
        if s > best.1 {
            best = (i, s);
        }
    }
    assert_eq!(best.0, 0, "expected the password doc to rank first");

    // Capacity is enforced, not trusted.
    let mut small = vec![0.0f32; dim - 1];
    let rc = cellm_embed_text(
        e,
        c_q.as_ptr(),
        CELLM_EMBED_QUERY,
        small.as_mut_ptr(),
        small.len(),
    );
    assert_eq!(rc, -1, "undersized buffer must be rejected");

    cellm_embedder_destroy(e);
    println!("\nPASS");
}
