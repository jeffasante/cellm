// Author: Jeffrey Asante (https://jeffasante.github.io/)
use std::ffi::c_char;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "vlm-smoke", about = "Smoke test the native .cellm VLM FFI call path")]
struct Args {
    /// Path to .cellm model file
    #[arg(long)]
    model: PathBuf,

    /// Session prompt for the image (text)
    #[arg(long, default_value = "What is in this image?")]
    prompt: String,

    /// Path to an image file (bytes passed through FFI)
    #[arg(long)]
    image: PathBuf,

    /// backend: cpu|metal
    #[arg(long, default_value = "metal")]
    backend: String,

    /// max generated tokens
    #[arg(long, default_value_t = 48)]
    gen: usize,

    /// sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// top-k for sampling
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// repetition penalty
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// repetition window
    #[arg(long, default_value_t = 96)]
    repeat_window: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model = args.model.to_string_lossy().to_string();
    let image_bytes = std::fs::read(&args.image)?;

    unsafe {
        let backend = if args.backend.eq_ignore_ascii_case("metal") {
            cellm_sdk::ffi::cellm_engine_create_v3(
                cstr(&model).as_ptr(),
                16,
                args.gen as u32,
                args.top_k as u32,
                args.temperature,
                args.repeat_penalty,
                args.repeat_window as u32,
                1,
                1,
            )
        } else {
            cellm_sdk::ffi::cellm_engine_create_v3(
                cstr(&model).as_ptr(),
                16,
                args.gen as u32,
                args.top_k as u32,
                args.temperature,
                args.repeat_penalty,
                args.repeat_window as u32,
                1,
                0,
            )
        };
        if backend == 0 {
            anyhow::bail!("engine_create failed: {}", last_error());
        }
        let session = cellm_sdk::ffi::cellm_session_create(backend);
        if session == 0 {
            cellm_sdk::ffi::cellm_engine_destroy(backend);
            anyhow::bail!("session_create failed: {}", last_error());
        }

        let mut out = vec![0i8; 32 * 1024];
        let rc = cellm_sdk::ffi::cellm_vlm_describe_image(
            backend,
            session,
            image_bytes.as_ptr(),
            image_bytes.len(),
            cstr(&args.prompt).as_ptr(),
            out.as_mut_ptr(),
            out.len(),
        );
        if rc == 0 {
            let mut bbuf = vec![0i8; 32];
            let _ = cellm_sdk::ffi::cellm_engine_backend_name(backend, bbuf.as_mut_ptr(), bbuf.len());
            let bname = std::ffi::CStr::from_ptr(bbuf.as_ptr()).to_string_lossy();
            let text = std::ffi::CStr::from_ptr(out.as_ptr())
                .to_string_lossy()
                .to_string();
            println!("Backend active: {bname}");
            let mut patch = 0.0f64;
            let mut encoder = 0.0f64;
            let mut decode = 0.0f64;
            let mut total = 0.0f64;
            let trc = cellm_sdk::ffi::cellm_vlm_last_timings_ms(
                &mut patch as *mut f64,
                &mut encoder as *mut f64,
                &mut decode as *mut f64,
                &mut total as *mut f64,
            );
            if trc == 0 {
                println!(
                    "Timings: patch={:.1}ms encoder={:.1}ms decode={:.1}ms total={:.1}ms",
                    patch, encoder, decode, total
                );
            }
            println!("VLM output:\n{text}");
        } else {
            println!("VLM returned error: {}", last_error());
        }

        let _ = cellm_sdk::ffi::cellm_session_cancel(backend, session);
        cellm_sdk::ffi::cellm_engine_destroy(backend);
    }

    Ok(())
}

fn cstr(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s).expect("no interior nulls")
}

unsafe fn last_error() -> String {
    let mut buf = vec![0i8; 4096];
    let n = cellm_sdk::ffi::cellm_last_error_message(buf.as_mut_ptr() as *mut c_char, buf.len());
    if n == 0 {
        return "unknown error".to_string();
    }
    std::ffi::CStr::from_ptr(buf.as_ptr())
        .to_string_lossy()
        .to_string()
}
