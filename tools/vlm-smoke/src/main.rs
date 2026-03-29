use std::ffi::c_char;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "vlm-smoke", about = "Smoke test the VLM FFI call path (currently not implemented)")]
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
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model = args.model.to_string_lossy().to_string();
    let image_bytes = std::fs::read(&args.image)?;

    unsafe {
        let engine = cellm_sdk::ffi::cellm_engine_create_v2(
            cstr(&model).as_ptr(),
            16,
            256,
            40,
            0.7,
            1.05,
            64,
            1,
        );
        if engine == 0 {
            anyhow::bail!("engine_create failed: {}", last_error());
        }
        let session = cellm_sdk::ffi::cellm_session_create(engine);
        if session == 0 {
            cellm_sdk::ffi::cellm_engine_destroy(engine);
            anyhow::bail!("session_create failed: {}", last_error());
        }

        let rc = cellm_sdk::ffi::cellm_vlm_describe_image(
            engine,
            session,
            image_bytes.as_ptr(),
            image_bytes.len(),
            cstr(&args.prompt).as_ptr(),
            std::ptr::null_mut(),
            0,
        );

        // We currently expect this to fail with a clear message.
        if rc == 0 {
            println!("VLM returned success (unexpected for now).");
        } else {
            println!("VLM returned error (expected for now): {}", last_error());
        }

        let _ = cellm_sdk::ffi::cellm_session_cancel(engine, session);
        cellm_sdk::ffi::cellm_engine_destroy(engine);
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

