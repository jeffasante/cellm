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
                256,
                40,
                0.7,
                1.15,
                128,
                1,
                1,
            )
        } else {
            cellm_sdk::ffi::cellm_engine_create_v3(
                cstr(&model).as_ptr(),
                16,
                256,
                40,
                0.7,
                1.15,
                128,
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
