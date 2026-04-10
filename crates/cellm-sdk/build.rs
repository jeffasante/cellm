fn main() {
    println!("cargo:rerun-if-env-changed=LITERT_LM_LIB_PATH");

    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if os == "macos" || os == "ios" {
        // Only link LiteRT explicitly when the caller provides a concrete library path.
        // Default iOS/macOS builds stay Python-free and framework-agnostic.
        if let Ok(path) = std::env::var("LITERT_LM_LIB_PATH") {
            let path = std::path::Path::new(&path);
            if let Some(parent) = path.parent() {
                println!("cargo:rustc-link-search=native={}", parent.display());
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let lib_name = stem.strip_prefix("lib").unwrap_or(stem);
                println!("cargo:rustc-link-lib=dylib={}", lib_name);
            }
        }
    }
}
