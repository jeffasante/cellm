// Author: Jeffrey Asante (https://jeffasante.github.io/)
use anyhow::Result;

fn main() -> Result<()> {
    cellm_kernels::MetalKernels::smoke_test_add_f32()?;
    println!("Metal smoke test: OK");
    Ok(())
}

