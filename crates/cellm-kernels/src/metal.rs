#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions};

pub struct MetalKernels;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub type MetalBuffer = Buffer;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct MetalBuffer;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalMatmul {
    queue: CommandQueue,
    _lib: Library,
    pso: ComputePipelineState,
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct MetalMatmul;

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalKernels {
    pub fn smoke_test_add_f32() -> anyhow::Result<()> {
        let device = Device::system_default()
            .or_else(|| Device::all().into_iter().next())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Metal: no device found (system_default and all() empty). \
                     If you are in a restricted/sandboxed shell, re-run outside sandbox."
                )
            })?;
        let queue = device.new_command_queue();

        let src = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void add_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            out[id] = a[id] + b[id];
        }
        "#;

        let (lib, pso) = build_pipeline(&device, src, "add_f32")?;

        let n = 1024usize;
        let bytes = (n * std::mem::size_of::<f32>()) as u64;
        let a = make_buf_f32(&device, n, |i| i as f32)?;
        let b = make_buf_f32(&device, n, |i| (2 * i) as f32)?;
        let out = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);

        dispatch_1d(&queue, &pso, n as u64, |enc| {
            enc.set_buffer(0, Some(&a), 0);
            enc.set_buffer(1, Some(&b), 0);
            enc.set_buffer(2, Some(&out), 0);
        })?;

        // Validate.
        let out_ptr = out.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, n) };
        for i in 0..n {
            let expected = (i as f32) + (2 * i) as f32;
            let got = out_slice[i];
            if (got - expected).abs() > 1e-5 {
                anyhow::bail!("Metal add_f32 mismatch at {i}: got={got} expected={expected}");
            }
        }

        // Keep lib referenced so the pipeline stays valid in debug builds.
        let _ = lib;
        Ok(())
    }

    pub fn create_matmul() -> anyhow::Result<MetalMatmul> {
        let device = Device::system_default()
            .or_else(|| Device::all().into_iter().next())
            .ok_or_else(|| anyhow::anyhow!("Metal: no device found"))?;
        let queue = device.new_command_queue();
        let src = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void matmul_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            constant uint& m [[buffer(3)]],
            constant uint& n [[buffer(4)]],
            constant uint& k [[buffer(5)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint row = gid.y;
            uint col = gid.x;
            if (row >= m || col >= n) return;
            float acc = 0.0f;
            for (uint kk = 0; kk < k; ++kk) {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            out[row * n + col] = acc;
        }
        "#;
        let (lib, pso) = build_pipeline(&device, src, "matmul_f32")?;
        Ok(MetalMatmul {
            queue,
            _lib: lib,
            pso,
        })
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalKernels {
    pub fn smoke_test_add_f32() -> anyhow::Result<()> {
        anyhow::bail!("MetalKernels only supported on macOS in this crate build")
    }

    pub fn create_matmul() -> anyhow::Result<MetalMatmul> {
        anyhow::bail!("Metal matmul only supported on Apple platforms")
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalMatmul {
    pub fn upload_f32(&self, src: &[f32]) -> anyhow::Result<MetalBuffer> {
        let device = self.queue.device();
        make_buf_from_f32(device, src)
    }

    pub fn matmul_row_major_f32(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        if a.len() != m * k || b.len() != k * n || out.len() != m * n {
            anyhow::bail!("Metal matmul shape mismatch");
        }
        let device = self.queue.device();
        let a_buf = make_buf_from_f32(device, a)?;
        let b_buf = make_buf_from_f32(device, b)?;
        self.matmul_row_major_f32_with_b_buffer(&a_buf, m, k, &b_buf, n, out)
    }

    pub fn matmul_row_major_f32_with_b_buffer(
        &self,
        a_buf: &MetalBuffer,
        m: usize,
        k: usize,
        b_buf: &MetalBuffer,
        n: usize,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        if out.len() != m * n {
            anyhow::bail!("Metal matmul shape mismatch for output");
        }
        let out_bytes = (out.len() * std::mem::size_of::<f32>()) as u64;
        let device = self.queue.device();
        let out_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let m_buf = make_buf_from_u32(device, &[m_u32])?;
        let n_buf = make_buf_from_u32(device, &[n_u32])?;
        let k_buf = make_buf_from_u32(device, &[k_u32])?;

        dispatch_2d(&self.queue, &self.pso, n as u64, m as u64, |enc| {
            enc.set_buffer(0, Some(a_buf), 0);
            enc.set_buffer(1, Some(b_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_buffer(3, Some(&m_buf), 0);
            enc.set_buffer(4, Some(&n_buf), 0);
            enc.set_buffer(5, Some(&k_buf), 0);
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, out.len()) };
        out.copy_from_slice(out_slice);
        Ok(())
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalMatmul {
    pub fn upload_f32(&self, _src: &[f32]) -> anyhow::Result<MetalBuffer> {
        anyhow::bail!("Metal matmul only supported on Apple platforms")
    }

    pub fn matmul_row_major_f32(
        &self,
        _a: &[f32],
        _m: usize,
        _k: usize,
        _b: &[f32],
        _n: usize,
        _out: &mut [f32],
    ) -> anyhow::Result<()> {
        anyhow::bail!("Metal matmul only supported on Apple platforms")
    }

    pub fn matmul_row_major_f32_with_b_buffer(
        &self,
        _a_buf: &MetalBuffer,
        _m: usize,
        _k: usize,
        _b_buf: &MetalBuffer,
        _n: usize,
        _out: &mut [f32],
    ) -> anyhow::Result<()> {
        anyhow::bail!("Metal matmul only supported on Apple platforms")
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn build_pipeline(device: &Device, src: &str, fn_name: &str) -> anyhow::Result<(Library, ComputePipelineState)> {
    let options = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(src, &options)
        .map_err(|e| anyhow::anyhow!("Metal: failed to compile library: {e:?}"))?;
    let func = lib
        .get_function(fn_name, None)
        .map_err(|e| anyhow::anyhow!("Metal: missing function {fn_name}: {e:?}"))?;
    let pso = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("Metal: failed to build pipeline: {e:?}"))?;
    Ok((lib, pso))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn make_buf_f32(
    device: &Device,
    n: usize,
    f: impl Fn(usize) -> f32,
) -> anyhow::Result<Buffer> {
    let bytes = (n * std::mem::size_of::<f32>()) as u64;
    let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    let ptr = buf.contents() as *mut f32;
    if ptr.is_null() {
        anyhow::bail!("Metal: buffer contents is null");
    }
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, n) };
    for i in 0..n {
        slice[i] = f(i);
    }
    Ok(buf)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn make_buf_from_f32(device: &metal::DeviceRef, src: &[f32]) -> anyhow::Result<Buffer> {
    let bytes = (std::mem::size_of_val(src)) as u64;
    let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    let ptr = buf.contents() as *mut f32;
    if ptr.is_null() {
        anyhow::bail!("Metal: buffer contents is null");
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, src.len()) };
    dst.copy_from_slice(src);
    Ok(buf)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn make_buf_from_u32(device: &metal::DeviceRef, src: &[u32]) -> anyhow::Result<Buffer> {
    let bytes = (std::mem::size_of_val(src)) as u64;
    let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    let ptr = buf.contents() as *mut u32;
    if ptr.is_null() {
        anyhow::bail!("Metal: buffer contents is null");
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, src.len()) };
    dst.copy_from_slice(src);
    Ok(buf)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn dispatch_1d(
    queue: &CommandQueue,
    pso: &ComputePipelineState,
    threads: u64,
    bind: impl FnOnce(&metal::ComputeCommandEncoderRef),
) -> anyhow::Result<()> {
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    bind(enc);

    let w = pso.thread_execution_width() as u64;
    let tg = metal::MTLSize {
        width: w.min(threads),
        height: 1,
        depth: 1,
    };
    let grid = metal::MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn dispatch_2d(
    queue: &CommandQueue,
    pso: &ComputePipelineState,
    width: u64,
    height: u64,
    bind: impl FnOnce(&metal::ComputeCommandEncoderRef),
) -> anyhow::Result<()> {
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    bind(enc);
    let w = pso.thread_execution_width() as u64;
    let tg = metal::MTLSize {
        width: w.max(1).min(width.max(1)),
        height: 1,
        depth: 1,
    };
    let grid = metal::MTLSize {
        width,
        height,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(())
}
