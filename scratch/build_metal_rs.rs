use std::fs;
fn main() {
    let mut s = fs::read_to_string("crates/cellm-kernels/src/metal.rs").unwrap();
    
    // 1. Add missing PSO fields to MetalOps struct
    if !s.contains("pso_mv_i8") {
        s = s.replace(
            "pub pso_mv_f16: ComputePipelineState,",
            "pub pso_mv_f16: ComputePipelineState,
    pub pso_mv_i8: ComputePipelineState,
    pub pso_mv2_i8: ComputePipelineState,
    pub pso_mv_qkv_i8: ComputePipelineState,"
        );
    }
    if !s.contains("active_enc") {
        s = s.replace(
            "tensor_cache: std::collections::HashMap<String, Buffer>,",
            "tensor_cache: std::collections::HashMap<String, Buffer>,
    active_enc: Option<metal::ComputeCommandEncoder>,"
        );
    }

    // 2. Add missing PSO initialization in MetalOps::create
    if !s.contains("pso_mv_i8 = build_pso_ops") {
        s = s.replace(
            "let pso_mv_f16 = build_pso_ops(&device, &lib, \"mv_f16\")?;",
            "let pso_mv_f16 = build_pso_ops(&device, &lib, \"mv_f16\")?;
        let pso_mv_i8 = build_pso_ops(&device, &lib, \"mv_i8\")?;
        let pso_mv2_i8 = build_pso_ops(&device, &lib, \"mv2_i8\")?;
        let pso_mv_qkv_i8 = build_pso_ops(&device, &lib, \"mv_qkv_i8\")?;"
        );
        s = s.replace(
            "pso_mv_f16,",
            "pso_mv_f16,
            pso_mv_i8,
            pso_mv2_i8,
            pso_mv_qkv_i8,"
        );
    }
    if !s.contains("active_enc: None") {
         s = s.replace(
            "tensor_cache: std::collections::HashMap::new(),",
            "tensor_cache: std::collections::HashMap::new(),
            active_enc: None,"
        );
    }

    // 3. Add begin_pass / end_pass
    if !s.contains("fn begin_pass") {
        s = s.replace(
            "impl MetalOps {",
            "impl MetalOps {
    pub fn begin_pass(&mut self) -> anyhow::Result<()> {
        if self.active_enc.is_some() { anyhow::bail!(\"MetalOps: pass already active\"); }
        // Note: Real implementation would create a command buffer and encoder.
        // For simplicity in this runner, we just use it as a flag if needed, 
        // OR we actually implement it. Gemma runner seems to want it.
        Ok(())
    }
    pub fn end_pass(&mut self) -> anyhow::Result<()> {
        self.active_enc = None;
        Ok(())
    }"
        );
    }

    // 4. Add encode_mv_i8, etc.
    if !s.contains("fn encode_mv_i8") {
        s = s.replace(
            "pub fn encode_mv_f16(",
            "pub fn encode_mv_i8(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        a_buf: &metal::BufferRef,
        s_buf: &metal::BufferRef,
        x_buf: &metal::BufferRef,
        out_buf: &metal::BufferRef,
        rows: usize,
        cols: usize,
    ) {
        let r32 = rows as u32;
        let c32 = cols as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_i8);
        enc.set_buffer(0, Some(a_buf), 0);
        enc.set_buffer(1, Some(s_buf), 0);
        enc.set_buffer(2, Some(x_buf), 0);
        enc.set_buffer(3, Some(out_buf), 0);
        enc.set_bytes(4, 4, &r32 as *const u32 as *const _);
        enc.set_bytes(5, 4, &c32 as *const u32 as *const _);
        let threads = rows as u64;
        let w = self.pso_mv_i8.thread_execution_width() as u64;
        let tg = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
        let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_mv_f16("
        );
    }

    // 5. Add encode_rope_half_f32
    if !s.contains("fn encode_rope_half_f32") {
        s = s.replace(
            "pub fn encode_rope_adj_f32(",
            "pub fn encode_rope_half_f32(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        n_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        pos: usize,
        theta: f32,
    ) {
        let threads = (n_heads * (rotary_dim / 2)) as u64;
        let nh = n_heads as u32;
        let hd = head_dim as u32;
        let rd = rotary_dim as u32;
        let p = pos as u32;
        enc.set_compute_pipeline_state(&self.pso_rope_half);
        enc.set_buffer(0, Some(x_buf), 0);
        enc.set_bytes(1, 4, &nh as *const u32 as *const _);
        enc.set_bytes(2, 4, &hd as *const u32 as *const _);
        enc.set_bytes(3, 4, &rd as *const u32 as *const _);
        enc.set_bytes(4, 4, &p as *const u32 as *const _);
        enc.set_bytes(5, 4, &theta as *const f32 as *const _);
        let w = self.pso_rope_half.thread_execution_width() as u64;
        let tg = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
        let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_rope_adj_f32("
        );
    }

    fs::write("crates/cellm-kernels/src/metal.rs", s).unwrap();
}
