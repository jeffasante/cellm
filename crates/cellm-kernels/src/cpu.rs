use cellm_core::{Backend, CoreError, TensorView};

/// Phase 1/2 placeholder backend.
///
/// The model forward pass currently doesn't execute real matmul/attention yet,
/// but we still need a `Backend` object to exercise paged KV cache integration.
#[derive(Debug, Default)]
pub struct CpuBackendStub;

impl CpuBackendStub {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackendStub {
    fn name(&self) -> &'static str {
        "cpu-stub"
    }

    fn matmul(
        &self,
        _a: &TensorView,
        _b: &TensorView,
        _out: &mut TensorView,
        _arena: &[u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend("matmul not implemented".into()))
    }

    fn rms_norm(
        &self,
        _x: &TensorView,
        _weight: &TensorView,
        _out: &mut TensorView,
        _eps: f32,
        _arena: &[u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend("rms_norm not implemented".into()))
    }

    fn rope_inplace(
        &self,
        _q: &mut TensorView,
        _k: &mut TensorView,
        _positions: &[usize],
        _theta: f32,
        _arena: &mut [u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend("rope_inplace not implemented".into()))
    }

    fn silu(
        &self,
        _x: &TensorView,
        _out: &mut TensorView,
        _arena: &[u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend("silu not implemented".into()))
    }

    fn add(
        &self,
        _a: &TensorView,
        _b: &TensorView,
        _out: &mut TensorView,
        _arena: &[u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend("add not implemented".into()))
    }

    fn mul(
        &self,
        _a: &TensorView,
        _b: &TensorView,
        _out: &mut TensorView,
        _arena: &[u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend("mul not implemented".into()))
    }

    fn softmax_inplace(&self, _x: &mut TensorView, _arena: &mut [u8]) -> Result<(), CoreError> {
        Err(CoreError::Backend("softmax_inplace not implemented".into()))
    }

    fn attention(
        &self,
        _q: &TensorView,
        _k: &TensorView,
        _v: &TensorView,
        _n_heads: usize,
        _n_kv_heads: usize,
        _out: &mut TensorView,
        _arena: &[u8],
        _scratch: &mut Vec<f32>,
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend("attention not implemented".into()))
    }
}

// Back-compat export name used in earlier stubs.
pub type SIMDKernels = CpuBackendStub;
