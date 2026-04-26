// Author: Jeffrey Asante (https://jeffasante.github.io/)
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("arena out of memory: requested {requested} bytes, available {available}")]
    ArenaOom { requested: usize, available: usize },

    #[error("invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DtypeMismatch {
        expected: crate::DType,
        got: crate::DType,
    },

    #[error("invalid storage handle: {0}")]
    InvalidHandle(u32),

    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("backend error: {0}")]
    Backend(String),
}