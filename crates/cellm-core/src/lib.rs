// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! cellm-core: Tensor abstraction, memory arena, and op dispatch.

pub mod arena;
pub mod dtype;
pub mod shape;
pub mod storage;
pub mod tensor;
pub mod backend;
pub mod error;
pub mod kv_cache;

pub use arena::{Arena, ArenaCheckpoint};
pub use dtype::DType;
pub use shape::{Shape, Strides};
pub use storage::{StorageHandle, StorageKind, StorageRegistry};
pub use tensor::TensorView;
pub use backend::Backend;
pub use error::CoreError;
pub use kv_cache::{KvCacheLayout, KvCacheReadView, KvCacheView};
