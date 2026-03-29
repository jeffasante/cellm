use crate::{DType, Shape, StorageHandle};
use crate::shape::Strides;

/// A lightweight view over a tensor. Does not own memory.
/// The actual bytes are referenced via `storage` through a StorageRegistry.
#[derive(Clone, Debug)]
pub struct TensorView {
    pub dtype: DType,
    pub shape: Shape,
    pub strides: Strides,
    /// Handle into the StorageRegistry.
    pub storage: StorageHandle,
    /// Byte offset from the start of the storage region.
    pub byte_offset: usize,
}

impl TensorView {
    pub fn new(
        dtype: DType,
        shape: Shape,
        storage: StorageHandle,
        byte_offset: usize,
    ) -> Self {
        let strides = shape.contiguous_strides();
        Self {
            dtype,
            shape,
            strides,
            storage,
            byte_offset,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Total bytes occupied by this tensor.
    pub fn nbytes(&self) -> usize {
        self.numel() * self.dtype.bytes_per_elem()
    }

    /// True if strides match row-major layout with no holes.
    pub fn is_contiguous(&self) -> bool {
        let expected = self.shape.contiguous_strides();
        self.strides == expected
    }

    /// Slice the last dimension: returns a view of row `i` in a 2-D tensor.
    pub fn row(&self, i: usize) -> Self {
        assert_eq!(self.shape.rank(), 2);
        let cols = self.shape.dims()[1];
        let elem_size = self.dtype.bytes_per_elem();
        Self {
            dtype: self.dtype,
            shape: Shape::new(&[cols]),
            strides: Strides::new(&[1]),
            storage: self.storage,
            byte_offset: self.byte_offset + i * cols * elem_size,
        }
    }

    /// Return a sub-view starting at token index `start` with `len` tokens.
    /// Only works on 2-D tensors [seq, dim].
    pub fn token_slice(&self, start: usize, len: usize) -> Self {
        assert_eq!(self.shape.rank(), 2);
        let dim = self.shape.dims()[1];
        let elem_size = self.dtype.bytes_per_elem();
        Self {
            dtype: self.dtype,
            shape: Shape::new(&[len, dim]),
            strides: self.strides.clone(),
            storage: self.storage,
            byte_offset: self.byte_offset + start * dim * elem_size,
        }
    }
}

impl std::fmt::Display for TensorView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor[{} {} handle={}]",
            self.dtype,
            self.shape,
            self.storage.0
        )
    }
}
