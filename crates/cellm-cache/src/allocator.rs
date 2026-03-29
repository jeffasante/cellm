use std::collections::VecDeque;

use crate::CacheError;

/// Fixed-size block allocator for paged KV cache.
///
/// The allocator manages *block ids* (u32). The actual KV bytes live in a
/// separate pool owned by the cache implementation / backend.
#[derive(Debug)]
pub struct BlockAllocator {
    total: u32,
    in_use: u32,
    free_list: VecDeque<u32>,
    is_free: Vec<bool>,
}

impl BlockAllocator {
    /// Create an allocator with `total_blocks` ids in the free list.
    pub fn new(total_blocks: usize) -> Self {
        assert!(
            total_blocks <= (u32::MAX as usize),
            "total_blocks must fit in u32"
        );

        let mut free_list = VecDeque::with_capacity(total_blocks);
        for i in 0..(total_blocks as u32) {
            free_list.push_back(i);
        }

        Self {
            total: total_blocks as u32,
            in_use: 0,
            free_list,
            is_free: vec![true; total_blocks],
        }
    }

    pub fn total_count(&self) -> usize {
        self.total as usize
    }

    pub fn in_use_count(&self) -> usize {
        self.in_use as usize
    }

    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Allocate one block id.
    pub fn alloc(&mut self) -> Option<u32> {
        let id = self.free_list.pop_front()?;
        debug_assert!(self.is_free[id as usize]);
        self.is_free[id as usize] = false;
        self.in_use += 1;
        Some(id)
    }

    /// Allocate `n` block ids or return an error without changing state.
    pub fn alloc_n(&mut self, n: usize) -> Result<Vec<u32>, CacheError> {
        if n > self.free_count() {
            return Err(CacheError::OutOfBlocks {
                requested: n,
                free: self.free_count(),
            });
        }

        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.alloc().expect("checked free_count"));
        }
        Ok(out)
    }

    /// Free a previously allocated block id.
    pub fn free(&mut self, block_id: u32) -> Result<(), CacheError> {
        if block_id >= self.total {
            return Err(CacheError::InvalidBlockId(block_id));
        }
        let idx = block_id as usize;
        if self.is_free[idx] {
            return Err(CacheError::DoubleFree(block_id));
        }
        self.is_free[idx] = true;
        self.free_list.push_back(block_id);
        self.in_use = self.in_use.saturating_sub(1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_roundtrip() {
        let mut a = BlockAllocator::new(3);
        assert_eq!(a.total_count(), 3);
        assert_eq!(a.free_count(), 3);
        assert_eq!(a.in_use_count(), 0);

        let b0 = a.alloc().unwrap();
        let b1 = a.alloc().unwrap();
        assert_ne!(b0, b1);
        assert_eq!(a.free_count(), 1);
        assert_eq!(a.in_use_count(), 2);

        a.free(b0).unwrap();
        a.free(b1).unwrap();
        assert_eq!(a.free_count(), 3);
        assert_eq!(a.in_use_count(), 0);
    }

    #[test]
    fn alloc_exhaustion() {
        let mut a = BlockAllocator::new(1);
        assert!(a.alloc().is_some());
        assert!(a.alloc().is_none());
    }

    #[test]
    fn double_free_errors() {
        let mut a = BlockAllocator::new(1);
        let id = a.alloc().unwrap();
        a.free(id).unwrap();
        let err = a.free(id).unwrap_err();
        assert_eq!(err, CacheError::DoubleFree(id));
    }

    #[test]
    fn invalid_block_id_errors() {
        let mut a = BlockAllocator::new(1);
        let err = a.free(99).unwrap_err();
        assert_eq!(err, CacheError::InvalidBlockId(99));
    }

    #[test]
    fn alloc_n_is_atomic() {
        let mut a = BlockAllocator::new(2);
        let err = a.alloc_n(3).unwrap_err();
        assert_eq!(
            err,
            CacheError::OutOfBlocks {
                requested: 3,
                free: 2
            }
        );
        assert_eq!(a.free_count(), 2);
        assert_eq!(a.in_use_count(), 0);
    }
}
