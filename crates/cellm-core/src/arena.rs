use crate::CoreError;

/// Checkpoint into an arena — lets you restore position after a block of work.
#[derive(Copy, Clone, Debug)]
pub struct ArenaCheckpoint(pub usize);

/// Bump allocator. Alloc is O(1), free-all is O(1).
///
/// Two intended usages:
///   - Persistent arena  → model weights, never reset
///   - Scratch arena     → activations per forward step, reset() each step
pub struct Arena {
    buf: Vec<u8>,
    cursor: usize,
    /// Track peak usage for profiling.
    peak: usize,
    label: &'static str,
}

impl Arena {
    pub fn new(capacity: usize, label: &'static str) -> Self {
        Self {
            buf: vec![0u8; capacity],
            cursor: 0,
            peak: 0,
            label,
        }
    }

    /// Allocate `size` bytes aligned to `align`. Returns byte offset into the buffer.
    pub fn alloc(&mut self, size: usize, align: usize) -> Result<usize, CoreError> {
        let aligned_cursor = align_up(self.cursor, align);
        let end = aligned_cursor + size;

        if end > self.buf.len() {
            return Err(CoreError::ArenaOom {
                requested: size,
                available: self.buf.len().saturating_sub(self.cursor),
            });
        }

        self.cursor = end;
        if self.cursor > self.peak {
            self.peak = self.cursor;
        }

        Ok(aligned_cursor)
    }

    /// Alloc and return a mutable byte slice.
    pub fn alloc_slice(&mut self, size: usize, align: usize) -> Result<&mut [u8], CoreError> {
        let offset = self.alloc(size, align)?;
        Ok(&mut self.buf[offset..offset + size])
    }

    /// Reset all allocations. O(1).
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Save position so you can restore later.
    pub fn checkpoint(&self) -> ArenaCheckpoint {
        ArenaCheckpoint(self.cursor)
    }

    /// Restore to a prior checkpoint. Frees everything allocated after it.
    pub fn restore(&mut self, cp: ArenaCheckpoint) {
        assert!(cp.0 <= self.cursor, "checkpoint is ahead of cursor");
        self.cursor = cp.0;
    }

    pub fn used_bytes(&self) -> usize {
        self.cursor
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    pub fn peak_bytes(&self) -> usize {
        self.peak
    }

    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Raw access to the underlying buffer (needed by backends).
    pub fn buffer(&self) -> &[u8] {
        &self.buf
    }

    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    /// Copy bytes into the arena and return the offset.
    pub fn copy_in(&mut self, data: &[u8], align: usize) -> Result<usize, CoreError> {
        let offset = self.alloc(data.len(), align)?;
        self.buf[offset..offset + data.len()].copy_from_slice(data);
        Ok(offset)
    }

    pub fn print_stats(&self) {
        log::debug!(
            "[arena:{}] used={} KB  peak={} KB  cap={} KB",
            self.label,
            self.cursor / 1024,
            self.peak / 1024,
            self.buf.len() / 1024,
        );
    }
}

fn align_up(val: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (val + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_alloc() {
        let mut arena = Arena::new(1024, "test");
        let off = arena.alloc(64, 16).unwrap();
        assert_eq!(off, 0);
        let off2 = arena.alloc(64, 16).unwrap();
        assert_eq!(off2, 64);
    }

    #[test]
    fn alignment() {
        let mut arena = Arena::new(1024, "test");
        let _ = arena.alloc(1, 1).unwrap(); // cursor = 1
        let off = arena.alloc(4, 4).unwrap(); // should align to 4
        assert_eq!(off, 4);
    }

    #[test]
    fn checkpoint_restore() {
        let mut arena = Arena::new(1024, "test");
        let _ = arena.alloc(64, 8).unwrap();
        let cp = arena.checkpoint();
        let _ = arena.alloc(128, 8).unwrap();
        assert_eq!(arena.used_bytes(), 192);
        arena.restore(cp);
        assert_eq!(arena.used_bytes(), 64);
    }

    #[test]
    fn oom_error() {
        let mut arena = Arena::new(32, "test");
        let result = arena.alloc(64, 8);
        assert!(result.is_err());
    }

    #[test]
    fn reset() {
        let mut arena = Arena::new(256, "test");
        arena.alloc(128, 8).unwrap();
        arena.reset();
        assert_eq!(arena.used_bytes(), 0);
        assert_eq!(arena.peak_bytes(), 128);
    }
}
