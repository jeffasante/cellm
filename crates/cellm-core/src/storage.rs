// Author: Jeffrey Asante (https://jeffasante.github.io/)
use std::collections::HashMap;
use crate::CoreError;

/// Opaque handle to a storage allocation. Cheap to copy.
/// Tensors hold handles, not raw pointers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StorageHandle(pub u32);

impl StorageHandle {
    pub const INVALID: StorageHandle = StorageHandle(u32::MAX);

    pub fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

/// Describes where a storage lives.
#[derive(Clone, Debug)]
pub enum StorageKind {
    /// Lives inside the Arena at byte offset.
    Arena { offset: usize, len: usize },
    /// Memory-mapped file region.
    Mmap { ptr: *const u8, len: usize },
    /// Owned heap allocation (for small/debug tensors).
    Owned(Vec<u8>),
}

// Safety: we never mutate Mmap through the raw pointer in safe code.
unsafe impl Send for StorageKind {}
unsafe impl Sync for StorageKind {}

/// Owns the mapping from handles → storage descriptors.
/// The actual bytes live elsewhere (in an Arena or mmap).
pub struct StorageRegistry {
    entries: HashMap<StorageHandle, StorageKind>,
    next_id: u32,
}

impl StorageRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            next_id: 0,
        }
    }

    fn next_handle(&mut self) -> StorageHandle {
        let h = StorageHandle(self.next_id);
        self.next_id += 1;
        h
    }

    /// Register an arena-backed storage.
    pub fn register_arena(&mut self, offset: usize, len: usize) -> StorageHandle {
        let h = self.next_handle();
        self.entries.insert(h, StorageKind::Arena { offset, len });
        h
    }

    /// Register a memory-mapped region.
    pub fn register_mmap(&mut self, ptr: *const u8, len: usize) -> StorageHandle {
        let h = self.next_handle();
        self.entries.insert(h, StorageKind::Mmap { ptr, len });
        h
    }

    /// Register owned bytes.
    pub fn register_owned(&mut self, data: Vec<u8>) -> StorageHandle {
        let h = self.next_handle();
        self.entries.insert(h, StorageKind::Owned(data));
        h
    }

    pub fn get(&self, h: StorageHandle) -> Result<&StorageKind, CoreError> {
        self.entries.get(&h).ok_or(CoreError::InvalidHandle(h.0))
    }

    /// Resolve a handle to a byte slice given arena memory.
    pub fn resolve<'a>(
        &'a self,
        h: StorageHandle,
        arena_buf: &'a [u8],
    ) -> Result<&'a [u8], CoreError> {
        match self.get(h)? {
            StorageKind::Arena { offset, len } => Ok(&arena_buf[*offset..*offset + *len]),
            StorageKind::Mmap { ptr, len } => {
                // Safety: ptr is valid for len bytes while the mmap is alive.
                Ok(unsafe { std::slice::from_raw_parts(*ptr, *len) })
            }
            StorageKind::Owned(v) => Ok(v.as_slice()),
        }
    }

    pub fn unregister(&mut self, h: StorageHandle) {
        self.entries.remove(&h);
    }
}

impl Default for StorageRegistry {
    fn default() -> Self {
        Self::new()
    }
}
