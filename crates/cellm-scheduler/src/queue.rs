use std::collections::VecDeque;

use crate::rr::SessionId;

#[derive(Debug, Clone, Default)]
pub struct Queue {
    q: VecDeque<SessionId>,
}

impl Queue {
    pub fn new() -> Self {
        Self { q: VecDeque::new() }
    }

    pub fn len(&self) -> usize {
        self.q.len()
    }

    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }

    pub fn contains(&self, id: SessionId) -> bool {
        self.q.contains(&id)
    }

    pub fn push_back_unique(&mut self, id: SessionId) {
        if !self.contains(id) {
            self.q.push_back(id);
        }
    }

    pub fn pop_front(&mut self) -> Option<SessionId> {
        self.q.pop_front()
    }

    pub fn remove(&mut self, id: SessionId) -> bool {
        if let Some(idx) = self.q.iter().position(|&x| x == id) {
            self.q.remove(idx);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_push_and_remove() {
        let mut q = Queue::new();
        q.push_back_unique(1);
        q.push_back_unique(1);
        q.push_back_unique(2);
        assert_eq!(q.len(), 2);
        assert!(q.remove(1));
        assert!(!q.remove(9));
        assert_eq!(q.pop_front(), Some(2));
        assert!(q.is_empty());
    }
}
