use std::collections::VecDeque;

pub type SessionId = u64;

/// Minimal round-robin scheduler for decode steps.
#[derive(Debug, Default)]
pub struct RoundRobinScheduler {
    q: VecDeque<SessionId>,
}

impl RoundRobinScheduler {
    pub fn new() -> Self {
        Self { q: VecDeque::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }

    pub fn add(&mut self, id: SessionId) {
        if self.q.contains(&id) {
            return;
        }
        self.q.push_back(id);
    }

    pub fn remove(&mut self, id: SessionId) {
        if let Some(idx) = self.q.iter().position(|&x| x == id) {
            self.q.remove(idx);
        }
    }

    /// Pop the next session id and push it to the back (round-robin).
    pub fn next(&mut self) -> Option<SessionId> {
        let id = self.q.pop_front()?;
        self.q.push_back(id);
        Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_rotates() {
        let mut rr = RoundRobinScheduler::new();
        rr.add(1);
        rr.add(2);
        rr.add(3);
        assert_eq!(rr.next(), Some(1));
        assert_eq!(rr.next(), Some(2));
        assert_eq!(rr.next(), Some(3));
        assert_eq!(rr.next(), Some(1));
    }

    #[test]
    fn remove_works() {
        let mut rr = RoundRobinScheduler::new();
        rr.add(1);
        rr.add(2);
        rr.remove(1);
        assert_eq!(rr.next(), Some(2));
        assert_eq!(rr.next(), Some(2));
    }
}

