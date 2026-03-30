use crate::rr::SessionId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Queued,
    Prefill,
    Decoding,
    Suspended,
    Terminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionError {
    InvalidTransition {
        from: SessionState,
        to: SessionState,
    },
}

#[derive(Debug, Clone)]
pub struct Session {
    id: SessionId,
    state: SessionState,
    priority: u8,
    last_touch_tick: u64,
    prompt_tokens: u64,
    generated_tokens: u64,
}

impl Session {
    pub fn new(id: SessionId) -> Self {
        Self {
            id,
            state: SessionState::Queued,
            priority: 0,
            last_touch_tick: 0,
            prompt_tokens: 0,
            generated_tokens: 0,
        }
    }

    pub fn id(&self) -> SessionId {
        self.id
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    pub fn priority(&self) -> u8 {
        self.priority
    }

    pub fn set_priority(&mut self, priority: u8) {
        self.priority = priority;
    }

    pub fn last_touch_tick(&self) -> u64 {
        self.last_touch_tick
    }

    pub fn prompt_tokens(&self) -> u64 {
        self.prompt_tokens
    }

    pub fn generated_tokens(&self) -> u64 {
        self.generated_tokens
    }

    pub fn touch(&mut self, tick: u64) {
        self.last_touch_tick = tick;
    }

    pub fn add_prompt_tokens(&mut self, n: usize) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(n as u64);
    }

    pub fn add_generated_token(&mut self) {
        self.generated_tokens = self.generated_tokens.saturating_add(1);
    }

    pub fn transition(&mut self, to: SessionState) -> Result<(), SessionError> {
        if can_transition(self.state, to) {
            self.state = to;
            Ok(())
        } else {
            Err(SessionError::InvalidTransition {
                from: self.state,
                to,
            })
        }
    }
}

fn can_transition(from: SessionState, to: SessionState) -> bool {
    use SessionState::*;
    match (from, to) {
        (Terminal, Terminal) => true,
        (Terminal, _) => false,
        (Queued, Prefill | Suspended | Terminal) => true,
        (Prefill, Decoding | Suspended | Terminal) => true,
        (Decoding, Suspended | Terminal) => true,
        (Suspended, Queued | Prefill | Decoding | Terminal) => true,
        (a, b) if a == b => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_transitions_work() {
        let mut s = Session::new(7);
        assert_eq!(s.state(), SessionState::Queued);
        s.transition(SessionState::Prefill).unwrap();
        s.transition(SessionState::Decoding).unwrap();
        s.transition(SessionState::Suspended).unwrap();
        s.transition(SessionState::Queued).unwrap();
    }

    #[test]
    fn terminal_is_sticky() {
        let mut s = Session::new(9);
        s.transition(SessionState::Terminal).unwrap();
        let err = s.transition(SessionState::Queued).unwrap_err();
        assert_eq!(
            err,
            SessionError::InvalidTransition {
                from: SessionState::Terminal,
                to: SessionState::Queued
            }
        );
    }
}
