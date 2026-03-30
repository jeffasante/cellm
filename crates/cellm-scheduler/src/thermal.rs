#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalLevel {
    Nominal,
    Elevated,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Copy)]
pub struct ThermalPolicy {
    level: ThermalLevel,
}

impl Default for ThermalPolicy {
    fn default() -> Self {
        Self {
            level: ThermalLevel::Nominal,
        }
    }
}

impl ThermalPolicy {
    pub fn new(level: ThermalLevel) -> Self {
        Self { level }
    }

    pub fn level(&self) -> ThermalLevel {
        self.level
    }

    pub fn set_level(&mut self, level: ThermalLevel) {
        self.level = level;
    }

    pub fn should_pause_decode(&self) -> bool {
        self.level == ThermalLevel::Emergency
    }

    pub fn max_active_decode_sessions(&self) -> usize {
        match self.level {
            ThermalLevel::Nominal => usize::MAX,
            ThermalLevel::Elevated => 2,
            ThermalLevel::Critical => 1,
            ThermalLevel::Emergency => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_budgets_match_level() {
        let mut t = ThermalPolicy::default();
        assert_eq!(t.max_active_decode_sessions(), usize::MAX);
        t.set_level(ThermalLevel::Elevated);
        assert_eq!(t.max_active_decode_sessions(), 2);
        t.set_level(ThermalLevel::Critical);
        assert_eq!(t.max_active_decode_sessions(), 1);
        t.set_level(ThermalLevel::Emergency);
        assert_eq!(t.max_active_decode_sessions(), 0);
        assert!(t.should_pause_decode());
    }
}
