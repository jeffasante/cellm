// Author: Jeffrey Asante (https://jeffasante.github.io/)
/// Supported data types for tensor elements.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    F8E4M3,
    I2,
    I8,
    U8,
    /// 8-bit symmetric block quantization (block size = 32)
    Q8_0,
}

impl DType {
    /// Size of one element in bytes.
    pub fn bytes_per_elem(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F8E4M3 => 1,
            DType::I2 => 1,
            DType::I8 => 1,
            DType::U8 => 1,
            // Q8_0: 32 int8 values + 1 f32 scale = 36 bytes per block of 32
            // Report per-element as approximate (actual storage is block-based)
            DType::Q8_0 => 1,
        }
    }

    /// True if this dtype requires a backend that understands block quantization.
    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::I2 | DType::Q8_0)
    }

    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F8E4M3 => "f8e4m3",
            DType::I2 => "i2",
            DType::I8 => "i8",
            DType::U8 => "u8",
            DType::Q8_0 => "q8_0",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl TryFrom<&str> for DType {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "f32" => Ok(DType::F32),
            "f16" => Ok(DType::F16),
            "bf16" => Ok(DType::BF16),
            "f8e4m3" => Ok(DType::F8E4M3),
            "i2" => Ok(DType::I2),
            "i8" => Ok(DType::I8),
            "u8" => Ok(DType::U8),
            "q8_0" => Ok(DType::Q8_0),
            other => Err(format!("unknown dtype: {other}")),
        }
    }
}
