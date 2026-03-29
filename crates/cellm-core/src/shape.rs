use smallvec::SmallVec;
use crate::CoreError;

/// Tensor shape. Uses stack-allocated SmallVec for typical <= 4 dims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: SmallVec<[usize; 4]>,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: SmallVec::from_slice(dims),
        }
    }

    pub fn scalar() -> Self {
        Self::new(&[])
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn dim(&self, i: usize) -> Result<usize, CoreError> {
        self.dims.get(i).copied().ok_or_else(|| {
            CoreError::InvalidShape(format!(
                "dim index {} out of range for rank-{} shape",
                i,
                self.rank()
            ))
        })
    }

    /// Compute row-major (C-contiguous) strides.
    pub fn contiguous_strides(&self) -> Strides {
        let mut strides = SmallVec::<[isize; 4]>::with_capacity(self.rank());
        let mut stride = 1isize;
        for &d in self.dims.iter().rev() {
            strides.push(stride);
            stride *= d as isize;
        }
        strides.reverse();
        Strides { dims: strides }
    }

    pub fn is_empty(&self) -> bool {
        self.dims.iter().any(|&d| d == 0)
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

/// Strides in elements (not bytes). Can be negative for reversed dims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Strides {
    pub dims: SmallVec<[isize; 4]>,
}

impl Strides {
    pub fn new(dims: &[isize]) -> Self {
        Self {
            dims: SmallVec::from_slice(dims),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel() {
        assert_eq!(Shape::new(&[2, 3, 4]).numel(), 24);
        assert_eq!(Shape::new(&[]).numel(), 1); // scalar
    }

    #[test]
    fn strides() {
        let s = Shape::new(&[2, 3, 4]);
        let st = s.contiguous_strides();
        assert_eq!(st.dims.as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn dim_oob() {
        let s = Shape::new(&[2, 3]);
        assert!(s.dim(2).is_err());
    }
}
