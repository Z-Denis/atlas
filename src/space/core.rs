use core::fmt;

/// Minimal description of a configuration domain.
///
/// A space defines the flat sample size, the primitive scalar type used for a
/// single sample, and how to validate a contiguous slice.
pub trait Space {
    type Scalar;

    fn sample_size(&self) -> usize;

    fn contains(&self, sample: &[Self::Scalar]) -> bool;
}

/// Extension trait for spaces that can expose zero-copy structured views.
///
/// The view type is free to encode whatever structure is useful for the space
/// and its algorithms, as long as it borrows from the flat sample.
pub trait ViewSpace: Space {
    type View<'a>
    where
        Self: 'a,
        Self::Scalar: 'a;

    fn view<'a>(&self, sample: &'a [Self::Scalar]) -> Self::View<'a>;
}

/// Trait for objects that provide a log density over a configuration space.
pub trait LogDensity<S: Space> {
    type Real;

    fn log_density(&self, space: &S, sample: &[S::Scalar]) -> Self::Real;
}

/// Canonical contiguous storage for samples.
///
/// The last axis is always a complete sample of length `sample_size`. Leading
/// axes are batch axes stored in row-major order. The element type is the
/// primitive scalar chosen by the space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Samples<T> {
    pub data: Vec<T>,
    pub batch_shape: Vec<usize>,
    pub sample_size: usize,
}

impl<T> Samples<T> {
    pub fn new(
        data: Vec<T>,
        batch_shape: Vec<usize>,
        sample_size: usize,
    ) -> Result<Self, SamplesError> {
        let expected_len = expected_len(&batch_shape, sample_size)?;
        if data.len() != expected_len {
            return Err(SamplesError::LengthMismatch {
                data_len: data.len(),
                expected_len,
            });
        }

        Ok(Self {
            data,
            batch_shape,
            sample_size,
        })
    }

    pub fn iter_samples(&self) -> SamplesIter<'_, T> {
        SamplesIter {
            chunks: self.data.chunks_exact(self.sample_size),
        }
    }
}

fn expected_len(batch_shape: &[usize], sample_size: usize) -> Result<usize, SamplesError> {
    batch_shape
        .iter()
        .try_fold(sample_size, |acc, &dim| acc.checked_mul(dim))
        .ok_or(SamplesError::ShapeOverflow)
}

/// Iterator over immutable flat samples.
pub struct SamplesIter<'a, T> {
    chunks: core::slice::ChunksExact<'a, T>,
}

impl<'a, T> Iterator for SamplesIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.chunks.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for SamplesIter<'a, T> {}

/// Error returned when a `Samples<T>` container violates its shape invariant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SamplesError {
    LengthMismatch {
        data_len: usize,
        expected_len: usize,
    },
    ShapeOverflow,
}

impl fmt::Display for SamplesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                data_len,
                expected_len,
            } => write!(
                f,
                "sample buffer has length {}, expected {}",
                data_len, expected_len
            ),
            Self::ShapeOverflow => write!(f, "sample shape overflowed"),
        }
    }
}

impl std::error::Error for SamplesError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_validate_shape_invariant() {
        let samples = Samples::new(vec![0u8, 1, 1, 0], vec![2], 2).unwrap();
        assert_eq!(samples.batch_shape, vec![2]);
        assert_eq!(samples.sample_size, 2);
        assert_eq!(samples.data.len(), 4);
    }

    #[test]
    fn samples_reject_incorrect_length() {
        let err = Samples::new(vec![0u8, 1, 1], vec![2], 2).unwrap_err();
        assert_eq!(
            err,
            SamplesError::LengthMismatch {
                data_len: 3,
                expected_len: 4
            }
        );
    }

    #[test]
    fn samples_iterate_over_flat_samples() {
        let samples = Samples::new(vec![0u8, 1, 1, 0, 1, 1], vec![3], 2).unwrap();
        let collected: Vec<&[u8]> = samples.iter_samples().collect();
        assert_eq!(collected, vec![&[0, 1], &[1, 0], &[1, 1]]);
    }

    #[test]
    fn samples_support_arbitrary_batch_axes() {
        let samples = Samples::new(vec![0u8, 1, 1, 0, 1, 1, 0, 0], vec![2, 2], 2).unwrap();

        assert_eq!(samples.batch_shape, vec![2, 2]);
        let collected: Vec<&[u8]> = samples.iter_samples().collect();
        assert_eq!(collected, vec![&[0, 1], &[1, 0], &[1, 1], &[0, 0]]);
    }
}
