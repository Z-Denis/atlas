use burn::tensor::{BasicOps, Bool, Numeric, Tensor, backend::Backend};
use burn_backend::Element;
use burn_backend::tensor::Ordered;

/// Minimal description of a configuration domain.
///
/// A space defines the flat sample size, the primitive scalar type used inside
/// each configuration, and how to validate tensorized samples. The last axis is
/// always the flat configuration axis; leading axes are batch axes.
pub trait Space {
    type Scalar;

    fn sample_size(&self) -> usize;

    fn contains<B, const D: usize, K>(&self, samples: Tensor<B, D, K>) -> Tensor<B, D, Bool>
    where
        B: Backend,
        K: BasicOps<B, Elem = Self::Scalar> + Ordered<B>,
        Self::Scalar: Clone + Element;
}

/// Marker trait for a structured local degree of freedom.
pub trait LocalSpace: Space {}

/// Extension trait for spaces that can generate initial states.
pub trait RandomState: Space {
    fn random_state<B, K>(&self, n_chains: usize, device: &B::Device) -> Tensor<B, 2, K>
    where
        B: Backend,
        K: Numeric<B, Elem = Self::Scalar>,
        Self::Scalar: Clone + Element;
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

/// Trait for Burn-backed objects that provide a real-valued log density over a
/// configuration space.
///
/// The input tensor stores flat samples on the last axis and may carry any
/// number of leading batch axes. The returned tensor keeps the same leading
/// axes and is real-valued on the backend.
pub trait LogDensity<S: Space> {
    fn log_density<B, const D: usize, K>(
        &self,
        space: &S,
        samples: Tensor<B, D, K>,
    ) -> Tensor<B, D>
    where
        B: Backend,
        K: BasicOps<B>,
        K::Elem: Element;
}

/// Canonical contiguous storage for samples.
///
/// The last axis is always a complete sample of length `sample_size`. Leading
/// axes are batch axes. Burn owns the backend representation; Atlas only keeps
/// the flat sample layout and scalar type.
pub type Samples<B, const D: usize, K = burn::tensor::Float> = Tensor<B, D, K>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::{Int, Tensor};

    fn dims<const D: usize>(tensor: Samples<NdArray, D, Int>) -> [usize; D] {
        tensor.dims()
    }

    #[test]
    fn samples_store_axes() {
        let device = Default::default();
        let data = Tensor::<NdArray, 2, Int>::from_data([[0, 1], [1, 0]], &device);
        assert_eq!(dims(data), [2, 2]);
    }

    #[test]
    fn samples_support_batch_axes() {
        let device = Default::default();
        let data = Tensor::<NdArray, 2, Int>::from_data([[0, 1], [1, 0], [1, 1], [0, 0]], &device);
        assert_eq!(dims(data), [4, 2]);
    }
}
