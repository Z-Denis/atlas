use burn::tensor::{BasicOps, Bool, Distribution, Numeric, Tensor, backend::Backend};
use burn_backend::Element;
use burn_backend::tensor::Ordered;
use num_traits::{Float, ToPrimitive};

use crate::utils::float_opts;

use super::core::{LocalSpace, RandomState, Space, ViewSpace};
use super::homogeneous::HomogeneousSpace;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Particles<'a, T> {
    sample: &'a [T],
    dim: usize,
}

impl<'a, T> Particles<'a, T> {
    pub fn new(sample: &'a [T], dim: usize) -> Self {
        assert!(dim > 0);
        assert_eq!(sample.len() % dim, 0);
        Self { sample, dim }
    }

    pub fn n_particles(&self) -> usize {
        self.sample.len() / self.dim
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn particle(&self, i: usize) -> &'a [T] {
        let start = i * self.dim;
        let end = start + self.dim;
        &self.sample[start..end]
    }
}

/// Bounded or unbounded continuous local space of fixed dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousSpace<T> {
    dim: usize,
    lower: T,
    upper: T,
}

impl<T: Float> ContinuousSpace<T> {
    pub fn new(lower: T, upper: T, dim: usize) -> Self {
        assert!(dim > 0);
        assert!(lower <= upper);
        Self { dim, lower, upper }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl<T: Float> Space for ContinuousSpace<T> {
    type Scalar = T;
    type DType = burn::tensor::Float;

    fn sample_size(&self) -> usize {
        self.dim
    }

    fn contains<B, const D: usize>(&self, samples: Tensor<B, D, Self::DType>) -> Tensor<B, D, Bool>
    where
        B: Backend,
        Self::DType: BasicOps<B, Elem = T> + Ordered<B>,
        T: Clone + Element,
    {
        let device = samples.device();
        let dims = samples.dims();
        let sample_size = dims[D - 1];
        let mut out_dims = dims;
        out_dims[D - 1] = 1;

        if sample_size != self.sample_size() {
            return Tensor::<B, D, Bool>::zeros(out_dims, &device);
        }

        let flat_size = dims[..D - 1].iter().product::<usize>();
        if sample_size == 0 {
            return Tensor::<B, D, Bool>::ones(out_dims, &device);
        }

        let flat = samples.reshape([flat_size, sample_size]);
        let mut valid = Tensor::<B, 2, Bool>::ones(flat.dims(), &device);

        if self.lower.is_finite() {
            valid = valid.bool_and(flat.clone().greater_equal_elem(self.lower));
        }
        if self.upper.is_finite() {
            valid = valid.bool_and(flat.clone().lower_equal_elem(self.upper));
        }

        valid.all_dim(1).reshape(out_dims)
    }
}

impl<T: Float + 'static> ViewSpace for ContinuousSpace<T> {
    type View<'a>
        = Particles<'a, T>
    where
        Self: 'a,
        Self::Scalar: 'a;

    fn view<'a>(&self, sample: &'a [Self::Scalar]) -> Self::View<'a> {
        debug_assert_eq!(sample.len(), self.sample_size());
        Particles::new(sample, self.dim)
    }
}

impl<T: Float> LocalSpace for ContinuousSpace<T> {}

impl<T: Float> RandomState for ContinuousSpace<T> {
    fn random_state<B>(&self, n_chains: usize, device: &B::Device) -> Tensor<B, 2, Self::DType>
    where
        B: Backend,
        Self::DType: Numeric<B, Elem = Self::Scalar>,
        Self::Scalar: Clone + Element,
    {
        if self.lower.is_finite() && self.upper.is_finite() {
            Tensor::<B, 2, Self::DType>::random(
                [n_chains, self.sample_size()],
                Distribution::Uniform(
                    ToPrimitive::to_f64(&self.lower).unwrap(),
                    ToPrimitive::to_f64(&self.upper).unwrap(),
                ),
                float_opts::<B>(device),
            )
        } else {
            Tensor::<B, 2, Self::DType>::random(
                [n_chains, self.sample_size()],
                Distribution::Normal(0.0, 1.0),
                float_opts::<B>(device),
            )
        }
    }
}

pub type ParticleSpace<T> = HomogeneousSpace<ContinuousSpace<T>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FloatTensor;
    use burn::backend::Flex;

    #[test]
    fn bounded_contains() {
        let space = ContinuousSpace::new(-1.0f32, 1.0, 2);
        let device = Default::default();
        let valid = FloatTensor::<Flex, 3>::from_data([[[0.0, 0.0]], [[1.0, -1.0]]], &device);
        let invalid = FloatTensor::<Flex, 3>::from_data([[[0.0, 2.0]], [[1.0, -1.0]]], &device);
        assert!(space.contains(valid).all().into_scalar());
        assert!(!space.contains(invalid).all().into_scalar());
    }

    #[test]
    fn unbounded_contains() {
        let space = ContinuousSpace::new(f32::NEG_INFINITY, f32::INFINITY, 2);
        let device = Default::default();
        let sample = FloatTensor::<Flex, 3>::from_data([[[0.0, 2.0]], [[1.0, -1.0]]], &device);
        assert!(space.contains(sample).all().into_scalar());
    }

    #[test]
    fn random_state_has_shape() {
        let space = ContinuousSpace::new(-1.0f32, 1.0, 3);
        let device = Default::default();
        let state = space.random_state::<Flex>(4, &device);
        assert_eq!(state.dims(), [4, 3]);
    }

    #[test]
    fn particles_view_chunks_flat_sample() {
        let view = Particles::new(&[0i32, 1, 2, 3, 4, 5], 2);
        assert_eq!(view.n_particles(), 3);
        assert_eq!(view.dim(), 2);
        assert_eq!(view.particle(1), &[2, 3]);
    }

    #[test]
    fn particles_view_supports_d_gt_1() {
        let space = ContinuousSpace::new(f32::NEG_INFINITY, f32::INFINITY, 2);
        let sample = [0.0f32, 1.0];
        let view = space.view(&sample);
        assert_eq!(view.n_particles(), 1);
        assert_eq!(view.particle(0), &[0.0, 1.0]);
    }
}
