use crate::layout::Layout;

use burn::tensor::{BasicOps, Bool, Distribution, Numeric, Tensor, backend::Backend};
use burn_backend::Element;
use burn_backend::tensor::Ordered;
use num_traits::Float;
use num_traits::ToPrimitive;

use crate::utils::float_opts;

use super::core::{Space, ViewSpace};

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

#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousSpace<L, T> {
    layout: L,
    lower: T,
    upper: T,
}

impl<L: Layout, T: Float> ContinuousSpace<L, T> {
    pub fn new(layout: L, lower: T, upper: T) -> Self {
        assert!(lower <= upper);
        Self {
            layout,
            lower,
            upper,
        }
    }
}

impl<L: Layout, T: Float> Space for ContinuousSpace<L, T> {
    type Scalar = T;

    fn sample_size(&self) -> usize {
        self.layout.len()
    }

    fn contains<B, const D: usize, K>(&self, samples: Tensor<B, D, K>) -> Tensor<B, D, Bool>
    where
        B: Backend,
        K: BasicOps<B, Elem = T> + Ordered<B>,
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

impl<L: Layout + 'static, T: Float + 'static> ViewSpace for ContinuousSpace<L, T> {
    type View<'a>
        = &'a [T]
    where
        Self: 'a,
        Self::Scalar: 'a;

    fn view<'a>(&self, sample: &'a [Self::Scalar]) -> Self::View<'a> {
        debug_assert_eq!(sample.len(), self.sample_size());
        sample
    }
}

impl<L: Layout, T: Float> ContinuousSpace<L, T> {
    pub fn random_state<B, K>(&self, n_chains: usize, device: &B::Device) -> Tensor<B, 2, K>
    where
        B: Backend,
        K: Numeric<B, Elem = T>,
        T: Clone + Element,
    {
        if self.lower.is_finite() && self.upper.is_finite() {
            Tensor::<B, 2, K>::random(
                [n_chains, self.sample_size()],
                Distribution::Uniform(
                    ToPrimitive::to_f64(&self.lower).unwrap(),
                    ToPrimitive::to_f64(&self.upper).unwrap(),
                ),
                float_opts::<B>(device),
            )
        } else {
            Tensor::<B, 2, K>::random(
                [n_chains, self.sample_size()],
                Distribution::Normal(0.0, 1.0),
                float_opts::<B>(device),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::Tensor;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Chain(usize);

    impl Layout for Chain {
        fn len(&self) -> usize {
            self.0
        }
    }

    #[test]
    fn bounded_contains() {
        let space = ContinuousSpace::new(Chain(2), -1.0f32, 1.0);
        let device = Default::default();
        let valid: Tensor<Flex, 3> = Tensor::from_data([[[0.0, 0.0]], [[1.0, -1.0]]], &device);
        let invalid: Tensor<Flex, 3> = Tensor::from_data([[[0.0, 2.0]], [[1.0, -1.0]]], &device);
        assert!(space.contains(valid).all().into_scalar());
        assert!(!space.contains(invalid).all().into_scalar());
    }

    #[test]
    fn unbounded_contains() {
        let space = ContinuousSpace::new(Chain(2), f32::NEG_INFINITY, f32::INFINITY);
        let device = Default::default();
        let sample: Tensor<Flex, 3> = Tensor::from_data([[[0.0, 2.0]], [[1.0, -1.0]]], &device);
        assert!(space.contains(sample).all().into_scalar());
    }

    #[test]
    fn random_state_has_shape() {
        let space = ContinuousSpace::new(Chain(3), -1.0f32, 1.0);
        let device = Default::default();
        let state: Tensor<Flex, 2> = space.random_state(4, &device);
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
        let space = ContinuousSpace::new(Chain(6), f32::NEG_INFINITY, f32::INFINITY);
        let sample = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let view = Particles::new(space.view(&sample), 2);
        assert_eq!(view.n_particles(), 3);
        assert_eq!(view.particle(0), &[0.0, 1.0]);
        assert_eq!(view.particle(2), &[4.0, 5.0]);
    }
}
