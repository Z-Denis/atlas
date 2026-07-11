use burn::tensor::{BasicOps, Bool, Int, Numeric, Tensor, backend::Backend};
use burn_backend::Element;
use burn_backend::tensor::Ordered;

use super::continuous::Particles;
use super::core::{LocalSpace, RandomState, Space, ViewSpace};

#[doc(hidden)]
pub trait HomogeneousValue<B: Backend>: burn_backend::tensor::TensorKind<B> + BasicOps<B> {}

impl<B, T> HomogeneousValue<B> for T
where
    B: Backend,
    T: burn_backend::tensor::TensorKind<B> + BasicOps<B>,
{
}

#[doc(hidden)]
pub trait HomogeneousState<B: Backend>: HomogeneousValue<B> + Numeric<B> + Ordered<B> {}

impl<B, T> HomogeneousState<B> for T
where
    B: Backend,
    T: HomogeneousValue<B> + Numeric<B> + Ordered<B>,
{
}

/// Extension trait for local spaces with a finite set of local states.
pub trait HomogeneousProductSpace: LocalSpace {
    fn local_states(&self) -> &[Self::Scalar];

    fn local_size(&self) -> usize {
        self.local_states().len()
    }

    fn indices_of<B>(&self, values: Tensor<B, 1, Self::DType>) -> Tensor<B, 1, Int>
    where
        B: Backend,
        Self::DType: HomogeneousValue<B> + BasicOps<B, Elem = Self::Scalar>,
        Self::Scalar: Clone + Element,
    {
        let device = values.device();
        let local_size = self.local_size();
        let n_values = values.dims()[0];
        let states = Tensor::<B, 1, Self::DType>::from_data(self.local_states(), &device);

        values
            .clone()
            .unsqueeze_dim::<2>(1)
            .expand([n_values, local_size])
            .equal(states.unsqueeze_dim::<2>(0).expand([n_values, local_size]))
            .int()
            .argmax(1)
            .squeeze_dim::<1>(1)
    }

    fn states_at<B>(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 1, Self::DType>
    where
        B: Backend,
        Self::DType: HomogeneousValue<B> + BasicOps<B, Elem = Self::Scalar>,
        Self::Scalar: Clone + Element,
    {
        let device = indices.device();
        let states = Tensor::<B, 1, Self::DType>::from_data(self.local_states(), &device);
        states.select(0, indices)
    }
}

/// Homogeneous product of a local space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HomogeneousSpace<L> {
    local: L,
    n: usize,
}

impl<L> HomogeneousSpace<L> {
    pub fn new(local: L, n: usize) -> Self {
        assert!(n > 0);
        Self { local, n }
    }

    pub fn local(&self) -> &L {
        &self.local
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.n
    }
}

impl<L: Space> Space for HomogeneousSpace<L> {
    type Scalar = L::Scalar;
    type DType = L::DType;

    fn sample_size(&self) -> usize {
        self.local.sample_size() * self.n
    }

    fn contains<B, const D: usize>(&self, samples: Tensor<B, D, Self::DType>) -> Tensor<B, D, Bool>
    where
        B: Backend,
        Self::DType: HomogeneousState<B> + BasicOps<B, Elem = Self::Scalar>,
        Self::Scalar: Clone + Element,
    {
        let device = samples.device();
        let dims = samples.dims();
        let sample_size = dims[D - 1];
        let mut out_dims = dims;
        out_dims[D - 1] = 1;

        if sample_size != self.sample_size() {
            return Tensor::<B, D, Bool>::zeros(out_dims, &device);
        }

        let local_size = self.local.sample_size();
        let flat_size = dims[..D - 1].iter().product::<usize>();
        if sample_size == 0 {
            return Tensor::<B, D, Bool>::ones(out_dims, &device);
        }

        let flat = samples.reshape([flat_size * self.n, local_size]);
        let valid = self.local.contains(flat);

        valid
            .reshape([flat_size, self.n])
            .all_dim(1)
            .reshape(out_dims)
    }
}

impl<L: Space + 'static> ViewSpace for HomogeneousSpace<L> {
    type View<'a>
        = Particles<'a, L::Scalar>
    where
        Self: 'a,
        Self::Scalar: 'a;

    fn view<'a>(&self, sample: &'a [Self::Scalar]) -> Self::View<'a> {
        debug_assert_eq!(sample.len(), self.sample_size());
        Particles::new(sample, self.local.sample_size())
    }
}

impl<L: LocalSpace> LocalSpace for HomogeneousSpace<L> {}

impl<L: Space + RandomState> RandomState for HomogeneousSpace<L> {
    fn random_state<B>(&self, n_chains: usize, device: &B::Device) -> Tensor<B, 2, Self::DType>
    where
        B: Backend,
        Self::DType: HomogeneousValue<B> + Numeric<B, Elem = Self::Scalar>,
        Self::Scalar: Clone + Element,
    {
        self.local
            .random_state::<B>(n_chains * self.n, device)
            .reshape([n_chains, self.sample_size()])
    }
}

impl<L: HomogeneousProductSpace> HomogeneousProductSpace for HomogeneousSpace<L> {
    fn local_states(&self) -> &[Self::Scalar] {
        self.local.local_states()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FloatTensor;
    use crate::space::{ContinuousSpace, Spin};
    use burn::backend::Flex;

    #[test]
    fn continuous_product_space_checks_domain() {
        let local = ContinuousSpace::new(-1.0f32, 1.0, 2);
        let space = HomogeneousSpace::new(local, 2);
        let device = Default::default();
        let valid = FloatTensor::<Flex, 3>::from_data([[[0.0, 0.0, 1.0, -1.0]]], &device);
        let invalid = FloatTensor::<Flex, 3>::from_data([[[0.0, 2.0, 1.0, -1.0]]], &device);
        assert!(space.contains(valid).all().into_scalar());
        assert!(!space.contains(invalid).all().into_scalar());
    }

    #[test]
    fn continuous_product_space_views_particles() {
        let local = ContinuousSpace::new(-1.0f32, 1.0, 2);
        let space = HomogeneousSpace::new(local, 3);
        let sample = [0.0f32, 1.0, -1.0, 0.5, 0.25, -0.25];
        let view = space.view(&sample);
        assert_eq!(view.n_particles(), 3);
        assert_eq!(view.dim(), 2);
        assert_eq!(view.particle(1), &[-1.0, 0.5]);
        assert_eq!(view.particle(2), &[0.25, -0.25]);
    }

    #[test]
    fn spin_product_space_views_flat_sample() {
        let local = Spin::half_integer(1);
        let space = HomogeneousSpace::new(local, 3);
        let sample = [-1i32, 1, -1];
        let view = space.view(&sample);
        assert_eq!(view.n_particles(), 3);
        assert_eq!(view.dim(), 1);
        assert_eq!(view.particle(1), &[1]);
    }

    #[test]
    fn product_space_random_state_has_shape() {
        let local = ContinuousSpace::new(-1.0f32, 1.0, 2);
        let space = HomogeneousSpace::new(local, 3);
        let device = Default::default();
        let state = space.random_state::<Flex>(4, &device);
        assert_eq!(state.dims(), [4, 6]);
    }
}
