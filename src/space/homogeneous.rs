use burn::tensor::{BasicOps, Bool, Int, Tensor, backend::Backend};
use burn_backend::Element;

use crate::layout::Layout;
use crate::utils::randint;

use super::core::{Space, ViewSpace};

pub trait HomogeneousProductSpace: Space {
    fn local_states(&self) -> &[Self::Scalar];

    fn local_size(&self) -> usize {
        self.local_states().len()
    }

    /// Return the local-state indices for each value.
    fn indices_of<B, K>(&self, values: Tensor<B, 1, K>) -> Tensor<B, 1, Int>
    where
        B: Backend,
        K: BasicOps<B, Elem = Self::Scalar>,
        Self::Scalar: Clone + Element,
    {
        let device = values.device();
        let local_size = self.local_size();
        let n_values = values.dims()[0];
        let states = Tensor::<B, 1, K>::from_data(self.local_states(), &device);

        values
            .clone()
            .unsqueeze_dim::<2>(1)
            .expand([n_values, local_size])
            .equal(states.unsqueeze_dim::<2>(0).expand([n_values, local_size]))
            .int()
            .argmax(1)
            .squeeze_dim::<1>(1)
    }

    /// Return the local states for each local-state index.
    fn states_at<B, K>(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 1, K>
    where
        B: Backend,
        K: BasicOps<B, Elem = Self::Scalar>,
        Self::Scalar: Clone + Element,
    {
        let device = indices.device();
        let states = Tensor::<B, 1, K>::from_data(self.local_states(), &device);
        states.select(0, indices)
    }

    /// Generate `n_chains` random configurations with shape `[n_chains, sample_size]`.
    fn random_state<B, K>(&self, n_chains: usize, device: &B::Device) -> Tensor<B, 2, K>
    where
        B: Backend,
        K: BasicOps<B, Elem = Self::Scalar>,
        Self::Scalar: Clone + Element,
    {
        let states = Tensor::<B, 1, K>::from_data(self.local_states(), device);
        let indices = randint::<B, 2>(
            [n_chains, self.sample_size()],
            0,
            self.local_size() as i64,
            device,
        );

        states.take::<2, 2>(0, indices)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HomogeneousSpace<L, T> {
    layout: L,
    local_states: Vec<T>,
}

impl<L: Layout, T: PartialEq> HomogeneousSpace<L, T> {
    pub fn new(layout: L, local_states: Vec<T>) -> Self {
        Self {
            layout,
            local_states,
        }
    }
}

impl<L: Layout, T: PartialEq> Space for HomogeneousSpace<L, T> {
    type Scalar = T;

    fn sample_size(&self) -> usize {
        self.layout.len()
    }

    fn contains<B, const D: usize, K>(&self, samples: Tensor<B, D, K>) -> Tensor<B, D, Bool>
    where
        B: Backend,
        K: BasicOps<B, Elem = T>,
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

        let local_size = self.local_size();
        let flat = samples.reshape([flat_size, sample_size]);
        let states = Tensor::<B, 1, K>::from_data(self.local_states(), &device)
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .expand([flat_size, sample_size, local_size]);

        flat.unsqueeze_dim::<3>(2)
            .expand([flat_size, sample_size, local_size])
            .equal(states)
            .any_dim(2)
            .all_dim(1)
            .reshape(out_dims)
    }
}

impl<L: Layout + 'static, T: PartialEq + 'static> ViewSpace for HomogeneousSpace<L, T> {
    type View<'a>
        = &'a [T]
    where
        Self: 'a,
        Self::Scalar: 'a;

    fn view<'a>(&self, sample: &'a [Self::Scalar]) -> Self::View<'a> {
        debug_assert_eq!(sample.len(), self.sample_size());
        debug_assert!(sample.iter().all(|x| self.local_states.contains(x)));
        sample
    }
}

impl<L: Layout, T: PartialEq> HomogeneousProductSpace for HomogeneousSpace<L, T> {
    fn local_states(&self) -> &[Self::Scalar] {
        &self.local_states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::{Int, Tensor};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Chain(usize);

    impl Layout for Chain {
        fn len(&self) -> usize {
            self.0
        }
    }

    #[test]
    fn checks_domain() {
        let space = HomogeneousSpace::new(Chain(4), vec![0i32, 1]);
        assert_eq!(space.sample_size(), 4);
        assert_eq!(space.local_size(), 2);
        let device = Default::default();
        let valid = Tensor::<Flex, 3, Int>::from_data([[[0, 1, 1, 0]], [[1, 0, 1, 0]]], &device);
        let invalid = Tensor::<Flex, 3, Int>::from_data([[[0, 1, 2, 1]], [[1, 0, 1, 0]]], &device);
        assert_eq!(space.contains(valid.clone()).dims(), [2, 1, 1]);
        assert!(space.contains(valid).all().into_scalar());
        assert!(!space.contains(invalid).all().into_scalar());
    }

    #[test]
    fn views_flat_sample() {
        let space = HomogeneousSpace::new(Chain(3), vec![0i32, 1, 2]);
        let sample = [1i32, 0, 2];
        let view = space.view(&sample);
        assert_eq!(view, &sample);
    }

    #[test]
    fn generates_random_state() {
        let device = Default::default();
        let space = HomogeneousSpace::new(Chain(4), vec![0i32, 1]);
        let state: Tensor<Flex, 2, Int> = space.random_state(3, &device);
        assert_eq!(state.dims(), [3, 4]);
    }
}
