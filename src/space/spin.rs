use burn::tensor::{BasicOps, Bool, Tensor, backend::Backend};
use burn_backend::Element;
use burn_backend::tensor::Ordered;

use crate::utils::randint;

use super::core::{LocalSpace, RandomState, Space, ViewSpace};
use super::homogeneous::{HomogeneousProductSpace, HomogeneousSpace};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spin {
    twice_s: usize,
    local_states: Vec<i32>,
}

impl Spin {
    pub fn half_integer(n: usize) -> Self {
        let twice_s = n;
        let local_states = (0..=twice_s)
            .map(|m| -(twice_s as i32) + 2 * m as i32)
            .collect();
        Self {
            twice_s,
            local_states,
        }
    }

    pub fn integer(n: usize) -> Self {
        let twice_s = 2 * n;
        let local_states = (0..=twice_s)
            .map(|m| -(twice_s as i32) + 2 * m as i32)
            .collect();
        Self {
            twice_s,
            local_states,
        }
    }

    pub fn local_size(&self) -> usize {
        self.local_states.len()
    }

    pub fn local_states(&self) -> &[i32] {
        &self.local_states
    }
}

impl Space for Spin {
    type Scalar = i32;

    fn sample_size(&self) -> usize {
        1
    }

    fn contains<B, const D: usize, K>(&self, samples: Tensor<B, D, K>) -> Tensor<B, D, Bool>
    where
        B: Backend,
        K: BasicOps<B, Elem = Self::Scalar> + Ordered<B>,
        Self::Scalar: Clone + Element,
    {
        let device = samples.device();
        let dims = samples.dims();
        let sample_size = dims[D - 1];
        let mut out_dims = dims;
        out_dims[D - 1] = 1;

        if sample_size != 1 {
            return Tensor::<B, D, Bool>::zeros(out_dims, &device);
        }

        let flat_size = dims[..D - 1].iter().product::<usize>();
        let flat = samples.reshape([flat_size, sample_size]);
        let states = Tensor::<B, 1, K>::from_data(self.local_states(), &device)
            .unsqueeze_dim::<2>(0)
            .expand([flat_size, self.local_size()]);

        flat.expand([flat_size, self.local_size()])
            .equal(states)
            .any_dim(1)
            .reshape(out_dims)
    }
}

impl ViewSpace for Spin {
    type View<'a>
        = &'a [i32]
    where
        Self: 'a,
        Self::Scalar: 'a;

    fn view<'a>(&self, sample: &'a [Self::Scalar]) -> Self::View<'a> {
        debug_assert_eq!(sample.len(), 1);
        sample
    }
}

impl LocalSpace for Spin {}

impl RandomState for Spin {
    fn random_state<B, K>(&self, n_chains: usize, device: &B::Device) -> Tensor<B, 2, K>
    where
        B: Backend,
        K: BasicOps<B, Elem = Self::Scalar>,
        Self::Scalar: Clone + Element,
    {
        let states = Tensor::<B, 1, K>::from_data(self.local_states(), device);
        let indices = randint::<B, 2>([n_chains, 1], 0, self.local_size() as i64, device);
        states.take::<2, 2>(0, indices)
    }
}

impl HomogeneousProductSpace for Spin {
    fn local_states(&self) -> &[Self::Scalar] {
        self.local_states()
    }
}

pub type SpinSpace = HomogeneousSpace<Spin>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::{Int, Tensor};

    #[test]
    fn spin_constructors_work() {
        assert_eq!(Spin::half_integer(1).local_size(), 2);
        assert_eq!(Spin::integer(1).local_size(), 3);
    }

    #[test]
    fn spin_space_uses_homogeneous_structure() {
        let local = Spin::integer(1);
        let space = HomogeneousSpace::new(local, 3);
        let device = Default::default();
        let sample: Tensor<Flex, 2, Int> = Tensor::from_data([[-2i32, 0, 2]], &device);
        assert_eq!(space.sample_size(), 3);
        assert!(space.contains(sample.clone()).into_scalar());
        assert_eq!(space.view(&[-2i32, 0, 2]).particle(1), &[0]);
    }
}
