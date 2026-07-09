use crate::layout::Layout;

use burn::tensor::{backend::Backend, BasicOps, Bool, Tensor};
use burn_backend::Element;

use super::core::{Space, ViewSpace};
use super::homogeneous::{HomogeneousProductSpace, HomogeneousSpace};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spin {
    twice_s: usize,
}

impl Spin {
    pub fn half_integer(n: usize) -> Self {
        Self { twice_s: n }
    }

    pub fn integer(n: usize) -> Self {
        Self { twice_s: 2 * n }
    }

    pub fn local_size(&self) -> usize {
        self.twice_s + 1
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpinSpace<L, T> {
    space: HomogeneousSpace<L, T>,
    spin: Spin,
}

impl<L: Layout, T: PartialEq> SpinSpace<L, T> {
    pub fn new(layout: L, spin: Spin, local_states: Vec<T>) -> Self {
        assert_eq!(local_states.len(), spin.local_size());
        Self {
            space: HomogeneousSpace::new(layout, local_states),
            spin,
        }
    }

    pub fn spin(&self) -> Spin {
        self.spin
    }
}

impl<L: Layout, T: PartialEq> Space for SpinSpace<L, T> {
    type Scalar = T;

    fn sample_size(&self) -> usize {
        self.space.sample_size()
    }

    fn contains<B, const D: usize, K>(&self, samples: Tensor<B, D, K>) -> Tensor<B, D, Bool>
    where
        B: Backend,
        K: BasicOps<B, Elem = Self::Scalar>,
        Self::Scalar: Clone + Element,
    {
        self.space.contains(samples)
    }
}

impl<L: Layout + 'static, T: PartialEq + 'static> ViewSpace for SpinSpace<L, T> {
    type View<'a>
        = &'a [T]
    where
        Self: 'a,
        Self::Scalar: 'a;

    fn view<'a>(&self, sample: &'a [Self::Scalar]) -> Self::View<'a> {
        self.space.view(sample)
    }
}

impl<L: Layout, T: PartialEq> HomogeneousProductSpace for SpinSpace<L, T> {
    fn local_states(&self) -> &[Self::Scalar] {
        self.space.local_states()
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
    fn spin_constructors_work() {
        assert_eq!(Spin::half_integer(1).local_size(), 2);
        assert_eq!(Spin::integer(1).local_size(), 3);
    }

    #[test]
    fn spin_space_uses_homogeneous_structure() {
        let space = SpinSpace::new(Chain(3), Spin::integer(1), vec![-1i32, 0, 1]);
        let device = Default::default();
        let sample: Tensor<Flex, 2, Int> = Tensor::from_data([[-1i32, 0, 1]], &device);
        assert_eq!(space.sample_size(), 3);
        assert_eq!(space.local_size(), 3);
        assert!(space.contains(sample.clone()).into_scalar());
        assert_eq!(space.view(&[-1i32, 0, 1]), &[-1i32, 0, 1]);
    }
}
