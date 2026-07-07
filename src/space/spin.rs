use crate::layout::Layout;

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
pub struct SpinSpace<L> {
    space: HomogeneousSpace<L>,
    spin: Spin,
}

impl<L> SpinSpace<L> {
    pub fn new(layout: L, spin: Spin) -> Self {
        Self {
            space: HomogeneousSpace::new(layout, spin.local_size()),
            spin,
        }
    }

    pub fn spin(&self) -> Spin {
        self.spin
    }
}

impl<L: Layout> Space for SpinSpace<L> {
    type Scalar = u8;

    fn sample_size(&self) -> usize {
        self.space.sample_size()
    }

    fn contains(&self, sample: &[Self::Scalar]) -> bool {
        self.space.contains(sample)
    }
}

impl<L: Layout + 'static> ViewSpace for SpinSpace<L> {
    type View<'a> = &'a [u8] where Self: 'a, Self::Scalar: 'a;

    fn view<'a>(&self, sample: &'a [Self::Scalar]) -> Self::View<'a> {
        self.space.view(sample)
    }
}

impl<L: Layout> HomogeneousProductSpace for SpinSpace<L> {
    fn local_size(&self) -> usize {
        self.spin.local_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let space = SpinSpace::new(Chain(3), Spin::integer(1));
        let sample = [0u8, 1, 2];
        assert_eq!(space.sample_size(), 3);
        assert_eq!(space.local_size(), 3);
        assert!(space.contains(&sample));
        assert_eq!(space.view(&sample), &sample);
    }
}
