use crate::layout::Layout;

use super::core::{Space, ViewSpace};

pub trait HomogeneousProductSpace: Space {
    fn local_size(&self) -> usize;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HomogeneousSpace<L> {
    layout: L,
    local_size: usize,
}

impl<L> HomogeneousSpace<L> {
    pub fn new(layout: L, local_size: usize) -> Self {
        Self { layout, local_size }
    }
}

impl<L: Layout> Space for HomogeneousSpace<L> {
    type Scalar = u8;

    fn sample_size(&self) -> usize {
        self.layout.len()
    }

    fn contains(&self, sample: &[Self::Scalar]) -> bool {
        sample.len() == self.sample_size() && sample.iter().all(|&x| usize::from(x) < self.local_size)
    }
}

impl<L: Layout + 'static> ViewSpace for HomogeneousSpace<L> {
    type View<'a> = &'a [u8] where Self: 'a, Self::Scalar: 'a;

    fn view<'a>(&self, sample: &'a [Self::Scalar]) -> Self::View<'a> {
        debug_assert!(self.contains(sample));
        sample
    }
}

impl<L: Layout> HomogeneousProductSpace for HomogeneousSpace<L> {
    fn local_size(&self) -> usize {
        self.local_size
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
    fn homogeneous_space_checks_local_state_domain() {
        let space = HomogeneousSpace::new(Chain(4), 3);
        assert_eq!(space.sample_size(), 4);
        assert_eq!(space.local_size(), 3);
        assert!(space.contains(&[0, 1, 2, 1]));
        assert!(!space.contains(&[0, 1, 3, 1]));
    }

    #[test]
    fn homogeneous_space_views_flat_sample() {
        let space = HomogeneousSpace::new(Chain(3), 2);
        let sample = [1u8, 0, 1];
        let view = space.view(&sample);
        assert_eq!(view, &sample);
    }
}
