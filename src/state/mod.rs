use burn::tensor::{BasicOps, Numeric, Tensor, TensorCreationOptions, backend::Backend};
use burn_backend::Element;
use burn_backend::tensor::Ordered;

use crate::model::Model;
use crate::sampler::{LogDensity, Metropolis, Proposal, SamplerState};
use crate::space::{RandomState, Samples, Space};

/// Marker for the ambient state space interpreted by a variational state.
///
/// The state space decides how to turn a model's `log_value` into a sampler
/// facing `log_density`.
pub trait StateSpace {
    fn log_density<B>(&self, log_value: Tensor<B, 1>) -> Tensor<B, 1>
    where
        B: Backend;
}

/// Simplex state space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Simplex;

impl StateSpace for Simplex {
    fn log_density<B>(&self, log_value: Tensor<B, 1>) -> Tensor<B, 1>
    where
        B: Backend,
    {
        log_value
    }
}

/// Hilbert state space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hilbert;

impl StateSpace for Hilbert {
    fn log_density<B>(&self, log_value: Tensor<B, 1>) -> Tensor<B, 1>
    where
        B: Backend,
    {
        // The quantum case is real-valued for now.
        2 * log_value
    }
}

/// Adapter that turns a model plus state space into a sampler-facing density.
#[derive(Clone, Copy, Debug)]
struct StateLogDensity<'a, M, SS> {
    model: &'a M,
    state_space: &'a SS,
}

impl<'a, M, SS> StateLogDensity<'a, M, SS> {
    fn new(model: &'a M, state_space: &'a SS) -> Self {
        Self { model, state_space }
    }
}

/// User-facing orchestration object.
///
/// A variational state owns the model, sampler, chain state, and collected
/// samples. It is the natural checkpoint boundary.
pub struct VariationalState<M, S: Space, B, K, P, SS = Simplex>
where
    B: Backend,
    K: BasicOps<B> + Numeric<B>,
{
    pub model: M,
    pub space: S,
    pub state_space: SS,
    pub sampler: Metropolis<P>,
    pub sampler_state: SamplerState<B, K>,
    pub samples: Samples<B, 2, K>,
    n_samples_per_chain: usize,
}

impl<M, S: Space, B, K, P, SS> VariationalState<M, S, B, K, P, SS>
where
    B: Backend,
    K: BasicOps<B> + Numeric<B>,
{
    pub fn new(
        model: M,
        space: S,
        state_space: SS,
        sampler: Metropolis<P>,
        sampler_state: SamplerState<B, K>,
        n_samples_per_chain: usize,
    ) -> Self {
        assert!(n_samples_per_chain > 0);

        let dims = sampler_state.chains.dims();
        let device = sampler_state.chains.device();
        let samples = Tensor::<B, 2, K>::zeros(
            [dims[0] * n_samples_per_chain, dims[1]],
            TensorCreationOptions::<B>::new(device),
        );

        Self {
            model,
            space,
            state_space,
            sampler,
            sampler_state,
            samples,
            n_samples_per_chain,
        }
    }

    /// Construct a variational state and initialize chains from the space.
    pub fn from_space(
        model: M,
        space: S,
        state_space: SS,
        sampler: Metropolis<P>,
        n_chains: usize,
        n_samples_per_chain: usize,
    ) -> Self
    where
        S: RandomState,
        K: burn::tensor::Numeric<B, Elem = S::Scalar>,
        S::Scalar: Clone + Element,
    {
        let device = Default::default();
        let sampler_state = SamplerState::from_space(&space, n_chains, &device);
        Self::new(
            model,
            space,
            state_space,
            sampler,
            sampler_state,
            n_samples_per_chain,
        )
    }

    pub fn sample(&mut self)
    where
        S: Space,
        B: Backend,
        K: BasicOps<B, Elem = S::Scalar> + Numeric<B> + Ordered<B>,
        S::Scalar: Clone + Element,
        P: Proposal<B, S, K>,
        M: Model<S, B>,
        SS: StateSpace,
    {
        let sweep_size = self.space.sample_size();
        let n_chains = self.sampler_state.chains.dims()[0];
        let device = self.sampler_state.chains.device();
        let sampler = &self.sampler;
        let log_density = StateLogDensity::new(&self.model, &self.state_space);

        for sample_idx in 0..self.n_samples_per_chain {
            for _ in 0..sweep_size {
                sampler.step(&self.space, &log_density, &mut self.sampler_state);
            }

            let start = sample_idx * n_chains;
            let end = start + n_chains;
            self.samples = self.samples.clone().slice_assign(
                start..end,
                self.sampler_state.chains.clone().to_device(&device),
            );
        }
    }

    /// Evaluate the model on the collected samples buffer.
    pub fn log_value(&self) -> Tensor<B, 1>
    where
        M: Model<S, B>,
    {
        self.model.log_value(&self.space, self.samples.clone())
    }

    /// Evaluate the model on arbitrary configurations.
    pub fn log_value_on(&self, samples: Tensor<B, 2, K>) -> Tensor<B, 1>
    where
        M: Model<S, B>,
    {
        self.model.log_value(&self.space, samples)
    }
}

impl<'a, M, S, B, K, SS> LogDensity<B, S, K> for StateLogDensity<'a, M, SS>
where
    S: Space,
    B: Backend,
    K: Numeric<B>,
    M: Model<S, B>,
    SS: StateSpace,
{
    fn log_density(&self, space: &S, samples: Tensor<B, 2, K>) -> Tensor<B, 1> {
        self.state_space
            .log_density(self.model.log_value(space, samples))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::LocalProposal;
    use crate::space::HomogeneousSpace;
    use crate::space::Spin;
    use crate::test_utils::ZeroModel;
    use burn::backend::Flex;
    use burn::tensor::{Int, Tensor};

    #[test]
    fn log_value_uses_collected_samples() {
        let space = HomogeneousSpace::new(Spin::half_integer(1), 1);
        let sampler = Metropolis::new(LocalProposal);
        let state: VariationalState<_, _, Flex, Int, _, Simplex> =
            VariationalState::from_space(ZeroModel, space, Simplex, sampler, 1, 2);

        let values = state.log_value();

        assert_eq!(values.dims(), [2]);
    }

    #[test]
    fn log_value_on_uses_arbitrary_samples() {
        let space = HomogeneousSpace::new(Spin::half_integer(1), 1);
        let sampler = Metropolis::new(LocalProposal);
        let state: VariationalState<_, _, Flex, Int, _, Simplex> =
            VariationalState::from_space(ZeroModel, space, Simplex, sampler, 1, 2);
        let samples = Tensor::<Flex, 2, Int>::from_data([[1], [0]], &Default::default());

        let values = state.log_value_on(samples);

        assert_eq!(values.dims(), [2]);
    }
}
