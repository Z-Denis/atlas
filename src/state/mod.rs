use burn::tensor::{BasicOps, Numeric, Tensor, TensorCreationOptions, backend::Backend};
use burn_backend::Element;
use burn_backend::tensor::Ordered;

use crate::model::Model;
use crate::sampler::{LogDensity, Metropolis, Proposal, SamplerState};
use crate::space::{RandomState, Samples, Space};
use crate::utils::FloatTensor;
use burn::tensor::FloatDType;
use burn_backend::tensor::{Float, TensorKind};

#[doc(hidden)]
pub trait IntoFloatTensor<B: Backend, const D: usize>: TensorKind<B> {
    fn into_float(tensor: Tensor<B, D, Self>, dtype: FloatDType) -> FloatTensor<B, D>;
}

impl<B: Backend, const D: usize> IntoFloatTensor<B, D> for Float {
    fn into_float(tensor: Tensor<B, D, Self>, dtype: FloatDType) -> FloatTensor<B, D> {
        tensor.cast(dtype)
    }
}

impl<B: Backend, const D: usize> IntoFloatTensor<B, D> for burn::tensor::Int {
    fn into_float(tensor: Tensor<B, D, Self>, dtype: FloatDType) -> FloatTensor<B, D> {
        tensor.cast(dtype)
    }
}

fn into_model_tensor<B, K>(samples: Tensor<B, 2, K>, dtype: FloatDType) -> FloatTensor<B, 2>
where
    B: Backend,
    K: IntoFloatTensor<B, 2>,
{
    K::into_float(samples, dtype)
}

/// Marker for the ambient state space interpreted by a variational state.
///
/// The state space decides how a model's `log_value` is exposed to the
/// sampler as a `log_density`.
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

/// Private adapter used by `VariationalState::sample()` to expose a model as
/// a sampler-facing density.
#[derive(Clone, Copy, Debug)]
struct StateLogDensity<'a, M, SS> {
    model: &'a M,
    state_space: &'a SS,
    param_dtype: FloatDType,
}

impl<'a, M, SS> StateLogDensity<'a, M, SS> {
    fn new(model: &'a M, state_space: &'a SS, param_dtype: FloatDType) -> Self {
        Self {
            model,
            state_space,
            param_dtype,
        }
    }
}

/// User-facing orchestration object.
///
/// A variational state owns the model, space, state space, sampler, chain
/// state, and collected samples. It is the natural checkpoint boundary.
pub struct VariationalState<M, S: Space, B, P, SS = Simplex>
where
    B: Backend,
    S::DType: BasicOps<B> + Numeric<B> + IntoFloatTensor<B, 2>,
{
    pub model: M,
    pub space: S,
    pub state_space: SS,
    pub sampler: Metropolis<P>,
    pub sampler_state: SamplerState<B, S::DType>,
    pub samples: Samples<B, 2, S::DType>,
    n_samples_per_chain: usize,
}

impl<M, S: Space, B, P, SS> VariationalState<M, S, B, P, SS>
where
    B: Backend,
    S::DType: BasicOps<B> + Numeric<B> + IntoFloatTensor<B, 2>,
{
    pub fn new(
        model: M,
        space: S,
        state_space: SS,
        sampler: Metropolis<P>,
        sampler_state: SamplerState<B, S::DType>,
        n_samples_per_chain: usize,
    ) -> Self {
        assert!(n_samples_per_chain > 0);

        let dims = sampler_state.chains.dims();
        let device = sampler_state.chains.device();
        let samples = Tensor::<B, 2, S::DType>::zeros(
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
        S::DType: burn::tensor::Numeric<B, Elem = S::Scalar>,
        S::Scalar: Clone + Element,
    {
        let device = Default::default();
        let sampler_state = SamplerState::<B, S::DType>::from_space(&space, n_chains, &device);
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
        S::DType: BasicOps<B, Elem = S::Scalar> + Numeric<B> + Ordered<B> + IntoFloatTensor<B, 2>,
        S::Scalar: Clone + Element,
        P: Proposal<B, S>,
        M: Model<B>,
        SS: StateSpace,
    {
        let sweep_size = self.space.sample_size();
        let n_chains = self.sampler_state.chains.dims()[0];
        let device = self.sampler_state.chains.device();
        let sampler = &self.sampler;
        // Build the private bridge from the model and state space.
        let log_density =
            StateLogDensity::new(&self.model, &self.state_space, self.model.param_dtype());

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
    pub fn log_value(&self) -> FloatTensor<B, 1>
    where
        M: Model<B>,
    {
        let samples = into_model_tensor(self.samples.clone(), self.model.param_dtype());
        self.model.log_value(samples)
    }

    /// Evaluate the model on arbitrary configurations.
    pub fn log_value_on(&self, samples: Tensor<B, 2, S::DType>) -> FloatTensor<B, 1>
    where
        M: Model<B>,
    {
        let samples = into_model_tensor(samples, self.model.param_dtype());
        self.model.log_value(samples)
    }
}

impl<'a, M, S, B, SS> LogDensity<B, S> for StateLogDensity<'a, M, SS>
where
    S: Space,
    B: Backend,
    S::DType: Numeric<B> + IntoFloatTensor<B, 2>,
    M: Model<B>,
    SS: StateSpace,
{
    fn log_density(&self, _space: &S, samples: Tensor<B, 2, S::DType>) -> FloatTensor<B, 1> {
        let samples = into_model_tensor(samples, self.param_dtype);
        self.state_space.log_density(self.model.log_value(samples))
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
        let state: VariationalState<_, _, Flex, _, Simplex> =
            VariationalState::from_space(ZeroModel, space, Simplex, sampler, 1, 2);

        let values = state.log_value();

        assert_eq!(values.dims(), [2]);
    }

    #[test]
    fn log_value_on_uses_arbitrary_samples() {
        let space = HomogeneousSpace::new(Spin::half_integer(1), 1);
        let sampler = Metropolis::new(LocalProposal);
        let state: VariationalState<_, _, Flex, _, Simplex> =
            VariationalState::from_space(ZeroModel, space, Simplex, sampler, 1, 2);
        let samples = Tensor::<Flex, 2, Int>::from_data([[1], [0]], &Default::default());

        let values = state.log_value_on(samples);

        assert_eq!(values.dims(), [2]);
    }
}
