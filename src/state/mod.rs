use burn::tensor::{BasicOps, Numeric, Tensor, TensorCreationOptions, backend::Backend};
use burn_backend::Element;
use burn_backend::tensor::Ordered;

use crate::model::Model;
use crate::operator::Operator;
use crate::sampler::{LogDensity, Metropolis, Proposal, SamplerState};
use crate::space::{RandomState, Samples, Space};
use crate::utils::{ComplexTensor, FloatTensor};
use burn::tensor::FloatDType;
use burn_backend::tensor::{Float, TensorKind};

#[doc(hidden)]
pub trait IntoFloatTensor<B: Backend, const D: usize>: TensorKind<B> {
    fn into_float(tensor: Tensor<B, D, Self>, dtype: FloatDType) -> FloatTensor<B, D>;
}

#[doc(hidden)]
pub trait StateDType<B: Backend>: BasicOps<B> + Numeric<B> + IntoFloatTensor<B, 2> {}

impl<B, T> StateDType<B> for T
where
    B: Backend,
    T: BasicOps<B> + Numeric<B> + IntoFloatTensor<B, 2>,
{
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

#[doc(hidden)]
#[doc(hidden)]
pub trait LocalValue<B: Backend>: Sized {
    type Connected;

    fn reshape_connected(flat: Self, batch: usize, n_conns: usize) -> Self::Connected;

    fn local_value(
        current: Self,
        connected: Self::Connected,
        mels: FloatTensor<B, 2>,
    ) -> FloatTensor<B, 1>;
}

impl<B: Backend> LocalValue<B> for FloatTensor<B, 1> {
    type Connected = FloatTensor<B, 2>;

    fn reshape_connected(flat: Self, batch: usize, n_conns: usize) -> Self::Connected {
        flat.reshape([batch, n_conns])
    }

    fn local_value(
        current: Self,
        connected: Self::Connected,
        mels: FloatTensor<B, 2>,
    ) -> FloatTensor<B, 1> {
        (mels * (connected - current.unsqueeze_dim(1)).exp())
            .sum_dim(1)
            .squeeze_dim::<1>(1)
    }
}

impl<B: Backend> LocalValue<B> for ComplexTensor<B, 1> {
    type Connected = ComplexTensor<B, 2>;

    fn reshape_connected(flat: Self, batch: usize, n_conns: usize) -> Self::Connected {
        ComplexTensor::new(
            flat.real().reshape([batch, n_conns]),
            flat.imag().reshape([batch, n_conns]),
        )
    }

    fn local_value(
        current: Self,
        connected: Self::Connected,
        mels: FloatTensor<B, 2>,
    ) -> FloatTensor<B, 1> {
        let ratio = ComplexTensor::new(
            connected.real() - current.real().unsqueeze_dim(1),
            connected.imag() - current.imag().unsqueeze_dim(1),
        )
        .exp();
        ComplexTensor::new(ratio.real() * mels.clone(), ratio.imag() * mels)
            .real()
            .sum_dim(1)
            .squeeze_dim::<1>(1)
    }
}

/// Marker for the ambient state space interpreted by a variational state.
///
/// The state space decides how a model's `log_value` is exposed to the
/// sampler as a `log_density`.
pub trait StateSpace<B: Backend, Out> {
    fn log_density(&self, log_value: Out) -> FloatTensor<B, 1>;
}

/// Simplex state space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Simplex;

impl<B: Backend> StateSpace<B, FloatTensor<B, 1>> for Simplex {
    fn log_density(&self, log_value: FloatTensor<B, 1>) -> FloatTensor<B, 1> {
        log_value
    }
}

/// Hilbert state space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hilbert;

impl<B: Backend> StateSpace<B, FloatTensor<B, 1>> for Hilbert {
    fn log_density(&self, log_value: FloatTensor<B, 1>) -> FloatTensor<B, 1> {
        log_value.mul_scalar(2)
    }
}

impl<B: Backend> StateSpace<B, ComplexTensor<B, 1>> for Hilbert {
    fn log_density(&self, log_value: ComplexTensor<B, 1>) -> FloatTensor<B, 1> {
        log_value.real().mul_scalar(2)
    }
}

/// Private adapter used by `VariationalState::sample()` to expose a model as
/// a sampler-facing density.
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
/// A variational state owns the model, space, state space, sampler, chain
/// state, and collected samples. It is the natural checkpoint boundary.
pub struct VariationalState<M, S: Space, B: Backend, P, SS = Simplex>
where
    M: Model<B>,
    S::DType: StateDType<B>,
{
    pub model: M,
    pub space: S,
    pub state_space: SS,
    pub sampler: Metropolis<P>,
    pub sampler_state: SamplerState<B, S::DType>,
    pub samples: Samples<B, 2, S::DType>,
    n_samples_per_chain: usize,
}

impl<M, S: Space, B: Backend, P, SS> VariationalState<M, S, B, P, SS>
where
    M: Model<B>,
    S::DType: StateDType<B>,
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
        S::DType: StateDType<B> + burn::tensor::Numeric<B, Elem = S::Scalar>,
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
        S::DType: StateDType<B> + BasicOps<B, Elem = S::Scalar> + Ordered<B>,
        S::Scalar: Clone + Element,
        P: Proposal<B, S>,
        SS: StateSpace<B, M::Output>,
    {
        let sweep_size = self.space.sample_size();
        let n_chains = self.sampler_state.chains.dims()[0];
        let device = self.sampler_state.chains.device();
        let sampler = &self.sampler;
        // Build the private bridge from the model and state space.
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
    pub fn log_value(&self) -> M::Output {
        let samples = into_model_tensor(self.samples.clone(), self.model.param_dtype());
        self.model.log_value(samples)
    }

    /// Evaluate the model on arbitrary configurations.
    pub fn log_value_on(&self, samples: Tensor<B, 2, S::DType>) -> M::Output {
        let samples = into_model_tensor(samples, self.model.param_dtype());
        self.model.log_value(samples)
    }

    /// Expectation value of an observable over the collected samples.
    pub fn expect<O>(&self, operator: &O) -> FloatTensor<B, 1>
    where
        O: Operator<B, S>,
        M::Output: LocalValue<B>,
    {
        let (conns, mels) = operator.get_conns_and_mels(&self.space, self.samples.clone());
        let batch = conns.dims()[0];
        let n_conns = conns.dims()[1];
        if n_conns == 0 {
            return FloatTensor::<B, 1>::zeros(
                [1],
                TensorCreationOptions::<B>::new(self.samples.device()),
            );
        }

        let sample_size = conns.dims()[2];
        let current = self.model.log_value(into_model_tensor(
            self.samples.clone(),
            self.model.param_dtype(),
        ));
        let connected = self.model.log_value(into_model_tensor(
            conns.reshape([batch * n_conns, sample_size]),
            self.model.param_dtype(),
        ));
        let connected = <M::Output as LocalValue<B>>::reshape_connected(connected, batch, n_conns);

        <M::Output as LocalValue<B>>::local_value(current, connected, mels).mean_dim(0)
    }
}

impl<'a, M, S, B: Backend, SS> LogDensity<B, S> for StateLogDensity<'a, M, SS>
where
    M: Model<B>,
    S: Space,
    S::DType: StateDType<B>,
    SS: StateSpace<B, M::Output>,
{
    fn log_density(&self, _space: &S, samples: Tensor<B, 2, S::DType>) -> FloatTensor<B, 1> {
        let samples = into_model_tensor(samples, self.model.param_dtype());
        self.state_space.log_density(self.model.log_value(samples))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Hilbert;
    use crate::IntTensor;
    use crate::model::Rbm;
    use crate::operator::{Magnetization, Operator, TransverseFieldIsing};
    use crate::sampler::LocalProposal;
    use crate::space::ContinuousSpace;
    use crate::space::HomogeneousSpace;
    use crate::space::Spin;
    use crate::space::SpinSpace;
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

    #[test]
    fn hilbert_accepts_real_log_value() {
        let space = HomogeneousSpace::new(Spin::half_integer(1), 1);
        let sampler = Metropolis::new(LocalProposal);
        let state: VariationalState<_, _, Flex, _, Hilbert> =
            VariationalState::from_space(ZeroModel, space, Hilbert, sampler, 1, 2);

        let values = state.log_value();

        assert_eq!(values.dims(), [2]);
    }

    #[test]
    fn expect_uses_observable_local_values() {
        struct FirstColumnOperator;

        impl<B, S> Operator<B, S> for FirstColumnOperator
        where
            B: burn::tensor::backend::Backend,
            S: Space<DType = burn::tensor::Float>,
        {
            fn get_conns_and_mels(
                &self,
                _space: &S,
                samples: Tensor<B, 2, S::DType>,
            ) -> (Tensor<B, 3, S::DType>, FloatTensor<B, 2>) {
                let device = Default::default();
                let mels = samples
                    .clone()
                    .select(1, Tensor::<B, 1, Int>::zeros([1], &device));
                (samples.unsqueeze_dim(1), mels)
            }
        }

        let space = ContinuousSpace::new(-1.0f32, 1.0, 2);
        let sampler = Metropolis::new(LocalProposal);
        let mut state: VariationalState<_, _, Flex, _, Simplex> =
            VariationalState::from_space(ZeroModel, space, Simplex, sampler, 1, 2);
        state.samples =
            FloatTensor::<Flex, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &Default::default());

        let value = state.expect(&FirstColumnOperator);

        assert_eq!(value.dims(), [1]);
        assert!((value.into_data().to_vec::<f32>().unwrap()[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn spin_magnetization_expectation_is_zero_for_symmetric_samples() {
        let device = Default::default();
        let space: SpinSpace = HomogeneousSpace::new(Spin::half_integer(1), 1);
        let sampler = Metropolis::new(LocalProposal);
        let model = Rbm::<Flex>::zero(1, 1, None, &device);
        let mut state: VariationalState<_, _, Flex, _, Simplex> =
            VariationalState::from_space(model, space, Simplex, sampler, 1, 2);
        state.samples = IntTensor::<Flex, 2>::from_data([[-1], [1]], &device);

        let value = state.expect(&Magnetization);

        assert_eq!(value.dims(), [1]);
        assert!((value.into_data().to_vec::<f32>().unwrap()[0]).abs() < 1e-6);
    }

    #[test]
    fn transverse_field_expectation_counts_flips() {
        let space: SpinSpace = HomogeneousSpace::new(Spin::half_integer(1), 2);
        let sampler = Metropolis::new(LocalProposal);
        let mut state: VariationalState<_, _, Flex, _, Simplex> =
            VariationalState::from_space(ZeroModel, space, Simplex, sampler, 1, 2);
        state.samples = IntTensor::<Flex, 2>::from_data([[-1, 1], [1, -1]], &Default::default());

        let value = state.expect(&TransverseFieldIsing::new(0.0, 1.0));

        assert_eq!(value.dims(), [1]);
        assert!((value.into_data().to_vec::<f32>().unwrap()[0] + 2.0).abs() < 1e-6);
    }

    #[test]
    fn transverse_field_expectation_matches_two_site_reference() {
        let space: SpinSpace = HomogeneousSpace::new(Spin::half_integer(1), 2);
        let sampler = Metropolis::new(LocalProposal);
        let mut state: VariationalState<_, _, Flex, _, Simplex> =
            VariationalState::from_space(ZeroModel, space, Simplex, sampler, 1, 2);
        state.samples = IntTensor::<Flex, 2>::from_data([[1, -1]], &Default::default());

        let value = state.expect(&TransverseFieldIsing::new(0.0, 0.5));

        assert_eq!(value.dims(), [1]);
        assert!((value.into_data().to_vec::<f32>().unwrap()[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_bias_rbm_has_zero_magnetization_on_symmetric_samples() {
        let device = Default::default();
        let space: SpinSpace = HomogeneousSpace::new(Spin::half_integer(1), 1);
        let sampler = Metropolis::new(LocalProposal);
        let model = Rbm::<Flex>::new(1, 1, None, &device);
        let mut state: VariationalState<_, _, Flex, _, Simplex> =
            VariationalState::from_space(model, space, Simplex, sampler, 1, 2);
        state.samples = IntTensor::<Flex, 2>::from_data([[-1], [1]], &device);

        let value = state.expect(&Magnetization);

        assert_eq!(value.dims(), [1]);
        assert!((value.into_data().to_vec::<f32>().unwrap()[0]).abs() < 1e-6);
    }
}
