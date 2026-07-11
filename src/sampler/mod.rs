use burn::tensor::{BasicOps, Bool, Distribution, IndexingUpdateOp, Int, Tensor, backend::Backend};
use burn_backend::Element;
use burn_backend::tensor::{Ordered, TensorKind};
use num_traits::ToPrimitive;

use crate::space::{HomogeneousProductSpace, RandomState, Samples, Space};
use crate::utils::{FloatTensor, IntTensor};
use crate::utils::{chain_indices, float_opts, int_opts, randint};

type ChainStats<B> = IntTensor<B, 1>;
type LogProb<B> = FloatTensor<B, 1>;

fn chain_update_indices<B: Backend>(
    n_chains: usize,
    local_indices: IntTensor<B, 1>,
    device: &B::Device,
) -> IntTensor<B, 2> {
    Tensor::cat(
        vec![
            chain_indices::<B>(n_chains, device).unsqueeze_dim(1),
            local_indices.unsqueeze_dim(1),
        ],
        1,
    )
}

/// Draw from `0..local_size-1` while skipping the current state index.
fn skip_index<B: Backend>(choice: IntTensor<B, 1>, forbidden: IntTensor<B, 1>) -> IntTensor<B, 1> {
    let forbidden = forbidden.cast(choice.dtype());
    choice.clone().mask_where(
        choice.clone().greater_equal(forbidden),
        choice.add_scalar(1),
    )
}

fn reject_outside_domain<B: Backend, S>(
    space: &S,
    samples: &Tensor<B, 2, S::DType>,
) -> Tensor<B, 1, Bool>
where
    S: Space,
    S::DType: TensorKind<B>,
    S::DType: BasicOps<B, Elem = S::Scalar> + Ordered<B>,
    S::Scalar: Clone + Element,
{
    let valid = space.contains(samples.clone());
    let n_chains = valid.dims()[0];
    valid.reshape([n_chains])
}

pub trait Proposal<B: Backend, S: Space>
where
    S::DType: TensorKind<B>,
{
    /// Propose one updated configuration per chain.
    fn propose(&self, space: &S, samples: Tensor<B, 2, S::DType>) -> Tensor<B, 2, S::DType>;
}

pub trait LogDensity<B: Backend, S: Space>
where
    S::DType: TensorKind<B>,
{
    /// Evaluate the log density for each chain.
    fn log_density(&self, space: &S, samples: Tensor<B, 2, S::DType>) -> LogProb<B>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocalProposal;

impl<S, B> Proposal<B, S> for LocalProposal
where
    S: HomogeneousProductSpace,
    S::Scalar: Clone + PartialEq + Element,
    B: Backend,
    S::DType: TensorKind<B>,
    S::DType: BasicOps<B, Elem = S::Scalar>,
{
    fn propose(&self, space: &S, samples: Tensor<B, 2, S::DType>) -> Tensor<B, 2, S::DType> {
        let sample_size = space.sample_size();
        let local_size = space.local_size();
        if sample_size == 0 || local_size <= 1 {
            return samples;
        }

        let device = samples.device();
        let n_chains = samples.dims()[0];
        let local_indices = randint::<B, 1>([n_chains], 0, sample_size as i64, &device);
        let indices = chain_update_indices::<B>(n_chains, local_indices, &device);

        let current = samples.clone().gather_nd::<2, 1>(indices.clone());
        let current_local_indices = space.indices_of(current);

        let proposal_local_indices =
            randint::<B, 1>([n_chains], 0, (local_size - 1) as i64, &device);
        let proposal_values = space.states_at(skip_index::<B>(
            proposal_local_indices,
            current_local_indices,
        ));

        samples.scatter_nd::<2, 1>(indices, proposal_values, IndexingUpdateOp::Assign)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianProposal<T> {
    sigma: T,
}

impl<T> GaussianProposal<T> {
    pub fn new(sigma: T) -> Self {
        Self { sigma }
    }
}

impl<S, B, T> Proposal<B, S> for GaussianProposal<T>
where
    S: Space<Scalar = T, DType = burn::tensor::Float>,
    B: Backend,
    T: Element + ToPrimitive + num_traits::Float,
{
    fn propose(&self, space: &S, samples: Tensor<B, 2, S::DType>) -> Tensor<B, 2, S::DType> {
        let device = samples.device();
        let n_chains = samples.dims()[0];
        let noise = Tensor::<B, 2, S::DType>::random(
            [n_chains, space.sample_size()],
            Distribution::Normal(0.0, num_traits::ToPrimitive::to_f64(&self.sigma).unwrap()),
            float_opts::<B>(&device),
        );
        samples.add(noise)
    }
}

#[derive(Clone, Debug)]
pub struct SamplerState<B, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    pub chains: Samples<B, 2, K>,
    pub accepted: ChainStats<B>,
    pub proposed: ChainStats<B>,
}

impl<B, K> SamplerState<B, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    pub fn new(chains: Samples<B, 2, K>) -> Self {
        let n_chains = chains.dims()[0];
        let device = chains.device();
        Self {
            chains,
            accepted: Tensor::<B, 1, Int>::zeros([n_chains], int_opts::<B>(&device)),
            proposed: Tensor::<B, 1, Int>::zeros([n_chains], int_opts::<B>(&device)),
        }
    }

    /// Seed the sampler state from a space-specific random chain state.
    pub fn from_space<S>(
        space: &S,
        n_chains: usize,
        device: &B::Device,
    ) -> SamplerState<B, S::DType>
    where
        S: RandomState,
        S::DType: TensorKind<B>,
        S::DType: burn::tensor::Numeric<B, Elem = S::Scalar>,
        S::Scalar: Clone + Element,
    {
        SamplerState::new(space.random_state::<B>(n_chains, device))
    }
}

#[derive(Clone, Debug)]
pub struct Metropolis<P> {
    proposal: P,
}

impl<P> Metropolis<P> {
    pub fn new(proposal: P) -> Self {
        Self { proposal }
    }
}

impl<P> Metropolis<P> {
    pub fn step<S, F, B>(&self, space: &S, log_density: &F, state: &mut SamplerState<B, S::DType>)
    where
        S: Space,
        B: Backend,
        S::DType: TensorKind<B>,
        S::DType: BasicOps<B, Elem = S::Scalar> + Ordered<B>,
        S::Scalar: Clone + Element,
        P: Proposal<B, S>,
        F: LogDensity<B, S>,
    {
        let device = state.chains.device();
        let n_chains = state.chains.dims()[0];

        let current = state.chains.clone();
        let proposal = self.proposal.propose(space, current.clone());
        let valid = reject_outside_domain(space, &proposal);

        let logp_current = log_density.log_density(space, current.clone());
        let logp_proposal = log_density.log_density(space, proposal.clone());

        let log_uniform = Tensor::<B, 1>::random(
            [n_chains],
            Distribution::Uniform(0.0, 1.0),
            float_opts::<B>(&device),
        )
        .log();

        let accept = log_uniform.lower(logp_proposal - logp_current);
        let accept = accept.bool_and(valid);
        let accept_mask = accept.clone().unsqueeze_dim::<2>(1).expand(current.shape());

        state.chains = current.mask_where(accept_mask, proposal);
        state.accepted = state.accepted.clone() + accept.int().cast(state.accepted.dtype());
        state.proposed =
            state.proposed.clone() + Tensor::<B, 1, Int>::ones([n_chains], int_opts::<B>(&device));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::{
        ContinuousSpace, HomogeneousProductSpace, HomogeneousSpace, Spin, ViewSpace,
    };
    use crate::test_utils::{ZeroModel, ints};
    use crate::{FloatTensor, IntTensor};
    use crate::{Simplex, VariationalState};
    use burn::backend::Flex;
    use burn::tensor::Float;
    use burn::tensor::backend::BackendTypes;

    #[test]
    fn sampler_state_tracks_chain_state() {
        let device: <Flex as BackendTypes>::Device = Default::default();
        let samples = IntTensor::<Flex, 2>::from_data([[0], [1]], &device);
        let state = SamplerState::new(samples);
        assert_eq!(state.chains.dims(), [2, 1]);
        assert_eq!(state.accepted.dims(), [2]);
        assert_eq!(state.proposed.dims(), [2]);
    }

    #[test]
    fn metropolis_updates_chain_state() {
        let space = HomogeneousSpace::new(Spin::half_integer(1), 1);
        let sampler = Metropolis::new(LocalProposal);
        let device: <Flex as BackendTypes>::Device = Default::default();
        let mut state: SamplerState<Flex, Int> =
            SamplerState::<Flex, Int>::from_space(&space, 1, &device);
        let before = ints(state.chains.clone());
        sampler.clone().step(&space, &ZeroModel, &mut state);

        let data = ints(state.chains.clone());
        assert_ne!(data, before);
        assert!(space.contains(state.chains.clone()).all().into_scalar());
        assert_eq!(ints(state.accepted.clone()), vec![1]);
        assert_eq!(ints(state.proposed.clone()), vec![1]);
    }

    #[test]
    fn variational_state_collects_batches() {
        let space = HomogeneousSpace::new(Spin::half_integer(1), 1);
        let sampler = Metropolis::new(LocalProposal);
        let mut state: VariationalState<_, _, Flex, _, Simplex> =
            VariationalState::from_space(ZeroModel, space, Simplex, sampler, 1, 2);

        state.sample();

        assert_eq!(state.samples.dims(), [2, 1]);
        assert_eq!(ints(state.sampler_state.accepted.clone()), vec![2]);
        assert_eq!(ints(state.sampler_state.proposed.clone()), vec![2]);
    }

    #[test]
    fn metropolis_sample_density_is_uniform() {
        let n_chains = 4;
        let n_samples_per_chain = 4;
        let space = HomogeneousSpace::new(Spin::half_integer(1), 1);
        let sampler = Metropolis::new(LocalProposal);
        let mut state: VariationalState<_, _, Flex, _, Simplex> = VariationalState::from_space(
            ZeroModel,
            space,
            Simplex,
            sampler,
            n_chains,
            n_samples_per_chain,
        );

        state.sample();

        let values = ints(state.samples.clone());
        let local_states = state.space.local_states().to_vec();
        let mut counts = vec![0usize; local_states.len()];

        for value in values {
            let idx = local_states
                .iter()
                .position(|state| *state == value)
                .unwrap();
            counts[idx] += 1;
        }

        let total = counts.iter().sum::<usize>() as f64;
        let density: Vec<f64> = counts.iter().map(|&count| count as f64 / total).collect();
        println!(
            "sample density: states={:?}, counts={:?}, density={:?}",
            local_states, counts, density
        );

        assert_eq!(counts, vec![8, 8]);
        assert_eq!(
            ints(state.sampler_state.accepted.clone()),
            vec![n_samples_per_chain as i32; n_chains]
        );
        assert_eq!(
            ints(state.sampler_state.proposed.clone()),
            vec![n_samples_per_chain as i32; n_chains]
        );
    }

    #[test]
    fn gaussian_proposal_updates_continuous_state() {
        let local = ContinuousSpace::new(f32::NEG_INFINITY, f32::INFINITY, 2);
        let space = HomogeneousSpace::new(local, 3);
        let sampler = Metropolis::new(GaussianProposal::new(0.1f32));
        let device: <Flex as BackendTypes>::Device = Default::default();
        let mut state: SamplerState<Flex, Float> =
            SamplerState::new(space.random_state(1, &device));
        let before = state.chains.clone();
        sampler.clone().step(&space, &ZeroModel, &mut state);

        assert_eq!(state.chains.dims(), before.dims());
        assert_ne!(
            state.chains.into_data().to_vec::<f32>().unwrap(),
            before.into_data().to_vec::<f32>().unwrap()
        );
        assert_eq!(ints(state.accepted.clone()), vec![1]);
        assert_eq!(ints(state.proposed.clone()), vec![1]);
    }

    #[test]
    fn local_spaces_integrate_with_sampler() {
        let device: <Flex as BackendTypes>::Device = Default::default();

        let continuous = ContinuousSpace::new(-1.0f32, 1.0, 2);
        assert_eq!(continuous.sample_size(), 2);
        let continuous_sample = FloatTensor::<Flex, 2>::from_data([[0.0f32, 1.0]], &device);
        assert!(continuous.contains(continuous_sample).all().into_scalar());
        assert_eq!(continuous.view(&[0.0f32, 1.0]).particle(0), &[0.0, 1.0]);
        assert_eq!(continuous.random_state::<Flex>(4, &device).dims(), [4, 2]);

        let mut continuous_state: SamplerState<Flex, burn::tensor::Float> =
            SamplerState::<Flex, burn::tensor::Float>::from_space(
                &HomogeneousSpace::new(continuous, 1),
                2,
                &device,
            );
        let continuous_sampler = Metropolis::new(GaussianProposal::new(0.1f32));
        continuous_sampler.clone().step(
            &HomogeneousSpace::new(ContinuousSpace::new(-1.0f32, 1.0, 2), 1),
            &ZeroModel,
            &mut continuous_state,
        );
        assert_eq!(continuous_state.chains.dims(), [2, 2]);

        let spin = Spin::half_integer(1);
        assert_eq!(spin.sample_size(), 1);
        let spin_sample = IntTensor::<Flex, 2>::from_data([[1i32]], &device);
        assert!(spin.contains(spin_sample).all().into_scalar());
        assert_eq!(spin.view(&[1i32]), &[1]);
        assert_eq!(spin.random_state::<Flex>(4, &device).dims(), [4, 1]);

        let mut spin_state: SamplerState<Flex, burn::tensor::Int> =
            SamplerState::<Flex, burn::tensor::Int>::from_space(
                &HomogeneousSpace::new(Spin::half_integer(1), 1),
                2,
                &device,
            );
        let spin_sampler = Metropolis::new(LocalProposal);
        spin_sampler.clone().step(
            &HomogeneousSpace::new(Spin::half_integer(1), 1),
            &ZeroModel,
            &mut spin_state,
        );
        assert_eq!(spin_state.chains.dims(), [2, 1]);
    }

    #[derive(Clone, Copy, Debug)]
    struct BadProposal;

    impl Proposal<Flex, HomogeneousSpace<Spin>> for BadProposal {
        fn propose(
            &self,
            _space: &HomogeneousSpace<Spin>,
            samples: IntTensor<Flex, 2>,
        ) -> IntTensor<Flex, 2> {
            IntTensor::<Flex, 2>::from_data([[0i32]], &samples.device())
        }
    }

    #[test]
    fn rejects_invalid_proposal() {
        let space = HomogeneousSpace::new(Spin::half_integer(1), 1);
        let sampler = Metropolis::new(BadProposal);
        let device: <Flex as BackendTypes>::Device = Default::default();
        let mut state: SamplerState<Flex, Int> =
            SamplerState::<Flex, Int>::from_space(&space, 1, &device);
        let before = state.chains.clone();
        sampler.clone().step(&space, &ZeroModel, &mut state);

        assert_eq!(ints(state.chains.clone()), ints(before));
        assert_eq!(ints(state.accepted.clone()), vec![0]);
        assert_eq!(ints(state.proposed.clone()), vec![1]);
    }
}
