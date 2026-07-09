use burn::tensor::{
    BasicOps, Distribution, IndexingUpdateOp, Int, Tensor, TensorCreationOptions, backend::Backend,
};
use burn_backend::Element;

use crate::space::{HomogeneousProductSpace, Samples, Space};
use crate::utils::{chain_indices, float_opts, int_opts, randint};

type SampleTensor<B, K> = Tensor<B, 2, K>;
type ChainStats<B> = Tensor<B, 1, Int>;
type LogProb<B> = Tensor<B, 1>;

fn chain_update_indices<B: Backend>(
    n_chains: usize,
    local_indices: Tensor<B, 1, Int>,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::cat(
        vec![
            chain_indices::<B>(n_chains, device).unsqueeze_dim(1),
            local_indices.unsqueeze_dim(1),
        ],
        1,
    )
}

/// Draw from `0..local_size-1` while skipping the current state index.
fn skip_index<B: Backend>(
    choice: Tensor<B, 1, Int>,
    forbidden: Tensor<B, 1, Int>,
) -> Tensor<B, 1, Int> {
    let forbidden = forbidden.cast(choice.dtype());
    choice.clone().mask_where(
        choice.clone().greater_equal(forbidden),
        choice.add_scalar(1),
    )
}

pub trait Proposal<B: Backend, S, K>
where
    K: BasicOps<B>,
{
    /// Propose one updated configuration per chain.
    fn propose(&self, space: &S, samples: SampleTensor<B, K>) -> SampleTensor<B, K>;
}

pub trait LogDensityBatch<B: Backend, S, K>
where
    K: BasicOps<B>,
{
    /// Evaluate the log density for each chain.
    fn log_density_batch(&self, space: &S, samples: SampleTensor<B, K>) -> LogProb<B>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocalProposal;

impl<S, B, K> Proposal<B, S, K> for LocalProposal
where
    S: HomogeneousProductSpace,
    S::Scalar: Clone + PartialEq + Element,
    B: Backend,
    K: BasicOps<B, Elem = S::Scalar>,
{
    fn propose(&self, space: &S, samples: SampleTensor<B, K>) -> SampleTensor<B, K> {
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
    pub fn from_space<S>(space: &S, n_chains: usize, device: &B::Device) -> Self
    where
        S: HomogeneousProductSpace,
        K: BasicOps<B, Elem = S::Scalar>,
        S::Scalar: Clone + Element,
    {
        Self::new(space.random_state::<B, K>(n_chains, device))
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
    pub fn step<S, M, B, K>(&mut self, space: &S, model: &M, state: &mut SamplerState<B, K>)
    where
        S: Space,
        B: Backend,
        K: BasicOps<B, Elem = S::Scalar>,
        S::Scalar: Clone + Element,
        P: Proposal<B, S, K>,
        M: LogDensityBatch<B, S, K>,
    {
        let device = state.chains.device();
        let n_chains = state.chains.dims()[0];

        let current = state.chains.clone();
        let proposal = self.proposal.propose(space, current.clone());
        let valid = space.contains(proposal.clone()).all().into_data();
        assert!(
            valid.to_vec::<bool>().unwrap()[0],
            "proposal left the space"
        );

        let logp_current = model.log_density_batch(space, current.clone());
        let logp_proposal = model.log_density_batch(space, proposal.clone());

        let log_uniform = Tensor::<B, 1>::random(
            [n_chains],
            Distribution::Uniform(0.0, 1.0),
            float_opts::<B>(&device),
        )
        .log();

        let accept = log_uniform.lower(logp_proposal - logp_current);
        let accept_mask = accept.clone().unsqueeze_dim::<2>(1).expand(current.shape());

        state.chains = current.mask_where(accept_mask, proposal);
        state.accepted = state.accepted.clone() + accept.int().cast(state.accepted.dtype());
        state.proposed =
            state.proposed.clone() + Tensor::<B, 1, Int>::ones([n_chains], int_opts::<B>(&device));
    }
}

pub struct VariationalState<M, S: Space, B, K, P>
where
    B: Backend,
    K: BasicOps<B>,
{
    pub model: M,
    pub space: S,
    pub sampler: Metropolis<P>,
    pub sampler_state: SamplerState<B, K>,
    pub samples: Samples<B, 2, K>,
    n_samples_per_chain: usize,
}

impl<M, S: Space, B, K, P> VariationalState<M, S, B, K, P>
where
    B: Backend,
    K: BasicOps<B>,
{
    pub fn new(
        model: M,
        space: S,
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
        sampler: Metropolis<P>,
        n_chains: usize,
        n_samples_per_chain: usize,
    ) -> Self
    where
        S: HomogeneousProductSpace,
        K: BasicOps<B, Elem = S::Scalar>,
        S::Scalar: Clone + Element,
    {
        let device = Default::default();
        let sampler_state = SamplerState::from_space(&space, n_chains, &device);
        Self::new(model, space, sampler, sampler_state, n_samples_per_chain)
    }
}

impl<M, S, B, K, P> VariationalState<M, S, B, K, P>
where
    S: Space,
    B: Backend,
    K: BasicOps<B, Elem = S::Scalar>,
    S::Scalar: Clone + Element,
    P: Proposal<B, S, K>,
    M: LogDensityBatch<B, S, K>,
{
    pub fn sample(&mut self) {
        let sweep_size = self.space.sample_size();
        let n_chains = self.sampler_state.chains.dims()[0];
        let device = self.sampler_state.chains.device();

        for sample_idx in 0..self.n_samples_per_chain {
            for _ in 0..sweep_size {
                self.sampler
                    .step(&self.space, &self.model, &mut self.sampler_state);
            }

            let start = sample_idx * n_chains;
            let end = start + n_chains;
            self.samples = self.samples.clone().slice_assign(
                start..end,
                self.sampler_state.chains.clone().to_device(&device),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Layout;
    use crate::space::{HomogeneousProductSpace, HomogeneousSpace, Spin, SpinSpace};
    use burn::backend::Flex;
    use burn::tensor::backend::{Backend, BackendTypes};
    use burn::tensor::{Int, Tensor};

    fn ints<const D: usize>(tensor: Tensor<Flex, D, Int>) -> Vec<i32> {
        tensor.into_data().to_vec::<i32>().unwrap()
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Chain(usize);

    impl Layout for Chain {
        fn len(&self) -> usize {
            self.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct ZeroModel;

    impl<S, B, K> LogDensityBatch<B, S, K> for ZeroModel
    where
        B: Backend,
        K: BasicOps<B>,
    {
        fn log_density_batch(&self, _space: &S, samples: Tensor<B, 2, K>) -> Tensor<B, 1> {
            Tensor::<B, 1>::zeros(
                [samples.dims()[0]],
                TensorCreationOptions::<B>::float().with_device(samples.device()),
            )
        }
    }

    #[test]
    fn sampler_state_tracks_chain_state() {
        let device: <Flex as BackendTypes>::Device = Default::default();
        let samples: Samples<Flex, 2, Int> = Tensor::from_data([[0], [1]], &device);
        let state = SamplerState::new(samples);
        assert_eq!(state.chains.dims(), [2, 1]);
        assert_eq!(state.accepted.dims(), [2]);
        assert_eq!(state.proposed.dims(), [2]);
    }

    #[test]
    fn metropolis_updates_chain_state() {
        let space = SpinSpace::new(Chain(1), Spin::half_integer(1), vec![-1i32, 1]);
        let model = ZeroModel;
        let sampler = Metropolis::new(LocalProposal);
        let device: <Flex as BackendTypes>::Device = Default::default();
        let mut state: SamplerState<Flex, Int> = SamplerState::from_space(&space, 1, &device);
        let before = ints(state.chains.clone());

        sampler.clone().step(&space, &model, &mut state);

        let data = ints(state.chains.clone());
        assert_ne!(data, before);
        assert!(space.contains(state.chains.clone()).all().into_scalar());
        assert_eq!(ints(state.accepted.clone()), vec![1]);
        assert_eq!(ints(state.proposed.clone()), vec![1]);
    }

    #[test]
    fn variational_state_collects_batches() {
        let space = HomogeneousSpace::new(Chain(1), vec![-1i32, 1]);
        let model = ZeroModel;
        let sampler = Metropolis::new(LocalProposal);
        let mut state: VariationalState<_, _, Flex, Int, _> =
            VariationalState::from_space(model, space, sampler, 1, 2);

        state.sample();

        assert_eq!(state.samples.dims(), [2, 1]);
        assert_eq!(ints(state.sampler_state.accepted.clone()), vec![2]);
        assert_eq!(ints(state.sampler_state.proposed.clone()), vec![2]);
    }

    #[test]
    fn metropolis_sample_density_is_uniform() {
        let n_chains = 4;
        let n_samples_per_chain = 4;
        let space = SpinSpace::new(Chain(1), Spin::half_integer(1), vec![-1i32, 1]);
        let model = ZeroModel;
        let sampler = Metropolis::new(LocalProposal);
        let mut state: VariationalState<_, _, Flex, Int, _> =
            VariationalState::from_space(model, space, sampler, n_chains, n_samples_per_chain);

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

    #[derive(Clone, Copy, Debug)]
    struct BadProposal;

    impl Proposal<Flex, SpinSpace<Chain, i32>, Int> for BadProposal {
        fn propose(
            &self,
            _space: &SpinSpace<Chain, i32>,
            samples: Tensor<Flex, 2, Int>,
        ) -> Tensor<Flex, 2, Int> {
            Tensor::from_data([[0i32]], &samples.device())
        }
    }

    #[test]
    #[should_panic(expected = "proposal left the space")]
    fn rejects_invalid_proposal() {
        let space = SpinSpace::new(Chain(1), Spin::half_integer(1), vec![-1i32, 1]);
        let model = ZeroModel;
        let sampler = Metropolis::new(BadProposal);
        let device: <Flex as BackendTypes>::Device = Default::default();
        let mut state: SamplerState<Flex, Int> = SamplerState::from_space(&space, 1, &device);

        sampler.clone().step(&space, &model, &mut state);
    }
}
