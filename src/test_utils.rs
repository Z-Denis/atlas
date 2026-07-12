use crate::model::Model;
use crate::sampler::LogDensity;
use crate::space::Space;
use crate::utils::{FloatTensor, IntTensor};
use burn::tensor::{BasicOps, Tensor, TensorCreationOptions, backend::Backend};
use burn_backend::tensor::{Float, TensorKind};

fn zero_log_values<B, K>(samples: &Tensor<B, 2, K>) -> FloatTensor<B, 1>
where
    B: Backend,
    K: TensorKind<B> + BasicOps<B>,
{
    FloatTensor::<B, 1>::zeros(
        [samples.dims()[0]],
        TensorCreationOptions::<B>::new(samples.device()),
    )
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ZeroModel;

impl<B> Model<B> for ZeroModel
where
    B: Backend,
    Float: TensorKind<B> + BasicOps<B>,
{
    fn log_value(&self, samples: FloatTensor<B, 2>) -> FloatTensor<B, 1> {
        zero_log_values(&samples)
    }
}

impl<S, B> LogDensity<B, S> for ZeroModel
where
    S: Space,
    B: Backend,
    S::DType: TensorKind<B> + BasicOps<B>,
{
    fn log_density(&self, _space: &S, samples: Tensor<B, 2, S::DType>) -> FloatTensor<B, 1> {
        zero_log_values(&samples)
    }
}

pub(crate) fn ints<const D: usize, B>(tensor: IntTensor<B, D>) -> Vec<i32>
where
    B: Backend,
{
    tensor.into_data().to_vec::<i32>().unwrap()
}
