use crate::model::Model;
use crate::sampler::LogDensity;
use crate::space::Space;
use burn::tensor::{BasicOps, Tensor, TensorCreationOptions, backend::Backend};
use burn_backend::tensor::TensorKind;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ZeroModel;

impl<S, B> Model<S, B> for ZeroModel
where
    S: Space,
    B: Backend,
    S::DType: TensorKind<B> + BasicOps<B>,
{
    type ParamDType = burn::tensor::Float;

    fn log_value(&self, _space: &S, samples: Tensor<B, 2, S::DType>) -> Tensor<B, 1> {
        Tensor::<B, 1>::zeros(
            [samples.dims()[0]],
            TensorCreationOptions::<B>::new(samples.device()),
        )
    }
}

pub(crate) fn ints<const D: usize, B>(tensor: Tensor<B, D, burn::tensor::Int>) -> Vec<i32>
where
    B: Backend,
{
    tensor.into_data().to_vec::<i32>().unwrap()
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ZeroLogDensity;

impl<S, B> LogDensity<B, S> for ZeroLogDensity
where
    S: Space,
    B: Backend,
    S::DType: TensorKind<B> + BasicOps<B>,
{
    fn log_density(&self, _space: &S, samples: Tensor<B, 2, S::DType>) -> Tensor<B, 1> {
        Tensor::<B, 1>::zeros(
            [samples.dims()[0]],
            TensorCreationOptions::<B>::new(samples.device()),
        )
    }
}
