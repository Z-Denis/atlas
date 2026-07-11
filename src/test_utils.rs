use crate::model::Model;
use crate::space::Space;
use burn::tensor::{Numeric, Tensor, TensorCreationOptions, backend::Backend};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ZeroModel;

impl<S, B> Model<S, B> for ZeroModel
where
    S: Space,
    B: Backend,
{
    fn log_value<K>(&self, _space: &S, samples: Tensor<B, 2, K>) -> Tensor<B, 1>
    where
        K: Numeric<B>,
    {
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

pub(crate) fn zero_log_density<S, B, K>(space: &S, samples: Tensor<B, 2, K>) -> Tensor<B, 1>
where
    S: Space,
    B: Backend,
    K: Numeric<B>,
{
    ZeroModel.log_value(space, samples)
}
