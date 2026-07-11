use burn::module::{Module, Param};
use burn::tensor::{
    Float, FloatDType, Tensor, TensorCreationOptions, activation::softplus, backend::Backend,
};

use super::Model;
use crate::space::Space;

/// Minimal restricted Boltzmann machine.
///
/// This is a first concrete Burn module definition. It currently targets
/// floating-point configuration tensors and exposes an unnormalized log
/// value.
#[derive(Module, Debug)]
pub struct Rbm<B: Backend> {
    pub visible_bias: Param<Tensor<B, 1>>,
    pub hidden_bias: Param<Tensor<B, 1>>,
    pub weight: Param<Tensor<B, 2>>,
    visible_size: usize,
    hidden_size: usize,
}

impl<B: Backend> Rbm<B> {
    pub fn new(
        visible_size: usize,
        hidden_size: usize,
        param_dtype: Option<FloatDType>,
        device: &B::Device,
    ) -> Self {
        let param_dtype = param_dtype.unwrap_or(FloatDType::F32);
        let opts = TensorCreationOptions::<B>::new(device.clone()).with_dtype(param_dtype.into());
        let visible_bias = Tensor::zeros([visible_size], opts.clone());
        let hidden_bias = Tensor::zeros([hidden_size], opts.clone());
        let weight = Tensor::random(
            [visible_size, hidden_size],
            burn::tensor::Distribution::Default,
            opts,
        );

        Self {
            visible_bias: Param::from_tensor(visible_bias),
            hidden_bias: Param::from_tensor(hidden_bias),
            weight: Param::from_tensor(weight),
            visible_size,
            hidden_size,
        }
    }
}

impl<S, B> Model<S, B> for Rbm<B>
where
    S: Space,
    B: Backend,
{
    type ParamDType = Float;

    fn param_dtype(&self) -> FloatDType {
        self.weight.val().dtype().into()
    }

    fn log_value(&self, space: &S, samples: Tensor<B, 2, Float>) -> Tensor<B, 1> {
        assert_eq!(space.sample_size(), self.visible_size);
        assert_eq!(
            self.weight.val().dims(),
            [self.visible_size, self.hidden_size]
        );
        let visible_bias = self.visible_bias.val().unsqueeze_dim(1);
        let hidden_bias = self.hidden_bias.val().unsqueeze_dim(0);

        let visible = samples.clone().matmul(visible_bias);
        let hidden = softplus(samples.matmul(self.weight.val()) + hidden_bias, 1.0).sum_dim(1);

        (visible + hidden).squeeze_dim::<1>(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::ContinuousSpace;
    use burn::backend::NdArray;
    use burn::tensor::Float;

    #[test]
    fn rbm_produces_log_value() {
        let device = Default::default();
        let space = ContinuousSpace::new(-1.0f32, 1.0, 4);
        let rbm = Rbm::<NdArray>::new(4, 3, None, &device);
        let samples = Tensor::<NdArray, 2, Float>::from_data([[0.0, 1.0, 0.0, 1.0]], &device);

        let density = rbm.log_value(&space, samples);

        assert_eq!(density.dims(), [1]);
    }
}
