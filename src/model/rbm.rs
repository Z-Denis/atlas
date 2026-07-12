use burn::module::{Module, Param};
use burn::tensor::{FloatDType, Tensor, TensorCreationOptions, backend::Backend};

use super::{
    Model,
    utils::{log_cosh, log_cosh_real},
};
use crate::utils::{ComplexTensor, FloatTensor};

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

impl<B> Model<B> for Rbm<B>
where
    B: Backend,
{
    type Output = FloatTensor<B, 1>;

    fn param_dtype(&self) -> FloatDType {
        self.weight.val().dtype().into()
    }

    fn log_value(&self, samples: FloatTensor<B, 2>) -> Self::Output {
        assert_eq!(
            self.weight.val().dims(),
            [self.visible_size, self.hidden_size]
        );
        let visible_bias = self.visible_bias.val().unsqueeze_dim(1);
        let hidden_bias = self.hidden_bias.val().unsqueeze_dim(0);

        let visible = samples.clone().matmul(visible_bias);
        let hidden = log_cosh_real(&(samples.matmul(self.weight.val()) + hidden_bias)).sum_dim(1);

        (visible + hidden).squeeze_dim::<1>(1)
    }
}

/// Complex-valued restricted Boltzmann machine.
#[derive(Module, Debug)]
pub struct ComplexRbm<B: Backend> {
    pub visible_bias_re: Param<Tensor<B, 1>>,
    pub visible_bias_im: Param<Tensor<B, 1>>,
    pub hidden_bias_re: Param<Tensor<B, 1>>,
    pub hidden_bias_im: Param<Tensor<B, 1>>,
    pub weight_re: Param<Tensor<B, 2>>,
    pub weight_im: Param<Tensor<B, 2>>,
    visible_size: usize,
    hidden_size: usize,
}

impl<B: Backend> ComplexRbm<B> {
    pub fn new(
        visible_size: usize,
        hidden_size: usize,
        param_dtype: Option<FloatDType>,
        device: &B::Device,
    ) -> Self {
        let param_dtype = param_dtype.unwrap_or(FloatDType::F32);
        let opts = TensorCreationOptions::<B>::new(device.clone()).with_dtype(param_dtype.into());

        let visible_bias_re = Tensor::zeros([visible_size], opts.clone());
        let visible_bias_im = Tensor::zeros([visible_size], opts.clone());
        let hidden_bias_re = Tensor::zeros([hidden_size], opts.clone());
        let hidden_bias_im = Tensor::zeros([hidden_size], opts.clone());
        let weight_re = Tensor::random(
            [visible_size, hidden_size],
            burn::tensor::Distribution::Default,
            opts.clone(),
        );
        let weight_im = Tensor::random(
            [visible_size, hidden_size],
            burn::tensor::Distribution::Default,
            opts,
        );

        Self {
            visible_bias_re: Param::from_tensor(visible_bias_re),
            visible_bias_im: Param::from_tensor(visible_bias_im),
            hidden_bias_re: Param::from_tensor(hidden_bias_re),
            hidden_bias_im: Param::from_tensor(hidden_bias_im),
            weight_re: Param::from_tensor(weight_re),
            weight_im: Param::from_tensor(weight_im),
            visible_size,
            hidden_size,
        }
    }

    fn amplitude(&self, samples: FloatTensor<B, 2>) -> ComplexTensor<B, 1> {
        assert_eq!(
            self.weight_re.val().dims(),
            [self.visible_size, self.hidden_size]
        );

        let visible_re = samples
            .clone()
            .matmul(self.visible_bias_re.val().unsqueeze_dim(1))
            .squeeze_dim::<1>(1);
        let visible_im = samples
            .clone()
            .matmul(self.visible_bias_im.val().unsqueeze_dim(1))
            .squeeze_dim::<1>(1);

        let hidden_re = samples.clone().matmul(self.weight_re.val())
            + self.hidden_bias_re.val().unsqueeze_dim(0);
        let hidden_im =
            samples.matmul(self.weight_im.val()) + self.hidden_bias_im.val().unsqueeze_dim(0);
        let hidden = log_cosh(&ComplexTensor::new(hidden_re, hidden_im));
        let hidden_re = hidden.re.sum_dim(1).squeeze_dim::<1>(1);
        let hidden_im = hidden.im.sum_dim(1).squeeze_dim::<1>(1);

        ComplexTensor::new(visible_re + hidden_re, visible_im + hidden_im)
    }
}

impl<B> Model<B> for ComplexRbm<B>
where
    B: Backend,
{
    type Output = ComplexTensor<B, 1>;

    fn param_dtype(&self) -> FloatDType {
        self.weight_re.val().dtype().into()
    }

    fn log_value(&self, samples: FloatTensor<B, 2>) -> Self::Output {
        self.amplitude(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn rbm_produces_log_value() {
        let device = Default::default();
        let rbm = Rbm::<NdArray>::new(4, 3, None, &device);
        let samples = FloatTensor::<NdArray, 2>::from_data([[0.0, 1.0, 0.0, 1.0]], &device);

        let density = rbm.log_value(samples);

        assert_eq!(density.dims(), [1]);
    }

    #[test]
    fn complex_rbm_produces_log_psi() {
        let device = Default::default();
        let rbm = ComplexRbm::<NdArray>::new(4, 3, None, &device);
        let samples = FloatTensor::<NdArray, 2>::from_data([[0.0, 1.0, 0.0, 1.0]], &device);

        let density = rbm.log_value(samples);

        assert_eq!(density.real().dims(), [1]);
        assert_eq!(density.imag().dims(), [1]);
    }

    #[test]
    fn zero_bias_rbm_is_spin_flip_symmetric() {
        let device = Default::default();
        let rbm = Rbm::<NdArray>::new(4, 3, None, &device);
        let samples = FloatTensor::<NdArray, 2>::from_data([[1.0, -1.0, 1.0, -1.0]], &device);

        let value = rbm.log_value(samples.clone());
        let flipped = rbm.log_value(samples.mul_scalar(-1.0));
        let diff = (value - flipped).into_data().to_vec::<f32>().unwrap()[0].abs();

        assert!(diff < 1e-6);
    }
}
