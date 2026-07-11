pub mod rbm;

use burn::tensor::{FloatDType, Numeric, backend::Backend};
use burn_backend::tensor::TensorKind;

use crate::utils::FloatTensor;

/// Minimal model interface.
///
/// A model is a Burn module that maps a batch of float configurations to a
/// batch of logarithmic values.
pub trait Model<B: Backend> {
    fn param_dtype(&self) -> FloatDType {
        FloatDType::F32
    }

    fn log_value(&self, samples: FloatTensor<B, 2>) -> FloatTensor<B, 1>
    where
        burn::tensor::Float: TensorKind<B> + Numeric<B>;
}

pub use rbm::Rbm;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::ZeroModel;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    #[test]
    fn model_provides_log_value() {
        let device = Default::default();
        let samples: FloatTensor<NdArray, 2> = Tensor::from_data([[0.0f32]], &device);
        let model = ZeroModel;

        let density = model.log_value(samples);

        assert_eq!(density.dims(), [1]);
    }
}
