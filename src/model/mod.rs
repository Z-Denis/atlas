pub mod rbm;

use crate::space::Space;
use burn::tensor::{FloatDType, Numeric, Tensor, backend::Backend};
use burn_backend::tensor::TensorKind;

/// Minimal model interface.
///
/// A model is defined on a configuration space and returns a backend-native
/// logarithmic value for each configuration batch.
pub trait Model<S: Space, B: Backend> {
    type ParamDType;

    fn param_dtype(&self) -> FloatDType {
        FloatDType::F32
    }

    fn log_value(&self, space: &S, samples: Tensor<B, 2, Self::ParamDType>) -> Tensor<B, 1>
    where
        Self::ParamDType: TensorKind<B> + Numeric<B>;
}

pub use rbm::Rbm;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::ContinuousSpace;
    use crate::test_utils::ZeroModel;
    use burn::backend::NdArray;
    use burn::tensor::{Float, Tensor};

    #[test]
    fn model_provides_log_value() {
        let device = Default::default();
        let space = ContinuousSpace::new(-1.0f32, 1.0, 1);
        let samples: Tensor<NdArray, 2, Float> = Tensor::from_data([[0.0f32]], &device);
        let model = ZeroModel;

        let density = model.log_value(&space, samples);

        assert_eq!(density.dims(), [1]);
    }
}
