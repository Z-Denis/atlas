pub mod rbm;

use crate::space::Space;
use burn::tensor::{Numeric, Tensor, backend::Backend};

/// Minimal model interface.
///
/// A model is defined on a configuration space and returns a logarithmic value
/// for each configuration.
pub trait Model<S: Space, B: Backend> {
    fn log_value<K>(&self, space: &S, samples: Tensor<B, 2, K>) -> Tensor<B, 1>
    where
        K: Numeric<B>;
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
