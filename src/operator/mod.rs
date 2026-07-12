pub mod spin;

use burn::tensor::{Tensor, backend::Backend};
use burn_backend::tensor::TensorKind;

use crate::space::Space;
use crate::utils::FloatTensor;

pub use spin::{IsingEnergy, Magnetization, TransverseFieldIsing};

/// Operator contract for local estimators.
///
/// Given a batch of samples, return all connected configurations and their
/// matrix elements.
pub trait Operator<B: Backend, S: Space> {
    fn get_conns_and_mels(
        &self,
        space: &S,
        samples: Tensor<B, 2, S::DType>,
    ) -> (Tensor<B, 3, S::DType>, FloatTensor<B, 2>)
    where
        S::DType: TensorKind<B>;
}
