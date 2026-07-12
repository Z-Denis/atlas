pub mod spin;

use burn::tensor::{Tensor, backend::Backend};
use burn_backend::tensor::TensorKind;

use crate::space::Space;
use crate::utils::FloatTensor;

pub use spin::{Magnetization, TransverseField};

/// User-defined operator or observable.
///
/// The observable provides connected configurations and their matrix
/// elements for a batch of samples. The minimal expectation path uses the
/// first local contribution for now and will grow into a full local estimator
/// as more operator families are added.
pub trait Observable<B: Backend, S: Space> {
    fn get_conns_and_mels(
        &self,
        space: &S,
        samples: Tensor<B, 2, S::DType>,
    ) -> (Tensor<B, 3, S::DType>, FloatTensor<B, 2>)
    where
        S::DType: TensorKind<B>;
}
