use burn::tensor::{FloatDType, Int, Tensor, TensorCreationOptions, backend::Backend};

use crate::observable::Observable;
use crate::space::SpinSpace;
use crate::utils::FloatTensor;

/// Total spin magnetization.
///
/// This is diagonal in the configuration basis, so the only connected
/// configuration is the input sample itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Magnetization;

impl<B> Observable<B, SpinSpace> for Magnetization
where
    B: Backend,
{
    fn get_conns_and_mels(
        &self,
        _space: &SpinSpace,
        samples: Tensor<B, 2, burn::tensor::Int>,
    ) -> (Tensor<B, 3, burn::tensor::Int>, FloatTensor<B, 2>) {
        let conns = samples.clone().unsqueeze_dim(1);
        let mels = samples.cast(FloatDType::F32).sum_dim(1);
        (conns, mels)
    }
}

/// Transverse-field spin flip operator.
///
/// Each connected configuration flips exactly one spin.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransverseField {
    strength: f32,
}

impl TransverseField {
    pub fn new(strength: f32) -> Self {
        Self { strength }
    }
}

impl<B> Observable<B, SpinSpace> for TransverseField
where
    B: Backend,
{
    fn get_conns_and_mels(
        &self,
        space: &SpinSpace,
        samples: Tensor<B, 2, burn::tensor::Int>,
    ) -> (Tensor<B, 3, burn::tensor::Int>, FloatTensor<B, 2>) {
        assert_eq!(space.local().local_size(), 2);

        let device = samples.device();
        let batch = samples.dims()[0];
        let n_sites = samples.dims()[1];
        let sites = Tensor::<B, 1, Int>::arange(0..n_sites as i64, &device).one_hot(n_sites);
        let flip = Tensor::<B, 2, Int>::ones(
            [n_sites, n_sites],
            TensorCreationOptions::<B>::int().with_device(device.clone()),
        ) - sites.mul_scalar(2);
        let mels = FloatTensor::<B, 2>::ones(
            [batch, n_sites],
            TensorCreationOptions::<B>::float().with_device(device.clone()),
        )
        .mul_scalar(self.strength);
        let conns = samples
            .unsqueeze_dim::<3>(1)
            .expand([batch, n_sites, n_sites])
            .mul(flip.unsqueeze_dim::<3>(0).expand([batch, n_sites, n_sites]));
        (conns, mels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntTensor;
    use crate::space::{HomogeneousSpace, Spin};
    use burn::backend::Flex;

    #[test]
    fn magnetization_returns_the_input_and_the_total_spin() {
        let device = Default::default();
        let space: SpinSpace = HomogeneousSpace::new(Spin::half_integer(1), 2);
        let samples = IntTensor::<Flex, 2>::from_data([[1, -1]], &device);

        let (conns, mels) = Magnetization.get_conns_and_mels(&space, samples);

        assert_eq!(conns.dims(), [1, 1, 2]);
        assert_eq!(conns.into_data().to_vec::<i32>().unwrap(), vec![1, -1]);
        assert_eq!(mels.dims(), [1, 1]);
        assert!((mels.into_data().to_vec::<f32>().unwrap()[0]).abs() < 1e-6);
    }

    #[test]
    fn transverse_field_flips_each_spin_once() {
        let device = Default::default();
        let space: SpinSpace = HomogeneousSpace::new(Spin::half_integer(1), 2);
        let samples = IntTensor::<Flex, 2>::from_data([[1, -1]], &device);

        let (conns, mels) = TransverseField::new(2.0).get_conns_and_mels(&space, samples);

        assert_eq!(conns.dims(), [1, 2, 2]);
        assert_eq!(
            conns.into_data().to_vec::<i32>().unwrap(),
            vec![-1, -1, 1, 1]
        );
        assert_eq!(mels.dims(), [1, 2]);
        assert_eq!(mels.into_data().to_vec::<f32>().unwrap(), vec![2.0, 2.0]);
    }
}
