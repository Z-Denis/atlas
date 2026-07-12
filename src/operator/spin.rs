use burn::tensor::{FloatDType, Int, Tensor, TensorCreationOptions, backend::Backend, s};

use crate::operator::Operator;
use crate::space::SpinSpace;
use crate::utils::FloatTensor;

fn ising_energy<B: Backend>(
    samples: &Tensor<B, 2, burn::tensor::Int>,
    coupling: f32,
    field: f32,
) -> FloatTensor<B, 1> {
    let device = samples.device();
    let batch = samples.dims()[0];
    let n_sites = samples.dims()[1];
    let field = samples
        .clone()
        .cast(FloatDType::F32)
        .sum_dim(1)
        .squeeze_dim::<1>(1)
        .mul_scalar(-field);
    let coupling = if n_sites <= 1 {
        FloatTensor::<B, 1>::zeros(
            [batch],
            TensorCreationOptions::<B>::float().with_device(device.clone()),
        )
    } else {
        let left = samples
            .clone()
            .slice(s![.., 0..n_sites - 1])
            .cast(FloatDType::F32);
        let right = samples
            .clone()
            .slice(s![.., 1..n_sites])
            .cast(FloatDType::F32);
        (left * right)
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .mul_scalar(-coupling)
    };

    field + coupling
}

/// Total spin magnetization.
///
/// This is diagonal in the configuration basis, so the only connected
/// configuration is the input sample itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Magnetization;

impl<B> Operator<B, SpinSpace> for Magnetization
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

/// Transverse-field Ising Hamiltonian.
///
/// The first connected configuration is the diagonal Ising energy.
/// The remaining connected configurations flip one spin at a time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransverseFieldIsing {
    coupling: f32,
    field: f32,
}

impl TransverseFieldIsing {
    pub fn new(coupling: f32, field: f32) -> Self {
        Self { coupling, field }
    }
}

impl<B> Operator<B, SpinSpace> for TransverseFieldIsing
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
        let diagonal = ising_energy(&samples, self.coupling, self.field).unsqueeze_dim(1);

        let offdiag_conns = samples
            .clone()
            .unsqueeze_dim::<3>(1)
            .expand([batch, n_sites, n_sites])
            .mul(flip.unsqueeze_dim::<3>(0).expand([batch, n_sites, n_sites]));
        let offdiag_mels = FloatTensor::<B, 2>::full(
            [batch, n_sites],
            -self.field,
            TensorCreationOptions::<B>::float().with_device(device.clone()),
        );

        (
            Tensor::cat(vec![samples.clone().unsqueeze_dim(1), offdiag_conns], 1),
            Tensor::cat(vec![diagonal, offdiag_mels], 1),
        )
    }
}

/// Diagonal Ising energy on an open 1D chain.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IsingEnergy {
    coupling: f32,
    field: f32,
}

impl IsingEnergy {
    pub fn new(coupling: f32, field: f32) -> Self {
        Self { coupling, field }
    }
}

impl<B> Operator<B, SpinSpace> for IsingEnergy
where
    B: Backend,
{
    fn get_conns_and_mels(
        &self,
        _space: &SpinSpace,
        samples: Tensor<B, 2, burn::tensor::Int>,
    ) -> (Tensor<B, 3, burn::tensor::Int>, FloatTensor<B, 2>) {
        let conns = samples.clone().unsqueeze_dim(1);
        (
            conns,
            ising_energy(&samples, self.coupling, self.field).unsqueeze_dim(1),
        )
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
    fn transverse_field_ising_returns_diagonal_and_flips() {
        let device = Default::default();
        let space: SpinSpace = HomogeneousSpace::new(Spin::half_integer(1), 2);
        let samples = IntTensor::<Flex, 2>::from_data([[1, -1]], &device);

        let (conns, mels) = TransverseFieldIsing::new(1.0, 2.0).get_conns_and_mels(&space, samples);

        assert_eq!(conns.dims(), [1, 3, 2]);
        assert_eq!(
            conns.into_data().to_vec::<i32>().unwrap(),
            vec![1, -1, -1, -1, 1, 1]
        );
        assert_eq!(mels.dims(), [1, 3]);
        assert_eq!(
            mels.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, -2.0, -2.0]
        );
    }

    #[test]
    fn ising_returns_diagonal_energy() {
        let device = Default::default();
        let space: SpinSpace = HomogeneousSpace::new(Spin::half_integer(1), 2);
        let samples = IntTensor::<Flex, 2>::from_data([[1, -1]], &device);

        let (conns, mels) = IsingEnergy::new(1.0, 0.5).get_conns_and_mels(&space, samples);

        assert_eq!(conns.dims(), [1, 1, 2]);
        assert_eq!(conns.into_data().to_vec::<i32>().unwrap(), vec![1, -1]);
        assert_eq!(mels.dims(), [1, 1]);
        assert!((mels.into_data().to_vec::<f32>().unwrap()[0] - 1.0).abs() < 1e-6);
    }
}
