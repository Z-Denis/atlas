use burn::tensor::{Distribution, Int, Tensor, TensorCreationOptions, backend::Backend};

pub type FloatTensor<B, const D: usize> = Tensor<B, D, burn::tensor::Float>;
pub type IntTensor<B, const D: usize> = Tensor<B, D, burn::tensor::Int>;

#[derive(Clone, Debug)]
pub struct ComplexTensor<B: Backend, const D: usize> {
    pub re: FloatTensor<B, D>,
    pub im: FloatTensor<B, D>,
}

impl<B: Backend, const D: usize> ComplexTensor<B, D> {
    pub fn new(re: FloatTensor<B, D>, im: FloatTensor<B, D>) -> Self {
        assert_eq!(re.dims(), im.dims());
        Self { re, im }
    }

    pub fn real(&self) -> FloatTensor<B, D> {
        self.re.clone()
    }

    pub fn imag(&self) -> FloatTensor<B, D> {
        self.im.clone()
    }

    pub fn conj(&self) -> Self {
        Self::new(self.re.clone(), -self.im.clone())
    }

    pub fn exp(&self) -> Self {
        let mag = self.re.clone().exp();
        Self::new(
            mag.clone() * self.im.clone().cos(),
            mag * self.im.clone().sin(),
        )
    }

    pub fn log(&self) -> Self {
        Self::new(
            self.abs2().log().mul_scalar(0.5),
            self.im.clone().atan2(self.re.clone()),
        )
    }

    pub fn abs2(&self) -> FloatTensor<B, D> {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }
}

pub(crate) fn int_opts<B: Backend>(device: &B::Device) -> TensorCreationOptions<B> {
    TensorCreationOptions::<B>::int().with_device(device.clone())
}

pub(crate) fn float_opts<B: Backend>(device: &B::Device) -> TensorCreationOptions<B> {
    TensorCreationOptions::<B>::float().with_device(device.clone())
}

pub(crate) fn chain_indices<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::arange(0..n as i64, int_opts::<B>(device))
}

/// Sample integer values from `[low, high)`.
pub(crate) fn randint<B, const D: usize>(
    shape: [usize; D],
    low: i64,
    high: i64,
    device: &B::Device,
) -> Tensor<B, D, Int>
where
    B: Backend,
{
    assert!(low < high, "randint requires low < high");

    Tensor::<B, D, Int>::random(
        shape,
        Distribution::Uniform(low as f64, high as f64),
        int_opts::<B>(device),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    #[test]
    fn randint_samples_inside_half_open_range() {
        let device = Default::default();
        let values: Tensor<Flex, 1, Int> = randint([128], 2, 7, &device);
        let data = values.into_data().to_vec::<i32>().unwrap();
        assert!(data.iter().all(|&x| (2..7).contains(&x)));
    }

    #[test]
    fn complex_tensor_tracks_parts() {
        let device = Default::default();
        let re = FloatTensor::<Flex, 1>::from_data([1.0f32, -2.0], &device);
        let im = FloatTensor::<Flex, 1>::from_data([3.0f32, -4.0], &device);
        let complex = ComplexTensor::new(re, im);

        assert_eq!(complex.re.dims(), [2]);
        assert_eq!(complex.im.dims(), [2]);
        assert_eq!(
            complex.abs2().into_data().to_vec::<f32>().unwrap(),
            vec![10.0, 20.0]
        );
        assert_eq!(
            complex.conj().im.into_data().to_vec::<f32>().unwrap(),
            vec![-3.0, 4.0]
        );
    }

    #[test]
    fn complex_tensor_log_is_principal_branch() {
        let device = Default::default();
        let re = FloatTensor::<Flex, 1>::from_data([-1.0f32], &device);
        let im = FloatTensor::<Flex, 1>::from_data([0.0f32], &device);
        let complex = ComplexTensor::new(re, im).log();

        let re = complex.re.into_data().to_vec::<f32>().unwrap();
        let im = complex.im.into_data().to_vec::<f32>().unwrap();

        assert!((re[0] - 0.0).abs() < 1e-6);
        assert!((im[0] - std::f32::consts::PI).abs() < 1e-6);
    }
}
