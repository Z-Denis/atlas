use burn::tensor::{Distribution, Int, Tensor, TensorCreationOptions, backend::Backend};

pub type FloatTensor<B, const D: usize> = Tensor<B, D, burn::tensor::Float>;
pub type IntTensor<B, const D: usize> = Tensor<B, D, burn::tensor::Int>;

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
}
