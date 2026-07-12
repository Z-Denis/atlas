use burn::tensor::backend::Backend;

use crate::utils::ComplexTensor;

/// Stable principal-branch `log(1 + z)` for complex `z = re + i im`.
pub fn log1p<B: Backend, const D: usize>(z: &ComplexTensor<B, D>) -> ComplexTensor<B, D> {
    let re = z.re.clone();
    let im = z.im.clone();
    let norm2_minus_one = re.clone() * (re.clone() + 2.0) + im.clone() * im.clone();

    ComplexTensor::new(norm2_minus_one.log1p().mul_scalar(0.5), im.atan2(re + 1.0))
}

/// Stable `log(cosh(z))` for complex `z = re + i im`.
pub fn log_cosh<B: Backend, const D: usize>(z: &ComplexTensor<B, D>) -> ComplexTensor<B, D> {
    let negative = z.re.clone().lower_elem(0);
    let x = ComplexTensor::new(
        z.re.clone().mask_where(negative.clone(), -z.re.clone()),
        z.im.clone().mask_where(negative, -z.im.clone()),
    );
    let y =
        log1p(&ComplexTensor::new(x.re.clone().mul_scalar(-2), x.im.clone().mul_scalar(-2)).exp());

    ComplexTensor::new(
        y.re + x.re.clone() - std::f64::consts::LN_2,
        y.im + x.im.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::FloatTensor;
    use burn::backend::Flex;

    #[test]
    fn complex_log1p_matches_principal_branch() {
        let device = Default::default();
        let re = FloatTensor::<Flex, 1>::from_data([-0.5f32], &device);
        let im = FloatTensor::<Flex, 1>::from_data([0.0f32], &device);
        let z = ComplexTensor::new(re, im);

        let value = log1p(&z);
        let re = value.re.into_data().to_vec::<f32>().unwrap();
        let im = value.im.into_data().to_vec::<f32>().unwrap();

        assert!((re[0] - (-std::f32::consts::LN_2)).abs() < 1e-6);
        assert!(im[0].abs() < 1e-6);
    }

    #[test]
    fn complex_log_cosh_matches_real_reference() {
        let device = Default::default();
        let re = FloatTensor::<Flex, 1>::from_data([0.3f32], &device);
        let im = FloatTensor::<Flex, 1>::from_data([0.0f32], &device);
        let z = ComplexTensor::new(re, im);

        let value = log_cosh(&z);
        let re = value.re.into_data().to_vec::<f32>().unwrap();
        let im = value.im.into_data().to_vec::<f32>().unwrap();

        assert!((re[0] - 0.3f32.cosh().ln()).abs() < 1e-5);
        assert!(im[0].abs() < 1e-6);
    }

    #[test]
    fn complex_log_cosh_matches_direct_complex_identity() {
        let device = Default::default();
        let re = FloatTensor::<Flex, 1>::from_data([0.2f32], &device);
        let im = FloatTensor::<Flex, 1>::from_data([0.3f32], &device);
        let z = ComplexTensor::new(re, im);

        let stable = log_cosh(&z);
        let direct = {
            let ez = z.exp();
            let emz = ComplexTensor::new(-z.re.clone(), -z.im.clone()).exp();
            let half = 0.5f32;
            ComplexTensor::new(
                (ez.re + emz.re).mul_scalar(half),
                (ez.im + emz.im).mul_scalar(half),
            )
            .log()
        };

        let stable_re = stable.re.into_data().to_vec::<f32>().unwrap();
        let stable_im = stable.im.into_data().to_vec::<f32>().unwrap();
        let direct_re = direct.re.into_data().to_vec::<f32>().unwrap();
        let direct_im = direct.im.into_data().to_vec::<f32>().unwrap();

        assert!((stable_re[0] - direct_re[0]).abs() < 1e-5);
        assert!((stable_im[0] - direct_im[0]).abs() < 1e-5);
    }

    #[test]
    fn complex_log_cosh_tracks_branch_cut_near_imaginary_pi_over_two() {
        let device = Default::default();
        let re = FloatTensor::<Flex, 1>::from_data([0.0f32], &device);
        let im = FloatTensor::<Flex, 1>::from_data([1.8f32], &device);
        let z = ComplexTensor::new(re, im);

        let value = log_cosh(&z);
        let re = value.re.into_data().to_vec::<f32>().unwrap();
        let im = value.im.into_data().to_vec::<f32>().unwrap();

        assert!(re[0].is_finite());
        assert!(im[0].is_finite());
        assert!((im[0].abs() - std::f32::consts::PI).abs() < 1e-4);
    }
}
