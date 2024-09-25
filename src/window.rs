use core::f32::consts::TAU;

// Hann window function: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows

#[inline(always)]
pub fn hann_window_const<const N: usize>(in_samples: &[f32; N], out_samples: &mut [f32; N]) {
    in_samples
        .iter()
        .zip(out_samples.iter_mut())
        .enumerate()
        .for_each(|(i, (in_s, out_s))| {
            *out_s = *in_s * 0.5 * (1.0 - (TAU * i as f32 / N as f32).cos())
        });
}

#[inline(always)]
pub fn hann_window(in_samples: &[f32]) -> Vec<f32> {
    let n = in_samples.len();
    let mut out_samples: Vec<f32> = Vec::with_capacity(n);
    in_samples.iter().enumerate().for_each(|(i, in_s)| {
        out_samples.push(*in_s * 0.5 * (1.0 - (TAU * i as f32 / n as f32).cos()))
    });
    out_samples
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_hann_const() {
        let in_samples = [1.0; 64];
        let mut out_samples = [0.0; 64];
        hann_window_const(&in_samples, &mut out_samples);
        assert_eq!(out_samples[0], 0.0);
        assert_eq!(out_samples[32], 1.0);
    }

    #[test]
    fn test_hann() {
        let in_samples = [1.0; 64];
        let out_samples = hann_window(&in_samples);
        assert_eq!(out_samples[0], 0.0);
        assert_eq!(out_samples[32], 1.0);
    }

    #[test]
    fn test_hann_vs_hann_const() {
        let in_samples = [1.0; 64];
        let mut out_samples = [0.0; 64];
        hann_window_const(&in_samples, &mut out_samples);
        assert_eq!(&out_samples, &hann_window(&in_samples)[..]);
    }
}
