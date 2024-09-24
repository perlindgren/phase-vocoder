use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use std::{
    f32::consts::{PI, TAU},
    fmt::Display,
    path::Path,
};

use std::fs::File;
use std::io::prelude::*;

pub fn gen_sin(f: f32, fs: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|k| (k as f32 * f * TAU / fs as f32).sin())
        .collect()
}

pub fn write_csv<T>(data: &Vec<T>, path: &impl AsRef<Path>)
where
    T: Display,
{
    let mut file = File::create(path).unwrap();

    data.iter().for_each(|k| write!(file, "{},", k).unwrap());
}

#[cfg(test)]
mod test {
    use super::*;
    use realfft::RealFftPlanner;

    #[test]
    fn test_gen_sin() {
        let sin = gen_sin(10_000.0, 48000, 1000);

        write_csv(&sin, &"sin_f32.data");
    }

    #[test]
    fn test_realfft() {
        const FS: usize = 1000;
        let f = 100.0;

        let mut sin_100 = gen_sin(f, FS, FS);
        write_csv(&sin_100, &"sin_100.data");

        let mut planner = RealFftPlanner::new();
        let fft_r2c = planner.plan_fft_forward(FS);
        let mut spectrum = fft_r2c.make_output_vec();

        fft_r2c.process(&mut sin_100, &mut spectrum).unwrap();

        println!(
            "data length {}, spectrum length {}",
            sin_100.len(),
            spectrum.len()
        );

        let fft_norm: Vec<_> = spectrum.iter().map(|c| c.norm()).collect();

        let mut spectrum2 = fft_r2c.make_output_vec(); // zeroes

        let shift = 0.5;
        // shift spectrum
        spectrum2.iter_mut().enumerate().for_each(|(i, bin_out)| {
            let to_bin: f32 = i as f32 * shift;
            let left = to_bin.floor();
            let right = (to_bin + 1.0).floor();
            let left_weight = 1.0 - (to_bin - left).abs();
            let right_weight = 1.0 - (to_bin - right).abs();
            // println!("i {}, to_bin {}", i, to_bin);
            let left_sample = if let Some(r) = spectrum.get(left as usize) {
                r.to_owned()
            } else {
                Complex::zero()
            };
            let right_sample = if let Some(r) = spectrum.get(right as usize) {
                r.to_owned()
            } else {
                Complex::zero()
            };

            *bin_out = left_sample * left_weight + right_sample * right_weight;

            // println!(
            //     "i {}, to_bin {}, left {}, right {}, left_weight {}, right_weight {}",
            //     i, to_bin, left, right, left_weight, right_weight
            // );
        });

        let fft_norm2: Vec<_> = spectrum2.iter().map(|c| c.norm()).collect();

        let ifft_cr = planner.plan_fft_inverse(FS);
        let mut ifft_data = ifft_cr.make_output_vec();

        ifft_cr.process(&mut spectrum2, &mut ifft_data); // .unwrap();

        write_csv(&fft_norm, &"spectrum.data");
        write_csv(&fft_norm2, &"spectrum2.data");
        write_csv(&ifft_data, &"ifft.data");
    }
}

// fn main() {
//     let length = 256;

//     // make a planner
//     let mut real_planner = RealFftPlanner::<f64>::new();

//     // create a FFT
//     let r2c = real_planner.plan_fft_forward(length);
//     // make a dummy real-valued signal (filled with zeros)
//     let mut indata = r2c.make_input_vec();
//     // make a vector for storing the spectrum
//     let mut spectrum = r2c.make_output_vec();

//     // Are they the length we expect?
//     assert_eq!(indata.len(), length);
//     assert_eq!(spectrum.len(), length / 2 + 1);

//     // forward transform the signal
//     r2c.process(&mut indata, &mut spectrum).unwrap();

//     // create an inverse FFT
//     let c2r = real_planner.plan_fft_inverse(length);

//     // create a vector for storing the output
//     let mut outdata = c2r.make_output_vec();
//     assert_eq!(outdata.len(), length);

//     // inverse transform the spectrum back to a real-valued signal
//     c2r.process(&mut spectrum, &mut outdata).unwrap();
// }
