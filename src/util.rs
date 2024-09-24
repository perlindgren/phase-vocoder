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
        let f = 15.0;

        let mut in_data = gen_sin(f, FS, FS);
        write_csv(&in_data, &"in.data");

        let mut planner = RealFftPlanner::new();
        let fft_r2c = planner.plan_fft_forward(FS);
        let mut in_fft = fft_r2c.make_output_vec();

        fft_r2c.process(&mut in_data, &mut in_fft).unwrap();

        println!(
            "in data length {}, in fft length {}",
            in_data.len(),
            in_fft.len()
        );

        let fft_norm: Vec<_> = in_fft.iter().map(|c| c.norm() / FS as f32).collect();
        write_csv(&fft_norm, &"in-fft-norm.data");

        // naive time stretching
        let stretch = 2.5;
        let stretch_len = (FS as f32 * stretch) as usize;
        let ifft_cr = planner.plan_fft_inverse(stretch_len);
        let mut ifft_spectrum = ifft_cr.make_input_vec(); // zeroes
        let mut ifft_data = ifft_cr.make_output_vec();
        println!(
            "ifft_data length {}, spectrum2 (stretched) length {}",
            ifft_data.len(),
            ifft_spectrum.len()
        );

        // stretch time
        in_fft.iter().enumerate().for_each(|(i, bin_in)| {
            let unwrapped_phase = i as f32 * TAU + bin_in.arg();
            let new_phase = unwrapped_phase * stretch;
            let new_frequency = new_phase / TAU;
            let to_bin = new_frequency.round();
            let new_phase_principal = new_phase % TAU;

            ifft_spectrum[to_bin as usize] =
                Complex::from_polar(bin_in.norm(), new_phase_principal);

            // println!(
            //     "i {}, to_bin {}, left {}, right {}, left_weight {}, right_weight {}",
            //     i, to_bin, left, right, left_weight, right_weight
            // );
        });

        let fft_norm2: Vec<_> = ifft_spectrum
            .iter()
            .map(|c| c.norm() / stretch_len as f32)
            .collect();

        write_csv(&fft_norm2, &"stretched-fft-norm.data");

        let _ = ifft_cr.process(&mut ifft_spectrum, &mut ifft_data); // .unwrap();

        write_csv(&ifft_data, &"stretched.data");

        // and back again
        let fft_r2c = planner.plan_fft_forward(FS);
        let mut spectrum = fft_r2c.make_output_vec();

        fft_r2c
            .process(&mut ifft_data[0..FS], &mut spectrum)
            .unwrap();
        let fft_norm: Vec<_> = spectrum
            .iter()
            .map(|c| c.norm() / stretch_len as f32)
            .collect();
        write_csv(&fft_norm, &"stretched-data-fft-norm.data");
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
