use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use std::f32::consts::{PI, TAU};

use std::fs::File;
use std::io::prelude::*;

fn gen_sin(f: f32, fs: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|k| (k as f32 * f * TAU / fs as f32).sin())
        .collect()
}

#[test]
fn test_gen_sin() {
    let mut file = File::create("sin_f32.data").unwrap();
    let sin = gen_sin(10_000.0, 48000, 1000);

    sin.iter().for_each(|k| write!(file, "{},", k).unwrap());
}

fn main() {
    let length = 256;

    // make a planner
    let mut real_planner = RealFftPlanner::<f64>::new();

    // create a FFT
    let r2c = real_planner.plan_fft_forward(length);
    // make a dummy real-valued signal (filled with zeros)
    let mut indata = r2c.make_input_vec();
    // make a vector for storing the spectrum
    let mut spectrum = r2c.make_output_vec();

    // Are they the length we expect?
    assert_eq!(indata.len(), length);
    assert_eq!(spectrum.len(), length / 2 + 1);

    // forward transform the signal
    r2c.process(&mut indata, &mut spectrum).unwrap();

    // create an inverse FFT
    let c2r = real_planner.plan_fft_inverse(length);

    // create a vector for storing the output
    let mut outdata = c2r.make_output_vec();
    assert_eq!(outdata.len(), length);

    // inverse transform the spectrum back to a real-valued signal
    c2r.process(&mut spectrum, &mut outdata).unwrap();
}
