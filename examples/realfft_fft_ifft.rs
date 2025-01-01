// small example of realfft

use phase_vocoder::util::*;
use realfft::{num_complex::Complex, RealFftPlanner};
use std::f32::consts::TAU;

fn main() {
    const FS: usize = 48000;
    let f = 1000.0;

    let mut in_data = gen_sin(f, FS, FS);
    write_csv(&in_data, &"in.data");

    let length = FS;

    // make a planner
    let mut real_planner = RealFftPlanner::<f32>::new();

    // create a FFT
    let r2c = real_planner.plan_fft_forward(length);
    // make a vector for storing the spectrum
    let mut spectrum = r2c.make_output_vec();

    // Are they the length we expect?
    assert_eq!(in_data.len(), length);
    assert_eq!(spectrum.len(), length / 2 + 1);

    // forward transform the signal
    r2c.process(&mut in_data, &mut spectrum).unwrap();

    // create an inverse FFT
    let c2r = real_planner.plan_fft_inverse(length);

    // create a vector for storing the output
    let mut out_data = c2r.make_output_vec();
    assert_eq!(out_data.len(), length);
    println!("out_data length {}", out_data.len());

    // inverse transform the spectrum back to a real-valued signal
    c2r.process(&mut spectrum, &mut out_data).unwrap();
    // normalize volume
    let scale = 1.0 / (out_data.len()) as f32;
    for v in out_data.iter_mut() {
        *v *= scale;
    }
    write_csv(&out_data, &"out.data");
}
