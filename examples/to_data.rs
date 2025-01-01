use anyhow::Result;
use phase_vocoder::util::*;
use realfft::{num_complex::Complex, RealFftPlanner};
use rodio::{Decoder, Source};
use std::f32::consts::TAU;
use std::{fs::File, path::Path, time::Duration};

fn main() {
    let path = "amazing_grace.mp3";
    let file = File::open(path).unwrap();
    let source = Decoder::new(file).unwrap();

    let sample_rate = source.sample_rate() as usize;
    let channels = source.channels() as usize;
    let samples_vec: Vec<f32> = source.convert_samples::<f32>().collect(); // all samples
    let samples: Vec<Vec<f32>> = samples_vec.chunks(channels).map(Vec::from).collect();
    let left: Vec<f32> = samples
        .iter()
        .take(sample_rate * 3) // 3 seconds
        .map(|s| *s.get(0).unwrap())
        .collect();

    println!("sample_rate {}, channels {}", sample_rate, channels);
    write_csv(&left, &"in.data");

    let mut in_data = left;

    let length = in_data.len();

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
