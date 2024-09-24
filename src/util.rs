use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use std::{
    f32::consts::{PI, TAU},
    fmt::Display,
    path::Path,
    sync::Arc,
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

pub struct Stretch {
    sample_rate: usize,
    frame_time: f32,
    stretch_factor: f32,
    in_buffer: Vec<f32>,
    fft_r2c: Arc<dyn RealToComplex<f32>>,
    fft_in: Vec<f32>,
    fft_out: Vec<Complex<f32>>,
    ifft_cr: Arc<dyn ComplexToReal<f32>>,
    ifft_in: Vec<Complex<f32>>,
    ifft_out: Vec<f32>,
}

impl Stretch {
    pub fn new(sample_rate: usize, frame_time: f32, stretch_factor: f32) -> Self {
        let mut planner = RealFftPlanner::new();
        let in_frame_size = (frame_time * sample_rate as f32) as usize;
        let out_frame_size = (in_frame_size as f32 * stretch_factor) as usize;
        let fft_r2c = planner.plan_fft_forward(in_frame_size);
        let in_buffer = fft_r2c.make_input_vec();
        let fft_in = fft_r2c.make_input_vec();
        let fft_out = fft_r2c.make_output_vec();

        let ifft_cr = planner.plan_fft_inverse(out_frame_size);
        let ifft_out = ifft_cr.make_output_vec();
        let ifft_in = ifft_cr.make_input_vec();

        Self {
            sample_rate,
            frame_time,
            stretch_factor,
            in_buffer,
            fft_r2c,
            fft_in,
            fft_out,
            ifft_cr,
            ifft_in,
            ifft_out,
        }
    }

    pub fn stretch(&mut self, in_samples: &[f32], out_sample: &mut [f32]) {
        // push in_samples into buffer
        let in_buffer_len = self.in_buffer.len();
        self.in_buffer.copy_within(in_samples.len().., 0);
        self.in_buffer[in_buffer_len - in_samples.len()..].copy_from_slice(in_samples);

        self.fft_in.copy_from_slice(&self.in_buffer);

        self.fft_r2c
            .process(&mut self.fft_in, &mut self.fft_out)
            .unwrap();

        // stretch time
        self.fft_out.iter().enumerate().for_each(|(i, bin_in)| {
            let unwrapped_phase = i as f32 * TAU + bin_in.arg();
            let new_phase = unwrapped_phase * self.stretch_factor;
            let new_frequency = new_phase / TAU;
            let to_bin = new_frequency.round();
            let new_phase_principal = new_phase % TAU;

            self.ifft_in[to_bin as usize] = Complex::from_polar(bin_in.norm(), new_phase_principal);
        });

        let _ = self.ifft_cr.process(&mut self.ifft_in, &mut self.ifft_out); // .unwrap();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use realfft::RealFftPlanner;

    #[test]
    fn test_stretch_buffer() {
        const SAMPLE_RATE: usize = 480;
        const FRAME_TIME: f32 = 0.50;
        const STRETCH_FACTOR: f32 = 2.0;
        const IN_LEN: usize = (FRAME_TIME * (SAMPLE_RATE as f32)) as usize;

        let mut stretch = Stretch::new(SAMPLE_RATE, FRAME_TIME, STRETCH_FACTOR);
        println!("sample_rate    {}", stretch.sample_rate);
        println!("frame_time     {}", stretch.frame_time);
        println!("stretch_factor {}", stretch.stretch_factor);
        println!("in_buffer.len  {}", stretch.in_buffer.len());
        assert_eq!(stretch.in_buffer.len(), IN_LEN);
        println!("fft_in.len     {}", stretch.fft_in.len());
        println!("fft_out.len    {}", stretch.fft_out.len());
        assert_eq!(stretch.fft_in.len(), stretch.in_buffer.len());
        assert_eq!(stretch.fft_out.len(), stretch.fft_in.len() / 2 + 1);

        println!("ifft_in.len    {}", stretch.ifft_in.len());
        println!("ifft_out.len   {}", stretch.ifft_out.len());
        assert_eq!(
            stretch.ifft_out.len(),
            (stretch.fft_in.len() as f32 * stretch.stretch_factor) as usize
        );

        const NR_IN_SAMPLES: usize = 20;
        const NR_OUT_SAMPLES: usize = (NR_IN_SAMPLES as f32 * STRETCH_FACTOR) as usize;

        let in_samples = [1.0; NR_IN_SAMPLES];
        let mut out_samples = [0.0; NR_OUT_SAMPLES];
        stretch.stretch(&in_samples, &mut out_samples);
        assert_eq!(
            stretch.in_buffer[0..IN_LEN - NR_IN_SAMPLES],
            [0.0; IN_LEN - NR_IN_SAMPLES]
        );
        assert_eq!(stretch.in_buffer[IN_LEN - NR_IN_SAMPLES..], in_samples);

        const NR_IN_SAMPLES_2: usize = 40;
        const NR_OUT_SAMPLES_2: usize = (NR_IN_SAMPLES as f32 * STRETCH_FACTOR) as usize;

        let in_samples_2 = [0.5; NR_IN_SAMPLES_2];
        let mut out_samples_2 = [0.0; NR_OUT_SAMPLES_2];

        println!("IN_LEN {}", IN_LEN);
        stretch.stretch(&in_samples_2, &mut out_samples_2);

        assert_eq!(
            stretch.in_buffer[0..IN_LEN - (NR_IN_SAMPLES_2 + NR_IN_SAMPLES)],
            [0.0; IN_LEN - (NR_IN_SAMPLES_2 + NR_IN_SAMPLES)]
        );

        assert_eq!(
            stretch.in_buffer[IN_LEN - (NR_IN_SAMPLES_2 + NR_IN_SAMPLES)..IN_LEN - NR_IN_SAMPLES_2],
            in_samples
        );

        assert_eq!(stretch.in_buffer[IN_LEN - NR_IN_SAMPLES_2..], in_samples_2);
    }

    #[test]
    fn test_gen_sin() {
        let sin = gen_sin(10_000.0, 48000, 1000);

        write_csv(&sin, &"sin_f32.data");
    }

    #[test]
    fn push() {
        let mut v = vec![];
        v.push(3);
        v.push(2);
        v.push(1);
        v.push(0);
        println!("v {:?}", v);
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
