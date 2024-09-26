use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use std::{
    f32::consts::{PI, TAU},
    fmt::Display,
    path::Path,
    str::from_boxed_utf8_unchecked,
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

// Following the ideas for a phase vocoder, along following steps
//
// 1) incoming samples are stored in an frame buffer holding the latest N samples
//
// 2) we compute the fft (spectrum) of frame is computed (input frame in frequency domain)
//
// 3) the spectrum is stretched (taking new phase into account)
//
// 4) we compute the inverse fft (ifft) to a corresponding (output frame in time domain)
//
// 5) we overlap and add the generated frame to the output buffer
//
// The amount of overlap is determined by the hop_size (equal to the number samples provided)
// The current implementation does not take into account phase correlation between
// overlapping frames, generating audible artifacts.
//
// The frame size determines the frequency resolution (nr of bins in the fft).
//

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
    out_accumulator: Vec<f32>,
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

        let out_accumulator = vec![0.0; out_frame_size * 2];
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
            out_accumulator,
        }
    }

    // in_buffer              |            |k         | << k = in_samples.len()
    //                        |k           |in_samples|
    //
    // fft_in [hann]          |f32, f32, ...          | (in_frame_size)
    // fft_out                |..., cn, ...           | (in_frame_size)
    //
    // ifft_in                |..., ..., stretch_factor *cn, ...| (out_frame_size)
    // ifft_out               |f32, f32, ...                    | (out_frame_size)
    //
    pub fn stretch(&mut self, in_samples: &[f32], out_samples: &mut [f32]) {
        assert_eq!(
            (in_samples.len() as f32 * self.stretch_factor) as usize,
            out_samples.len()
        );
        // push in_samples into buffer
        let in_buffer_len = self.in_buffer.len();
        self.in_buffer.copy_within(in_samples.len().., 0);
        self.in_buffer[in_buffer_len - in_samples.len()..].copy_from_slice(in_samples);

        self.fft_in.copy_from_slice(&self.in_buffer);

        self.fft_r2c
            .process(&mut self.fft_in, &mut self.fft_out)
            .unwrap();

        // stretch time
        self.fft_out.take(siter().enumerate().for_each(|(i, bin_in)| {
            let unwrapped_phase = i as f32 * TAU + bin_in.arg();
            let new_phase = unwrapped_phase * self.stretch_factor;
            let new_frequency = new_phase / TAU;
            let to_bin = new_frequency.round();
            let new_phase_principal = new_phase % TAU;

            self.ifft_in[to_bin as usize] = Complex::from_polar(bin_in.norm(), new_phase_principal);
        });

        let _ = self.ifft_cr.process(&mut self.ifft_in, &mut self.ifft_out); // .unwrap();

        let frame_size = self.ifft_out.len();
        let hop_size = out_samples.len();
        let to_index = frame_size - hop_size;

        println!(
            "frame_size {}, hop_size {}, to_index {}",
            frame_size, hop_size, to_index
        );

        // shift out_accumulator left, out_accumulator.len() = 2 * frame_size
        //       |           |          x| << hop_size
        //       |          x|           |
        self.out_accumulator.copy_within(frame_size.., to_index);

        // accumulate old and new
        self.out_accumulator[frame_size..frame_size + hop_size]
            .iter_mut()
            .zip(self.ifft_out[..frame_size - hop_size].iter())
            .for_each(|(old, new)| {
                *old = (*old + *new) / 2.0;
            });

        // plain copy of new
        self.out_accumulator[2 * frame_size - hop_size..]
            .copy_from_slice(&self.ifft_out[frame_size - hop_size..]);
        //
        println!("out_samples_len {}", out_samples.len());
        out_samples.copy_from_slice(&self.out_accumulator[to_index..frame_size]);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use realfft::RealFftPlanner;

    #[test]
    fn test_stretch_buffer() {
        const SAMPLE_RATE: usize = 1000;
        const FRAME_TIME: f32 = 0.50;
        const STRETCH_FACTOR: f32 = 1.0;
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

        let in_samples = gen_sin(100.0, 1000, NR_IN_SAMPLES);
        let mut out_samples = [0.0; NR_OUT_SAMPLES];
        stretch.stretch(&in_samples, &mut out_samples);
        assert_eq!(
            stretch.in_buffer[0..IN_LEN - NR_IN_SAMPLES],
            [0.0; IN_LEN - NR_IN_SAMPLES]
        );
        // assert_eq!(stretch.in_buffer[IN_LEN - NR_IN_SAMPLES..], in_samples);

        // const NR_IN_SAMPLES_2: usize = 40;
        // const NR_OUT_SAMPLES_2: usize = (NR_IN_SAMPLES_2 as f32 * STRETCH_FACTOR) as usize;

        // let in_samples_2 = [0.5; NR_IN_SAMPLES_2];
        // let mut out_samples_2 = [0.0; NR_OUT_SAMPLES_2];

        // println!("IN_LEN {}", IN_LEN);
        // stretch.stretch(&in_samples_2, &mut out_samples_2);

        // assert_eq!(
        //     stretch.in_buffer[0..IN_LEN - (NR_IN_SAMPLES_2 + NR_IN_SAMPLES)],
        //     [0.0; IN_LEN - (NR_IN_SAMPLES_2 + NR_IN_SAMPLES)]
        // );

        // assert_eq!(
        //     stretch.in_buffer[IN_LEN - (NR_IN_SAMPLES_2 + NR_IN_SAMPLES)..IN_LEN - NR_IN_SAMPLES_2],
        //     in_samples
        // );

        // assert_eq!(stretch.in_buffer[IN_LEN - NR_IN_SAMPLES_2..], in_samples_2);

        // // stretch.stretch(&in_samples_2, &mut out_samples_2);
        // // stretch.stretch(&in_samples_2, &mut out_samples_2);
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
