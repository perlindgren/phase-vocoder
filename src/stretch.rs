use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{num_complex::Complex, num_traits::Zero};

use std::{f32::consts::TAU, sync::Arc};

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

        println!("in_frame_size {}", in_frame_size);
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
        assert!((in_samples.len() as f32 * self.stretch_factor) as usize == out_samples.len());
        // push in_samples into buffer
        let in_buffer_len = self.in_buffer.len();
        // check that in_samples fit in buffer
        assert!(in_samples.len() <= in_buffer_len);

        println!("in_samples {:?}", &in_samples[..10]);
        self.in_buffer.copy_within(in_samples.len().., 0);
        self.in_buffer[in_buffer_len - in_samples.len()..].copy_from_slice(in_samples);

        self.fft_in.copy_from_slice(&self.in_buffer);

        self.fft_r2c
            .process(&mut self.fft_in, &mut self.fft_out)
            .unwrap();

        self.ifft_in.iter_mut().for_each(|c| *c = Complex::zero());
        // stretch time
        self.fft_out
            .iter()
            .take(self.fft_out.len() - 1)
            .enumerate()
            .for_each(|(i, bin_in)| {
                let unwrapped_phase = i as f32 * TAU + bin_in.arg();
                let new_phase = unwrapped_phase * self.stretch_factor;
                let new_frequency = new_phase / TAU;
                let to_bin = new_frequency.round();
                let new_phase_principal = new_phase % TAU;

                self.ifft_in[to_bin as usize] =
                    Complex::from_polar(bin_in.norm(), new_phase_principal);
            });

        let _ = self.ifft_cr.process(&mut self.ifft_in, &mut self.ifft_out); // .unwrap();

        let in_frame_size = self.fft_in.len();
        let out_frame_size = self.ifft_out.len();
        self.ifft_out
            .iter_mut()
            .for_each(|r| *r /= in_frame_size as f32);

        let hop_size = out_samples.len();
        let to_index = out_frame_size - hop_size;

        println!(
            "out_frame_size {}, hop_size {}, to_index {}",
            out_frame_size, hop_size, to_index
        );

        // shift out_accumulator left, out_accumulator.len() = 2 * frame_size
        //       |           |          x| << hop_size
        //       |          x|           |
        self.out_accumulator.copy_within(out_frame_size.., to_index);

        // accumulate old and new
        self.out_accumulator[out_frame_size..out_frame_size + hop_size]
            .iter_mut()
            .zip(self.ifft_out[..out_frame_size - hop_size].iter())
            .for_each(|(old, new)| {
                *old = (*old + *new) / 2.0;
            });

        // plain copy of new
        self.out_accumulator[2 * out_frame_size - hop_size..]
            .copy_from_slice(&self.ifft_out[out_frame_size - hop_size..]);
        //
        println!("out_samples_len {}", out_samples.len());
        out_samples.copy_from_slice(&self.out_accumulator[to_index..out_frame_size]);
        println!("out_samples {:?}", &out_samples[..10]);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util;

    #[test]
    fn test_stretch_buffer() {
        const SAMPLE_RATE: usize = 1000;
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

        const NR_IN_SAMPLES: usize = 500;
        const NR_OUT_SAMPLES: usize = (NR_IN_SAMPLES as f32 * STRETCH_FACTOR) as usize;

        let in_samples = util::gen_sin(100.0, 1000, NR_IN_SAMPLES);
        let mut out_samples = [0.0; NR_OUT_SAMPLES];
        stretch.stretch(&in_samples, &mut out_samples);
        println!("1st iteration out {:?}", out_samples);

        stretch.stretch(&in_samples, &mut out_samples);
        println!("2nd iteration out {:?}", out_samples);

        util::write_csv(&out_samples, &"2nd.data");

        // assert_eq!(
        //     stretch.in_buffer[0..IN_LEN - NR_IN_SAMPLES],
        //     [0.0; IN_LEN - NR_IN_SAMPLES]
        // );
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
}
