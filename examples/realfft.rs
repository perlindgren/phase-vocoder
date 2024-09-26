// small example of realfft

use phase_vocoder::util::*;
use realfft::{num_complex::Complex, RealFftPlanner};
use std::f32::consts::TAU;

fn main() {
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

        ifft_spectrum[to_bin as usize] = Complex::from_polar(bin_in.norm(), new_phase_principal);

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
