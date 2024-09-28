use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use phase_vocoder::stretch::Stretch;
use phase_vocoder::track::Track;
use std::time::Duration;

fn main() {
    let host = cpal::default_host();
    let device = host.default_output_device().unwrap();
    println!("Output device: {}", device.name().unwrap());

    let mut track = Track::new(&"./amazing_grace.mp3").unwrap();

    let config = cpal::StreamConfig {
        sample_rate: cpal::SampleRate(track.sample_rate as u32),
        channels: 1,                                   // we want mono channels,
        buffer_size: cpal::BufferSize::Fixed(512 * 4), // in bytes 256 f32
    };

    let stretch_factor = 2.0;
    let frame_time_ms = 0.05; // 100ms
    println!("sample_rate {}", track.sample_rate);
    let mut stretch = Stretch::new(track.sample_rate, frame_time_ms, stretch_factor);

    println!("Config: {:?}", config);
    let err_fn = |err| eprintln!("Error:{}", err);

    let mut in_samples = [0.0f32; 10000];

    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if data.len() > (track.sample_rate as f32 * frame_time_ms) as usize {
                    println!(
                        "too big, data {} > fs * ft {}",
                        data.len(),
                        track.sample_rate as f32 * frame_time_ms
                    );
                } else {
                    let in_slice = &mut in_samples[..(data.len() as f32 / stretch_factor) as usize];
                    println!("data len {}, in_slice len {}, ", data.len(), in_slice.len());
                    in_slice
                        .iter_mut()
                        .for_each(|in_s| *in_s = track.next_sample(0).unwrap());
                    stretch.stretch(in_slice, data);
                }
            },
            err_fn,
            None,
        )
        .unwrap();

    stream.play().ok();

    std::thread::sleep(Duration::from_secs(10));
}
