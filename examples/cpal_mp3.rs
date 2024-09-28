use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use phase_vocoder::track::Track;
use std::time::Duration;

fn main() {
    let host = cpal::default_host();
    let device = host.default_output_device().unwrap();
    println!("Output device: {}", device.name().unwrap());

    let mut track = Track::new(&"./amazing_grace.mp3").unwrap();

    let config = cpal::StreamConfig {
        sample_rate: cpal::SampleRate(track.sample_rate as u32),
        channels: 1,                                // we want mono channels,
        buffer_size: cpal::BufferSize::Fixed(1024), // in bytes, thus 256 f32
    };
    let err_fn = |err| eprintln!("Error:{}", err);

    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                data.iter_mut()
                    .for_each(|s| *s = track.next_sample(0).unwrap());
            },
            err_fn,
            None,
        )
        .unwrap();

    stream.play().ok();

    std::thread::sleep(Duration::from_secs(5));
}
