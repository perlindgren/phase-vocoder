use std::{
    borrow::{Borrow, BorrowMut},
    fs::File,
    ops::Mul,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Sample,
};
use phase_vocoder::stretch::{self, Stretch};
use rodio::{Decoder, Source};
use std::path::Path;
use std::time::Duration;

fn main() {
    let host = cpal::default_host();

    let device = host.default_output_device().unwrap();

    println!("Output device: {}", device.name().unwrap());

    let file = File::open(Path::new("./amazing_grace.mp3")).unwrap();
    play(&device, file);
}

fn play(device: &cpal::Device, file: File) {
    let mut sample_clock = 0;

    let source = Decoder::new(file).unwrap();
    let sample_rate = source.sample_rate();
    println!("sample_rate {}", sample_rate);
    let channels = source.channels();
    let samples = source.convert_samples::<f32>();
    let mut samples_vec = vec![];
    for sample in samples {
        samples_vec.push(sample);
    }
    let mut track = Track::new(sample_rate as usize, samples_vec, channels as usize);

    let config = cpal::StreamConfig {
        sample_rate: cpal::SampleRate(sample_rate),
        channels: 1,                                // we want mono channels,
        buffer_size: cpal::BufferSize::Fixed(1024), // 256 f32
    };

    let stretch_factor = 1.5;
    println!("sample_rate {}", sample_rate);
    let mut stretch = Stretch::new(sample_rate as usize, 0.025, stretch_factor);

    println!("Config: {:?}", config);
    let err_fn = |err| eprintln!("Error:{}", err);

    let mut in_samples = [0.0f32; 10000];

    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let in_slice = &mut in_samples[..(data.len() as f32 / stretch_factor) as usize];
                println!("data len {}, in_slice len {}, ", data.len(), in_slice.len());
                in_slice
                    .iter_mut()
                    .for_each(|in_s| *in_s = track.next_sample());
                stretch.stretch(in_slice, data);
                // write_data(data, track.channels, &mut || track.next_sample())
            },
            err_fn,
            None,
        )
        .unwrap();

    stream.play().ok();

    std::thread::sleep(Duration::from_secs(10));
}

fn write_data(output: &mut [f32], channels: usize, next_sample: &mut dyn FnMut() -> f32) {
    for frame in output.chunks_mut(channels) {
        let value: f32 = f32::from_sample(next_sample());
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}

pub struct Track {
    pub playhead: usize,    // in samples
    pub sample_rate: usize, // in Hz
    pub samples: Vec<f32>,
    pub channels: usize,
}

impl Track {
    pub fn new(sample_rate: usize, samples: Vec<f32>, channels: usize) -> Track {
        Track {
            playhead: 0,
            sample_rate,
            samples,
            channels,
        }
    }

    pub fn next_sample(&mut self) -> f32 {
        let sample = self.samples[self.playhead];
        self.playhead += self.channels;
        sample
    }

    pub fn set_playhead(&mut self, transport: Duration) {
        let secs = transport.as_secs_f32();
        let sample_idx = (secs * self.sample_rate as f32) as usize;
        self.playhead = sample_idx;
    }
}
