use anyhow::Result;
use rodio::{Decoder, Source};
use std::{fs::File, path::Path, time::Duration};

pub struct Track {
    pub playhead: usize,    // in samples
    pub sample_rate: usize, // in Hz
    pub channels: usize,
    pub samples: Vec<Vec<f32>>,
}

impl Track {
    pub fn new(path: &impl AsRef<Path>) -> Result<Track> {
        let file = File::open(path)?;
        let source = Decoder::new(file)?;
        let sample_rate = source.sample_rate() as usize;
        let channels = source.channels() as usize;
        println!("sample_rate {}, channels {}", sample_rate, channels);
        let samples_vec: Vec<f32> = source.convert_samples::<f32>().collect(); // all samples
        let samples: Vec<Vec<f32>> = samples_vec.chunks(channels).map(Vec::from).collect();
        let playhead = 0;

        Ok(Track {
            playhead,
            sample_rate,
            channels,
            samples,
        })
    }

    // Returns Some<f32> or None, in case channel, playhead out of range
    // TODO: should perhaps implement iterator, but we keep it simple for now
    pub fn next_sample(&mut self, channel: usize) -> Option<f32> {
        let sample_chunk = &self.samples.get(self.playhead)?;
        self.playhead += 1;
        sample_chunk.get(channel).copied()
    }

    pub fn set_playhead(&mut self, transport: Duration) {
        let secs = transport.as_secs_f32();
        let sample_idx = (secs * self.sample_rate as f32) as usize;
        self.playhead = sample_idx;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_track() {
        let mut track = Track::new(&"./amazing_grace.mp3").unwrap();
        let mut l = 0;
        while track.next_sample(0).is_some() {
            l += 1;
        }
        assert_eq!(l / track.sample_rate, 25); // amazing grace is a horrifying 25 seconds
    }
}
