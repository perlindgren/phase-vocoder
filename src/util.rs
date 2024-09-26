// utilities

use std::{f32::consts::TAU, fmt::Display, fs::*, io::prelude::*, path::*};

pub fn gen_sin(f: f32, fs: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|k| (k as f32 * f * TAU / fs as f32).sin())
        .collect()
}

pub fn write_csv<T>(data: &[T], path: &impl AsRef<Path>)
where
    T: Display,
{
    let mut file = File::create(path).unwrap();

    data.iter().for_each(|k| write!(file, "{},", k).unwrap());
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_gen_sin() {
        let sin = gen_sin(10_000.0, 48000, 1000);

        write_csv(&sin, &"sin_f32.data");
    }

    #[test]
    fn push() {
        #[allow(clippy::vec_init_then_push)]
        let mut v = vec![];
        v.push(3);
        v.push(2);
        v.push(1);
        v.push(0);
        assert_eq!(v, [3, 2, 1, 0]);
    }
}
