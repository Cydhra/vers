#![allow(dead_code)]

use criterion::PlotConfiguration;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use vers_vecs::{RsVec, RsVectorBuilder};

pub const SIZES: [usize; 10] = [
    1 << 8,
    1 << 10,
    1 << 12,
    1 << 14,
    1 << 16,
    1 << 18,
    1 << 20,
    1 << 22,
    1 << 24,
    1 << 26,
];

pub fn construct_vers_vec(rng: &mut ThreadRng, len: usize) -> RsVec {
    let sample = Uniform::new(0, u64::MAX);

    let mut bit_vec = RsVectorBuilder::new();
    for _ in 0..len / 64 {
        bit_vec.append_word(sample.sample(rng));
    }

    bit_vec.build()
}

pub fn fill_random_vec(rng: &mut ThreadRng, len: usize) -> Vec<u64> {
    let sample = Uniform::new(0, u64::MAX);

    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(sample.sample(rng));
    }

    vec
}

pub fn plot_config() -> PlotConfiguration {
    PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic)
}
