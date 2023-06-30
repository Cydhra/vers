#![allow(dead_code)]

use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use vers::{FastBitVector, RsVectorBuilder};

pub fn construct_vers_vec(rng: &mut ThreadRng, len: usize) -> FastBitVector {
    let sample = Uniform::new(0, u64::MAX);

    let mut bit_vec = RsVectorBuilder::<FastBitVector>::new();
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