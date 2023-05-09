use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use vers::{BitVectorBuilder, FastBitVector};

pub fn construct_vers_vec(rng: &mut ThreadRng, len: usize) -> FastBitVector {
    let sample = Uniform::new(0, u64::MAX);

    let mut bit_vec = BitVectorBuilder::<FastBitVector>::new();
    for _ in 0..len / 64 {
        bit_vec.append_word(sample.sample(rng));
    }

    bit_vec.build()
}
