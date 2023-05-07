use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::ThreadRng;
use vers::BitVector;

fn construct_vec(rng: &mut ThreadRng, len: usize) -> BitVector {
    let sample = Uniform::new(0, u64::MAX);

    let mut bit_vec = BitVector::new();
    for _ in 0..len {
        bit_vec.append_word(sample.sample(rng));
    }

    bit_vec
}

fn bench_rank(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("rank");
    for l in [2 << 8, 2 << 10, 2 << 12, 2 << 14, 2 << 16, 2 << 18] {
        let bit_vec = construct_vec(&mut rng, l);
        let sample = Uniform::new(0, bit_vec.len());
        group.bench_with_input(format!("{} elements", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(bit_vec.rank0(e)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rank);
criterion_main!(benches);
