use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::distributions::{Distribution, Uniform};
use vers::BitVector;

fn bench_rank(b: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let sample = Uniform::new(0, u64::MAX);

    let mut group = b.benchmark_group("rank");
    for l in [2 << 8, 2 << 10, 2 << 12, 2 << 14, 2 << 16, 2 << 18] {
        let mut bit_vec = BitVector::new();
        for _ in 0..l {
            bit_vec.append_word(sample.sample(&mut rng));
        }

        let sample = Uniform::new(0, bit_vec.len() as usize);
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
