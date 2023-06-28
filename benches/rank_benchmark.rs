use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Uniform};
use vers::RsVector;

mod common;

fn bench_rank(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("vers rank");
    for l in [2 << 8, 2 << 10, 2 << 12, 2 << 14, 2 << 16, 2 << 18, 2 << 20] {
        let bit_vec = common::construct_vers_vec(&mut rng, l);
        let sample = Uniform::new(0, bit_vec.len());
        group.bench_with_input(BenchmarkId::new("rank", l), &l, |b, _| {
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
