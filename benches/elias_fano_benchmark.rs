use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::distributions::{Distribution, Standard};
use rand::{Rng, thread_rng};
use vers::{EliasFanoVec, FastBitVector};

fn bench_ef(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("predecessor");
    for l in [2 << 8, 2 << 10, 2 << 12, 2 << 14, 2 << 16, 2 << 18, 2 << 20] {
        let ef_vec = EliasFanoVec::<FastBitVector>::new(&thread_rng().sample_iter(Standard).take(l).collect::<Vec<u64>>());

        group.bench_with_input(format!("{} elements", l), &l, |b, _| {
            b.iter_batched(
                || Standard.sample(&mut rng),
                |e| black_box(ef_vec.pred(e)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ef);
criterion_main!(benches);
