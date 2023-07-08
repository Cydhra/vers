use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{thread_rng, Rng};
use vers::EliasFanoVec;

fn bench_ef(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("vers elias fano");
    for l in [
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
    ] {
        let mut sequence = thread_rng()
            .sample_iter(Standard)
            .take(l)
            .collect::<Vec<u64>>();
        sequence.sort_unstable();
        let ef_vec = EliasFanoVec::new(&sequence);

        let pred_sample = Uniform::new(ef_vec.get(0), u64::MAX);

        group.bench_with_input(BenchmarkId::new("predecessor", l), &l, |b, _| {
            b.iter_batched(
                || pred_sample.sample(&mut rng),
                |e| black_box(ef_vec.pred(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("bin search", l), &l, |b, _| {
            b.iter_batched(
                || pred_sample.sample(&mut rng),
                |e| black_box(sequence.partition_point(|&x| x <= e) - 1),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ef);
criterion_main!(benches);
