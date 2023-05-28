use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, BenchmarkId};
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{thread_rng, Rng};
use vers::{EliasFanoVec, FastBitVector};

fn bench_ef(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("vers elias fano");
    for l in [2 << 8, 2 << 10, 2 << 12, 2 << 14, 2 << 16, 2 << 18, 2 << 20] {
        let mut sequence = thread_rng()
            .sample_iter(Standard)
            .take(l)
            .collect::<Vec<u64>>();
        sequence.sort_unstable();
        let ef_vec = EliasFanoVec::<FastBitVector>::new(&sequence);

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
