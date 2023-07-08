use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use elias_fano::EliasFano;
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{thread_rng, Rng};
use vers::EliasFanoVec;

mod common;

fn bench_ef(b: &mut Criterion) {
    let mut group = b.benchmark_group("random-access");
    let mut rng = rand::thread_rng();

    for l in common::SIZES {
        let mut sequence = thread_rng()
            .sample_iter(Standard)
            .take(l)
            .collect::<Vec<u64>>();
        sequence.sort_unstable();

        let ef_vec = EliasFanoVec::new(&sequence);
        let mut comparison_ef_vec =
            EliasFano::new(sequence[sequence.len() - 1], sequence.len() as u64);
        comparison_ef_vec.compress(sequence.iter());

        let sample = Uniform::new(0, sequence.len());

        group.bench_with_input(BenchmarkId::new("vers vector", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(ef_vec.get(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("elias fano vector", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(comparison_ef_vec.visit(e as u64)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();

    let mut group = b.benchmark_group("in-order access");
    for l in [2 << 8, 2 << 10, 2 << 12, 2 << 14, 2 << 16, 2 << 18, 2 << 20] {
        let mut sequence = thread_rng()
            .sample_iter(Standard)
            .take(l)
            .collect::<Vec<u64>>();
        sequence.sort_unstable();

        let ef_vec = EliasFanoVec::new(&sequence);
        let mut comparison_ef_vec =
            EliasFano::new(sequence[sequence.len() - 1], sequence.len() as u64);
        comparison_ef_vec.compress(sequence.iter());

        group.bench_with_input(BenchmarkId::new("vers vector", l), &l, |b, _| {
            b.iter(|| {
                for idx in 0..l {
                    black_box(ef_vec.get(idx));
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("elias fano vector", l), &l, |b, _| {
            b.iter(|| {
                comparison_ef_vec.reset();
                for _ in 0..l - 1 {
                    black_box(comparison_ef_vec.next().unwrap());
                }
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ef);
criterion_main!(benches);
