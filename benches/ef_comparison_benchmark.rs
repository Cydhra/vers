use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use elias_fano::EliasFano;
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{thread_rng, Rng};
use std::time::{Duration, Instant};
use vers_vecs::EliasFanoVec;

mod common;

fn bench_ef(b: &mut Criterion) {
    let mut group = b.benchmark_group("Elias-Fano Comparison: random-access");
    group.plot_config(common::plot_config());

    let mut rng = rand::thread_rng();

    for l in common::SIZES {
        let mut sequence = thread_rng()
            .sample_iter(Standard)
            .take(l)
            .collect::<Vec<u64>>();
        sequence.sort_unstable();

        let ef_vec = EliasFanoVec::from_slice(&sequence);
        let mut comparison_ef_vec =
            EliasFano::new(sequence[sequence.len() - 1], sequence.len() as u64);
        comparison_ef_vec.compress(sequence.iter());

        let sample = Uniform::new(0, sequence.len());

        group.bench_with_input(BenchmarkId::new("vers vector", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(ef_vec.get_unchecked(e)),
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

    let mut group = b.benchmark_group("Elias-Fano Comparison: in-order-access");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let mut sequence = thread_rng()
            .sample_iter(Standard)
            .take(l)
            .collect::<Vec<u64>>();
        sequence.sort_unstable();

        let ef_vec = EliasFanoVec::from_slice(&sequence);
        let mut comparison_ef_vec =
            EliasFano::new(sequence[sequence.len() - 1], sequence.len() as u64);
        comparison_ef_vec.compress(sequence.iter());

        group.bench_with_input(BenchmarkId::new("vers vector", l), &l, |b, _| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                let mut i = 0;

                while i < iters {
                    let iter = ef_vec.iter().take((iters - i) as usize);
                    let start = Instant::now();
                    for e in iter {
                        black_box(e);
                        i += 1;
                    }
                    time += start.elapsed();
                }

                time
            })
        });

        group.bench_with_input(BenchmarkId::new("elias fano vector", l), &l, |b, _| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                let mut i = 0;

                while i < iters {
                    comparison_ef_vec.reset();
                    let start = Instant::now();
                    while comparison_ef_vec.next().is_ok() && i < iters {
                        i += 1;
                    }
                    time += start.elapsed();
                }

                time
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ef);
criterion_main!(benches);
