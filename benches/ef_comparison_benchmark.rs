use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use elias_fano::EliasFano;
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{thread_rng, Rng};
use std::time::{Duration, Instant};
use sucds::mii_sequences::EliasFanoBuilder;
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
        let sample = Uniform::new(0, sequence.len());

        let ef_vec = EliasFanoVec::from_slice(&sequence);
        group.bench_with_input(BenchmarkId::new("vers vector", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(ef_vec.get_unchecked(e)),
                BatchSize::SmallInput,
            )
        });
        drop(ef_vec);

        let mut elias_fano_vec =
            EliasFano::new(sequence[sequence.len() - 1], sequence.len() as u64);
        elias_fano_vec.compress(sequence.iter());
        group.bench_with_input(BenchmarkId::new("elias fano vector", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(elias_fano_vec.visit(e as u64)),
                BatchSize::SmallInput,
            )
        });
        drop(elias_fano_vec);

        let mut sucds_ef_vec =
            EliasFanoBuilder::new(*sequence.last().unwrap() as usize + 1, sequence.len())
                .expect("Failed to create sucds Elias-Fano builder");

        sucds_ef_vec
            .extend(sequence.iter().map(|e| *e as usize))
            .expect("Failed to extend sucds Elias-Fano builder");
        let sucds_ef_vec = sucds_ef_vec.build();
        group.bench_with_input(
            BenchmarkId::new("sucds elias fano vector", l),
            &l,
            |b, _| {
                b.iter_batched(
                    || sample.sample(&mut rng),
                    |e| black_box(sucds_ef_vec.select(e)),
                    BatchSize::SmallInput,
                )
            },
        );
        drop(sucds_ef_vec);
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
        drop(ef_vec);

        let mut elias_fano_vec =
            EliasFano::new(sequence[sequence.len() - 1], sequence.len() as u64);
        elias_fano_vec.compress(sequence.iter());
        group.bench_with_input(BenchmarkId::new("elias fano vector", l), &l, |b, _| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                let mut i = 0;

                while i < iters {
                    elias_fano_vec.reset();
                    let start = Instant::now();
                    while elias_fano_vec.next().is_ok() && i < iters {
                        i += 1;
                    }
                    time += start.elapsed();
                }

                time
            })
        });
        drop(elias_fano_vec);

        let mut sucds_ef_vec =
            EliasFanoBuilder::new(*sequence.last().unwrap() as usize + 1, sequence.len())
                .expect("Failed to create sucds Elias-Fano builder");
        sucds_ef_vec
            .extend(sequence.iter().map(|e| *e as usize))
            .expect("Failed to extend sucds Elias-Fano builder");
        let sucds_ef_vec = sucds_ef_vec.build();
        group.bench_with_input(
            BenchmarkId::new("sucds elias fano vector", l),
            &l,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    let mut i = 0;

                    while i < iters {
                        let iter = sucds_ef_vec.iter(0).take((iters - i) as usize);
                        let start = Instant::now();
                        for e in iter {
                            black_box(e);
                            i += 1;
                        }
                        time += start.elapsed();
                    }

                    time
                })
            },
        );
        drop(sucds_ef_vec);
    }
    group.finish();
}

criterion_group!(benches, bench_ef);
criterion_main!(benches);
