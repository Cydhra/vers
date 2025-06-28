use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::{Duration, Instant};

mod common;

fn bench_select_iter(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("Select Iterator: Randomized Input");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let bit_vec = common::construct_vers_vec(&mut rng, l);

        group.bench_with_input(BenchmarkId::new("select queries", l), &l, |b, _| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                let mut i = 0;
                let rank1 = bit_vec.rank1(bit_vec.len());

                let start = Instant::now();
                while (i) < iters {
                    black_box(bit_vec.select1(i % rank1));
                    i += 1;
                }
                time += start.elapsed();

                time
            })
        });

        group.bench_with_input(BenchmarkId::new("select iterator", l), &l, |b, _| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                let mut i = 0;

                while i < iters {
                    let iter = bit_vec.iter1().take((iters - i) as usize);
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

        #[cfg(all(
            feature = "simd",
            target_arch = "x86_64",
            target_feature = "avx",
            target_feature = "avx2",
            target_feature = "avx512f",
            target_feature = "avx512bw",
        ))]
        group.bench_with_input(BenchmarkId::new("bitset iterator", l), &l, |b, _| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                let mut i = 0;

                while i < iters {
                    let iter = bit_vec.bit_set_iter1().take((iters - i) as usize);
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
    }
    group.finish();
}

criterion_group!(benches, bench_select_iter);
criterion_main!(benches);
