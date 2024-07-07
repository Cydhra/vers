use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::distributions::Standard;
use rand::{thread_rng, Rng};

use vers_vecs::EliasFanoVec;

mod common;

fn bench_ef(b: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = b.benchmark_group("Elias-Fano: Iteration");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let mut sequence = (&mut rng)
            .sample_iter(Standard)
            .take(l)
            .collect::<Vec<u64>>();
        sequence.sort_unstable();
        let ef_vec = EliasFanoVec::from_slice(&sequence);

        group.bench_with_input(BenchmarkId::new("manual indexing", l), &l, |b, _| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                let mut i = 0;

                let start = Instant::now();
                while i < iters {
                    black_box(ef_vec.get_unchecked(i as usize % l));
                    i += 1;
                }
                time += start.elapsed();

                time
            })
        });

        group.bench_with_input(BenchmarkId::new("iterator", l), &l, |b, _| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                let mut i = 0;

                while i < iters {
                    let start = Instant::now();
                    for e in ef_vec.iter().take((iters - i) as usize) {
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

criterion_group!(benches, bench_ef);
criterion_main!(benches);
