use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distr::StandardUniform;
use rand::{rng, RngExt};
use std::hint::black_box;
use vers_vecs::EliasFanoVec;

mod common;

fn bench_ef(b: &mut Criterion) {
    let mut rng = rng();

    let mut group = b.benchmark_group("Elias-Fano: Construction");

    for &l in common::SIZES[0..8].iter() {
        group.bench_with_input(BenchmarkId::new("construction", l), &l, |b, _| {
            b.iter_batched(
                || {
                    let mut sequence = (&mut rng)
                        .sample_iter(StandardUniform)
                        .take(l)
                        .collect::<Vec<u64>>();
                    sequence.sort_unstable();
                    sequence
                },
                |e| black_box(EliasFanoVec::from_slice(&e)),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ef);
criterion_main!(benches);
