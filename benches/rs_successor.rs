//! Benchmark compare performance of bit-vector successor call with select(rank(p) + 1).

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Uniform};

mod common;

fn bench_successor(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("Successor");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let bit_vec = common::construct_vers_vec(&mut rng, l);
        let sample = Uniform::new(0, bit_vec.len() / 4);
        group.bench_with_input(BenchmarkId::new("successor", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(bit_vec.successor1(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("select", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(bit_vec.select1(bit_vec.rank1(e) + 1)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_successor);
criterion_main!(benches);
