use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distr::Distribution;
use rand::distr::Uniform;
use std::hint::black_box;

mod common;

fn bench_select(b: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = b.benchmark_group("Select: Randomized Input");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let bit_vec = common::construct_vers_vec(&mut rng, l);
        let sample = Uniform::new(0, bit_vec.len() / 4).unwrap();
        group.bench_with_input(BenchmarkId::new("select", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(bit_vec.select0(e)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_select);
criterion_main!(benches);
