use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::ThreadRng;
use rand::Rng;
use rsdict::RsDict;
use vers::RsVector;

mod common;

fn construct_rsdict_vec(rng: &mut ThreadRng, len: usize) -> RsDict {
    let mut rs_dict = RsDict::with_capacity(len * 64);
    for _ in 0..len {
        rs_dict.push(rng.gen_bool(0.5));
    }
    rs_dict
}

fn compare_ranks(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("comparison");

    for l in [2 << 8, 2 << 10, 2 << 12, 2 << 14, 2 << 16, 2 << 18] {
        let vers_vec = common::construct_vers_vec(&mut rng, l);
        let rsdict = construct_rsdict_vec(&mut rng, l);
        let sample = Uniform::new(0, l);

        group.bench_with_input(BenchmarkId::new("vers-rank", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(vers_vec.rank0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("rsdict-rank", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(rsdict.rank(e, false)),
                BatchSize::SmallInput,
            )
        });

        let sample = Uniform::new(0, l / 3);

        group.bench_with_input(BenchmarkId::new("vers-select", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(vers_vec.select0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("rsdict-select", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(rsdict.select(e, false)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, compare_ranks);
criterion_main!(benches);
