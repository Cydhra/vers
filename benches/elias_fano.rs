use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{thread_rng, Rng};
use vers_vecs::EliasFanoVec;

mod common;

fn bench_ef(b: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = b.benchmark_group("Elias-Fano: Randomized Input");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let mut sequence = (&mut rng)
            .sample_iter(Standard)
            .take(l)
            .collect::<Vec<u64>>();
        sequence.sort_unstable();
        let ef_vec = EliasFanoVec::from_slice(&sequence);
        let pred_sample = Uniform::new(ef_vec.get_unchecked(0), sequence.last().unwrap());

        group.bench_with_input(BenchmarkId::new("predecessor", l), &l, |b, _| {
            b.iter_batched(
                || pred_sample.sample(&mut rng),
                |e| black_box(ef_vec.predecessor_unchecked(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("successor", l), &l, |b, _| {
            b.iter_batched(
                || pred_sample.sample(&mut rng),
                |e| black_box(ef_vec.successor_unchecked(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("rank", l), &l, |b, _| {
            b.iter_batched(
                || pred_sample.sample(&mut rng),
                |e| black_box(ef_vec.rank(e)),
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
