use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use elias_fano::EliasFano;
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{thread_rng, Rng};
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
}

criterion_group!(benches, bench_ef);
criterion_main!(benches);
