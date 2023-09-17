use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{thread_rng, Rng};
use sucds::mii_sequences::EliasFanoBuilder;
use vers_vecs::EliasFanoVec;

mod common;

fn bench_ef(b: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = b.benchmark_group("Elias-Fano Benchmark: Randomized Input");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let mut sequence = (&mut rng)
            .sample_iter(Standard)
            .take(l)
            .collect::<Vec<u64>>();
        sequence.sort_unstable();

        let pred_sample = Uniform::new(sequence.first().unwrap(), sequence.last().unwrap());

        let ef_vec = EliasFanoVec::from_slice(&sequence);
        group.bench_with_input(BenchmarkId::new("vers predecessor", l), &l, |b, _| {
            b.iter_batched(
                || pred_sample.sample(&mut rng),
                |e| black_box(ef_vec.predecessor_unchecked(e)),
                BatchSize::SmallInput,
            )
        });
        drop(ef_vec);

        let mut sucds_ef_vec =
            EliasFanoBuilder::new(*sequence.last().unwrap() as usize + 1, sequence.len())
                .expect("Failed to create sucds Elias-Fano builder");
        sucds_ef_vec
            .extend(sequence.iter().map(|e| *e as usize))
            .expect("Failed to extend sucds Elias-Fano builder");
        let sucds_ef_vec = sucds_ef_vec.build().enable_rank();
        group.bench_with_input(BenchmarkId::new("sucds predecessor", l), &l, |b, _| {
            b.iter_batched(
                || pred_sample.sample(&mut rng),
                |e| black_box(sucds_ef_vec.predecessor(e as usize)),
                BatchSize::SmallInput,
            )
        });
        drop(sucds_ef_vec);

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
