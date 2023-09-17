use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{thread_rng, Rng};
use sucds::mii_sequences::EliasFanoBuilder;
use vers_vecs::EliasFanoVec;

mod common;

fn bench_ef(b: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = b.benchmark_group("Elias-Fano Benchmark: Worst Case Input");
    group.plot_config(common::plot_config());

    let dist_high = Uniform::new(u64::MAX / 2 - 200, u64::MAX / 2 - 1);
    for l in common::SIZES {
        // a distribution clustered at the low end with some but not too many duplicates
        let dist_low = Uniform::new(0, l as u64);
        let query_distribution = Uniform::new(0, l);

        // prepare a uniformly distributed sequence
        let mut sequence = (&mut rng)
            .sample_iter(Standard)
            .take(l)
            .collect::<Vec<u64>>();
        sequence.sort_unstable();
        let uniform_ef_vec = EliasFanoVec::from_slice(&sequence);

        // query random values from the actual sequences, to be equivalent to the worst case
        // benchmark below
        group.bench_with_input(BenchmarkId::new("vers uniform input", l), &l, |b, _| {
            b.iter_batched(
                || sequence[query_distribution.sample(&mut rng)],
                |e| black_box(uniform_ef_vec.predecessor_unchecked(e)),
                BatchSize::SmallInput,
            )
        });
        drop(uniform_ef_vec);

        // prepare a sequence of low values with a few high values at the end
        let mut sequence = (&mut rng)
            .sample_iter(dist_low)
            .take(l - 100)
            .collect::<Vec<u64>>();
        sequence.sort_unstable();
        let mut sequence_top = (&mut rng)
            .sample_iter(dist_high)
            .take(100)
            .collect::<Vec<u64>>();
        sequence_top.sort_unstable();
        sequence.append(&mut sequence_top);

        let bad_ef_vec = EliasFanoVec::from_slice(&sequence);
        // query random values from the actual sequences, to force long searches in the lower vec
        group.bench_with_input(BenchmarkId::new("vers clustered input", l), &l, |b, _| {
            b.iter_batched(
                || sequence[query_distribution.sample(&mut rng)],
                |e| black_box(bad_ef_vec.predecessor_unchecked(e)),
                BatchSize::SmallInput,
            )
        });
        drop(bad_ef_vec);

        let mut sucds_ef_vec =
            EliasFanoBuilder::new(*sequence.last().unwrap() as usize + 1, sequence.len())
                .expect("Failed to create sucds Elias-Fano builder");
        sucds_ef_vec
            .extend(sequence.iter().map(|e| *e as usize))
            .expect("Failed to extend sucds Elias-Fano builder");
        let sucds_ef_vec = sucds_ef_vec.build().enable_rank();
        group.bench_with_input(BenchmarkId::new("sucds clustered input", l), &l, |b, _| {
            b.iter_batched(
                || query_distribution.sample(&mut rng),
                |e| black_box(sucds_ef_vec.predecessor(e)),
                BatchSize::SmallInput,
            )
        });
        drop(sucds_ef_vec);

        group.bench_with_input(BenchmarkId::new("bin search", l), &l, |b, _| {
            b.iter_batched(
                || sequence[query_distribution.sample(&mut rng)],
                |e| black_box(sequence.partition_point(|&x| x <= e) - 1),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ef);
criterion_main!(benches);
