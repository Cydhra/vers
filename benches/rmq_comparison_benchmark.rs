use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use librualg::segment_tree::RmqMin;
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{thread_rng, Rng};

mod common;

fn bench_rmq(b: &mut Criterion) {
    let mut group = b.benchmark_group("RMQ Comparison: Randomized Input");
    let mut rng = rand::thread_rng();

    for l in common::SIZES {
        let sequence = thread_rng()
            .sample_iter(Standard)
            .take(l)
            .collect::<Vec<u64>>();
        let sample = Uniform::new(0, sequence.len());

        let rmq = vers::FastRmq::new(sequence.clone());
        let ru_rmq = RmqMin::new(&sequence.iter().map(|x| *x as usize).collect::<Vec<_>>());
        let creates_rmq = range_minimum_query::Rmq::from_iter(sequence);

        group.bench_with_input(BenchmarkId::new("vers", l), &l, |b, _| {
            b.iter_batched(
                || {
                    let a = sample.sample(&mut rng);
                    let b = sample.sample(&mut rng);
                    if a < b {
                        (a, b)
                    } else {
                        (b, a)
                    }
                },
                |e| black_box(rmq.range_min(e.0, e.1)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("librualg", l), &l, |b, _| {
            b.iter_batched(
                || {
                    let a = sample.sample(&mut rng);
                    let b = sample.sample(&mut rng);
                    if a < b {
                        (a, b)
                    } else {
                        (b, a)
                    }
                },
                |e| black_box(ru_rmq.query(e.0, e.1)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("crates rmq", l), &l, |b, _| {
            b.iter_batched(
                || {
                    let a = sample.sample(&mut rng);
                    let b = sample.sample(&mut rng);
                    if a < b {
                        (a, b)
                    } else {
                        (b, a)
                    }
                },
                |e| black_box(creates_rmq.range_minimum(e.0..=e.1)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rmq);
criterion_main!(benches);
