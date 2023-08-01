use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use vers_vecs::rmq::fast_rmq::FastRmq;

mod common;

fn bench_rmq(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("RMQ Benchmark: Randomized Input");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let rmq = FastRmq::new(common::fill_random_vec(&mut rng, l));
        let sample = Uniform::new(0, rmq.len());
        group.bench_with_input(BenchmarkId::new("range_min", l), &l, |b, _| {
            b.iter_batched(
                || {
                    let begin = sample.sample(&mut rng);
                    let end = begin + rng.gen_range(0..rmq.len() - begin);
                    (begin, end)
                },
                |e| black_box(rmq.range_min(e.0, e.1)),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_rmq);
criterion_main!(benches);
