use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

mod common;

fn bench_iter(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("Rank Comparison: Randomized Input");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let vers_vec = common::construct_vers_vec(&mut rng, l);
        let mut iter = vers_vec.iter();

        group.bench_with_input(BenchmarkId::new("vers", l), &l, |b, _| {
            b.iter(|| {
                black_box(if iter.next().is_none() {
                    iter = vers_vec.iter();
                });
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_iter);
criterion_main!(benches);
