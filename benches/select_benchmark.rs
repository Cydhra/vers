use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::distributions::{Distribution, Uniform};
use vers::RsVector;

mod common;

fn bench_select(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("select");
    for l in [2 << 8, 2 << 10, 2 << 12, 2 << 14, 2 << 16, 2 << 18, 2 << 20] {
        let bit_vec = common::construct_vers_vec(&mut rng, l);
        let sample = Uniform::new(0, bit_vec.len() / 4);
        group.bench_with_input(format!("{} elements", l), &l, |b, _| {
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
