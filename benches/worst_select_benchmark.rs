use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::thread_rng;
use vers::RsVectorBuilder;

mod common;

fn bench_ef(b: &mut Criterion) {
    let mut rng = thread_rng();
    let mut group = b.benchmark_group("vers select worst case");
    group.plot_config(common::plot_config());

    for l in [
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 24,
        1 << 26,
    ] {
        // uniformly distributed sequence
        let bit_vec = common::construct_vers_vec(&mut rng, l);
        group.bench_with_input(BenchmarkId::new("select uniform", l), &l, |b, _| {
            b.iter(|| black_box(bit_vec.select1((1 << 13) - 1)))
        });
        drop(bit_vec);

        // construct a vector with only one select block and put its last one bit at the end
        // of the vector
        let mut bit_vec_builder = RsVectorBuilder::with_capacity(l / 64);
        for _ in 0..(1usize << 13) / 64 - 1 {
            bit_vec_builder.append_word(u64::MAX);
        }
        bit_vec_builder.append_word(u64::MAX >> 1);

        for _ in 0..(l - (1 << 13)) / 64 - 1 {
            bit_vec_builder.append_word(0);
        }
        bit_vec_builder.append_word(2);
        let bit_vec = bit_vec_builder.build();

        group.bench_with_input(BenchmarkId::new("select bad", l), &l, |b, _| {
            b.iter(|| black_box(bit_vec.select1((1 << 13) - 1)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ef);
criterion_main!(benches);
