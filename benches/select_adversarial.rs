use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use vers_vecs::{BitVec, RsVec};

mod common;

fn select_worst_case(b: &mut Criterion) {
    let mut rng = thread_rng();
    let mut group = b.benchmark_group("Select: Adversarial Input");
    group.plot_config(common::plot_config());

    for order_of_magnitude in [14, 16, 18, 20, 22, 24, 26] {
        let length = 1 << order_of_magnitude;

        // uniformly distributed sequence
        let bit_vec = common::construct_vers_vec(&mut rng, length);
        let uniform_sample = Uniform::new(
            bit_vec.rank0(bit_vec.len()) / 4 * 3,
            bit_vec.rank0(bit_vec.len()),
        );
        group.bench_with_input(
            BenchmarkId::new("uniform input", length),
            &length,
            |b, _| {
                b.iter_batched(
                    || rng.sample(uniform_sample),
                    |e| black_box(bit_vec.select1(e)),
                    BatchSize::SmallInput,
                )
            },
        );
        drop(bit_vec);

        // construct a vector with only one select block and put its last one bit at the end
        // of the vector

        let mut bit_vec = BitVec::with_capacity(length / 64);
        for _ in 0..(1usize << 13) / 64 - 1 {
            bit_vec.append_word(u64::MAX);
        }
        bit_vec.append_word(u64::MAX >> 1);

        for _ in 0..(length - (1 << 13)) / 64 - 1 {
            bit_vec.append_word(0);
        }
        bit_vec.append_word(2);
        let bit_vec = RsVec::from_bit_vec(bit_vec);

        group.bench_with_input(
            BenchmarkId::new("worst case input", length),
            &length,
            |b, _| b.iter(|| black_box(bit_vec.select1((1 << 13) - 1))),
        );
        drop(bit_vec);
    }
    group.finish();
}

criterion_group!(benches, select_worst_case);
criterion_main!(benches);
