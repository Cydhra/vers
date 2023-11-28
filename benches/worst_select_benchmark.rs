use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::thread_rng;
use rsdict::RsDict;
// use sucds::bit_vectors::darray::DArray as SucDArray;
// use sucds::bit_vectors::{BitVector as SucBitVec, Select};
use vers_vecs::{BitVec, RsVec};

mod common;

fn bench_ef(b: &mut Criterion) {
    let mut rng = thread_rng();
    let mut group = b.benchmark_group("Select Benchmark: Worst Case Input");
    group.plot_config(common::plot_config());

    for order_of_magnitude in [14, 16, 18, 20, 22, 24, 26] {
        let length = 1 << order_of_magnitude;

        // uniformly distributed sequence
        let bit_vec = common::construct_vers_vec(&mut rng, length);
        group.bench_with_input(
            BenchmarkId::new("uniform input", length),
            &length,
            |b, _| b.iter(|| black_box(bit_vec.select1((1 << 13) - 1))),
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

        let mut rs_dict = RsDict::with_capacity(length / 64);
        for _ in 0..(1usize << 13) - 1 {
            rs_dict.push(true);
        }
        rs_dict.push(false);

        for _ in 0..(length - (1 << 13)) - 1 {
            rs_dict.push(false);
        }
        rs_dict.push(true);
        rs_dict.push(true);
        rs_dict.push(false);
        rs_dict.push(false);

        group.bench_with_input(
            BenchmarkId::new("rsdict worst case input", length),
            &length,
            |b, _| b.iter(|| black_box(rs_dict.select1((1 << 13) - 1))),
        );
        drop(rs_dict);

        // the following benchmark does not trigger the worst-case for sucd reliably,
        // because it utilizes compressed arrays.
        // TODO: find out how to trigger a worst-case for sucd (if there is one)

        // let mut suc_bv = SucBitVec::with_capacity(length / 64);
        // for _ in 0..(1usize << 13) / 64 - 1 {
        //     suc_bv
        //         .push_bits(u64::MAX as usize, 64)
        //         .expect("push_bits failed");
        // }
        // suc_bv
        //     .push_bits((u64::MAX >> 1) as usize, 64)
        //     .expect("push_bits failed");
        //
        // for _ in 0..(length - (1 << 13)) / 64 - 1 {
        //     suc_bv.push_bits(0, 64).expect("push_bits failed");
        // }
        // suc_bv.push_bits(2, 64).expect("push_bits failed");
        //
        // let sucds_darray = SucDArray::from_bits(suc_bv.iter()).enable_rank();
        // group.bench_with_input(BenchmarkId::new("sucds worst case input", length), &length, |b, _| {
        //     b.iter(|| black_box(sucds_darray.select1((1 << 13) - 1)))
        // });
        // drop(sucds_darray);
    }
    group.finish();
}

criterion_group!(benches, bench_ef);
criterion_main!(benches);
