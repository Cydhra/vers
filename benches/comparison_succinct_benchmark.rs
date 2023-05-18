use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Standard, Uniform};
use rand::rngs::ThreadRng;
use rand::Rng;
use rsdict::RsDict;
use succinct::{BitRankSupport, BitVecPush, BitVector as SuccinctVec, Rank9};
use vers::RsVector;

mod common;

fn construct_rsdict_vec(rng: &mut ThreadRng, len: usize) -> RsDict {
    let mut rs_dict = RsDict::with_capacity(len * 64);
    for _ in 0..len {
        rs_dict.push(rng.gen_bool(0.5));
    }
    rs_dict
}

fn construct_rank9_vec(rng: &mut ThreadRng, len: usize) -> Rank9<SuccinctVec<u64>> {
    let mut bit_vec = SuccinctVec::with_capacity(len as u64);
    for _ in 0..len / 8 {
        bit_vec.push_block(rng.sample(Standard))
    }
    Rank9::new(bit_vec)
}

fn compare_ranks(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("rank-against-succinct");
    // higher measurement time for reduced noise
    group.measurement_time(std::time::Duration::from_secs(10));

    for l in [2 << 8, 2 << 10, 2 << 12, 2 << 14, 2 << 16, 2 << 18] {
        let vers_vec = common::construct_vers_vec(&mut rng, l);
        let rank9_vec = construct_rank9_vec(&mut rng, l);
        let rsdict = construct_rsdict_vec(&mut rng, l);

        let sample = Uniform::new(0, l);

        group.bench_with_input(BenchmarkId::new("vers-rank", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(vers_vec.rank0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("succinct-rank9-rank", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(rank9_vec.rank0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("rsdict-rank", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(rsdict.rank(e, false)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, compare_ranks);
criterion_main!(benches);
