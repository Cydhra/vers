use bio::data_structures::rank_select::RankSelect;
use bv::BitVec as BioVec;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use fid::{BitVector as FidVec, FID};
use indexed_bitvec::IndexedBits;
use rand::distributions::{Distribution, Standard, Uniform};
use rand::rngs::ThreadRng;
use rand::Rng;
use rsdict::RsDict;
use succinct::{BitVecPush, BitVector as SuccinctVec, Rank9, BitRankSupport, BinSearchSelect};
use RankSelect as BioRsVec;
use succinct::select::Select0Support;

mod common;

fn construct_rsdict_vec(rng: &mut ThreadRng, len: usize) -> RsDict {
    let mut rs_dict = RsDict::with_capacity(len * 64);
    for _ in 0..len {
        rs_dict.push(rng.gen_bool(0.5));
    }
    rs_dict
}

fn construct_bio_vec(rng: &mut ThreadRng, len: usize) -> BioRsVec {
    let mut bio_vec = BioVec::new_fill(false, len as u64);
    for i in 0..len {
        bio_vec.set(i as u64, rng.gen_bool(0.5));
    }
    // k chosen to be succinct after Cray's definition as outlined in the documentation
    BioRsVec::new(
        bio_vec,
        ((len.trailing_zeros() * len.trailing_zeros()) / 32) as usize,
    )
}

fn construct_fair_bio_vec(rng: &mut ThreadRng, len: usize) -> BioRsVec {
    let mut bio_vec = BioVec::new_fill(false, len as u64);
    for i in 0..len {
        bio_vec.set(i as u64, rng.gen_bool(0.5));
    }
    // k chosen to be a fair comparison to vers
    BioRsVec::new(bio_vec, 512 / 32)
}

fn construct_fid_vec(rng: &mut ThreadRng, len: usize) -> FidVec {
    let mut fid_vec = FidVec::new();
    for _ in 0..len {
        fid_vec.push(rng.gen_bool(0.5));
    }
    fid_vec
}

fn construct_ind_bit_vec(rng: &mut ThreadRng, len: usize) -> IndexedBits<Vec<u8>> {
    let vec = rng.sample_iter(Standard).take(len / 8).collect::<Vec<u8>>();
    IndexedBits::build_from_bytes(vec, len as u64).unwrap()
}

fn construct_rank9_vec(rng: &mut ThreadRng, len: usize) -> Rank9<SuccinctVec<u64>> {
    let mut bit_vec = SuccinctVec::with_capacity(len as u64);
    for _ in 0..len / 8 {
        bit_vec.push_block(rng.sample(Standard))
    }
    Rank9::new(bit_vec)
}

fn construct_rank9_select_vec(rng: &mut ThreadRng, len: usize) -> BinSearchSelect<Rank9<SuccinctVec<u64>>> {
    BinSearchSelect::new(construct_rank9_vec(rng, len))
}

fn compare_ranks(b: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = b.benchmark_group("Rank Comparison: Randomized Input");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let vers_vec = common::construct_vers_vec(&mut rng, l);
        let rsdict = construct_rsdict_vec(&mut rng, l);
        let bio_vec = construct_bio_vec(&mut rng, l);
        let fair_bio_vec = construct_fair_bio_vec(&mut rng, l);
        let fid_vec = construct_fid_vec(&mut rng, l);
        let ind_bit_vec = construct_ind_bit_vec(&mut rng, l);
        let rank9_vec = construct_rank9_vec(&mut rng, l);

        let sample = Uniform::new(0, l);

        group.bench_with_input(BenchmarkId::new("vers", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(vers_vec.rank0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("rsdict", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(rsdict.rank(e, false)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("bio", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(bio_vec.rank_0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("fair bio", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(fair_bio_vec.rank_0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("fid", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(fid_vec.rank0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("indexed bitvector", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(ind_bit_vec.rank_zeros(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("rank9", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(rank9_vec.rank0(e)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();

    let mut group = b.benchmark_group("Select Comparison: Randomized Input");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let vers_vec = common::construct_vers_vec(&mut rng, l);
        let rsdict = construct_rsdict_vec(&mut rng, l);
        let bio_vec = construct_bio_vec(&mut rng, l);
        let fair_bio_vec = construct_fair_bio_vec(&mut rng, l);
        let fid_vec = construct_fid_vec(&mut rng, l);
        let ind_bit_vec = construct_ind_bit_vec(&mut rng, l);
        let rank9_vec = construct_rank9_select_vec(&mut rng, l);
        let sample = Uniform::new(0, l / 3);

        group.bench_with_input(BenchmarkId::new("vers", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng),
                |e| black_box(vers_vec.select0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("rsdict", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(rsdict.select(e, false)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("bio", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(bio_vec.select_0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("fair bio", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(fair_bio_vec.select_0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("fid", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(fid_vec.select0(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("indexed bitvector", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(ind_bit_vec.select_zeros(e)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("rank9", l), &l, |b, _| {
            b.iter_batched(
                || sample.sample(&mut rng) as u64,
                |e| black_box(rank9_vec.select0(e)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, compare_ranks);
criterion_main!(benches);
