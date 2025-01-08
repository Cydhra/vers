use criterion::measurement::{Measurement, ValueFormatter};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::ThreadRng;
use rand::seq::index::sample;
use rand::Rng;
use std::time::Instant;
use vers_vecs::{BitVec, RsVec};

mod common;

pub const SIZES: [usize; 7] = [
    1 << 14,
    1 << 16,
    1 << 18,
    1 << 20,
    1 << 22,
    1 << 24,
    1 << 26,
];

/// How full the vector is filled with ones.
const FILL_FACTORS: [f64; 6] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5];

/// Generate a bitvector with `fill_factors` percent ones at random positions
fn generate_vector_with_fill(rng: &mut ThreadRng, len: usize, fill_factor: f64) -> BitVec {
    let mut bit_vec1 = BitVec::from_zeros(len);

    // flip exactly fill-factor * len bits so the equality check is not trivial
    sample(rng, len, (fill_factor * len as f64) as usize)
        .iter()
        .for_each(|i| {
            bit_vec1.flip_bit(i);
        });

    bit_vec1
}

fn bench(b: &mut Criterion<TimeDiff>) {
    let mut rng = rand::thread_rng();

    for len in SIZES {
        let mut group = b.benchmark_group(format!("Equals Benchmark: {}", len));
        group.plot_config(common::plot_config());

        for fill_factor in FILL_FACTORS {
            group.bench_with_input(
                BenchmarkId::new("sparse overhead equal", fill_factor),
                &fill_factor,
                |b, _| {
                    b.iter_custom(|iters| {
                        let mut time_diff = TimeDiff.zero();

                        for _ in 0..iters {
                            let vec = generate_vector_with_fill(&mut rng, len, fill_factor);
                            let vec = RsVec::from_bit_vec(vec);

                            let start_full = TimeDiff.start();
                            black_box(vec.full_equals(&vec));
                            time_diff -= TimeDiff.end(start_full);

                            let start_sparse = TimeDiff.start();
                            black_box(vec.sparse_equals::<false>(&vec));
                            time_diff += TimeDiff.end(start_sparse);
                        }

                        time_diff
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("sparse overhead unequal", fill_factor),
                &fill_factor,
                |b, _| {
                    b.iter_custom(|iters| {
                        let mut time_diff = TimeDiff.zero();

                        for _ in 0..iters {
                            let vec = generate_vector_with_fill(&mut rng, len, fill_factor);
                            let mut vec2 = vec.clone();
                            let vec = RsVec::from_bit_vec(vec);

                            vec2.flip_bit(vec.select1(vec.rank1(len) - 1));
                            vec2.flip_bit(vec.select0(rng.gen_range(0..(vec.rank0(len) - 1))));
                            let vec2 = RsVec::from_bit_vec(vec2);

                            let start_full = TimeDiff.start();
                            black_box(vec.full_equals(&vec2));
                            time_diff -= TimeDiff.end(start_full);

                            let start_sparse = TimeDiff.start();
                            black_box(vec.sparse_equals::<false>(&vec2));
                            time_diff += TimeDiff.end(start_sparse);
                        }

                        time_diff
                    });
                },
            );
        }

        group.finish();
    }
}

/// Measurement for differential time measurements.
struct TimeDiff;

impl Measurement for TimeDiff {
    type Intermediate = Instant;
    type Value = isize;

    fn start(&self) -> Self::Intermediate {
        Instant::now()
    }

    fn end(&self, i: Self::Intermediate) -> Self::Value {
        i.elapsed().as_nanos() as isize
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        v1 + v2
    }

    fn zero(&self) -> Self::Value {
        0
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        *value as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &NanoSecondFormatter
    }
}

struct NanoSecondFormatter;

impl ValueFormatter for NanoSecondFormatter {
    fn format_value(&self, value: f64) -> String {
        let absolute = value.abs();
        if absolute < 1.0 {
            // ns = time in nanoseconds per iteration
            format!("{:.2} ps", value * 1e3)
        } else if absolute < 10f64.powi(3) {
            format!("{:.2} ns", value)
        } else if absolute < 10f64.powi(6) {
            format!("{:.2} us", value / 1e3)
        } else if absolute < 10f64.powi(9) {
            format!("{:.2} ms", value / 1e6)
        } else {
            format!("{:.2} s", value / 1e9)
        }
    }

    fn format_throughput(&self, _throughput: &Throughput, _value: f64) -> String {
        unimplemented!("throughput formatting not supported")
    }

    fn scale_values(&self, _typical_value: f64, _values: &mut [f64]) -> &'static str {
        "ns"
    }

    fn scale_throughputs(
        &self,
        _typical_value: f64,
        _throughput: &Throughput,
        _values: &mut [f64],
    ) -> &'static str {
        unimplemented!("throughput scaling not supported")
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "ns"
    }
}

fn differential_measuring() -> Criterion<TimeDiff> {
    Criterion::default().with_measurement(TimeDiff)
}

criterion_group! {
    name = benches;
    config=differential_measuring();
    targets = bench
}
criterion_main!(benches);
