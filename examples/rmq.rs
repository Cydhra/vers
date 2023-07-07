use rand::distributions::{Distribution, Uniform};
use rand::{thread_rng, Rng};
use vers::rmq::fast_rmq::FastRmq;

const LEN: usize = 1 << 24;

fn main() {
    let sample = Uniform::new(0, u64::MAX);
    let mut rng = thread_rng();

    let mut data = Vec::with_capacity(LEN);
    for _ in 0..LEN {
        data.push(sample.sample(&mut rng));
    }

    let rmq = FastRmq::new(data);
    let sample = Uniform::new(0, rmq.len());
    loop {
        let begin = sample.sample(&mut rng);
        let end = begin + rng.gen_range(0..rmq.len() - begin);
        rmq.range_min(begin, end);
    }
}
