use std::cmp::min_by;

struct SmallNaiveRmq {
    data: Vec<u64>,
    results: Vec<Vec<usize>>,
}

impl SmallNaiveRmq {
    fn new(data: Vec<u64>) -> Self {
        // todo: can we flatten this in a one-dimensional vector of predicted size?
        //  this /might/ save us cache misses later
        let mut results = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            results.push(Vec::with_capacity(
                (data.len() - i).next_power_of_two().trailing_zeros() as usize,
            ));
            results[i].push(i);
        }

        for i in 0..data.len().next_power_of_two().trailing_zeros() {
            let i = i as usize;
            for j in 0..data.len() {
                let offset = 1 << i;
                let arg_min = if j + offset < data.len() {
                    if data[results[j][i]] < data[results[j + offset][i]] {
                        results[j][i]
                    } else {
                        results[j + offset][i]
                    }
                } else {
                    if data[results[j][i]] < data[results[data.len() - offset - 1][i + 1]] {
                        results[j][i]
                    } else {
                        results[data.len() - offset - 1][i + 1]
                    }
                };

                results[j].push(arg_min);
            }
        }

        Self { data, results }
    }

    fn range_min(&self, i: usize, j: usize) -> usize {
        let log_dist = (usize::BITS - (j - i).leading_zeros()).saturating_sub(1) as usize;
        let dist = (1 << log_dist) - 1;
        min_by(
            self.results[i][log_dist],
            self.results[j - dist][log_dist],
            |a, b| self.data[*a].cmp(&self.data[*b]),
        )
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn small_naive_rmq_test() {
        let rmq = super::SmallNaiveRmq::new(vec![9, 6, 10, 4, 0, 8, 3, 7, 1, 2, 5]);

        assert_eq!(rmq.range_min(0, 0), 0);
        assert_eq!(rmq.range_min(0, 1), 1);
        assert_eq!(rmq.range_min(0, 2), 1);
        assert_eq!(rmq.range_min(0, 3), 3);
        assert_eq!(rmq.range_min(5, 8), 8);
        assert_eq!(rmq.range_min(5, 9), 8);
        assert_eq!(rmq.range_min(9, 10), 9);
        assert_eq!(rmq.range_min(0, 10), 4);
    }
}
