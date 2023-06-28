use std::cmp::min_by;

struct SmallNaiveRmq {
    data: Vec<u64>,
    results: Vec<Vec<usize>>,
}

impl SmallNaiveRmq {
    fn new(data: Vec<u64>) -> Self {
        // todo: can we flatten this in a one-dimensional vector of predicted size?
        //  this /might/ save us cache misses later
        let mut results = vec![Vec::new(); data.len()];

        // todo using a dynamic programming approach, this can be done in O(n log n) time instead of O(n^2)
        // only store each 2^n-th element in the results
        for i in 0..data.len() {
            let mut min = i;
            for j in i..data.len() {
                if data[j] < data[min] {
                    min = j;
                }

                if (j - i).count_ones() == 1 {
                    results[i].push(min);
                }
            }
        }

        Self { data, results }
    }

    fn range_min(&self, i: usize, j: usize) -> usize {
        if j - i == 0 { return i }
        let log_dist = (usize::BITS - 1 - (j - i).leading_zeros()) as usize;
        let dist = 1 << log_dist;
        min_by(self.results[i][log_dist], self.results[j - dist][log_dist], |a, b| self.data[*a].cmp(&self.data[*b]))
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
    }
}
