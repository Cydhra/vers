struct NaiveRmq {
    results: Vec<Vec<usize>>,
}

impl NaiveRmq {
    fn new(data: &[u64]) -> Self {
        let mut results = vec![vec![0; data.len()]; data.len()];
        for i in 0..data.len() {
            results[i][i] = i;
        }
        for i in 0..data.len() {
            for j in i + 1..data.len() {
                results[i][j] = if data[results[i][j - 1]] < data[j] {
                    results[i][j - 1]
                } else {
                    j
                };
            }
        }
        Self { results }
    }

    fn range_min(&self, i: usize, j: usize) -> usize {
        self.results[i][j]
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn naive_rmq_test() {
        let rmq = super::NaiveRmq::new(&[9, 6, 10, 4, 0, 8, 3, 7, 1, 2, 5]);

        assert_eq!(rmq.range_min(0, 0), 0);
        assert_eq!(rmq.range_min(0, 1), 1);
        assert_eq!(rmq.range_min(0, 2), 1);
        assert_eq!(rmq.range_min(0, 3), 3);
        assert_eq!(rmq.range_min(5, 8), 8);
        assert_eq!(rmq.range_min(5, 9), 8);
    }
}
