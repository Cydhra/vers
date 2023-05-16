use crate::bit_vec::{BitVec, BuildingStrategy};
use crate::RsVector;

pub struct EliasFanoVec<B: RsVector> {
    upper_vec: B,
    lower_vec: BitVec,
    lower_len: usize,
}

impl<B: RsVector + BuildingStrategy<Vector = B>> EliasFanoVec<B> {
    /// Create a new Elias-Fano vector by compressing the given data.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(data: &Vec<u64>) -> Self {
        let lower_width = data.len().leading_zeros() as usize;

        let mut upper_vec = BitVec::from_zeros(data.len() * 2 + 1);
        let mut lower_vec = BitVec::with_capacity(data.len() * lower_width);

        for (i, &word) in data.iter().enumerate() {
            let upper = (word >> lower_width) as usize;
            let lower = word & ((1 << lower_width) - 1);

            upper_vec.flip_bit(upper + i);
            lower_vec.append_bits(lower, lower_width);
        }

        Self {
            upper_vec: B::from_bit_vec(upper_vec),
            lower_vec,
            lower_len: lower_width,
        }
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.lower_vec.len() / self.lower_len
    }

    /// Returns true if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the element at the given index.
    #[allow(clippy::cast_possible_truncation)]
    pub fn get(&self, index: usize) -> u64 {
        let upper = self.upper_vec.select1(index) - index;
        let lower = self
            .lower_vec
            .get_bits(index * self.lower_len, self.lower_len);
        (upper << self.lower_len) as u64 | lower
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn pred(&self, n: u64) -> u64 {
        let upper = n >> self.lower_len;
        let p = self.upper_vec.select1(upper as usize);

        let lower = n & ((1 << self.lower_len) - 1);
        let index = self.upper_vec.rank1(p);
        let mut lower_index = index * self.lower_len;
        let mut lower_candidate = self.lower_vec.get_bits(lower_index, self.lower_len);

        loop {
            let next_candidate = self
                .lower_vec
                .get_bits(lower_index + self.lower_len, self.lower_len);

            if next_candidate > lower {
                break;
            }

            lower_candidate = next_candidate;
            lower_index += self.lower_len;

            if lower_index + self.lower_len >= self.lower_vec.len() {
                break;
            }
        }

        (p << self.lower_len) as u64 | lower_candidate
    }
}

#[cfg(test)]
mod tests {
    use crate::{EliasFanoVec, FastBitVector};

    #[test]
    fn test_elias_fano() {
        let ef = EliasFanoVec::<FastBitVector>::new(&vec![0, 1, 4, 7]);
        assert_eq!(ef.len(), 4);
        assert_eq!(ef.get(0), 0);
        assert_eq!(ef.get(1), 1);
        assert_eq!(ef.get(2), 4);
        assert_eq!(ef.get(3), 7);

        assert_eq!(ef.pred(0), 0);
        assert_eq!(ef.pred(1), 1);
        assert_eq!(ef.pred(2), 1);
        assert_eq!(ef.pred(5), 4);
        assert_eq!(ef.pred(8), 7);
    }
}
