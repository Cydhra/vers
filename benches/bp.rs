use std::collections::HashSet;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{thread_rng, Rng, SeedableRng};

mod common;

fn generate_tree<R: Rng>(rng: &mut R, nodes: usize) {
    // generate prüfer sequence
    let mut sequence = vec![0; nodes - 2];
    for i in 0..nodes - 2 {
        sequence[i] = rng.gen_range(0..nodes - 1);
    }

    // decode prüfer sequence
    let mut degrees = vec![1; nodes];
    sequence.iter().for_each(|i| degrees[*i] += 1);

    let mut prefix_sum = vec![0; nodes];
    let mut sum = 0;
    degrees.iter().enumerate().for_each(|(i, d)| {
        prefix_sum[i] = sum;
        sum += d;
    });

    let mut children = vec![0; sum];
    let mut assigned_children = vec![0; nodes];
    sequence.iter().for_each(|i| {
        let j = degrees.iter().enumerate().find(|(idx, x)| *idx != *i && **x == 1).unwrap().0;
        children[prefix_sum[*i] + assigned_children[*i]] = j;
        children[prefix_sum[j] + assigned_children[j]] = *i;
        degrees[*i] -= 1;
        degrees[j] -= 1;
        assigned_children[*i] += 1;
        assigned_children[j] += 1;
    });

    assert_eq!(degrees.iter().sum::<usize>(), 2);
    let u = degrees.iter().enumerate().find(|(i, x)| **x == 1).unwrap().0;
    let v = degrees.iter().enumerate().rev().find(|(i, x)| **x == 1).unwrap().0;

    children[prefix_sum[u] + assigned_children[u]] = v;
    children[prefix_sum[v] + assigned_children[v]] = u;

    let mut stack = Vec::new();
    let mut visited = HashSet::with_capacity(nodes);
    visited.insert(0);
    stack.push((0, 0));
    while let Some((depth, node)) = stack.pop() {

        for c in prefix_sum[node]..*prefix_sum.get(node + 1).unwrap_or(&children.len()) {
            if visited.insert(children[c]) {
                stack.push((depth + 1, children[c]))
            }
        }
    }
}

fn bench_bp(b: &mut Criterion) {

}

criterion_group!(benches, bench_bp);
criterion_main!(benches);
