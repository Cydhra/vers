use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::{thread_rng, Rng};
use std::collections::HashSet;
use vers_vecs::trees::bp::builder::BpDfsBuilder;
use vers_vecs::trees::bp::BpTree;
use vers_vecs::trees::{DfsTreeBuilder, Tree};

mod common;

// TODO this function has quadratic runtime, which is way too much for larger trees
fn generate_tree<R: Rng>(rng: &mut R, nodes: u64) -> BpTree {
    // generate prüfer sequence
    let mut sequence = vec![0; (nodes - 2) as usize];
    for i in 0..nodes - 2 {
        sequence[i as usize] = rng.gen_range(0..nodes - 1);
    }

    // decode prüfer sequence
    let mut degrees = vec![1; nodes as usize];
    sequence.iter().for_each(|i| degrees[*i as usize] += 1);

    let mut prefix_sum = vec![0; nodes as usize];
    let mut sum = 0;
    degrees.iter().enumerate().for_each(|(i, d)| {
        prefix_sum[i] = sum;
        sum += d;
    });

    let mut children = vec![0u64; sum];
    let mut assigned_children = vec![0; nodes as usize];
    sequence.iter().for_each(|i| {
        let j = degrees.iter().enumerate().find(|(idx, x)| *idx != *i as usize && **x == 1).unwrap().0 as u64;
        children[prefix_sum[*i as usize] + assigned_children[*i as usize]] = j;
        children[prefix_sum[j as usize] + assigned_children[j as usize]] = *i;
        degrees[*i as usize] -= 1;
        degrees[j as usize] -= 1;
        assigned_children[*i as usize] += 1;
        assigned_children[j as usize] += 1;
    });

    assert_eq!(degrees.iter().sum::<usize>(), 2);
    let u = degrees.iter().enumerate().find(|(_, x)| **x == 1).unwrap().0;
    let v = degrees.iter().enumerate().rev().find(|(_, x)| **x == 1).unwrap().0;

    children[prefix_sum[u] + assigned_children[u]] = v as u64;
    children[prefix_sum[v] + assigned_children[v]] = u as u64;

    // build tree
    let mut bpb = BpDfsBuilder::with_capacity(nodes);
    let mut stack = Vec::new();
    let mut visited = HashSet::with_capacity(nodes as usize);
    visited.insert(0);
    stack.push((0, 0u64, true));
    while let Some((depth, node, enter)) = stack.pop() {
        if enter {
            bpb.enter_node();
            stack.push((depth, node, false));
            for c in prefix_sum[node as usize]..*prefix_sum.get(node as usize + 1).unwrap_or(&children.len()) {
                if visited.insert(children[c]) {
                    stack.push((depth + 1, children[c], true))
                }
            }
        } else {
            bpb.leave_node();
        }
    }

    bpb.build().unwrap()
}

fn bench_parent(b: &mut Criterion) {
    let mut rng = thread_rng();

    for l in common::SIZES {
        let bp = generate_tree(&mut rng, l as u64);
        let node_handles = (0..l).map(|i| bp.node_handle(i)).collect::<Vec<_>>();

        b.bench_function("parent", |b| {
            b.iter_batched(|| node_handles[rng.gen_range(0..node_handles.len())], |h| {
                black_box(bp.parent(h))
            }, BatchSize::SmallInput)
        });
    }
}

criterion_group!(benches, bench_parent);
criterion_main!(benches);
