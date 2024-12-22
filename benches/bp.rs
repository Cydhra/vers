use std::cmp::Reverse;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::{thread_rng, Rng};
use std::collections::{BinaryHeap, HashSet};
use vers_vecs::trees::bp::builder::BpDfsBuilder;
use vers_vecs::trees::bp::BpTree;
use vers_vecs::trees::{DfsTreeBuilder, Tree};

mod common;

// TODO this function has nlogn runtime, which is a bit too much for the largest trees
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

    // keep a priority queue of nodes with degree one to reduce runtime from O(n^2) to O(n log n)
    let mut degree_one_set = BinaryHeap::new();
    degrees.iter().enumerate().filter(|(_, &v)| v == 1).for_each(|(idx, _)| degree_one_set.push(Reverse(idx as u64)));

    sequence.iter().for_each(|&i| {
        let j = degree_one_set.pop().unwrap().0;
        children[prefix_sum[i as usize] + assigned_children[i as usize]] = j;
        children[prefix_sum[j as usize] + assigned_children[j as usize]] = i;
        degrees[i as usize] -= 1;
        if degrees[i as usize] == 1 {
            degree_one_set.push(Reverse(i))
        }

        degrees[j as usize] -= 1;
        if degrees[j as usize] == 1 {
            degree_one_set.push(Reverse(j))
        }

        assigned_children[i as usize] += 1;
        assigned_children[j as usize] += 1;
    });

    assert_eq!(degrees.iter().sum::<usize>(), 2);
    let u = degree_one_set.pop().unwrap().0;
    let v = degree_one_set.pop().unwrap().0;

    children[prefix_sum[u as usize] + assigned_children[u as usize]] = v;
    children[prefix_sum[v as usize] + assigned_children[v as usize]] = u;

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

fn bench_navigation(b: &mut Criterion) {
    let mut rng = thread_rng();
    let mut b = b.benchmark_group("bp");

    for l in common::SIZES {
        let bp = generate_tree(&mut rng, l as u64);
        let node_handles = (0..l).map(|i| bp.node_handle(i)).collect::<Vec<_>>();

        b.bench_with_input(BenchmarkId::new("parent", l), &l, |b, _| {
            b.iter_batched(|| node_handles[rng.gen_range(0..node_handles.len())], |h| {
                black_box(bp.parent(h))
            }, BatchSize::SmallInput)
        });

        b.bench_with_input(BenchmarkId::new("last_child", l), &l, |b, _| {
            b.iter_batched(|| node_handles[rng.gen_range(0..node_handles.len())], |h| {
                black_box(bp.last_child(h))
            }, BatchSize::SmallInput)
        });

        b.bench_with_input(BenchmarkId::new("next_sibling", l), &l,|b, _| {
            b.iter_batched(|| node_handles[rng.gen_range(0..node_handles.len())], |h| {
                black_box(bp.next_sibling(h))
            }, BatchSize::SmallInput)
        });

        b.bench_with_input(BenchmarkId::new("prev_sibling", l), &l,|b, _| {
            b.iter_batched(|| node_handles[rng.gen_range(0..node_handles.len())], |h| {
                black_box(bp.previous_sibling(h))
            }, BatchSize::SmallInput)
        });
    }
}

criterion_group!(benches, bench_navigation);
criterion_main!(benches);
