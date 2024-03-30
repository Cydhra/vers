//! A custom benchmark that compares heap sizes for vers and the fastest competitor libraries.
//! It generates a plot using plotters and stores it under target/heap.svg.
//! The legend in the plot contains the percent overhead for the largest vector size measured
//! (because I assume that overhead converges against a number)

use crate::Measure::*;
use plotters::backend::SVGBackend;
use plotters::prelude::*;
use plotters::style::full_palette::{ORANGE, PURPLE};
use rand::distributions::{Standard, Uniform};
use rand::prelude::{Distribution, ThreadRng};
use rand::{thread_rng, Rng};
use rsdict::RsDict;
use std::collections::HashMap;
use plotters::prelude::full_palette::GREEN_800;
use sucds::bit_vectors::darray::DArray as SucDArray;
use sucds::bit_vectors::rank9sel::Rank9Sel as SucRank9Vec;
use sucds::bit_vectors::BitVector as SucBitVec;
use succinct::{BitVecPush, BitVector as SuccinctVec, Rank9, SpaceUsage};
use sucds::Serializable;
use vers_vecs::{BitVec, RsVec};

/// Build a random vers vector of given size
fn construct_vers_vec(rng: &mut ThreadRng, len: usize) -> RsVec {
    let sample = Uniform::new(0, u64::MAX);

    let mut bit_vec = BitVec::with_capacity(len / 64);
    for _ in 0..len / 64 {
        bit_vec.append_word(sample.sample(rng));
    }

    RsVec::from_bit_vec(bit_vec)
}

/// Build a random rsdict vector of given size
fn construct_rsdict_vec(rng: &mut ThreadRng, len: usize) -> RsDict {
    let mut rs_dict = RsDict::with_capacity(len);
    for _ in 0..len {
        rs_dict.push(rng.gen_bool(0.5));
    }
    rs_dict
}

fn construct_sucds_rank9_vec(rng: &mut ThreadRng, len: usize) -> SucRank9Vec {
    let mut suc_bv = SucBitVec::with_capacity(len);
    for _ in 0..len / 64 {
        suc_bv
            .push_bits(rng.sample(Standard), 64)
            .expect("Failed to push bits into sucds bitvector");
    }

    SucRank9Vec::new(suc_bv).select0_hints()
}

fn construct_sucds_darray(rng: &mut ThreadRng, len: usize) -> SucDArray {
    let mut suc_bv = SucBitVec::with_capacity(len);
    for _ in 0..len / 64 {
        suc_bv
            .push_bits(rng.sample(Standard), 64)
            .expect("Failed to push bits into sucds bitvector");
    }

    SucDArray::from_bits(suc_bv.iter()).enable_rank()
}

fn construct_rank9_vec(rng: &mut ThreadRng, len: usize) -> Rank9<SuccinctVec<u64>> {
    let mut bit_vec = SuccinctVec::with_capacity(len as u64);
    for _ in 0..len {
        bit_vec.push_bit(rng.sample(Standard))
    }
    Rank9::new(bit_vec)
}

#[derive(Eq, PartialEq, Hash, Debug, Copy, Clone)]
enum Measure {
    Vers,
    RsDict,
    SucdR9,
    SucdDa,
    Rank9
}

// select a color for each measure for the plot
impl From<&Measure> for RGBColor {
    fn from(value: &Measure) -> Self {
        match value {
            Vers => GREEN_800,
            RsDict => BLUE,
            SucdR9 => ORANGE,
            SucdDa => BLACK,
            Rank9 => PURPLE,
        }
    }
}

// all measures
static MEASURES: &[Measure] = &[Vers, RsDict, SucdR9, SucdDa, Rank9];

fn main() {
    let mut rng = thread_rng();
    const LENGTHS: [usize; 7] = [
        1 << 12,
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 24,
    ];

    let mut measurements = HashMap::<Measure, Vec<usize>>::new();
    for measure in MEASURES {
        measurements.insert(*measure, Vec::with_capacity(LENGTHS.len()));
    }

    for len in LENGTHS {
        measurements
            .get_mut(&Vers)
            .unwrap()
            .push(construct_vers_vec(&mut rng, len).heap_size());
        measurements
            .get_mut(&RsDict)
            .unwrap()
            .push(construct_rsdict_vec(&mut rng, len).heap_size());
        measurements
            .get_mut(&SucdR9)
            .unwrap()
            .push(construct_sucds_rank9_vec(&mut rng, len).size_in_bytes());
        measurements
            .get_mut(&SucdDa)
            .unwrap()
            .push(construct_sucds_darray(&mut rng, len).size_in_bytes());
        measurements
            .get_mut(&Rank9)
            .unwrap()
            .push(construct_rank9_vec(&mut rng, len).heap_bytes());
        println!("finished {} bytes measurement", len)
    }

    // draw plot
    let root_area = SVGBackend::new("target/heap.svg", (600, 400)).into_drawing_area();
    root_area.fill(&RGBColor(0xFF, 0xFA, 0xF0)).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Heap Size comparison", ("sans-serif", 16))
        .build_cartesian_2d(
            (LENGTHS[0] as i32..*LENGTHS.last().unwrap() as i32).log_scale(),
            0..100,
        )
        .unwrap();

    ctx.configure_mesh()
        .y_desc("% Overhead")
        .x_desc("vector size (bytes)")
        .disable_mesh()
        .draw()
        .unwrap();

    for measure in MEASURES {
        let measurement = &measurements[measure];
        ctx.draw_series(LineSeries::new::<_, RGBColor>(
            LENGTHS.iter().enumerate().map(|(idx, &len)| {
                (
                    len as i32,
                    (100.0 * measurement[idx] as f64 / (len / 8) as f64) as i32 - 100,
                )
            }),
            measure.into(),
        ))
        .unwrap()
        .label(format!(
            "{:?}: {}%",
            measure,
            ((10000.0 * *measurement.last().unwrap() as f64 / (LENGTHS.last().unwrap() / 8) as f64)
                - 10000.0) as usize as f64
                / 100.0
        ))
        .legend(|(x, y)| {
            PathElement::new::<_, RGBColor>(vec![(x, y), (x + 20, y)], measure.into())
        });
    }

    ctx.configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();
}
