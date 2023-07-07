#![feature(buf_read_has_data_left)]

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;
use std::{env, io};
use vers::{EliasFanoVec, FastBitVector};
use vers::rmq::fast_rmq::FastRmq;

fn main() {
    let mut args = env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| print_usage());
    let input_file = args.next().unwrap_or_else(|| print_usage());
    let output_file = args.next().unwrap_or_else(|| print_usage());

    let result = if mode == "pd" {
        handle_predecessor_benchmark(input_file, output_file)
    } else if mode == "rmq" {
        handle_rmq_benchmark(input_file, output_file)
    } else {
        print_usage();
    };

    match result {
        Ok((time, heap_size)) => {
            println!(
                "RESULT algo={} namejohannes_hengstler time={} space={}",
                mode, time, heap_size
            );
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}

fn handle_predecessor_benchmark(
    input_file: String,
    output_file: String,
) -> Result<(u128, usize), io::Error> {
    let mut input_reader = BufReader::new(File::open(input_file)?);

    let mut line = String::new();
    input_reader.read_line(&mut line)?;
    let size = line.trim().parse::<usize>().unwrap();
    let pred_buffer = read_n_numbers(size, &mut input_reader)?;
    let mut req_buffer = Vec::new();

    while input_reader.has_data_left()? {
        line.clear();
        input_reader.read_line(&mut line)?;
        req_buffer.push(line.trim().parse::<u64>().unwrap());
    }

    let start = Instant::now();
    let elias_fano_vec = EliasFanoVec::<FastBitVector>::new(&pred_buffer);

    for req in req_buffer.iter_mut() {
        *req = elias_fano_vec.pred(*req);
    }

    let time = start.elapsed().as_millis();

    let mut output_file = File::create(output_file)?;
    let output = req_buffer
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    output_file.write_all(output.as_bytes())?;

    Ok((time, elias_fano_vec.heap_size() * 8))
}

fn handle_rmq_benchmark(
    input_file: String,
    output_file: String,
) -> Result<(u128, usize), io::Error> {
    let mut input_reader = BufReader::new(File::open(input_file)?);

    let mut line = String::new();
    input_reader.read_line(&mut line)?;
    let size = line.trim().parse::<usize>().unwrap();

    let rmq_buffer = read_n_numbers(size, &mut input_reader)?;
    let mut req_buffer = Vec::new();

    while input_reader.has_data_left()? {
        line.clear();
        input_reader.read_line(&mut line)?;
        let mut range = line.split(',');
        let begin = range.next().unwrap().trim().parse::<usize>().unwrap();
        let end = range.next().unwrap().trim().parse::<usize>().unwrap();
        req_buffer.push((begin, end));
    }

    let start = Instant::now();
    let rmq = FastRmq::new(rmq_buffer);

    for (begin, end) in req_buffer.iter_mut() {
        *begin = rmq.range_min(*begin, *end);
    }

    let time = start.elapsed().as_millis();

    let mut output_file = File::create(output_file)?;
    let output = req_buffer
        .into_iter()
        .map(|(begin, _)| format!("{}", begin))
        .collect::<Vec<_>>()
        .join("\n");

    output_file.write_all(output.as_bytes())?;

    Ok((time, 8))
}

/// Read n lines and treat each as a 64 bit number.
fn read_n_numbers(n: usize, reader: &mut BufReader<File>) -> Result<Vec<u64>, io::Error> {
    let mut buffer = Vec::with_capacity(n);
    let mut line = String::new();
    for _ in 0..n {
        line.clear();
        reader.read_line(&mut line)?;
        buffer.push(line.trim().parse::<u64>().unwrap());
    }
    Ok(buffer)
}

fn print_usage() -> ! {
    println!("Usage: ads_project <mode> <args...>");
    println!("Modes:");
    println!("  pd <input file> <output file>");
    println!("  rmq <input file> <output file>");
    std::process::exit(1);
}
