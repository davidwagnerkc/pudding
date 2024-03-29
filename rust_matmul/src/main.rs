use ndarray::{Array, Array1, Array2, Zip, LinalgScalar};
use ndarray::parallel::par_azip;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::{Duration, Instant};
use std::env;
use std::sync::{Arc, Mutex};


fn dot<S>(arr1: &ArrayBase<S, Ix1>, arr2: &ArrayBase<S, Ix1>) -> i64
where
    S: Data<Elem = i64>,
{
    let sum = Zip::from(arr1)
                  .and(arr2)
                  .fold(0, |accum, a, b| accum + a * b );
    sum
}

fn matmul<S>(arr1: &ArrayBase<S, Ix2>, arr2: &ArrayBase<S, Ix2>) -> Array2<i64>
where
    S: Data<Elem = i64>
{
    let m = arr1.nrows();
    let n = arr2.ncols();
    let mut result = Array2::<i64>::zeros((m,  n));
    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = dot(&arr1.slice(s![i, ..]), &arr2.slice(s![.., j]));
        }
    }
    result
}

fn par_matmul<S>(arr1: &ArrayBase<S, Ix2>, arr2: &ArrayBase<S, Ix2>) -> Array2<i64>
where
    S: Data<Elem = i64> + std::marker::Sync
{
    let m = arr1.nrows();
    let n = arr2.ncols();
    let collected: Vec<_> = arr1.axis_iter(Axis(0))
        .into_par_iter()
        .flat_map(|row| {
            arr2.axis_iter(Axis(1))
                .map(|col| {
                    dot(&row, &col)
                }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
    let result = Array::from_shape_vec((m, n), collected).unwrap();
    result
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args[1].trim().parse().unwrap();
    let lower = -i64::pow(2, 16);
    let upper = i64::pow(2, 16);
    let A = Array::random((n, n), Uniform::new(lower, upper));
    let B = Array::random((n, n), Uniform::new(lower, upper));

    let start = Instant::now();
    let result = matmul(&A, &B);
    let duration = start.elapsed();
    // println!("{:?}", result);
    println!("{:?}", duration);

    let start = Instant::now();
    let expect = A.dot(&B);
    let duration = start.elapsed();
    // println!("{:?}", expect);
    println!("{:?}", duration);

    let start = Instant::now();
    let result = par_matmul(&A, &B);
    let duration = start.elapsed();
    // println!("{:?}", result);
    println!("{:?}", duration);
}
