use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{GlobalPodBuffer, PodStack};
use faer_core::{c64, ComplexField};
use reborrow::*;

use faer_core::{Mat, Parallelism};
use nalgebra::DMatrix;

pub fn cholesky(c: &mut Criterion) {
    use faer_cholesky::{ldlt_diagonal, llt};
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Warn)
        .init();

    for n in [6, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 4096] {
        c.bench_function(&format!("faer-st-ldlt-{n}"), |b| {
            let mut mat = Mat::new();

            mat.resize_with(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
            let mut mem = GlobalPodBuffer::new(
                ldlt_diagonal::compute::raw_cholesky_in_place_req::<f64>(
                    n,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            b.iter(|| {
                ldlt_diagonal::compute::raw_cholesky_in_place(
                    mat.as_mut(),
                    Default::default(),
                    Parallelism::None,
                    stack.rb_mut(),
                    Default::default(),
                );
            })
        });

        c.bench_function(&format!("faer-mt-ldlt-{n}"), |b| {
            let mut mat = Mat::new();

            mat.resize_with(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
            let mut mem = GlobalPodBuffer::new(
                ldlt_diagonal::compute::raw_cholesky_in_place_req::<f64>(
                    n,
                    Parallelism::Rayon(rayon::current_num_threads()),
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            b.iter(|| {
                ldlt_diagonal::compute::raw_cholesky_in_place(
                    mat.as_mut(),
                    Default::default(),
                    Parallelism::Rayon(rayon::current_num_threads()),
                    stack.rb_mut(),
                    Default::default(),
                );
            })
        });

        c.bench_function(&format!("faer-st-llt-{n}"), |b| {
            let mut mat = Mat::new();

            mat.resize_with(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
            let mut mem = GlobalPodBuffer::new(
                llt::compute::cholesky_in_place_req::<f64>(
                    n,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            b.iter(|| {
                llt::compute::cholesky_in_place(
                    mat.as_mut(),
                    Default::default(),
                    Parallelism::None,
                    stack.rb_mut(),
                    Default::default(),
                )
                .unwrap();
            })
        });

        c.bench_function(&format!("faer-mt-llt-{n}"), |b| {
            let mut mat = Mat::new();

            mat.resize_with(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
            let mut mem = GlobalPodBuffer::new(
                llt::compute::cholesky_in_place_req::<f64>(
                    n,
                    Parallelism::Rayon(rayon::current_num_threads()),
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            b.iter(|| {
                llt::compute::cholesky_in_place(
                    mat.as_mut(),
                    Default::default(),
                    Parallelism::Rayon(rayon::current_num_threads()),
                    stack.rb_mut(),
                    Default::default(),
                )
                .unwrap();
            })
        });

        c.bench_function(&format!("faer-st-cplx-llt-{n}"), |b| {
            let mut mat = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    c64::faer_one()
                } else {
                    c64::faer_zero()
                }
            });

            let mut mem = GlobalPodBuffer::new(
                llt::compute::cholesky_in_place_req::<c64>(
                    n,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            b.iter(|| {
                llt::compute::cholesky_in_place(
                    mat.as_mut(),
                    Default::default(),
                    Parallelism::None,
                    stack.rb_mut(),
                    Default::default(),
                )
                .unwrap();
            })
        });

        c.bench_function(&format!("faer-mt-cplx-llt-{n}"), |b| {
            let mut mat = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    c64::faer_one()
                } else {
                    c64::faer_zero()
                }
            });

            let mut mem = GlobalPodBuffer::new(
                llt::compute::cholesky_in_place_req::<c64>(
                    n,
                    Parallelism::Rayon(rayon::current_num_threads()),
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            b.iter(|| {
                llt::compute::cholesky_in_place(
                    mat.as_mut(),
                    Default::default(),
                    Parallelism::Rayon(rayon::current_num_threads()),
                    stack.rb_mut(),
                    Default::default(),
                )
                .unwrap();
            })
        });

        c.bench_function(&format!("nalg-st-llt-{n}"), |b| {
            let mut mat = DMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                mat[(i, i)] = 1.0;
            }

            b.iter(|| {
                let _ = mat.clone().cholesky();
            })
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);
    targets = cholesky
);
criterion_main!(benches);
