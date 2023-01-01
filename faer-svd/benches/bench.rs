use criterion::{criterion_group, criterion_main, Criterion};
use faer_svd::bidiagonalize_in_place;
use std::time::Duration;

use dyn_stack::*;
use rand::random;

use faer_core::{Mat, Parallelism};

pub fn bidiag(c: &mut Criterion) {
    for (m, n) in [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (10000, 128),
        (10000, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ] {
        c.bench_function(&format!("faer-st-bidiag-{m}x{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), m, n);
            let mut householder_left = Mat::with_dims(|_, _| random::<f64>(), n, 1);
            let mut householder_right = Mat::with_dims(|_, _| random::<f64>(), n, 1);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                bidiagonalize_in_place(
                    mat.as_mut(),
                    householder_left.as_mut().col(0),
                    householder_right.as_mut().col(0),
                    Parallelism::None,
                    stack.rb_mut(),
                )
            })
        });

        c.bench_function(&format!("faer-mt-bidiag-{m}x{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), m, n);
            let mut householder_left = Mat::with_dims(|_, _| random::<f64>(), n, 1);
            let mut householder_right = Mat::with_dims(|_, _| random::<f64>(), n, 1);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                bidiagonalize_in_place(
                    mat.as_mut(),
                    householder_left.as_mut().col(0),
                    householder_right.as_mut().col(0),
                    Parallelism::Rayon(0),
                    stack.rb_mut(),
                )
            })
        });
    }

    let _c = c;
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);
    targets = bidiag
);
criterion_main!(benches);
