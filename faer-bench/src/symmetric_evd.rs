use super::timeit;
use crate::random;
use dyn_stack::{GlobalPodBuffer, PodStack, ReborrowMut};
use faer::{linalg::evd as faer_evd, Mat, Parallelism};
use ndarray_linalg::Eigh;
use std::time::Duration;

pub fn ndarray<T: ndarray_linalg::Lapack>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = ndarray::Array::<T, _>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    c[(i, j)] = random();
                    if i == j {
                        c[(i, j)] = c[(i, j)] + c[(i, j)].conj();
                    }
                }
            }

            let time = timeit(|| {
                c.eigh(ndarray_linalg::UPLO::Lower).unwrap();
            });

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}

pub fn nalgebra<T: nalgebra::ComplexField>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = nalgebra::DMatrix::<T>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    c[(i, j)] = random();
                }
            }

            let time = timeit(|| {
                c.clone().symmetric_eigen();
            });

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}

pub fn faer<T: faer::ComplexField>(sizes: &[usize], parallelism: Parallelism) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = Mat::<T>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    c.write(i, j, random());
                }
            }
            let mut s = Mat::<T>::zeros(n, n);
            let mut u = Mat::<T>::zeros(n, n);

            let mut mem = GlobalPodBuffer::new(
                faer_evd::compute_hermitian_evd_req::<T>(
                    n,
                    faer_evd::ComputeVectors::Yes,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            let time = timeit(|| {
                faer_evd::compute_hermitian_evd(
                    c.as_ref(),
                    s.as_mut()
                        .submatrix_mut(0, 0, n, n)
                        .diagonal_mut()
                        .column_vector_mut()
                        .as_2d_mut(),
                    Some(u.as_mut()),
                    parallelism,
                    stack.rb_mut(),
                    Default::default(),
                );
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
