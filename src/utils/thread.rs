use crate::*;

#[inline]
pub fn join_raw(
    op_a: impl Send + FnOnce(Parallelism),
    op_b: impl Send + FnOnce(Parallelism),
    parallelism: Parallelism,
) {
    fn implementation(
        op_a: &mut (dyn Send + FnMut(Parallelism)),
        op_b: &mut (dyn Send + FnMut(Parallelism)),
        parallelism: Parallelism,
    ) {
        match parallelism {
            Parallelism::None => (op_a(parallelism), op_b(parallelism)),
            #[cfg(feature = "rayon")]
            Parallelism::Rayon(n_threads) => {
                if n_threads == 1 {
                    (op_a(Parallelism::None), op_b(Parallelism::None))
                } else {
                    let n_threads = if n_threads > 0 {
                        n_threads
                    } else {
                        rayon::current_num_threads()
                    };
                    let parallelism = Parallelism::Rayon(n_threads - n_threads / 2);
                    rayon::join(|| op_a(parallelism), || op_b(parallelism))
                }
            }
        };
    }
    let mut op_a = Some(op_a);
    let mut op_b = Some(op_b);
    implementation(
        &mut |parallelism| (op_a.take().unwrap())(parallelism),
        &mut |parallelism| (op_b.take().unwrap())(parallelism),
        parallelism,
    )
}

#[inline]
pub fn for_each_raw(n_tasks: usize, op: impl Send + Sync + Fn(usize), parallelism: Parallelism) {
    fn implementation(
        n_tasks: usize,
        op: &(dyn Send + Sync + Fn(usize)),
        parallelism: Parallelism,
    ) {
        if n_tasks == 1 {
            op(0);
            return;
        }

        match parallelism {
            Parallelism::None => (0..n_tasks).for_each(op),
            #[cfg(feature = "rayon")]
            Parallelism::Rayon(n_threads) => {
                let n_threads = if n_threads > 0 {
                    n_threads
                } else {
                    rayon::current_num_threads()
                };

                use rayon::prelude::*;
                let min_len = n_tasks / n_threads;
                (0..n_tasks)
                    .into_par_iter()
                    .with_min_len(min_len)
                    .for_each(op);
            }
        }
    }
    implementation(n_tasks, &op, parallelism);
}

pub struct Ptr<T>(pub *mut T);
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}
impl<T> Copy for Ptr<T> {}
impl<T> Clone for Ptr<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

#[inline]
pub fn parallelism_degree(parallelism: Parallelism) -> usize {
    match parallelism {
        Parallelism::None => 1,
        #[cfg(feature = "rayon")]
        Parallelism::Rayon(0) => rayon::current_num_threads(),
        #[cfg(feature = "rayon")]
        Parallelism::Rayon(n_threads) => n_threads,
    }
}

#[inline]
pub fn par_split_indices(n: usize, idx: usize, chunk_count: usize) -> (usize, usize) {
    let chunk_size = n / chunk_count;
    let rem = n % chunk_count;

    let idx_to_col_start = move |idx| {
        if idx < rem {
            idx * (chunk_size + 1)
        } else {
            rem + idx * chunk_size
        }
    };

    let start = idx_to_col_start(idx);
    let end = idx_to_col_start(idx + 1);
    (start, end - start)
}
