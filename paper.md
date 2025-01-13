---
title: 'faer: A linear algebra library for the Rust programming language'
tags:
  - Rust
  - linear algebra
  - math
authors:
  - name: Sarah El Kazdadi
    orcid: 0000-0002-5657-0710
    affiliation: 1
affiliations:
 - name: Independent Researcher, France
   index: 1
date: 5 October 2023
bibliography: paper.bib
---

# Summary

`faer` is a portable high performance dense linear algebra library written in Rust.
The library offers a convenient high level API for performing matrix
decompositions and solving linear systems. This API is built on top of
a lower level API that gives the user more control over the memory allocation
and multithreading settings.

Supported platforms include the ones supported by Rust.
Explicit SIMD instructions are currently used for x86-64 and Aarch64 (NEON),
with plans for SVE/SME and RVV optimizations once intrinsics for those are stabilized in Rust,
possibly earlier than that if we allow usage of a Just-in-Time (JIT) backend[^1].

The library provides a `Mat` type, allowing for quick and simple construction
and manipulation of matrices, as well as lightweight view types `MatRef` and
`MatMut` for building memory views over existing data.

These views are currently used to represent different kinds of matrices,
such as generic rectangular matrices, symmetric/Hermitian/triangular 
(where only half of the matrix is stored) square matrices.
In the future, we plan to make use of the robust Rust type-system to better
express the properties of those matrices, and prevent accidental misuse of the library's API. 

Multiple scalar types are supported, and the library code is generic over the
data type. Native floating point types `f32`, `f64`[^2], `c32`, and `c64` are
supported out of the box, as well as any user-defined types that satisfy the
requested interface, such as extended precision real numbers (double-double or multi-precision floats),
complex numbers using the aforementioned types as the base element, dual/hyper-dual numbers[^3]


[^1]: Inline assembly is not entirely appropriate for our use case since it's hard to make it generic enough for all the operations and types that we wish to support.
[^2]: IEEE 754-2008, with no implicit `fusedMultiplyAdd` contractions and with slight differences around NaN handling. See the [float semantics](https://github.com/rust-lang/rust/issues/128288) tracking issue for more information.
[^3]: These support at least for the simpler matrix decompositions (Cholesky, LU, QR). It's not clear yet how to handle iterative algorithms like the SVD and Eigendecomposition.

# Statement of need

Rust was chosen as a language for the library since it allows full control
over the memory layout of data and exposes low level CPU intrinsics for
SIMD[^4] computations. Additionally, its memory safety features make it a
perfect candidate for writing efficient and parallel code, since the compiler
statically checks for errors that are common in other low level languages,
such as data races and fatal use-after-free errors.

Rust also allows compatibility with the C ABI, allowing for simple interoperability
with C, and most other languages by extension. Once a design has been properly fleshed out,
we plan to expose a C API, along with bindings to other languages (C, C++, Python and Julia bindings are currently planned).

Aside from `faer`, the Rust ecosystem lacks high performance matrix factorization
libraries that aren't C library wrappers, which presents a distribution
challenge and can impede generic programming.

[^4]: Single instruction, multiple data operations that CPUs can use to parallelize data processing at the instruction level.

# Features

`faer` exposes a central `Entity` trait that allows users to describe how their
data should be laid out in memory. For example, native floating point types are
laid out contiguously in memory to make use of SIMD instructions that prefer this layout,
while complex types have the option of either being laid out contiguously or in a split format.
The latter is also called a zomplex data type in CHOLMOD [@cholmod].
An example of a type that benefits immensely from this is the double-double type, which is
composed of two `f64` components, stored in separate containers. This separate
storage scheme allows us to load each chunk individually to a SIMD register,
opening new avenues for generic vectorization.

The library generically implements algorithms for matrix multiplication, based
on the approach of @BLIS1. For native types, `faer` uses explicit SIMD
depending on the detected CPU features, that dispatch to several precompiled
variants for operations that can make use of these features.
An interesting alternative would be to compile the code Just-in-Time, which could improve compilation times and reduce binary size.
But there are also possible downsides that have to be weighed against these advantages,
such as increasing the startup time to optimize and assemble the code,
as well as the gap in maturity between ahead-of-time compilation (currently backed by LLVM),
and JIT compilation, for which the Rust ecosystem is still developing.
The library then uses matrix multiplication as a building block to implement commonly used matrix
decompositions, based on state of the art algorithms in order to guarantee
numerical robustness:  
- Cholesky (LLT, LDLT and Bunch-Kaufman LDLT),  
- QR (with and without column pivoting),  
- LU (with partial and full pivoting),  
- SVD (with or without singular vectors, thin or full),  
- eigenvalue decomposition (with or without eigenvectors).

For algorithms that are memory-bound and don't make much use of matrix multiplication,
`faer` uses optimized fused kernels[^5]. This can immensely improve the performance of the
QR decomposition with column pivoting, the LU decomposition with full pivoting,
as well as the reduction to condensed form to prepare matrices for the SVD or
eigenvalue decomposition, as described by @10.1145/2382585.2382587.

State of the art algorithms are used for each decomposition, allowing performance
that matches or even surpasses other low level libraries such as OpenBLAS
[@10.1145/2503210.2503219], LAPACK [@lapack99], and Eigen [@eigenweb].

To achieve high performance parallelism, `faer` uses the Rayon library [@rayon] as a
backend, and has shown to be competitive with other frameworks such as OpenMP [@chandra2001parallel]
and Intel Thread Building Blocks [@tbb].

[^5]: For example, computing $A x$ and $A^T y$ with a single pass over $A$, rather than two.

# Performance

Here we present the benchmarks for a representative subset of operations that
showcase our improvements over the current state of the art.

The benchmarks were run on an 11th Gen Intel(R) Core(TM) i5-11400 @ 2.60GHz with 12 threads.
Eigen is compiled with the `-fopenmp` flag to enable parallelism.

![$n^3$ over run time of matrix multiplication. Higher is better](https://github.com/sarah-ek/faer-rs/files/13344473/matmul.pdf){#matmul_perf width="100%"}

![$n^3$ over run time of QR decomposition. Higher is better](https://github.com/sarah-ek/faer-rs/files/13344474/qr.pdf){#qr_perf width="100%"}

![$n^3$ over run time of eigenvalue decomposition. Higher is better](https://github.com/sarah-ek/faer-rs/files/13344472/evd.pdf){#evd_perf width="100%"}

# Future work
We have so far focused mainly on dense matrix algorithms, which will eventually form
the foundation of supernodal sparse decompositions.
Sparse algorithm implementations are still a work in progress and will be
showcased in a future paper.

# References

