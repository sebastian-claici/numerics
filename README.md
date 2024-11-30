# Numerical Methods in Rust

Working my way through the third edition of [Numerical Recipes](https://numerical.recipes/), but rewriting everything in Rust.

This is not meant to be blazingly fast. For that, you want something like [faer](https://github.com/sarah-quinones/faer-rs), or better yet, just use [BLAS](https://www.netlib.org/blas/) if you have it available.

Currently working on matrix decomposition methods, including:
- [ ] Cholesky decomposition
- [ ] LU decomposition
- [ ] QR decomposition
- [ ] Forward and backward substitution methods
- [ ] Full solvers
