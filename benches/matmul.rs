use rand::Rng;

use numerics::core::gemm::matmul;
use numerics::core::matrix::Matrix;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench(args = [128, 512, 1024])]
fn transpose_bench(bencher: divan::Bencher, n: usize) {
    let gen: fn(usize, usize) -> f32 = |_, _| rand::thread_rng().gen();

    let a = Matrix::from_gen(n, n, gen);

    bencher.bench(|| {
        a.transpose();
    })
}

#[divan::bench(sample_count=20, args = [128, 512, 1024, 2048])]
fn matmul_bench(bencher: divan::Bencher, n: usize) {
    let gen: fn(usize, usize) -> f32 = |i, j| ((i + j) as f32) / 2048.0;

    let a = Matrix::from_gen(n, n, gen);
    let b = Matrix::from_gen(n, n, gen);

    bencher.bench(|| {
        matmul(&a, &b);
    })
}
