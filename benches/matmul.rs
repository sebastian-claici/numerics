use rand::Rng;

use numerics::gemm::matmul;
use numerics::Matrix;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench(args = [128, 512, 1024])]
fn transpose_bench(bencher: divan::Bencher, n: usize) {
    let gen: fn() -> f32 = || rand::thread_rng().gen();

    let a = Matrix::from_gen(n, n, gen);

    bencher.bench(|| {
        a.transpose();
    })
}

#[divan::bench(args = [128, 512, 1024])]
fn matmul_bench(bencher: divan::Bencher, n: usize) {
    let gen: fn() -> f32 = || rand::thread_rng().gen();

    let a = Matrix::from_gen(n, n, gen);
    let b = Matrix::from_gen(n, n, gen);

    bencher.bench(|| {
        matmul(&a, &b);
    })
}
