use numerics::gemm::matmul;
use numerics::Matrix;

fn main() {
    let gen: fn(usize, usize) -> f32 = |i, j| ((i + j) as f32) / 2048.0;

    let n = 2048;
    let a = Matrix::from_gen(n, n, gen);
    let b = Matrix::from_gen(n, n, gen);

    let c = matmul(&a, &b);
    println!("Output matrix has: {} rows, {} cols", c.n_rows, c.n_cols);
}
