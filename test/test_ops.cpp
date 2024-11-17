#include "ops.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinRel;

TEST_CASE("Dot product of two vectors", "[dot]") {
  Vector<double> a{1.0, 2.0, 3.0};
  Vector<double> b{4.0, 5.0, 6.0};

  REQUIRE_THAT(dot(a, b), WithinRel(32.0));
}

TEST_CASE("Matrix-vector multiplication", "[matmul]") {
  Matrix<double> A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  Vector<double> v{1.0, 2.0, 3.0};

  Vector<double> result = matmul(A, v);

  REQUIRE_THAT(result[0], WithinRel(14.0));
  REQUIRE_THAT(result[1], WithinRel(32.0));
  REQUIRE_THAT(result[2], WithinRel(50.0));
}

TEST_CASE("Benchmark matrix-vector multiplication", "[matmul][!benchmark]") {
  const size_t n = 1000;
  Matrix<double> rM = rand(n, n);
  Vector<double> rv = rand(n);

  BENCHMARK("Big matrix-vector multiply") { return matmul(rM, rv); };
}

TEST_CASE("Matrix-matrix multiplication", "[matmul]") {
  Matrix<double> A{{1.0, 2.0}, {3.0, 4.0}};
  Matrix<double> B{{5.0, 6.0}, {7.0, 8.0}};

  Matrix<double> result = matmul(A, B);

  REQUIRE(result.m_rows == A.m_rows);
  REQUIRE(result.m_cols == B.m_cols);

  REQUIRE_THAT(result[0][0], WithinRel(19.0));
  REQUIRE_THAT(result[0][1], WithinRel(22.0));
  REQUIRE_THAT(result[1][0], WithinRel(43.0));
  REQUIRE_THAT(result[1][1], WithinRel(50.0));
}

TEST_CASE("Benchmark matrix-matrix multiplication", "[matmul][!benchmark]") {
  const size_t n = 1000;
  Matrix<double> rA = rand(n, n);
  Matrix<double> rB = rand(n, n);

  BENCHMARK("Big matrix-matrix multiply") { return matmul(rA, rB); };
}
