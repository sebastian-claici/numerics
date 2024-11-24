#include "linalg/cholesky.h"
#include "linalg/solve.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinRel;

TEST_CASE("Cholesky decomposition", "[decomposition]") {
  Matrix<double> A{
      {4.0, 12.0, -16.0}, {12.0, 37.0, -43.0}, {-16.0, -43.0, 98.0}};

  Cholesky chol(A);
  REQUIRE_THAT(chol.L(0, 0), WithinRel(2.0));
  REQUIRE_THAT(chol.L(1, 0), WithinRel(6.0));
  REQUIRE_THAT(chol.L(2, 0), WithinRel(-8.0));
  REQUIRE_THAT(chol.L(0, 1), WithinRel(0.0));
  REQUIRE_THAT(chol.L(1, 1), WithinRel(1.0));
  REQUIRE_THAT(chol.L(2, 1), WithinRel(5.0));
  REQUIRE_THAT(chol.L(0, 2), WithinRel(0.0));
  REQUIRE_THAT(chol.L(1, 2), WithinRel(0.0));
  REQUIRE_THAT(chol.L(2, 2), WithinRel(3.0));
}

TEST_CASE("Backward substitution (vector)", "[solve]") {
  Matrix<double> A{{1, -2, 1}, {0, 1, 6}, {0, 0, 1}};
  Vector<double> b{4, -1, 2};

  auto x = bsub(A, b);
  REQUIRE_THAT(x[0], WithinRel(-24.0));
  REQUIRE_THAT(x[1], WithinRel(-13.0));
  REQUIRE_THAT(x[2], WithinRel(2.0));
}

TEST_CASE("Backward substitution (matrix)", "[solve]") {
  Matrix<double> A{{1, -2, 1}, {0, 1, 6}, {0, 0, 1}};
  Matrix<double> b{{4, 4}, {-1, -1}, {2, 2}};

  auto x = bsub(A, b);
  REQUIRE_THAT(x(0, 0), WithinRel(-24.0));
  REQUIRE_THAT(x(1, 0), WithinRel(-13.0));
  REQUIRE_THAT(x(2, 0), WithinRel(2.0));
  REQUIRE_THAT(x(0, 1), WithinRel(-24.0));
  REQUIRE_THAT(x(1, 1), WithinRel(-13.0));
  REQUIRE_THAT(x(2, 1), WithinRel(2.0));
}

TEST_CASE("Forward substitution (vector)", "[solve]") {
  Matrix<double> A{{1, 0, 0}, {6, 1, 0}, {1, -2, 1}};
  Vector<double> b{2, -1, 4};

  auto x = fsub(A, b);
  REQUIRE_THAT(x[0], WithinRel(2.0));
  REQUIRE_THAT(x[1], WithinRel(-13.0));
  REQUIRE_THAT(x[2], WithinRel(-24.0));
}

TEST_CASE("Forward substitution (matrix)", "[solve]") {
  Matrix<double> A{{1, 0, 0}, {6, 1, 0}, {1, -2, 1}};
  Matrix<double> b{{2, 2}, {-1, -1}, {4, 4}};

  auto x = fsub(A, b);
  REQUIRE_THAT(x(0, 0), WithinRel(2.0));
  REQUIRE_THAT(x(1, 0), WithinRel(-13.0));
  REQUIRE_THAT(x(2, 0), WithinRel(-24.0));
  REQUIRE_THAT(x(0, 1), WithinRel(2.0));
  REQUIRE_THAT(x(1, 1), WithinRel(-13.0));
  REQUIRE_THAT(x(2, 1), WithinRel(-24.0));
}

TEST_CASE("Cholesky solver", "[solve]") {
  Matrix<double> A{
      {4.0, 12.0, -16.0}, {12.0, 37.0, -43.0}, {-16.0, -43.0, 98.0}};
  Vector<double> b{1.0, 2.0, 3.0};

  auto x = solve(A, b);
  auto bp = matmul(A, x);
  REQUIRE_THAT(bp[0], WithinRel(b[0], 1e-6));
  REQUIRE_THAT(bp[1], WithinRel(b[1], 1e-6));
  REQUIRE_THAT(bp[2], WithinRel(b[2], 1e-6));
}
