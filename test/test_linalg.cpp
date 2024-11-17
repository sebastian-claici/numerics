#include "linalg/cholesky.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinRel;

TEST_CASE("Cholesky decomposition", "[decomposition]") {
  Matrix<double> A{
      {4.0, 12.0, -16.0}, {12.0, 37.0, -43.0}, {-16.0, -43.0, 98.0}};

  Cholesky chol(A);
  REQUIRE_THAT(chol.L[0][0], WithinRel(2.0));
  REQUIRE_THAT(chol.L[1][0], WithinRel(6.0));
  REQUIRE_THAT(chol.L[2][0], WithinRel(-8.0));
  REQUIRE_THAT(chol.L[0][1], WithinRel(0.0));
  REQUIRE_THAT(chol.L[1][1], WithinRel(1.0));
  REQUIRE_THAT(chol.L[2][1], WithinRel(5.0));
  REQUIRE_THAT(chol.L[0][2], WithinRel(0.0));
  REQUIRE_THAT(chol.L[1][2], WithinRel(0.0));
  REQUIRE_THAT(chol.L[2][2], WithinRel(3.0));
}
