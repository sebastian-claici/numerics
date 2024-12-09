#include "core.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Vector indexing", "[vector]") {
  Vector<int> v1{1, 2, 3};

  REQUIRE(v1[0] == 1);
  REQUIRE(v1[1] == 2);

  REQUIRE(v1[2] == 3);
}

TEST_CASE("Matrix indexing", "[matrix]") {
  Matrix<int> A1{{1, 2}, {3, 4}};

  REQUIRE(A1(0, 0) == 1);
  REQUIRE(A1(0, 1) == 2);
  REQUIRE(A1(1, 0) == 3);
  REQUIRE(A1(1, 1) == 4);
}

TEST_CASE("Matrix transpose", "[matrix]") {
  Matrix<int> A{{1, 2}, {3, 4}};
  auto At = A.transpose();

  REQUIRE(A(0, 0) == At(0, 0));
  REQUIRE(A(0, 1) == At(1, 0));
  REQUIRE(A(1, 0) == At(0, 1));
  REQUIRE(A(1, 1) == At(1, 1));
}

TEST_CASE("Vector equality operator", "[vector]") {
  Vector<int> v1{1, 2, 3};
  Vector<int> v2{1, 2, 3};
  Vector<int> v3{1, 2, 4};

  REQUIRE(v1 == v2);
  REQUIRE(v1 != v3);
}

TEST_CASE("Matrix equality operator", "[matrix]") {
  Matrix<int> A1{{1, 2}, {3, 4}};
  Matrix<int> A2{{1, 2}, {3, 4}};
  Matrix<int> A3{{1, 2}, {5, 6}};

  REQUIRE(A1 == A2);
  REQUIRE(A1 != A3);
}
