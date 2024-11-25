#pragma once

#include "core.h"
#include <cmath>
#include <stdexcept>

template <class T> struct Cholesky {
  size_t n;
  Matrix<T> L;

  Cholesky(const Matrix<T> &A) : n(A.m_rows), L(A.m_rows, A.m_cols) {
    // assumes A is square and positive semi-definite
    for (size_t i = 0; i < n; ++i) {
      double diag = A(i, i);
      for (size_t k = 0; k < i; ++k) {
        diag -= L(i, k) * L(i, k);
      }
      if (diag < 0.0)
        throw std::invalid_argument("Matrix is not positive semi-definite.");

      L(i, i) = sqrt(diag);

#pragma omp parallel for
      for (size_t j = i + 1; j < n; ++j) {
        double off_diag = A(i, j);
        for (size_t k = 0; k < i; ++k) {
          off_diag -= L(i, k) * L(j, k);
        }
        L(j, i) = 1. / L(i, i) * off_diag;
      }
    }
  }
};
