#ifndef CHOLESKY_H
#define CHOLESKY_H

#include "core.h"
#include <cmath>

struct Cholesky {
  int n;
  Matrix<double> L;

  Cholesky(const Matrix<double> &A) : n(A.m_rows), L(A.m_rows, A.m_cols, 0.0) {
    // assumes A is square and positive semi-definite
    for (size_t i = 0; i < n; ++i) {
      double diag = A(i, i);
      for (size_t k = 0; k < i; ++k) {
        diag -= L(i, k) * L(i, k);
      }
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

#endif // !CHOLESKY_H
