#pragma once

#include "core.h"

template <class T>
void bsub(const Matrix<T> &U, const Vector<T> &b, Vector<T> &x) {
  Vector<T> b_aux(b);

  // Assumes U is upper triangular
  for (int i = U.m_rows - 1; i >= 0; --i) {
    x[i] = b_aux[i] / U[i][i];
#pragma omp parallel for
    for (int j = i - 1; j >= 0; --j) {
      b_aux[j] -= b_aux[i] * U[j][i] / U[i][i];
    }
  }
}
