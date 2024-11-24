#pragma once

#include "core.h"

template <class T> Vector<T> bsub(const Matrix<T> &U, const Vector<T> &b) {
  Vector<T> x(U.m_rows);
  Vector<T> b_aux(b);

  // Assumes U is upper triangular
  for (int i = U.m_rows - 1; i >= 0; --i) {
    x[i] = b_aux[i] / U(i, i);
#pragma omp parallel for
    for (int j = i - 1; j >= 0; --j) {
      b_aux[j] -= b_aux[i] * U(j, i) / U(i, i);
    }
  }

  return x;
}

template <class T> Matrix<T> bsub(const Matrix<T> &U, const Matrix<T> &b) {
  Matrix<T> x(U.m_rows, b.m_cols);
  Matrix<T> b_aux(b);

  // Assumes U is upper triangular
  for (int i = U.m_rows - 1; i >= 0; --i) {
    for (size_t k = 0; k < x.m_cols; ++k) {
      x(i, k) = b_aux(i, k) / U(i, i);
    }

#pragma omp parallel for
    for (int j = i - 1; j >= 0; --j) {
      for (size_t k = 0; k < x.m_cols; ++k) {
        b_aux(j, k) -= b_aux(i, k) * U(j, i) / U(i, i);
      }
    }
  }

  return x;
}
