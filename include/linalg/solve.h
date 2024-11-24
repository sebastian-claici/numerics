#pragma once

#include <stdexcept>

#include "core.h"
#include "linalg/cholesky.h"
#include "ops.h"

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

template <class T> Vector<T> fsub(const Matrix<T> &L, const Vector<T> &b) {
  Vector<T> x(L.m_rows);
  Vector<T> b_aux(b);

  // Assumes L is lower triangular
  for (int i = 0; i < L.m_rows; ++i) {
    x[i] = b_aux[i] / L(i, i);
#pragma omp parallel for
    for (int j = i + 1; j < L.m_rows; ++j) {
      b_aux[j] -= b_aux[i] * L(j, i) / L(i, i);
    }
  }

  return x;
}

template <class T> Matrix<T> fsub(const Matrix<T> &L, const Matrix<T> &b) {
  Matrix<T> x(L.m_rows, b.m_cols);
  Matrix<T> b_aux(b);

  // Assumes L is lower triangular
  for (int i = 0; i < L.m_rows; i++) {
    for (size_t k = 0; k < x.m_cols; ++k) {
      x(i, k) = b_aux(i, k) / L(i, i);
    }

#pragma omp parallel for
    for (int j = i + 1; j < L.m_rows; ++j) {
      for (size_t k = 0; k < x.m_cols; ++k) {
        b_aux(j, k) -= b_aux(i, k) * L(j, i) / L(i, i);
      }
    }
  }

  return x;
}

template <class T> Vector<T> solve(const Matrix<T> &A, const Vector<T> &b) {
  // Currently only works for symmetric, positive definite matrices
  if (!symmetric(A)) {
    throw std::logic_error("Solver only implemented for symmetric matrices.");
  }

  try {
    Cholesky chol(A);
    auto Lt = chol.L.transpose();
    // Forward substitution to solve Ly = b
    auto y = fsub(chol.L, b);
    // Backward substitution to solve L^T x = y
    return bsub(Lt, y);
  } catch (const std::invalid_argument &e) {
    throw std::logic_error("Solver not implemented for non-PSD matrices.");
  }
}

template <class T> Matrix<T> solve(const Matrix<T> &A, const Matrix<T> &b);
