#include "core.h"

template <class T> T dot(const Vector<T> &a, const Vector<T> &b) {
  if (a.m_n != b.m_n) {
    std::runtime_error("Vector dimensions are incompatible");
  }

  T result = a[0] * b[0];

#pragma omp parallel for
  for (size_t i = 1; i < a.m_n; ++i) {
    result += a[i] * b[i];
  }

  return result;
}

template <class T> Vector<T> matmul(const Matrix<T> &A, const Vector<T> &v) {
  if (A.m_cols != v.m_n) {
    std::runtime_error("Dimensions are incompatible");
  }

  Vector<T> result(A.m_rows, 0);

#pragma omp parallel for
  for (size_t i = 0; i < A.m_rows; ++i) {
    for (size_t j = 0; j < A.m_cols; ++j) {
      result[i] += A[i][j] * v[j];
    }
  }

  return result;
}

template <class T> Matrix<T> matmul(const Matrix<T> &A, const Matrix<T> &B) {
  if (A.m_cols != B.m_rows) {
    std::runtime_error("Dimensions are incompatible");
  }
  // transpose B for contiguous mem access
  auto Bt = B.transpose();

  Matrix<T> result(A.m_rows, B.m_cols, 0);

#pragma omp parallel for
  for (size_t i = 0; i < A.m_rows; ++i) {
    for (size_t k = 0; k < B.m_cols; ++k) {
      for (size_t j = 0; j < A.m_cols; ++j) {
        result[i][k] += A[i][j] * Bt[k][j];
      }
    }
  }

  return result;
}
