#include "core.h"

template <class T> T dot(const Vector<T> &a, const Vector<T> &b) {
  if (a.m_n != b.m_n) {
    std::runtime_error("Vector dimensions are incompatible");
  }

  T result = a[0] * b[0];
  for (size_t i = 1; i < a.m_n; ++i) {
    result += a[i] * b[i];
  }

  return result;
}

template <class T> Vector<T> matmul(const Matrix<T> &A, const Vector<T> &v) {
  if (A.m_cols != v.m_n) {
    std::runtime_error("Dimensions are incompatible");
  }

  Vector<T> result;
  return result;
}
