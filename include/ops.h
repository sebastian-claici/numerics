#include "core.h"

#include <random>

inline Vector<double> rand(size_t n) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dist;

  auto gen = [&]() { return dist(mersenne_engine); };

  return Vector<double>(n, gen);
}

inline Matrix<double> rand(size_t rows, size_t cols) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dist;

  auto gen = [&]() { return dist(mersenne_engine); };

  return Matrix<double>(rows, cols, gen);
}

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

  Vector<T> result(A.m_rows, 0.0);

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

  // TODO: This needs to be grid searched for each CPU
  const size_t tile_size = 64;

  Matrix<T> result(A.m_rows, B.m_cols, 0.0);
  for (size_t tile = 0; tile < A.m_cols; tile += tile_size) {
#pragma omp parallel for
    for (size_t i = 0; i < A.m_rows; ++i) {
      size_t tile_end = std::min(A.m_cols, tile + tile_size);
      for (size_t j = tile; j < tile_end; ++j) {
        for (size_t k = 0; k < B.m_cols; ++k) {
          result[i][k] += A[i][j] * B[j][k];
        }
      }
    }
  }

  return result;
}
