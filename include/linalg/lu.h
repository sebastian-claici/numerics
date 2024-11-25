#pragma once

#include "core.h"
#include <stdexcept>

template <class T> struct LU {
  Matrix<T> L;
  Matrix<T> U;

  LU(const Matrix<T> &A) : L(A.m_rows, A.m_cols), U(A.m_rows, A.m_cols) {
    if (A.m_rows != A.m_cols) {
      throw std::invalid_argument(
          "LU decomposition only implemented for square matrices.");
    }

    for (size_t i = 0; i < A.m_rows; ++i) {
      L(i, i) = 1.0;
    }
    for (size_t j = 0; j < A.m_rows; ++j) {
      for (size_t i = 0; i <= j; ++i) {
        T s_term{};
        for (size_t k = 0; k < i; ++k) {
          s_term += L(i, k) * U(k, j);
        }
        U(i, j) = A(i, j) - s_term;
      }

      for (size_t i = j + 1; i < A.m_rows; ++i) {
        T s_term{};
        for (size_t k = 0; k < i; ++k) {
          s_term += L(i, k) * U(k, j);
        }
        L(i, j) = 1.0 / U(j, j) * (A(i, j) - s_term);
      }
    }
  }
};
