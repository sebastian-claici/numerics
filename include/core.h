#ifndef CORE_H_
#define CORE_H_

#include <initializer_list>
#include <ostream>
#include <vector>

template <class T> struct Vector {
  size_t m_n;
  std::vector<T> m_data;

  explicit Vector(size_t n);
  explicit Vector(size_t n, const T &a);
  Vector(size_t n, const T *a);
  Vector(const Vector &rhs);
  Vector(std::initializer_list<T> il);
  Vector &operator=(const Vector &rhs);

  friend std::ostream &operator<<(std::ostream &stream,
                                  const Vector<T> &vector) {
    stream << "[";
    for (size_t i = 0; i < vector.m_n; ++i) {
      stream << vector[i];
      if (i < vector.m_n - 1) {
        stream << ", ";
      }
    }
    stream << "]";

    return stream;
  }

  inline T &operator[](const size_t i);
  inline const T &operator[](const size_t i) const;

  class Iterator {
  public:
    Iterator(T *ptr) : m_ptr(ptr) {}
    Iterator operator++() {
      ++m_ptr;
      return *this;
    }
    bool operator!=(const Iterator &other) const {
      return m_ptr != other.m_ptr;
    }
    const T &operator*() const { return *m_ptr; }

  private:
    T *m_ptr;
  };

  Iterator begin() const { return Iterator(m_data); }
  Iterator end() const { return Iterator(m_data + m_n); }

  void push(std::initializer_list<T> values);

  ~Vector();
};

template <class T> Vector<T>::Vector(size_t n) : m_n(n), m_data(n) {}

template <class T>
Vector<T>::Vector(size_t n, const T &a) : m_n(n), m_data(n, a) {}

template <class T>
Vector<T>::Vector(size_t n, const T *a) : m_n(n), m_data(a) {}

template <class T>
Vector<T>::Vector(const Vector &rhs) : m_n(rhs.m_n), m_data(rhs.m_data) {}

template <class T>
Vector<T>::Vector(std::initializer_list<T> il) : m_n(il.size()), m_data(il) {}

template <class T> Vector<T> &Vector<T>::operator=(const Vector<T> &rhs) {
  m_n = rhs.m_n;
  m_data.resize(m_n);

#pragma omp parallel for
  for (size_t i = 0; i < rhs.m_n; ++i) {
    m_data[i] = rhs.m_data[i];
  }
}

template <class T> inline T &Vector<T>::operator[](size_t i) {
  if (i >= m_n) {
    throw std::runtime_error("Out of bounds");
  }
  return m_data[i];
}

template <class T> inline const T &Vector<T>::operator[](const size_t i) const {
  if (i >= m_n) {
    throw std::runtime_error("Out of bounds");
  }
  return m_data[i];
}

template <class T> void Vector<T>::push(std::initializer_list<T> values) {
  size_t idx = 0;
  for (T value : values) {
    // Explicitly error out here
    if (idx >= m_n) {
      throw std::runtime_error("Too many values to add to vector.");
    }

    m_data[idx++] = value;
  }
}

template <class T> Vector<T>::~Vector() {}

template <class T> struct Matrix {
  size_t m_rows, m_cols;
  std::vector<std::vector<T>> m_data;

  explicit Matrix(size_t rows, size_t cols);
  explicit Matrix(size_t rows, size_t cols, const T &a);
  Matrix(size_t rows, size_t cols, T **a);
  Matrix(const Matrix &rhs);
  Matrix(std::initializer_list<std::initializer_list<T>> il);
  Matrix &operator=(const Matrix &rhs);

  friend std::ostream &operator<<(std::ostream &stream,
                                  const Matrix<T> &matrix) {
    stream << "[";
    for (size_t i = 0; i < matrix.m_rows; ++i) {
      i == 0 ? stream << "[" : stream << " [";
      for (size_t j = 0; j < matrix.m_cols; ++j) {
        stream << matrix[i][j];
        if (j < matrix.m_cols - 1) {
          stream << ", ";
        }
      }
      i < matrix.m_rows - 1 ? stream << "],\n" : stream << "]";
    }
    stream << "]";

    return stream;
  }

  inline std::vector<T> &operator[](const size_t i);
  inline const std::vector<T> &operator[](const size_t i) const;

  Vector<T> row(const size_t i) const;
  Vector<T> col(const size_t i) const;
  Vector<T> diag() const;
  Matrix<T> transpose() const;

  void push(std::initializer_list<T> values);

  ~Matrix();
};

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols)
    : m_rows(rows), m_cols(cols),
      m_data(std::vector<std::vector<T>>(m_rows, std::vector<T>(m_cols))) {
  if (rows == 0 || cols == 0) {
    throw std::runtime_error("Matrix cannot have 0 dimension.");
  }
}

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T &a)
    : m_rows(rows), m_cols(cols),
      m_data(std::vector<std::vector<T>>(m_rows, std::vector<T>(m_cols, a))) {
  if (rows == 0 || cols == 0) {
    throw std::runtime_error("Matrix cannot have 0 dimension.");
  }
}

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols, T **a)
    : m_rows(rows), m_cols(cols), m_data(a) {
  if (rows == 0 || cols == 0) {
    throw std::runtime_error("Matrix cannot have 0 dimension.");
  }

  if (m_data.size() != m_rows) {
    throw std::invalid_argument("Invalid number of rows in input data.");
  }
  for (auto row : m_data) {
    if (row.size() != m_cols) {
      throw std::invalid_argument("Invalid number of cols in input data.");
    }
  }
}

template <class T>
Matrix<T>::Matrix(const Matrix &rhs)
    : m_rows(rhs.m_rows), m_cols(rhs.m_cols), m_data(rhs.m_data) {}

template <class T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> il)
    : m_rows(il.size()), m_cols(il.begin()->size()) {
  for (const auto &row : il) {
    if (row.size() != m_cols) {
      throw std::invalid_argument(
          "All rows must have the same number of columns.");
    }
  }

  m_data = std::vector<std::vector<T>>(m_rows);
  for (const auto &row : il) {
    m_data.push_back(row);
  }
}

template <class T> Matrix<T> &Matrix<T>::operator=(const Matrix &rhs) {
  m_rows = rhs.m_rows;
  m_cols = rhs.m_cols;
  m_data = std::vector<std::vector<T>>(rhs.m_data);
}

template <class T>
inline std::vector<T> &Matrix<T>::operator[](const size_t i) {
  if (i >= m_rows) {
    throw std::runtime_error("Out of bounds.");
  }

  return m_data[i];
}

template <class T>
inline const std::vector<T> &Matrix<T>::operator[](const size_t i) const {
  if (i >= m_rows) {
    throw std::runtime_error("Out of bounds");
  }
  return m_data[i];
}

template <class T> Vector<T> Matrix<T>::row(const size_t i) const {
  if (i >= m_rows) {
    throw std::runtime_error("Out of bounds");
  }
  return Vector<T>(m_cols, m_data[i]);
}

template <class T> Vector<T> Matrix<T>::col(const size_t i) const {
  if (i >= m_cols) {
    throw std::runtime_error("Out of bounds");
  }

  std::vector<T> data(m_rows);
  for (size_t j = 0; j < m_rows; ++j) {
    data[j] = m_data[j][i];
  }

  Vector<T> result(m_rows, data);

  return result;
}

template <class T> Vector<T> Matrix<T>::diag() const {
  if (m_rows != m_cols) {
    throw std::runtime_error("Only implemented for square matrices.");
  }

  std::vector<T> data(m_rows);
  for (size_t j = 0; j < m_rows; ++j) {
    data[j] = m_data[j][j];
  }

  Vector<T> result(m_rows, data);
  return result;
}

template <class T> Matrix<T> Matrix<T>::transpose() const {
  Matrix<T> result(m_cols, m_rows);

#pragma omp parallel for
  for (size_t i = 0; i < m_rows; ++i) {
    for (size_t j = 0; j < m_cols; ++j) {
      result[j][i] = m_data[i][j];
    }
  }

  return result;
}

template <class T> void Matrix<T>::push(std::initializer_list<T> values) {
  size_t idx = 0;
  for (T value : values) {
    // Explicitly error out here
    if (idx >= m_rows * m_cols) {
      throw std::runtime_error("Too many values to add to matrix.");
    }

    size_t idx_row = idx / m_cols;
    size_t idx_col = idx % m_cols;
    m_data[idx_row][idx_col] = value;
    idx++;
  }
}

template <class T> Matrix<T>::~Matrix() {}

#endif // CORE_H_
