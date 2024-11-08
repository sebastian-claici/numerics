#include "core.h"
#include <iostream>
#include <numeric>

int main(int argc, char *argv[]) {
  Vector<int> v(2);
  v.push({1, 2});

  std::cout << v << std::endl;
  std::cout << std::accumulate(v.begin(), v.end(), 0) << std::endl;

  Matrix<int> m(2, 2);
  m.push({1, 2, 3, 4});
  std::cout << m << std::endl;

  return 0;
}
