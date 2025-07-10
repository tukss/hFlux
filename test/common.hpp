#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>

using Real = double;

template <std::size_t D>
struct Vec : std::array<Real, D>
{
  friend Vec<D> operator+(Vec<D> l, const Vec<D> &r)
  {
    for(decltype(D) i = 0; i < D; ++i) {
      l[i] += r[i];
    }
    return l;
  }
  Vec<D>& operator+=(const Vec<D> &r)
  {
    for(decltype(D) i = 0; i < D; ++i) {
      (*this)[i] += r[i];
    }
    return *this;
  }
};

using Dim2 = Vec<2>;
using Dim3 = Vec<3>;
using Dim5 = Vec<5>;

KOKKOS_INLINE_FUNCTION
void cross_product(const Dim3& A, const Dim3& B, Dim3& cross){
    cross[0] = A[1] * B[2] - A[2] * B[1];
    cross[1] = A[2] * B[0] - A[0] * B[2];
    cross[2] = A[0] * B[1] - A[1] * B[0];
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename T::value_type dot_product(const T& A, const T& B){
    typename T::value_type ret = 0.0;
    for (std::size_t i = 0; i < A.size(); ++i) {
        ret += A[i] * B[i];
    }
    return ret;
}

typedef enum ERROR_CODE_ENUM
{
   SUCCESS = 0,
   ERROR_OUT_OF_BOUNDS,
   LA_NO_PARTICLE,
   WALL_IMPACT,
   MOMENTUM_CUTOFF,
   STILL_BORN,
   DEFRAG,
   TIME_INTERVAL_VIOLATION,
   BEYOND_FIRST_WALL,
   DIMENSION_IS_ZERO,
   H_BELOW_MIN
} ERROR_CODE;
