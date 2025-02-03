#pragma once
#define MDSPAN_USE_BRACKET_OPERATOR 0
#include <algorithm>
#include <vector>
#include <array>
#include <numeric>
#include <iomanip>
#include <sstream>

using Real = double;
using Vector = std::vector<Real>;
using Dim2 = std::array<Real,2>; // R, Z
using Dim3 = std::array<Real,3>; // R, phi, Z
using Dim5 = std::array<Real,5>; // p,xi, R, phi,Z
using Dim6 = std::array<Real,6>; // R phi Z and,  R phi Z  end
using IntDim2 = std::array<int,2>;

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
  BEYOND_FIRST_WALL
} ERROR_CODE;

namespace Flags {
  constexpr bool TimeInterpolation = false;
}

template <typename T>
struct RZIndex; // Primary template (undefined)

template <>
struct RZIndex<Dim2> {
    static constexpr std::size_t R = 0;
    static constexpr std::size_t Z = 1;
};
template <>
struct RZIndex<Dim3> {
    static constexpr std::size_t R = 0;
    static constexpr std::size_t phi = 1;
    static constexpr std::size_t Z = 2;
};

template <>
struct RZIndex<Dim5> {
    static constexpr std::size_t p = 0;
    static constexpr std::size_t xi = 1;
    static constexpr std::size_t R = 2;
    static constexpr std::size_t phi = 3;
    static constexpr std::size_t Z = 4;
};

template <typename ArrayT>
double& getR(ArrayT& arr) {
    return arr[RZIndex<ArrayT>::R];
}

template <typename ArrayT>
double& getZ(ArrayT& arr) {
    return arr[RZIndex<ArrayT>::Z];
}

template <typename ArrayT>
const double& getR(const ArrayT& arr) {
    return arr[RZIndex<ArrayT>::R];
}

template <typename ArrayT>
const double& getZ(const ArrayT& arr) {
    return arr[RZIndex<ArrayT>::Z];
}

inline void cross_product(const Dim3& A, const Dim3& B, Dim3& cross){
    cross[0] = A[1] * B[2] - A[2] * B[1];
    cross[1] = A[2] * B[0] - A[0] * B[2];
    cross[2] = A[0] * B[1] - A[1] * B[0];
}

inline Real dot_product(const Dim3& A, const Dim3& B){
    return std::inner_product(A.cbegin(), A.cend(), B.cbegin(), 0.0);
}


template <int NumDigits, typename T>
std::string pad_with_zeros(T number) {
    static_assert(std::is_integral<T>::value, "T must be an integer type.");

    std::stringstream ss;
    ss << std::setw(NumDigits) << std::setfill('0') << number;
    return ss.str();
}
   
