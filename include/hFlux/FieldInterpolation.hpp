#pragma once
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <vector>
#include "FiniteDifferenceWeights.hpp"
#include "common.hpp"

using ExecSpace = Kokkos::DefaultExecutionSpace; 
template<int m, class ViewType>
KOKKOS_INLINE_FUNCTION
void cleanDivergence(ViewType hermite_data, const double hR, const double hZ) {
  // This process will modify a F_Z component of the input field F to make sure the div(F) = 0 analytcally
  // Taking a center of integration to be in the middle of computational domain
  const int nR_hermite_data = hermite_data.extent(0);
  const int nZ_hermite_data = hermite_data.extent(1);
  int iZ0 = nZ_hermite_data / 2;

  auto RBR = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 0);
  auto RBZ = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 2);

  // Make RF_Z(R,Z) = RBZ(R,Z_0) - match at the center with the original interpolating polynomial
  for (int iR = 0; iR < nR_hermite_data; ++iR)
    for (int idR = 0; idR < 2*m+2; ++idR) {
      for (int iZ = 0; iZ < nZ_hermite_data; ++iZ) {
          RBZ(iR, iZ, idR, 0) = RBZ(iR, iZ0, idR, 0); // Constant coefficeint match
          for (int idZ = 1; idZ < 2*m+2; ++idZ)
            RBZ(iR, iZ, idR, idZ) = 0.0;  // Non-constant coefficients 0
        }
        // Fill RBZ coefficients with local - Int dBRdR dZ
        for (int iZ = 0; iZ < nZ_hermite_data; ++iZ)
          for (int idZ = 1; idZ < 2*m+3; ++idZ) {
            if(idR + 1 < 2*m+2)
              RBZ(iR, iZ, idR, idZ) = - RBR(iR, iZ, idR + 1, idZ - 1) * hZ * static_cast<Real>(idR + 1) / hR / static_cast<Real>(idZ);
          }
        // Constant part with area from center to edges in cell iZ0
        for (int iZ = iZ0+1; iZ < nZ_hermite_data; ++iZ) {
          Real II = 0.0; // Integral over the entire cell
          for (int idZ = 1; idZ < 2*m+3; ++idZ) {
            II += RBZ(iR, iZ, idR, idZ) * (std::pow(0.5,idZ) - std::pow(-0.5,idZ));
            RBZ(iR, iZ, idR, 0) += RBZ(iR, iZ0, idR, idZ) * std::pow( 0.5, idZ);
            RBZ(iR, iZ, idR, 0) -= RBZ(iR, iZ,  idR, idZ) * std::pow(-0.5, idZ);
          }
          // Carry out integral to the end of the domain ammending the constant coefficient in Taylor expantion
          for (int iiZ = iZ+1; iiZ < nZ_hermite_data; ++iiZ)
            RBZ(iR, iiZ, idR, 0) += II;
        }
        // Repeat line integration towards bottm
        for (int iZ = 0; iZ < iZ0; ++iZ) {
          Real II = 0.0;
          for (int idZ = 1; idZ < 2*m+3; ++idZ) {
            II += RBZ(iR, iZ, idR, idZ) * (std::pow(-0.5,idZ) - std::pow(0.5,idZ));
            RBZ(iR, iZ, idR, 0) += RBZ(iR, iZ0, idR, idZ) * std::pow(-0.5, idZ);
            RBZ(iR, iZ, idR, 0) -= RBZ(iR, iZ , idR, idZ) * std::pow( 0.5, idZ);
          }
          for (int iiZ = 0; iiZ < iZ; ++iiZ)
            RBZ(iR, iiZ, idR, 0) += II;
        }
      }
}

template<int m, class T>
KOKKOS_INLINE_FUNCTION
void interpolateInPlace1D(T& data)
{
  static_assert(data.rank == 1);

  static const int sz = m+1;
  Kokkos::Array<Kokkos::Array<Real, 2*sz>, 2*sz> NT;

  for (int i = 0; i < sz; ++i)
    for (int idx = 0; idx < sz - i; ++idx)
    {
      NT[i][idx] = data(i);
      NT[i][idx + sz] = data(i + sz);
    }

  //Fill in missing values between known data.
  for (int i = 1; i < sz; ++i)
    for (int idx = sz - i; idx < sz;  ++idx)
      NT[i][idx] = NT[i-1][idx+1] - NT[i-1][idx];

  //Fill in final part of a table
  for (int i = sz; i < 2*sz; ++i)
    for (int idx = 0; idx < 2*sz - i; ++idx)
      NT[i][idx] = NT[i-1][idx+1] - NT[i-1][idx];

  //Get coeffictions
  for(int i = 0; i < 2*sz; ++i)
    data(i)= NT[i][0];

  // Change basis
  for (int k = 2*sz-2; k >= 0; --k)
    for (int j = k; j < 2*sz-1; ++j)
      if (k < sz)
        data(j) += 0.5 * data(j+1);
      else
        data(j) -= 0.5 * data(j+1);
}

template<int m, class ViewHermiteDataType>
KOKKOS_INLINE_FUNCTION
void interpolate2D(ViewHermiteDataType view_hermite_data) {
  assert(view_hermite_data.rank == 2);
  assert(view_hermite_data.extent(0) == 2*m+2);
  assert(view_hermite_data.extent(1) == 2*m+3);

  for (int i = 0; i < 2*m+2; ++i) {
    auto dd1 = Kokkos::subview(view_hermite_data, Kokkos::ALL, i);
    interpolateInPlace1D<m>(dd1);
  }

  for (int i = 0; i < 2*m+2; ++i) {
    auto dd1 = Kokkos::subview(view_hermite_data, i, Kokkos::ALL);
    interpolateInPlace1D<m>(dd1);
  }

}


template<int m, int swidth, class ViewDataType, class ViewHermiteDataType>
KOKKOS_INLINE_FUNCTION
void computeDerivativesStencil(ViewDataType view_data, ViewHermiteDataType view_hermite_data, const Real ratioR, const Real ratioZ) {
  assert(view_data.rank == 2);
  assert(view_data.extent(0) == swidth);
  assert(view_data.extent(1) == swidth);

  assert(view_hermite_data.rank == 2);
  assert(view_hermite_data.extent(0) == m+1);
  assert(view_hermite_data.extent(1) == m+1);

  constexpr auto D = fdw<swidth>();

  Real sclx = 1.0;
  for (int idx = 0; idx < m+1; ++idx)
  {
    Real scly = 1.0;
    for (int idy = 0; idy < m+1; ++idy)
    {
      auto& hdof = view_hermite_data(idx, idy);
      hdof = 0.0;
      for (int i = 0; i < swidth; ++i) {
        for (int j = 0; j < swidth; ++j) {
          hdof += D[idx][i] * D[idy][j] * view_data(i,j) * sclx * scly;
        }
      }
      scly *= ratioZ / (idy + 1);
    }
    sclx *= ratioR / (idx + 1);
  }
}

template<int m, int swidth = 7>
struct FieldInterpolation {
  const int nR_data, nZ_data;
  const int nfields;
  const int ndims = 3;
  const int nphi_data;
  const int nt;

  const Real R0, Z0;
  const Real dR, dZ;

  const int nR_hermite_data, nZ_hermite_data;
  const Real hR0, hZ0;
  const Real hR, hZ;

  Kokkos::View<Real******, ExecSpace> data;
  Kokkos::View<Real********, ExecSpace> hermite_data;





  void compute_derivatives() {
      auto data_ = data;
      auto hermite_data_ = hermite_data;
      Real ratioR = hR / dR, ratioZ = hZ / dZ;
      Kokkos::parallel_for("compute_derivatives",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0,0,0,0,0,0}, {nR_hermite_data,nZ_hermite_data,nfields,ndims,nphi_data,nt}),
      KOKKOS_LAMBDA(int i, int j, int fi, int di, int k, int ti) {
        for (int offx = 0; offx < 2; ++offx) {
          for (int offy = 0; offy < 2; ++offy) {
            int ii = (i + offx) * (swidth-1);
            int jj = (j + offy) * (swidth-1);
            int idx = (m+1) * offx;
            int idy = (m+1) * offy;

            auto sbv_data = Kokkos::subview(data_, Kokkos::make_pair(ii, ii + swidth),
                                                  Kokkos::make_pair(jj, jj + swidth), fi, di, k, ti);
            auto sbv_hermite_data = Kokkos::subview(hermite_data_, i, j, Kokkos::make_pair(idx, idx + m+1),
                                                                        Kokkos::make_pair(idy, idy + m+1), fi, di, k, ti);
            computeDerivativesStencil<m, swidth>(sbv_data, sbv_hermite_data, ratioR, ratioZ);
          }
        }
      });
  }


  public:

  FieldInterpolation(const int nR_data, const int nZ_data, const int nfields, const int nphi_data, const int nt,
                     const Real R0, const Real Z0, const Real dR, const Real dZ) :
    nR_data(nR_data), nZ_data(nZ_data), nfields(nfields), nphi_data(nphi_data), nt(nt),
    R0(R0), Z0(Z0), dR(dR), dZ(dZ),
    nR_hermite_data((nR_data-1) / swidth), nZ_hermite_data((nZ_data-1) / swidth),
    hR0(R0 + (swidth-1)/2*dR), hZ0(Z0 + (swidth-1)/2 *dZ),
    hR(dR * (swidth - 1)), hZ(dZ * (swidth - 1)),
    data("data", nR_data, nZ_data, nfields, ndims, nphi_data, nt),
    hermite_data("hermite_data", nR_hermite_data, nZ_hermite_data, 2*m+2, 2*m+3, nfields, ndims, nphi_data, nt) {
  };

  void interpolate() {
    compute_derivatives();

    auto hermite_data_ = hermite_data;

    Kokkos::parallel_for("interpolate",
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0,0,0,0,0,0}, {nR_hermite_data,nZ_hermite_data, nfields,ndims,nphi_data,nt}),
    KOKKOS_LAMBDA(int i, int j, int fi, int di, int k, int ti){
      auto sbv_hermite_data = Kokkos::subview(hermite_data_, i, j, Kokkos::ALL, Kokkos::ALL, fi, di, k, ti);
      interpolate2D<m>(sbv_hermite_data);
    });

    Real hR_ = hR, hZ_ = hZ;

    Kokkos::parallel_for("cleandiv",
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0,0}, {nphi_data,nt}),
    KOKKOS_LAMBDA(int k, int ti){
      auto sbv_hermite_data = Kokkos::subview(hermite_data_, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 0, Kokkos::ALL, k, ti);
      cleanDivergence<m>(sbv_hermite_data, hR_, hZ_);
    });
  }


  template<class ViewVals>
  KOKKOS_INLINE_FUNCTION
  ERROR_CODE operator()(ViewVals vals, Dim5 X) const
  {
    Real r =  X[2] - hR0;
    Real z =  X[4] - hZ0;
    int ii = static_cast<int> (floor(r / hR));
    int jj = static_cast<int> (floor(z / hZ));

    r = r/hR - ii - 0.5;
    z = z/hZ - jj - 0.5;

    KOKKOS_ASSERT(std::abs(r) <= 0.5);
    KOKKOS_ASSERT(std::abs(z) <= 0.5);
    KOKKOS_ASSERT(hermite_data.extent(0) > ii && ii >= 0);
    KOKKOS_ASSERT(hermite_data.extent(1) > jj && jj >= 0);

    auto sbv = Kokkos::subview(hermite_data, ii, jj, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    for (int fi = 0; fi < nfields; ++fi)
      for (int di = 0; di < ndims; ++di)
        for (int k = 0; k < nphi_data; ++k)
          for (int ti = 0; ti < nt; ++ti) {
            vals(fi, di, k, ti) = 0.0;
            Real sclr = 1.0;
            for (int i = 0; i < sbv.extent(0); ++i) {
                Real sclz = 1.0;
                for (int j = 0; j < sbv.extent(1); ++j) {
                    Real mon = sclr * sclz;
                            vals(fi, di, k, ti) += mon * sbv(i, j, fi, di, k, ti);
                    sclz *= z;
                }
                sclr *= r;
            }
          }

    return ERROR_CODE::SUCCESS;
  }


  decltype(auto) getDataRef() & {
    return data;
  }

  std::array<Real, 4> getCorners() {
    return {hR0, hR0 + nR_hermite_data * hR, hZ0, hZ0 + nZ_hermite_data * hZ};
  }
};

