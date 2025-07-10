#include <cmath>
#include <vector>
#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "AnalyticField.hpp"
#include "common.hpp"
#include "FiniteDifferenceWeights.hpp"


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

  Kokkos::View<Real******, Kokkos::HostSpace> data;
  Kokkos::View<Real********, Kokkos::HostSpace> hermite_data;


  template<class ViewDataType, class ViewHermiteDataType>
  KOKKOS_INLINE_FUNCTION
  void computeDerivativesStencil(ViewDataType view_data, ViewHermiteDataType view_hermite_data) {
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
        scly *= hZ / dZ / (idy + 1);
      }
      sclx *= hR / dR / (idx + 1);
    }
  }


  template<class T>
  KOKKOS_INLINE_FUNCTION
  void interpolateInPlace1D(T& data)
  {
    static_assert(data.rank == 1);

    static const int sz = m+1;
    std::array<Real, 4*sz*sz > table;
    Kokkos::View<double[2*sz][2*sz], Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        NT(table.data(), table.size());

    for (int i = 0; i < sz; ++i)
      for (int idx = 0; idx < sz - i; ++idx)
      {
        NT(i,idx) = data(i);
        NT(i,idx + sz) = data(i + sz);
      }

    //Fill in missing values between known data.
    for (int i = 1; i < sz; ++i)
      for (int idx = sz - i; idx < sz;  ++idx)
        NT(i, idx) = NT(i-1, idx+1) - NT(i-1, idx);

    //Fill in final part of a table
    for (int i = sz; i < 2*sz; ++i)
      for (int idx = 0; idx < 2*sz - i; ++idx)
        NT(i,idx) = NT(i-1,idx+1) - NT(i-1,idx);

    //Get coeffictions
    for(int i = 0; i < 2*sz; ++i)
      data(i)= NT(i,0);

    // Change basis
    for (int k = 2*sz-2; k >= 0; --k)
      for (int j = k; j < 2*sz-1; ++j)
        if (k < sz)
          data(j) += 0.5 * data(j+1);
        else
          data(j) -= 0.5 * data(j+1);
  }

  template<class ViewHermiteDataType>
  KOKKOS_INLINE_FUNCTION
  void interpolate2D(ViewHermiteDataType view_hermite_data) {
    assert(view_hermite_data.rank == 2);
    assert(view_hermite_data.extent(0) == 2*m+2);
    assert(view_hermite_data.extent(1) == 2*m+3);

    for (int i = 0; i < 2*m+2; ++i) {
      auto dd1 = Kokkos::subview(view_hermite_data, Kokkos::ALL, i);
      interpolateInPlace1D(dd1);
    }

    for (int i = 0; i < 2*m+2; ++i) {
      auto dd1 = Kokkos::subview(view_hermite_data, i, Kokkos::ALL);
      interpolateInPlace1D(dd1);
    }

  }

  void compute_derivatives() {
      Kokkos::parallel_for("compute_derivatives",
      Kokkos::MDRangePolicy<Kokkos::Rank<6>>({0,0,0,0,0,0}, {nR_hermite_data,nZ_hermite_data,nfields,ndims,nphi_data,nt}),
      KOKKOS_LAMBDA(int i, int j, int fi, int di, int k, int ti) {
        for (int offx = 0; offx < 2; ++offx) {
          for (int offy = 0; offy < 2; ++offy) {
            int ii = (i + offx) * (swidth-1);
            int jj = (j + offy) * (swidth-1);
            int idx = (m+1) * offx;
            int idy = (m+1) * offy;

            auto sbv_data = Kokkos::subview(data, std::make_pair(ii, ii + swidth),
                                                  std::make_pair(jj, jj + swidth), fi, di, k, ti);
            auto sbv_hermite_data = Kokkos::subview(hermite_data, i, j, std::make_pair(idx, idx + m+1),
                                                                        std::make_pair(idy, idy + m+1), fi, di, k, ti);
            computeDerivativesStencil(sbv_data, sbv_hermite_data);
          }
        }
      });
  }

  template<class ViewType>
  KOKKOS_INLINE_FUNCTION
  void cleanDivergence(ViewType hermite_data) {
    // This process will modify a F_Z component of the input field F to make sure the div(F) = 0 analytcally
    // Taking a center of integration to be in the middle of computational domain
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

    Kokkos::parallel_for("interpolate",
    Kokkos::MDRangePolicy<Kokkos::Rank<6>>({0,0,0,0,0,0}, {nR_hermite_data,nZ_hermite_data, nfields,ndims,nphi_data,nt}),
    KOKKOS_LAMBDA(int i, int j, int fi, int di, int k, int ti){
      auto sbv_hermite_data = Kokkos::subview(hermite_data, i, j, Kokkos::ALL, Kokkos::ALL, fi, di, k, ti);
      interpolate2D(sbv_hermite_data);
    });

    Kokkos::parallel_for("cleandiv",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nphi_data,nt}),
    KOKKOS_LAMBDA(int k, int ti){
      auto sbv_hermite_data = Kokkos::subview(hermite_data, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 0, Kokkos::ALL, k, ti);
      cleanDivergence(sbv_hermite_data);
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

void run(int nR_data, int nZ_data, Real& hR, std::array<Real, 3>& l2err) {
  static const int m = 2;
  int nfields = 2;
  int nphi_data = 1;
  int nt = 1;
  Real R0 = 1.525;
  Real Z0 = -2.975;
  Real dR = 0.0345;
  Real dZ = 0.02975;

  Real R1 = R0 + 99 * dR;
  Real Z1 = Z0 + 199 * dZ;

  dR = (R1 - R0) / (nR_data-1);
  dZ = (Z1 - Z0) / (nZ_data-1);

  Real q0 = 2.1;
  Real q2 = 2.0;
  Real R_a = 3.0;
  Real E_0 = 70.0;

  AnalyticField af(q0, q2, R_a, E_0);
  FieldInterpolation<m> field_interpolation(nR_data, nZ_data, nfields, nphi_data, nt, R0, Z0, dR, dZ);
  auto field_data = field_interpolation.getDataRef();

  using policy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  Kokkos::parallel_for("setfields",
  policy2D({0,0}, {nR_data,nZ_data}),
  KOKKOS_LAMBDA(int i, int j){
    // linearize: row-major numbering
    auto sbv = Kokkos::subview(field_data,
             i, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Real R = R0 + dR * i, Z = Z0 + dZ * j;
    Dim5 X = {};
    X[2] = R; X[4] = Z;
    Dim3 vB = {}, dBdR = {}, dBdZ = {}, curlB = {}, E = {};
    Real t = 0.0;
    af(X, t, vB, curlB, dBdR, dBdZ, E);
    for (int fi = 0; fi < sbv.extent(0); ++fi) {
      for (int di = 0; di < sbv.extent(1); ++di) {
        for (int k = 0; k < sbv.extent(2); ++k) {
          for (int ti = 0; ti < sbv.extent(3); ++ti) {
            sbv(fi,di,k,ti) = vB[di] * R;
          }
        }
      }
    }
  });

  field_interpolation.interpolate();

  int nR_pl = 400;
  int nZ_pl = 800;
  auto corners = field_interpolation.getCorners();
  Real eps = 1e-8;
  Real R0_pl = corners[0] + eps;
  Real Z0_pl = corners[2] + eps;
  Real dR_pl = (corners[1] - eps - (corners[0] + eps)) / (nR_pl-1);
  Real dZ_pl = (corners[3] - eps - (corners[2] + eps)) / (nZ_pl-1);

  Kokkos::View<Real******> view_pl("plot", nR_pl, nZ_pl, nfields, 3, nphi_data, nt);

  l2err = {};
  Kokkos::parallel_reduce("eval",
  policy2D({0,0}, {nR_pl,nZ_pl}),
  KOKKOS_LAMBDA(int i, int j, Dim3& err){
    // linearize: row-major numbering
    auto sbv = Kokkos::subview(view_pl,
             i, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Real R = R0_pl + dR_pl * i, Z = Z0_pl + dZ_pl * j;
    Dim5 X = {};
    X[2] = R; X[4] = Z;
    field_interpolation(sbv, X);
    Dim3 vB = {}, dBdR = {}, dBdZ = {}, curlB = {}, E = {};
    Real t = 0.0;
    af(X, t, vB, curlB, dBdR, dBdZ, E);

    err[0] += pow(vB[0] - sbv(0, 0, 0, 0) / R, 2);
    err[1] += pow(vB[1] - sbv(0, 1, 0, 0) / R, 2);
    err[2] += pow(vB[2] - sbv(0, 2, 0, 0) / R, 2);
  }, l2err);

  l2err[0] = sqrt(l2err[0] * dR_pl * dZ_pl);
  l2err[1] = sqrt(l2err[1] * dR_pl * dZ_pl);
  l2err[2] = sqrt(l2err[2] * dR_pl * dZ_pl);

  hR = field_interpolation.hR;
}

int main() {
  Kokkos::initialize();
  int NR = 100;
  Dim3 l2err;
  Real hR;
  Real order = (2*2+2);
  for (int ix = 0; ix < 5; ++ix) {
    Real hR_new;
    Dim3 l2err_new;

    run(NR, 2*NR, hR_new, l2err_new);
    if (ix > 0) {
      if (std::abs(pow(l2err[0] / l2err_new[0], 1.0 / (order - 0.0))  - hR / hR_new) > 5.e-3) {
        std::fprintf(stderr, "B_R interpolation did not converge with order %f\n", order);
        Kokkos::finalize();
        return 1;
      }
      if (std::abs(pow(l2err[2] / l2err_new[2], 1.0 / (order - 1.0)) - hR / hR_new) > 4.e-2) {
        std::fprintf(stderr, "B_Z interpolation did not converge with order %f\n", order - 1.0);
        Kokkos::finalize();
        return 2;
      }
    }

    l2err = l2err_new;
    hR = hR_new;

    NR *= 1.5;
  }
  Kokkos::finalize();
  return 0;
}

