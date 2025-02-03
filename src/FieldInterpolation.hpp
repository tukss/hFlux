#pragma once
#include <ranges>
#include <cmath>
#include <functional>
#include "common.hpp"
#include "mdspan.hpp"

namespace vw = std::views;
namespace stdex = Kokkos;

namespace FiniteDifferenceTable {
    // Order 1 (First derivative)
    static const std::vector<std::vector<std::vector<Real>>> fdw = {
      {
        {0.0, 0.0, 1.0, 0.0, 0.0},
        {1.0 / 12, -2.0 / 3, 0, 2.0 / 3, -1.0 / 12}, // Accuracy 4
        {-1.0 / 12, 4.0 / 3, -5.0 / 2, 4.0 / 3, -1.0 / 12}, // Accuracy 4
        {-0.5, 1, 0, -1, 0.5},                     // Accuracy 2
        {1, -4, 6, -4, 1},                          // Accuracy 2
        {0.0, 0.0, 0.0, 0.0, 0.0}
      },
      {
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
        {-1.0 / 60, 3.0 / 20, -3.0 / 4, 0, 3.0 / 4, -3.0 / 20, 1.0 / 60}, // Accuracy 6
        {1.0 / 90, -3.0 / 20, 3.0 / 2, -49.0 / 18, 3.0 / 2, -3.0 / 20, 1.0 / 90}, // Accuracy 6
        {1.0 / 8, -1, 13.0 / 8, 0, -13.0 / 8, 1, -1.0 / 8}, // Accuracy 4
        {-1.0 / 6, 2.0, -13.0 / 2, 28.0 / 3, -13.0 / 2, 2.0, -1.0 / 6},  // Accuracy 4
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
      },
      {
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 280, -4.0 / 105, 1.0 / 5, -4.0 / 5, 0, 4.0 / 5, -1.0 / 5, 4.0 / 105, -1.0 / 280}, // Accuracy 8
        {-1.0 / 560, 8.0 / 315, -1.0 / 5, 8.0 / 5, -205.0 / 72, 8.0 / 5, -1.0 / 5, 8.0 / 315, -1.0 / 560}, // Accuracy 8
        {-7.0 / 240, 3.0 / 10, -169.0 / 120, 61.0 / 30, 0, -61.0 / 30, 169.0 / 120, -3.0 / 10, 7.0 / 240}, // Accuracy 6
        {7.0 / 240, -2.0 / 5, 169.0 / 60, -122.0 / 15, 91.0 / 8, -122.0 / 15, 169.0 / 60, -2.0 / 5, 7.0 / 240}, // Accuracy 6
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
      }
    };
}

template<int STENCIL_RADIUS, int M>
class FieldInterpolation
{
    public:
        FieldInterpolation(const IntDim2 N_, const Dim2 x0_, const Dim2 d_);
        ~FieldInterpolation(){};
        void setRawGridData(const Vector& data);
        double* getRawFieldGridDataPtr();
        void readRawGridData(std::string dataset_name, std::string filaname = "../../InitialData/input.h5");
        void updateJre();

        void interpolate();
        void operator()(Dim6& B, Dim6& dBdR, Dim6& dBdZ,
                        Dim6& E,
                        const Dim2& X);
        void operator()(Dim6& B, Dim6& dBdR, Dim6& dBdZ,
                        Dim6& E,
                        Dim6& V,
                        Dim2& Psi,
                        const Dim2& X);
        void operator()(Vector& B, Vector& dBdR, Vector& dBdZ,
                        Vector& E,
                        const Vector& X);
        ERROR_CODE operator()(Vector& B, Vector& curlB, Vector& dBdR, Vector& dBdZ,
                        Vector& E,
                        Vector& V,
                        Vector& Psi,
                        const Vector& X);
        ERROR_CODE analytic(Vector& B, Vector& curlB, Vector& dBdR, Vector& dBdZ,
                        Vector& E,
                        Vector& V,
                        Vector& Psi,
                        const Vector& X);
        template<size_t Xdim>
        ERROR_CODE getFDIndex(const std::array<Real, Xdim>& X, int& index, Dim2& loc);
        template<size_t Xdim>
        ERROR_CODE getHermiteIndex(const std::array<Real, Xdim>& X, int& index, Dim2& loc);
        template<size_t Xdim>
        ERROR_CODE checkWall(const std::array<Real, Xdim>& X);

        void curlB(std::array<Real,6>& curlB_, const std::array<Real, 2>& X);
        template <class T>
        ERROR_CODE evalDiv0Fields(Dim3& B, Dim3& curlB, Dim3& dBdR, Dim3& dBdZ, Dim3& E, const T& X, const Real& t);
        template <class T>
        ERROR_CODE evalBFourier(Dim3& B, const T& X, const Real& t);
        template <class T>
        ERROR_CODE evalVJre(Dim3& V, Dim3& Jre, const T& X, const Real& t);
        template <class T>
        ERROR_CODE evalPsi(Real& Psi, const T& X, const Real& t);
        void setTimeFrame(const Real& tl_, const Real& tr_);

        ERROR_CODE plot();
        ERROR_CODE plotAnalyticFields();
        void error(std::array<Real, 9> & aerr, std::array<Real, 9> & rerr);
        void writeGrids();
        void checkDims() const;

        static constexpr int dim_in = 2;  // R , Z
        #ifdef FOURIER
        static constexpr int dim_time = 9;  // t begin t end
        #else // NOT FOURIER
        static constexpr int dim_time = 2;  // t begin t end
        #endif
        static constexpr int dim_out = 3; // R , phi , Z
        static constexpr int nFields = 4; // B, gradB, V, E
        static constexpr IntDim2 m{M,M};
        static constexpr IntDim2 nDoF{2*m[0]+2, 2*m[1]+2};
        static constexpr IntDim2 Npl{400, 800};


        using gridDataExtents = stdex::extents<int, std::dynamic_extent, std::dynamic_extent, nFields, dim_out, dim_time>;
        using gridDataMDSpan = stdex::mdspan<Real, gridDataExtents>;
        using hermiteDataExtents = stdex::extents<int, std::dynamic_extent, std::dynamic_extent, nDoF[0], nDoF[1], nFields, dim_out, dim_time>;
        using hermiteDataMDSpan = stdex::mdspan<Real, hermiteDataExtents>;
        using psiExtents = stdex::extents<int, std::dynamic_extent, std::dynamic_extent, nDoF[0]+1, nDoF[1]+1, dim_time>;
        using psiMDSpan = stdex::mdspan<Real, psiExtents>;
        using FExtents = stdex::extents<int, std::dynamic_extent, std::dynamic_extent, nDoF[0], nDoF[1]+1, 3, dim_out, dim_time>; // B, V, J_re
        using FMDSpan = stdex::mdspan<Real, FExtents>;
        using levelMDSpan = stdex::mdspan<int, stdex::extents<int, std::dynamic_extent, std::dynamic_extent>>;
        using jreMDSpan  = stdex::mdspan<Real, stdex::extents<int, std::dynamic_extent, std::dynamic_extent, 1, 3, 1>>;
        using hermiteJreMDSpan  = stdex::mdspan<Real, stdex::extents<int, std::dynamic_extent, std::dynamic_extent, nDoF[0], nDoF[1], 1, 3, 1>>;

        static constexpr int FCellSize = FExtents::static_extent(2) * FExtents::static_extent(3) * FExtents::static_extent(4) * FExtents::static_extent(5) * FExtents::static_extent(6);
        static constexpr int psiCellSize = psiExtents::static_extent(2) * psiExtents::static_extent(3) * psiExtents::static_extent(4);

        IntDim2 N;
        Dim2 d; 
        Dim2 x0fd, x0, h;
        IntDim2 nCells;

        Vector rawHermiteData, rawPsi, rawF;
        std::vector<int> rawLevel; 
        Vector rawJre;
        
        Real tl, tr;

        #ifdef FOURIER
        Vector rawChi;
        #endif
    private:
        template<class fdDataMDS, class hDataMDS>
        void computeDerivatives(Vector& grid, Vector& hermite);
        template<class hDataMDS>
        void interpolateHermite(Vector& hermite);
        void computeFlux(Vector& raw, int intZsource = 0, bool doLineRIntegration = true);
        void computeBZ();
        void assemble();
        void assembleJRE();
        void readLevelFile();

        Vector rawGridData;
    public:
        int PlotCounter;
};

struct TimeInterpolation
{ 
    Real c1, c2;
    TimeInterpolation(const Real& ta, const Real& tb, const Real& t):
    c1((tb - t)/(tb - ta)),
    c2((t - ta)/(tb - ta)){}
    Real operator()(const Real& a, const Real& b){
      if constexpr (Flags::TimeInterpolation) return c1*a + c2*b;
      return a;
    }
 
};

template<int STENCIL_RADIUS, int M>
template <class T>
ERROR_CODE FieldInterpolation<STENCIL_RADIUS,M>::evalDiv0Fields(Dim3& B, Dim3& curlB, Dim3& dBdR, Dim3& dBdZ, Dim3& E, const T& X, const Real& t)
{
    B = {}; dBdR = {}; dBdZ = {}; E = {}; curlB = {};
    Dim2 Xloc{}; 
    int index = -1; 
    ERROR_CODE ret = getHermiteIndex(X, index, Xloc);
    if (ret != SUCCESS){
      printf("Error out of bounds!\n");
      return ret;
    } 

    TimeInterpolation ti(tl, tr, t);

    auto F = stdex::mdspan(rawF.data() + index * FCellSize, 
                           FExtents::static_extent(2), FExtents::static_extent(3), FExtents::static_extent(4), FExtents::static_extent(5), FExtents::static_extent(6));
    
    Dim3 V = {}, J_re = {};

    Real sclr = 1.0;
    for (int i = 0; i < static_cast<int>(F.extent(0)); ++i) {
        Real sclz = 1.0;
        for (int j = 0; j < static_cast<int>(F.extent(1)); ++j) {
            Real mon = sclr * sclz;
            for (int k = 0; k < 3; ++k) {
                B[k] += ti(F(i,j,0,k,0), F(i,j,0,k,1)) * mon;
                if (i+1 < F.extent(0)) dBdR[k] += static_cast<Real>(i+1) * ti(F(i+1,j,0,k,0), F(i+1,j,0,k,1)) * mon;
                if (j+1 < F.extent(1)) dBdZ[k] += static_cast<Real>(j+1) * ti(F(i,j+1,0,k,0), F(i,j+1,0,k,1)) * mon;

                V[k] += ti(F(i,j,1,k,0), F(i,j,1,k,1)) * mon;
                J_re[k] += F(i,j,2,k,0) * mon;
            }
            sclz *= Xloc[1];
        }
        sclr *= Xloc[0];
    }

    const Real& XR = getR(X);
 
    std::transform(B.begin(), B.end(), B.begin(),
               std::bind(std::divides(), std::placeholders::_1, XR));
    curlB[2] = dBdR[1] / XR / h[0];

    std::transform(dBdR.begin(), dBdR.end(), B.begin(), dBdR.begin(), [&](Real& a, Real& b){
        return (a / h[0] - b) / XR;
    });

    std::transform(dBdZ.begin(), dBdZ.end(), dBdZ.begin(),
               std::bind(std::divides(), std::placeholders::_1, XR * h[1]));

    curlB[0] = -dBdZ[1];
    curlB[1] = dBdZ[0] - dBdR[2];
    cross_product(B,V,E);
    // E = - VxB + \eta / mu0 (\nabla x B - muJre)
//    for (auto k: vw::iota(0,3))
//      E[k]   = PhysicalParameters::E_n * (E[k] + PhysicalParameters::eta_mu0aVa*(curlB[k]) - J_re[k]);

    return ERROR_CODE::SUCCESS;
}

template<int STENCIL_RADIUS, int M>
template <class T>
ERROR_CODE FieldInterpolation<STENCIL_RADIUS,M>::evalVJre(Dim3& V, Dim3& J_re, const T& X, const Real& t)
{
    V = {}; J_re = {};
    Dim2 Xloc{}; 
    int index = -1; 
    ERROR_CODE ret = getHermiteIndex(X, index, Xloc);
    if (ret != SUCCESS) return ret;

    TimeInterpolation ti(tl, tr, t);

    auto F = stdex::mdspan(rawF.data() + index * FCellSize, 
                           FExtents::static_extent(2), FExtents::static_extent(3), FExtents::static_extent(4), FExtents::static_extent(5), FExtents::static_extent(6));

    Real sclr = 1.0;
    for (auto i : vw::iota(0, static_cast<int>(F.extent(0))))
    {
        Real sclz = 1.0;
        for (auto j : vw::iota(0, static_cast<int>(F.extent(1))))
        {
            Real mon = sclr * sclz;
            for (auto k: vw::iota(0,3))
            {
                V[k] += ti(F(i,j,1,k,0), F(i,j,1,k,1)) * mon;
                J_re[k] += F(i,j,2,k,0) * mon;
            }
            sclz *= Xloc[1];
        }
        sclr *= Xloc[0];
    }

    return ERROR_CODE::SUCCESS;
}

template<int STENCIL_RADIUS, int M>
template<class T>
ERROR_CODE FieldInterpolation<STENCIL_RADIUS,M>::evalPsi(Real& Psi, const T& X, const Real& t)
{
    Dim2 Xloc; int index; 
    Psi = 0.0;
    ERROR_CODE ret = getHermiteIndex(X, index, Xloc);
    if (ret != SUCCESS) return ret;

    TimeInterpolation ti(tl, tr, t);

    auto psi = stdex::mdspan(rawPsi.data() + index * psiCellSize, psiExtents::static_extent(2), psiExtents::static_extent(3), psiExtents::static_extent(4));

    Real sclr = 1.0;
    for (auto i : vw::iota(0, static_cast<int>(psi.extent(0)))){
        Real sclz = 1.0;
        for (auto j : vw::iota(0, static_cast<int>(psi.extent(1)))){
            Real mon = sclr * sclz;
            Psi += ti(psi(i,j,0),psi(i,j,1)) * mon;
            sclz *= Xloc[1];
        }
        sclr *= Xloc[0];
    }

    return SUCCESS;
}

#ifdef FOURIER
template<int STENCIL_RADIUS, int M>
template <class T>
ERROR_CODE FieldInterpolation<STENCIL_RADIUS,M>::evalBFourier(Dim3& B, const T& X, const Real& t)
{
    B = {}; 
    Dim2 Xloc{}; 
    int index = -1; 
    ERROR_CODE ret = getHermiteIndex(X, index, Xloc);
    if (ret != SUCCESS){
      printf("Error out of bounds!\n");
      return ret;
    } 
    auto F = stdex::mdspan(rawF.data() + index * FCellSize, 
                           FExtents::static_extent(2), FExtents::static_extent(3), FExtents::static_extent(4), FExtents::static_extent(5), FExtents::static_extent(6));
    auto chi = stdex::mdspan(rawChi.data() + index * psiCellSize, psiExtents::static_extent(2), psiExtents::static_extent(3), psiExtents::static_extent(4));
    const Real& XR = getR(X);
    
    Real sclr = 1.0;
    for (int i = 0; i < static_cast<int>(F.extent(0)); ++i) {
        Real sclz = 1.0;
        for (int j = 0; j < static_cast<int>(F.extent(1)); ++j) {
            Real mon = sclr * sclz;
            for (int k = 0; k < 3; ++k) {
                B[k] += F(i,j,0,k,0) * mon;
                for (int kk = 1; kk < 5; ++kk) {
                  B[k] += F(i,j,0,k,kk) * mon * cos(kk * t);
                  B[k] -= F(i,j,0,k,kk+4) * mon * sin(kk * t);
                }
            }
            sclz *= Xloc[1];
        }
        sclr *= Xloc[0];
    }

    sclr = 1.0;
    for (auto i : vw::iota(0, static_cast<int>(chi.extent(0)))){
        Real sclz = 1.0;
        for (auto j : vw::iota(0, static_cast<int>(chi.extent(1)))){
            Real mon = sclr * sclz;
            for (int kk = 1; kk < 5; ++kk) {
                B[2] += chi(i, j, kk) * mon * sin(kk * t) * kk / XR;
                B[2] += chi(i, j, kk + 4) * mon * cos(kk * t) * kk / XR;
            }
            sclz *= Xloc[1];
        }
        sclr *= Xloc[0];
    }

 
    std::transform(B.begin(), B.end(), B.begin(),
               std::bind(std::divides(), std::placeholders::_1, XR));

    return ERROR_CODE::SUCCESS;
}
 #endif

#ifdef FOURIER
template<int STENCIL_RADIUS, int M>
template <class T>
ERROR_CODE FieldInterpolation<STENCIL_RADIUS,M>::evalPsiFourier(Real& Psi, const T& X, const Real& t)
{
    Dim2 Xloc; int index; 
    Psi = 0.0;
    ERROR_CODE ret = getHermiteIndex(X, index, Xloc);
    if (ret != SUCCESS) return ret;

    auto psi = stdex::mdspan(rawPsi.data() + index * psiCellSize, psiExtents::static_extent(2), psiExtents::static_extent(3), psiExtents::static_extent(4));

    Real sclr = 1.0;
    for (auto i : vw::iota(0, static_cast<int>(psi.extent(0)))){
        Real sclz = 1.0;
        for (auto j : vw::iota(0, static_cast<int>(psi.extent(1)))){
            Real mon = sclr * sclz;
            Psi += psi(i,j,0) * mon;
            for (int kk = 1; kk < 5; ++kk) {
              Psi += psi(i,j,kk) * mon * cos(kk * t);
              Psi -= psi(i,j,kk+4) * mon * sin(kk * t);
            }
            sclz *= Xloc[1];
        }
        sclr *= Xloc[0];
    }

    return SUCCESS;
}
#endif




template<size_t Xdim>
inline ERROR_CODE getIndex(const Dim2& corner, const Dim2& cellWidth, const IntDim2& Ncells,
                           const std::array<Real, Xdim>& X, int& index, Dim2& loc) {
    index = -1; loc = {};
    IntDim2 ii;
    {   
        constexpr int dim = 0;
        const Real& var = getR(X);
        loc[dim] = (var - corner[dim]) / cellWidth[dim];
        ii[dim] = std::floor(loc[dim]);
        if (ii[dim] < 0 or ii[dim] > Ncells[dim]-1) 
            return ERROR_OUT_OF_BOUNDS;
        loc[dim] -= static_cast<Real> (ii[dim]) + 0.5;
    }

    {   
        constexpr int dim = 1;
        const Real& var = getZ(X);
        loc[dim] = (var - corner[dim]) / cellWidth[dim];
        ii[dim] = std::floor(loc[dim]);
        if (ii[dim] < 0 or ii[dim] > Ncells[dim]-1) 
            return ERROR_OUT_OF_BOUNDS;
        loc[dim] -= static_cast<Real> (ii[dim]) + 0.5;
    }

    index = ii[0] * Ncells[1] + ii[1];
    return SUCCESS;
}

template<int STENCIL_RADIUS, int M>
template<size_t Xdim>
ERROR_CODE FieldInterpolation<STENCIL_RADIUS,M>::getFDIndex(const std::array<Real, Xdim>& X, int& index, Dim2& loc) {
    return getIndex(x0fd, d, N, X, index, loc);
}

template<int STENCIL_RADIUS, int M>
template<size_t Xdim>
ERROR_CODE FieldInterpolation<STENCIL_RADIUS,M>::getHermiteIndex(const std::array<Real, Xdim>& X, int& index, Dim2& loc) {
    return getIndex(x0, h, nCells, X, index, loc);
}


template<int STENCIL_RADIUS, int M>
template<size_t Xdim>
ERROR_CODE FieldInterpolation<STENCIL_RADIUS,M>::checkWall(const std::array<Real, Xdim>& X) {
    int i; Dim2 loc;
    auto ret = getFDIndex(X, i, loc);
    if (rawLevel[i] < 0) {
        return ERROR_CODE::BEYOND_FIRST_WALL;
    } 
    else if (rawLevel[i] < 1) {
        return ERROR_CODE::WALL_IMPACT;
    }
    return SUCCESS;
}


template<int STENCIL_RADIUS, int M>
template<class fdDataMDS, class hDataMDS>
void FieldInterpolation<STENCIL_RADIUS,M>::computeDerivatives(Vector& grid, Vector& hermite)
{
  auto D = FiniteDifferenceTable::fdw[STENCIL_RADIUS-2];
  // D[__M__] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  auto hh = hDataMDS(hermite.data(), nCells[0], nCells[1]);
  auto f = fdDataMDS(grid.data(), N[0], N[1]);
  // Assemble data for hermite interpolation
  #pragma omp parallel for
  for (auto ii : vw::iota(0,hh.extent(0)+1))
    for (auto jj : vw::iota(0,hh.extent(1)+1))
      for (auto k1 : vw::iota(0,hh.extent(4)))
      for (auto k2 : vw::iota(0,hh.extent(5)))
      for (auto k3 : vw::iota(0,hh.extent(6)))
      {
        auto i = ii*STENCIL_RADIUS * 2;
        auto j = jj*STENCIL_RADIUS * 2;

        assert(i < f.extent(0));
        assert(j < f.extent(1));
        assert(i >= 0);
        assert(j >= 0);

        for (auto sti : {0,1})
          for (auto stj : {0,1})
          {
            if((ii - sti < nCells[0]) and (jj - stj < nCells[1])
            and(ii - sti >=  0) and (jj - stj >= 0))
            {
              Real sclx = 1.0;
              for (auto idx: vw::iota(0, m[0]+1)) {
                Real scly = 1.0;
                for (auto idy: vw::iota(0, m[1]+1)) {
                    auto& hcof = hh(ii-sti,jj-stj, idx + ((m[0])+1)*sti, idy + ((m[1])+1)*stj, k1,k2,k3);
                    hcof = 0.0;
                    for (auto iidx: vw::iota(0, 2 * STENCIL_RADIUS + 1))
                    {
                        for (auto iidy: vw::iota(0, 2 * STENCIL_RADIUS + 1))
                        {
                            hcof += f(i+iidx,j+iidy,k1,k2,k3) * D[idx][iidx] * D[idy][iidy] * sclx * scly;
                        }
                    }
                    scly *= h[1] / d[1] / (idy + 1);
                }
                sclx *= h[0] / d[0] / (idx + 1);
              }
            }
          }
      }
}



template<int STENCIL_RADIUS, int M, class T>
void interpolateInPlace1D(T& data)
{
  assert(data.size() % 2 == 0);
  const int sz = data.size()/2;
  assert((sz == FieldInterpolation<STENCIL_RADIUS,M>::m[0]+1) or (sz == FieldInterpolation<STENCIL_RADIUS,M>::m[1]+1));

  Vector arr(4*sz*sz);
  stdex::mdspan NT{arr.data(), 2*sz, 2*sz};

  for (int i = 0; i < sz; ++i)
    for (int idx = 0; idx < sz - i; ++idx)
    {
      NT(i,idx) = data[i];
      NT(i,idx + sz) = data[i + sz];
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
    data[i]= NT(i,0);

  // Change basis
  for (int k = 2*sz-2; k >= 0; --k)
    for (int j = k; j < 2*sz-1; ++j)
      if (k < sz)
        data[j] += 0.5 * data[j+1];
      else
        data[j] -= 0.5 * data[j+1];
}

template<int STENCIL_RADIUS, int M>
template<class hDataMDS>
void FieldInterpolation<STENCIL_RADIUS,M>::interpolateHermite(Vector& hermite)
{
  auto hh = hDataMDS(hermite.data(), nCells[0], nCells[1]);

  #pragma omp parallel for
  for (auto ii : vw::iota(0,hh.extent(0)))
    for (auto jj : vw::iota(0,hh.extent(1)))
    {
      std::array<Real, nDoF[0]> dd0;
      std::array<Real, nDoF[1]> dd1;
      for (auto k1  : vw::iota(0, hh.extent(4)))
      for (auto k2  : vw::iota(0, hh.extent(5)))
      for (auto k3  : vw::iota(0, hh.extent(6)))
      {
        for (auto i : vw::iota(0,hh.extent(2)))
        {
          for (auto j : vw::iota(0,hh.extent(3)))
          {
            dd1[j] = hh(ii,jj,i,j,k1,k2,k3);
          }
          interpolateInPlace1D(dd1);
          for (auto j : vw::iota(0,hh.extent(3)))
            hh(ii,jj,i,j,k1,k2,k3) = dd1[j];
        }

        for (auto j : vw::iota(0,hh.extent(3)))
        {
          for (auto i : vw::iota(0,hh.extent(2))){
            dd0[i] = hh(ii,jj,i,j,k1,k2,k3);
          }
          interpolateInPlace1D(dd0);
          for (auto i : vw::iota(0,hh.extent(2))){
            hh(ii,jj,i,j,k1,k2,k3) = dd0[i];
          }
        }

      }
    }
}

const Real Rplmin = 1.526316789473684;
const Real Zplmin = -4.487178487179487;

constexpr double HF_R0 = 1.61125;
constexpr double HF_Z0 = -2.900625;

template<class T>
void linspace(T& v, const Real& a, const Real& w)
{
  Real d = w / static_cast<Real>(v.size()-1);
  Real a_ = a-d;
  std::ranges::generate(v, [&a_, d]() mutable {return a_ += d;} );
  // assert(*(v.end()-1) dd== a + w);
}

template<int STENCIL_RADIUS, int M>
FieldInterpolation<STENCIL_RADIUS,M>::FieldInterpolation(const IntDim2 N_, const Dim2 x0_, const Dim2 d_):
  N(N_),
  d(d_),
  x0fd(x0_),
  x0({x0_[0] + STENCIL_RADIUS * d[0], x0_[1] + STENCIL_RADIUS * d[1]}),
  h({STENCIL_RADIUS * 2 * d[0], STENCIL_RADIUS * 2 * d[1]}),
  nCells({(N[0]-1) / (2* STENCIL_RADIUS)-1, (N[1]-1) / (2*STENCIL_RADIUS)-1}),
  rawHermiteData(nCells[0] * nCells[1] * nDoF[0] * nDoF[1] * 12 * dim_time, 0.0),
  rawPsi(nCells[0] * nCells[1] * (nDoF[0]+1) * (nDoF[1]+1) * dim_time, 0.0),
  rawF(nCells[0] * nCells[1] * nDoF[0] * (nDoF[1]+1) * 9 * dim_time, 0.0),
  rawLevel(N[0] * N[1], 0),
  rawJre((N[0]) * (N[1]) * 1*3*1),
  tl(0.0), tr(1.0),
  rawGridData((N[0]) * (N[1]) * nFields*dim_out*dim_time, 0.0),
  PlotCounter(0)
  #ifdef FOURIER
  ,rawChi(nCells[0] * nCells[1] * (nDoF[0]+1) * (nDoF[1]+1) * dim_time, 0.0)
  #endif
{
  readLevelFile();
}

template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::setRawGridData(const Vector& data)
{
  rawGridData.assign(data.begin(), data.end());
}

template<int STENCIL_RADIUS, int M>
Real* FieldInterpolation<STENCIL_RADIUS,M>::getRawFieldGridDataPtr() {
  unsigned long ptr = (unsigned long)rawGridData.data();
  ptr = (unsigned long)&(rawGridData[0]);
  // printf("%d %d\n",  sizeof(double), sizeof(unsigned long));
  return (Real*) ptr;
}


template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::interpolate()
{
  computeDerivatives<gridDataMDSpan, hermiteDataMDSpan>(rawGridData, rawHermiteData);
  interpolateHermite<hermiteDataMDSpan>(rawHermiteData);
  assemble();
  computeBZ();
  computeFlux(rawPsi);
  #ifdef FOURIER
  computeFlux(rawChi, 2, false);
  #endif
}

template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::updateJre()
{
  computeDerivatives<jreMDSpan, hermiteJreMDSpan>(rawJre, rawHermiteData);
  interpolateHermite<hermiteJreMDSpan>(rawHermiteData);
  assembleJRE();
}


template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::checkDims() const
{
}



template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::computeFlux(Vector& raw, int intZsource, bool doLineRIntegration)
{
  auto psi = psiMDSpan(raw.data(), nCells[0], nCells[1]);
  auto hh = hermiteDataMDSpan(rawHermiteData.data(), nCells[0], nCells[1]);
  std::fill(rawPsi.begin(), rawPsi.end(), 0.0);

  const IntDim2 ic = {(nCells[0])/2, (nCells[1])/2};


  for (auto k: vw::iota(0, dim_time))
  {
    for (auto iR: vw::iota(0, nCells[0]))
    {
        // Z integration of BR (3*k) index in hermite data hh
        // Compute hermite interpolation integral in the center cell
        for (auto idR: vw::iota(0, nDoF[0]))
        {
            Real IIplus = 0.0, IIminus = 0.0;
            {
              auto iZ = ic[1];
              for (auto idZ: vw::iota(0, nDoF[1]))
              {
                  Real& Psi = psi(iR, iZ, idR, (idZ)+1,k);
                  const Real& RBR = hh(iR, iZ, idR, idZ, 0, intZsource, k);
                  Real IR = - RBR * (h[1]) / static_cast<Real>(idZ+1);
                  Psi = IR;
                  IIplus  += IR * pow( 0.5,(idZ)+1);
                  IIminus += IR * pow(-0.5,(idZ)+1);
              }
              for (auto jj: vw::iota(iZ+1, nCells[1]))
                  psi(iR, jj, idR, 0, k) += IIplus;
              for (auto jj: vw::iota(0, iZ))
                  psi(iR, jj, idR, 0, k) += IIminus;
            }

            for (auto iZ: vw::iota(ic[1]+1, nCells[1]))
            {
                IIplus = 0.0;
                Real& Psi_0 = psi(iR, iZ, idR, 0,k);
                for (auto idZ: vw::iota(0, nDoF[1]))
                {
                    Real& Psi = psi(iR, iZ, idR, (idZ)+1,k);
                    const Real& RBR = hh(iR, iZ, idR, idZ, 0, intZsource, k);
                    Real IR = - RBR * (h[1]) / static_cast<Real>(idZ+1);
                    Psi = IR;
                    IIplus += IR * pow( 0.5,(idZ)+1);
                    IIplus -= IR * pow(-0.5,(idZ)+1);
                    Psi_0  -= IR * pow(-0.5,(idZ)+1);
                }
                for (auto jj: vw::iota(iZ+1, nCells[1]))
                    psi(iR, jj, idR, 0,k) += IIplus;
            }        

            for (auto iZ: vw::iota(0, ic[1]))
            {
                IIminus = 0.0;
                Real& Psi_0 = psi(iR, iZ, idR, 0,k);
                for (auto idZ: vw::iota(0, nDoF[1]))
                {
                    Real& Psi = psi(iR, iZ, idR, (idZ)+1,k);
                    const Real& RBR = hh(iR, iZ, idR, idZ, 0, intZsource, k);
                    Real IR = -RBR * (h[1]) / static_cast<Real>(idZ+1);
                    Psi = IR;
                    IIminus -= IR * pow( 0.5,(idZ)+1);
                    IIminus += IR * pow(-0.5,(idZ)+1);
                    Psi_0   -= IR * pow(0.5,(idZ)+1);
                }
                for (auto jj: vw::iota(0, iZ))
                    psi(iR, jj, idR, 0,k) += IIminus;
            }        
        }
    }
    // R integration of BZ(Z_0) (3*k) index in hermite data hh

    if (!doLineRIntegration) return;

    for (auto iZ: vw::iota(0, nCells[1]))
    {
        // Z integration of BR (3*k) index in hermite data hh
        // Compute hermite interpolation integral in the center cell
        Real IIplus = 0.0, IIminus = 0.0;
        {
          auto iR = ic[0];
          for (auto idR: vw::iota(0, nDoF[0]))
          {
              Real& Psi = psi(iR, iZ, idR+1, 0, k);
              const Real& RBZ = hh(iR, ic[1], idR, 0, 0, 2, k);
              Real IR = RBZ * (h[0]) / static_cast<Real>(idR+1);
              Psi += IR;
              IIplus  += IR * pow( 0.5,idR+1);
              IIminus += IR * pow(-0.5,idR+1);
          }
          for (auto jj: vw::iota(iR+1, nCells[0]))
              psi(jj, iZ, 0, 0, k) += IIplus;
          for (auto jj: vw::iota(0, iR))
              psi(jj, iZ, 0, 0, k) += IIminus;
        }

        for (auto iR: vw::iota(ic[0]+1, nCells[0]))
        {
            IIplus = 0.0;
            Real& Psi_0 = psi(iR, iZ, 0, 0,k);
            for (auto idR: vw::iota(0, nDoF[0]))
            {
                Real& Psi = psi(iR, iZ, idR+1, 0,k);
                const Real& RBZ = hh(iR, ic[1], idR, 0, 0, 2, k);
                Real IR = RBZ * (h[0]) / static_cast<Real>(idR+1);
                Psi += IR; 
                IIplus += IR * pow( 0.5,(idR)+1);
                IIplus -= IR * pow(-0.5,(idR)+1);
                Psi_0  -= IR * pow(-0.5,(idR)+1);
            }
            for (auto jj: vw::iota(iR+1, nCells[0]))
                psi(jj, iZ, 0, 0,k) += IIplus;
        }        

        for (auto iR: vw::iota(0, ic[0]))
        {
            IIminus = 0.0;
            Real& Psi_0 = psi(iR, iZ, 0, 0,k);
            for (auto idR: vw::iota(0, nDoF[0]))
            {
                Real& Psi = psi(iR, iZ, idR+1, 0,k);
                const Real& RBZ = hh(iR, ic[1], idR, 0,  0, 2, k);
                Real IR = RBZ * (h[0]) / static_cast<Real>(idR+1);
                Psi += IR;
                IIminus -= IR * pow( 0.5,(idR)+1);
                IIminus += IR * pow(-0.5,(idR)+1);
                Psi_0   -= IR * pow(0.5,(idR)+1);
            }
            for (auto jj: vw::iota(0, iR))
                psi(jj, iZ, 0, 0,k) += IIminus;
        }        
    }

  }
}


template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::operator()(Dim6& B,
                                    Dim6& dBdR,
                                    Dim6& dBdZ,
                                    Dim6& E,
                              const Dim2& X)
{
  // int ii = std::floor(X[0]/h[0]);
  // Real lc0 = X[0]/h[0] - ii - 0.5;
  // int jj = std::floor(X[1]/h[1]);
  // Real lc1 = X[1]/h[1] - jj - 0.5;
  // for(auto& f: B) f = 0.0;
  // for(auto& f: dBdR) f = 0.0;
  // for(auto& f: dBdZ) f = 0.0;
  // for(auto& f: E) f = 0.0;


  // auto hh = hermiteDataMDSpan(rawHermiteData.data(), nCells[0], nCells[1]); // [nCells[0..1], nDims]
  // Real sclr = 1.0;
  // assert(ii < hh.extent(0));
  // assert(jj < hh.extent(1));
  // for (auto i : vw::iota(0, hh.extent(2)))
  // {
  //   Real sclz = 1.0;
  //   for (auto j : vw::iota(0, hh.extent(3)))
  //   {
  //     for (auto k: vw::iota(0, 6))
  //     {
  //       B[k] += hh(ii,jj,i,j,k) * sclr * sclz;
  //       if(i+1 < hh.extent(2)) dBdR[k] += (i+1)*hh(ii,jj,i+1,j,k) * sclr * sclz;
  //       if(j+1 < hh.extent(3)) dBdZ[k] += (j+1)*hh(ii,jj,i,j+1,k) * sclr * sclz;

  //       E[k] += hh(ii,jj,i,j,k+6) * sclr * sclz;
  //     }
  //     sclz *= lc1;
  //   }
  //   sclr *= lc0;
  // }
  // for(auto& f: dBdR) f /= h[0];
  // for(auto& f: dBdZ) f /= h[1];
}

template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::operator()(Dim6& B,
                                    Dim6& dBdR,
                                    Dim6& dBdZ,
                                    Dim6& E,
                                    Dim6& V,
                                    Dim2& Psi,
                              const Dim2& X)
{
  // int ii = std::floor(X[0]/h[0]);
  // Real lc0 = X[0]/h[0] - ii - 0.5;
  // int jj = std::floor(X[1]/h[1]);
  // Real lc1 = X[1]/h[1] - jj - 0.5;
  // for(auto& f: B) f = 0.0;
  // for(auto& f: dBdR) f = 0.0;
  // for(auto& f: dBdZ) f = 0.0;
  // for(auto& f: E) f = 0.0;
  // for(auto& f: V) f = 0.0;
  // for(auto& f: Psi) f = 0.0;

  
  // auto hh = hermiteDataMDSpan(rawHermiteData.data(), nCells[0], nCells[1]); // [nCells[0..1], nDims]
  // auto psi = psiMDSpan(rawPsi.data(), nCells[0], nCells[1]);
   
  // assert(ii < hh.extent(0));
  // assert(jj < hh.extent(1));

  // Real sclr = 1.0;
  // for (auto i : vw::iota(0, hh.extent(2)))
  // {
  //   Real sclz = 1.0;
  //   for (auto j : vw::iota(0, hh.extent(3)))
  //   {
  //     for (auto k: vw::iota(0, 6))
  //     {
  //       B[k] += hh(ii,jj,i,j,k) * sclr * sclz;
  //       if(i+1 < hh.extent(2)) dBdR[k] += (i+1)*hh(ii,jj,i+1,j,k) * sclr * sclz;
  //       if(j+1 < hh.extent(3)) dBdZ[k] += (j+1)*hh(ii,jj,i,j+1,k) * sclr * sclz;

  //       E[k] += hh(ii,jj,i,j,k+6) * sclr * sclz;
  //       V[k] += hh(ii,jj,i,j,k+12) * sclr * sclz;
  //     }
  //     sclz *= lc1;
  //   }
  //   sclr *= lc0;
  // }

  // sclr = 1.0;
  // for (auto i : vw::iota(0, psi.extent(2)))
  // {
  //   Real sclz = 1.0;
  //   for (auto j : vw::iota(0, psi.extent(3)))
  //   {
  //     for (auto k: vw::iota(0, 2))
  //     {
  //       Psi[k] += psi(ii,jj,i,j,k) * sclr * sclz;
  //     }
  //     sclz *= lc1;
  //   }
  //   sclr *= lc0;
  // }


  // for(auto& f: dBdR) f /= h[0];
  // for(auto& f: dBdZ) f /= h[1];
}

template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::operator()(Vector& B,
                                    Vector& dBdR,
                                    Vector& dBdZ,
                                    Vector& E,
                              const Vector& X)
{

  // for (int i = 0; i < X.size()-1; i+=2)
  // {
  //     Dim2 X_ = {X[i], X[i+1]};
  //     Dim6 B_, dBdR_, dBdZ_, E_;
  //     operator()(B_, dBdR_, dBdZ_, E_, X_);
  //     B.insert(B.end(), B_.cbegin(), B_.cend());
  //     dBdR.insert(dBdR.end(), dBdR_.cbegin(), dBdR_.cend());
  //     dBdZ.insert(dBdZ.end(), dBdZ_.cbegin(), dBdZ_.cend());
  //     E.insert(E.end(), E_.cbegin(), E_.cend());
  // }

  // assert(B.size() == dBdR.size());
  // assert(dBdR.size() == dBdZ.size());
  // assert(dBdZ.size() == E.size());
  // assert(E.size() == X.size()*3);
}

template<int STENCIL_RADIUS, int M>
ERROR_CODE FieldInterpolation<STENCIL_RADIUS,M>::operator()(Vector& B,
                                    Vector& curlB,
                                    Vector& dBdR,
                                    Vector& dBdZ,
                                    Vector& E,
                                    Vector& V,
                                    Vector& Psi,
                              const Vector& X)
{

    for (int i = 0; i < X.size()-1; i+=2)
    {
        Dim2 X_ = {X[i], X[i+1]}; 
        Dim3 B_, curlB_, dBdR_, dBdZ_, E_, V_;
        Real Psi_;
        ERROR_CODE ret = evalPsi(Psi_, X_, tl);
        if (ret != ERROR_CODE::SUCCESS)
        {
            return ret;
        } 
        ret = evalDiv0Fields(B_, curlB_, dBdR_, dBdZ_, E_, X_, tl);
        #ifdef TESTS
        if (TestInputDeck::getInstance()->getData("TestName") == "GrowthRate")
        {
            // highjack the electic field replacing it by slab configuration
            E_[0] = 0.0;
            E_[1] = std::stod(TestInputDeck::getInstance()->getData("Ephi"));
            E_[2] = 0.0;
        }
        #endif
        if (ret != ERROR_CODE::SUCCESS) return ret;
        B.insert(B.end(), B_.cbegin(), B_.cend());
        curlB.insert(curlB.end(), curlB_.cbegin(), curlB_.cend());
        dBdR.insert(dBdR.end(), dBdR_.cbegin(), dBdR_.cend());
        dBdZ.insert(dBdZ.end(), dBdZ_.cbegin(), dBdZ_.cend());
        E.insert(E.end(), E_.cbegin(), E_.cend());
        V.insert(V.end(), V_.cbegin(), V_.cend());
        Psi.push_back(Psi_);
    }

    assert(B.size() == dBdR.size());
    assert(dBdR.size() == dBdZ.size());
    assert(dBdZ.size() == E.size());
    assert(E.size() == X.size()/2*3);
    assert(Psi.size() == X.size()/2);
    return SUCCESS;
}

template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::computeBZ()
{
  auto F = FMDSpan(rawF.data(), nCells[0], nCells[1]);
  int iZ0 = (nCells[1]) / 2;

  for (auto k: vw::iota(0, dim_time))
    // Make RBZ(R,Z) = RBZ(R,Z_0);
    for (auto iR: vw::iota(0, F.extent(0)))
      for (auto idR: vw::iota(0, F.extent(2)))
      {
        for (auto iZ: vw::iota(0, F.extent(1)))
        {
          F(iR, iZ, idR, 0, 0, 2, k) = F(iR, iZ0, idR, 0, 0, 2, k);
          for (auto idZ: vw::iota(1, F.extent(3)))
            F(iR, iZ, idR, idZ, 0, 2, k) = 0.0;
        }
    //   Fill RBZ coefficients with local - Int dBRdR dZ
        for (auto iZ: vw::iota(0, F.extent(1)))
          for (auto idZ: vw::iota(1, F.extent(3)))
          {
            if(idR + 1 < F.extent(2))
              F(iR, iZ, idR, idZ, 0, 2, k) = - F(iR, iZ, idR + 1, idZ - 1, 0, 0, k) * (h[1]) * static_cast<Real>(idR + 1) / (h[0]) / static_cast<Real>(idZ); 
          }
    // Constant part with area from center to edges in cell iZ0
        for (auto iZ: vw::iota(iZ0+1,F.extent(1)))
        {
          Real II = 0.0;
          for (auto idZ: vw::iota(1, F.extent(3)))
          {
            II += F(iR, iZ, idR, idZ, 0, 2, k) * (std::pow(0.5,idZ) - std::pow(-0.5,idZ));
            F(iR, iZ, idR, 0,  0, 2, k) += F(iR, iZ0, idR, idZ, 0, 2, k) *  std::pow( 0.5, idZ);
            F(iR, iZ, idR, 0,  0, 2, k) -= F(iR, iZ,  idR, idZ, 0, 2, k) *  std::pow(-0.5, idZ);
          }
          for (auto iiZ: vw::iota(iZ+1,F.extent(1)))
            F(iR, iiZ, idR, 0, 0, 2 ,k) += II;
        }
        for (auto iZ: vw::iota(0,iZ0))
        {
          Real II = 0.0;
          for (auto idZ: vw::iota(1, F.extent(3)))
          {
            II += F(iR, iZ, idR, idZ, 0, 2, k) * (std::pow(-0.5,idZ) - std::pow(0.5,idZ));
            F(iR, iZ, idR, 0, 0, 2, k) += F(iR, iZ0, idR, idZ, 0, 2, k) * std::pow(-0.5, idZ);
            F(iR, iZ, idR, 0, 0, 2, k) -= F(iR, iZ , idR, idZ, 0, 2, k) * std::pow( 0.5, idZ);
          }
          for (auto iiZ: vw::iota(0,iZ))
            F(iR, iiZ, idR, 0, 0, 2, k) += II;
        }
      }
    
}

template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::assemble()
{
    auto F  = FMDSpan(rawF.data(), nCells[0], nCells[1]);
    auto hh = hermiteDataMDSpan(rawHermiteData.data(), nCells[0], nCells[1]);
    for (auto iR: vw::iota(0, hh.extent(0)))
    {
      for (auto iZ: vw::iota(0, hh.extent(1)))
      {
        for (auto jR: vw::iota(0, hh.extent(2)))
        {
          for (auto jZ: vw::iota(0, hh.extent(3)))
          {
            for (auto dm: vw::iota(0, 3))
            for (auto it: vw::iota(0, 2))
            {
              F(iR, iZ, jR, jZ, 0, dm ,it) = hh(iR, iZ, jR, jZ, 0, dm, it);
              F(iR, iZ, jR, jZ, 1, dm, it) = hh(iR, iZ, jR, jZ, 2, dm, it);
            }
          }
        }
      }
    }
}


template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::assembleJRE()
{
    auto F  = FMDSpan(rawF.data(), nCells[0], nCells[1]);
    auto hh = hermiteJreMDSpan(rawHermiteData.data(), nCells[0], nCells[1]);
    for (auto iR: vw::iota(0, hh.extent(0)))
    {
      for (auto iZ: vw::iota(0, hh.extent(1)))
      {
        for (auto jR: vw::iota(0, hh.extent(2)))
        {
          for (auto jZ: vw::iota(0, hh.extent(3)))
          {
            for (auto dm: vw::iota(0, 3))
            {
              F(iR, iZ, jR, jZ, 2, dm ,0) = hh(iR, iZ, jR, jZ, 0, dm, 0);
            }
          }
        }
      }
    }
}

#include <fstream>
template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::readLevelFile()
{
    std::fstream lvlfile("../../InitialData/lvl.txt", std::ios_base::in);
    auto md = levelMDSpan(rawLevel.data(), N[0], N[1]);
    for(auto j: vw::iota(0, N[1]))
      for(int k = 0; k < 2; ++k)
        for(auto i: vw::iota(0, N[0])) 
          lvlfile >> md(i,j);
    lvlfile.close();
}

template<int STENCIL_RADIUS, int M>
void FieldInterpolation<STENCIL_RADIUS,M>::setTimeFrame(const Real& tl_, const Real& tr_)
{
  tl = tl_;
  tr = tr_;
}


