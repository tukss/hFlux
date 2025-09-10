#include "hFlux/c_wrapper.h"

#include "hFlux/FieldInterpolation.hpp"

constexpr int m = 2;

void hflux_kokkos_init() {
  Kokkos::initialize();
}

void hflux_kokkos_finalize() {
  Kokkos::finalize();
}

void hflux_init(
    const int nR_data,
    const int nZ_data,
    const int nfields,
    const int nphi_data,
    const int nt,
    const double R0,
    const double Z0,
    const double dR,
    const double dZ,
    void ** fi) {
  *fi = (void*) new FieldInterpolation<m>(nR_data, nZ_data, nfields, nphi_data, nt, R0, Z0, dR, dZ);
}

void hflux_interpolate(
    void* fi, double* raw_field_data) {

    auto pFi = static_cast<FieldInterpolation<m>*>(fi);

    Kokkos::View<double ******, Kokkos::LayoutRight, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_view(raw_field_data, pFi->nR_data, pFi->nZ_data, pFi->nfields,
               pFi->ndims, pFi->nphi_data, pFi->nt);
    Kokkos::deep_copy(h_view, pFi->getDataRef());
    Kokkos::fence();

    pFi->interpolate();
}

void hflux_compute_poincare(
    void* fi,
    const double r0,
    const double dr,
    const int n_r,
    const int n_theta,
    const double* poincare_data) {

}

void hflux_field_eval(
    void* fi,
    const int N,
    const double* R_mesh,
    const double* phi_mesh,
    const double* Z_mesh,
    const double* t_mesh,
    double* mesh_value) {

  const auto pFi = *static_cast<FieldInterpolation<m>*>(fi);
  using MeshView = Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using MeshValueView = Kokkos::View<double*****, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const MeshView R_h(R_mesh, N);
  const MeshView phi_h(phi_mesh, N);
  const MeshView Z_h(Z_mesh, N);
  const MeshView t_h(t_mesh, N);

  using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
  auto R = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, R_h);
  auto phi = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, phi_h);
  auto Z = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, Z_h);
  auto t = Kokkos::create_mirror_view_and_copy(DevMemSpace{}, t_h);

  MeshValueView B_h(mesh_value, N, pFi.nfields, pFi.ndims, pFi.nphi_data, pFi.nt);
  auto B = Kokkos::create_mirror_view(DevMemSpace{}, B_h);

  Kokkos::parallel_for("eval", N,
  KOKKOS_LAMBDA(int i){
    Kokkos::Array<Real, 5> X = {};
    X[2] = R(i); X[4] = Z(i);
    auto sbv = Kokkos::subview(B,
             i, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    pFi(sbv, X);
  });

  Kokkos::fence();
  Kokkos::deep_copy(B, B_h);

}

void hflux_destroy(void* fi) {
  auto pFi = static_cast<FieldInterpolation<m> *>(fi);

  delete pFi;
}
