#ifndef HFLUX_C_WRAPPER_H_
#define HFLUX_C_WRAPPER_H_

#ifdef __cplusplus
extern "C" {
#endif

  void hflux_kokkos_init();
  void hflux_kokkos_finalize();

  void hflux_init(const int nR_data, const int nZ_data, const int nfields,
                  const int nphi_data, const int nt, const double R0,
                  const double Z0, const double dR, const double dZ, void **fi);

  void hflux_interpolate(void *fi, double *raw_field_data);

  void hflux_compute_poincare(void* fi,
                              const double r0,
                              const double dr,
                              const int n_r,
                              const int n_theta,
                              const double* poincare_data);

  void hflux_field_eval(void *fi, const int N, const double *R_mesh,
                        const double *phi_mesh, const double *Z_mesh,
                        const double *t_mesh, double *mesh_value);

  void hflux_destroy(void* fi);

#ifdef __cplusplus
}
#endif

#endif
