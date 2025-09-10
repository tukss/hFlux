#ifndef HFLUX_C_WRAPPER_H_
#define HFLUX_C_WRAPPER_H_

void hflux_init(
    const int nR_data,
    const int nZ_data,
    const int nfields,
    const int nphi_data,
    const int nt,
    const double R0,
    const double Z0,
    const double dR,
    const double dZ
    void ** fi);

void hflux_compute_poincare(
    void* fi,
    const double r0,
    const double dr,
    const int n_r
    const int n_theta,
    const double* poincare_data);

void hflux_field_eval(
    void* fi,
    double R,
    double phi,
    double Z,
    double t,
    double* B,
    double* curlB,
    double* dBdR,
    double* dBdZ,
    double* E);
#endif
