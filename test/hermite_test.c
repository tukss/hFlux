#include <math.h>
#include <stdio.h>
#include <string.h>

#include "hFlux/c_wrapper.h"

typedef double Dim3[3];

void run(int nR_data, int nZ_data, double hR, Dim3 l2err) {
  void *fi_data;
  
  int nfields = 2;
  int nphi_data = 1;
  int nt = 1;
  double R0 = 1.525;
  double Z0 = -2.975;
  double dR = 0.0345;
  double dZ = 0.02975;

  double R1 = R0 + 99 * dR;
  double Z1 = Z0 + 199 * dZ;

  dR = (R1 - R0) / (nR_data-1);
  dZ = (Z1 - Z0) / (nZ_data-1);

  double q0 = 2.1;
  double q2 = 2.0;
  double R_a = 3.0;
  double E_0 = 70.0;

  hflux_init(nR_data, nZ_data, nfields, nphi_data, nt, R0, Z0, dR, dZ,
             &fi_data);

  // add code for testing the interpolation calls

  hflux_destroy(fi_data);
}

int main(int argc, char **argv) {
  int NR = 100;
  Dim3 l2err;
  double hR;
  double order = (2 * 2 + 2);

  hflux_kokkos_init();

  for (int ix = 0; ix < 5; ++ix) {
    double hR_new;
    Dim3 l2err_new;

    run(NR, 2*NR, hR_new, l2err_new);
    if (ix > 0) {
      if (fabs(pow(l2err[0] / l2err_new[0], 1.0 / (order - 0.0))  - hR / hR_new) > 5.e-3) {
        fprintf(stderr, "B_R interpolation did not converge with order %f\n", order);
        hflux_kokkos_finalize();
        return 1;
      }
      if (fabs(pow(l2err[2] / l2err_new[2], 1.0 / (order - 1.0)) - hR / hR_new) > 4.e-2) {
        fprintf(stderr, "B_Z interpolation did not converge with order %f\n", order - 1.0);
        hflux_kokkos_finalize();
        return 2;
      }
    }

    memcpy(l2err, l2err_new, sizeof l2err);
    hR = hR_new;

    NR *= 1.5;
  }

  hflux_kokkos_finalize();
  return 0;
}

