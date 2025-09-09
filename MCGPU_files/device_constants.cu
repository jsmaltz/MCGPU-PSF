#include <cuda_runtime.h>

// Definitions (exactly once in the whole program). No initializers.
__constant__ int    c_nx, c_ny, c_nz;
__constant__ int    c_have_rho;
__constant__ float* c_d_rho;

__constant__ float* c_x_edges;
__constant__ float* c_y_edges;
__constant__ float* c_z_edges;
