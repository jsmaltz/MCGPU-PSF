// grid_device_symbols.cu
#include <cuda_runtime.h>

extern "C" {
  __device__ __constant__ int c_nx;
  __device__ __constant__ int c_ny;
  __device__ __constant__ int c_nz;

  __device__ __constant__ int c_have_rho;

  __device__ __constant__ const float* c_d_rho;
  __device__ __constant__ const float* c_x_edges;
  __device__ __constant__ const float* c_y_edges;
  __device__ __constant__ const float* c_z_edges;
}
