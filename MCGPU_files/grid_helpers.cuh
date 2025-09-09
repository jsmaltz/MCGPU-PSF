#pragma once
#include <cuda_runtime.h>

extern "C" {
  extern __constant__ int    c_nx;
  extern __constant__ int    c_ny;
  extern __constant__ int    c_nz;
  extern __constant__ int    c_have_rho;

  extern __constant__ float* c_d_rho;

  extern __constant__ float* c_x_edges;
  extern __constant__ float* c_y_edges;
  extern __constant__ float* c_z_edges;
}

#if __CUDA_ARCH__ >= 350
  #define LDG(p) __ldg(p)
#else
  #define LDG(p) (*(p))
#endif

static __device__ __forceinline__ int nx_dev(){ return c_nx; }
static __device__ __forceinline__ int ny_dev(){ return c_ny; }
static __device__ __forceinline__ int nz_dev(){ return c_nz; }
static __device__ __forceinline__ float x_edge(int i){ return LDG(&c_x_edges[i]); }
static __device__ __forceinline__ float y_edge(int i){ return LDG(&c_y_edges[i]); }
static __device__ __forceinline__ float z_edge(int i){ return LDG(&c_z_edges[i]); }

// safe INF
static __device__ __forceinline__ float finf(){ return __int_as_float(0x7f800000); }

// density fetch
static __device__ __forceinline__
float fetch_density(int ix,int iy,int iz,int material_id){
    if (c_have_rho){
        const size_t idx = (size_t)ix + (size_t)nx_dev()*((size_t)iy + (size_t)ny_dev()*(size_t)iz);
        return LDG(&c_d_rho[idx]);
    } else {
        // your existing LUT path symbol
        return density_LUT_CONST[material_id];
    }
}
