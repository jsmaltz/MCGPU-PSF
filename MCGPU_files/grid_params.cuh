#pragma once
#include <cuda_runtime.h>

struct GridParams {
    int nx, ny, nz;
    int have_rho;
    const float* d_rho;      // device ptr (or nullptr)
    const float* x_edges;    // device ptr
    const float* y_edges;    // device ptr
    const float* z_edges;    // device ptr
};

__device__ __forceinline__ float gp_x_edge(const GridParams* gp, int i) { return gp->x_edges[i]; }
__device__ __forceinline__ float gp_y_edge(const GridParams* gp, int i) { return gp->y_edges[i]; }
__device__ __forceinline__ float gp_z_edge(const GridParams* gp, int i) { return gp->z_edges[i]; }
__device__ __forceinline__ float gp_density(const GridParams* gp, unsigned int idx, int material0)
{
    // if no rho cube, fall back to your existing LUT
    return (gp->have_rho && gp->d_rho) ? gp->d_rho[idx] : density_LUT_CONST[material0];
}
