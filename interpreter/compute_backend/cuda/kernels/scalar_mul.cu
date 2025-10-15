#include <cuda_runtime.h>

// CUDA kernel for scalar multiplication
__global__ void scalar_mul_kernel(const float* vec, float* out, float scalar, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = vec[idx] * scalar;
    }
}

// Host function to launch the kernel
extern "C" void launch_scalar_mul_kernel(const float* vec, float* out, float scalar, int count) {
    int threads_per_block = 256;
    int num_blocks = (count + threads_per_block - 1) / threads_per_block;
    scalar_mul_kernel<<<num_blocks, threads_per_block>>>(vec, out, scalar, count);
}
