#include <cuda_runtime.h>

// CUDA kernel for vector subtraction
__global__ void vector_sub_kernel(const float* a, const float* b, float* out, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = a[idx] - b[idx];
    }
}

// Host function to launch the kernel
extern "C" void launch_vector_sub_kernel(const float* a, const float* b, float* out, int count) {
    int threads_per_block = 256;
    int num_blocks = (count + threads_per_block - 1) / threads_per_block;
    vector_sub_kernel<<<num_blocks, threads_per_block>>>(a, b, out, count);
}
