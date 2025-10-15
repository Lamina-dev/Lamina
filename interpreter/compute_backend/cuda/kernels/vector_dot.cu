#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// CUDA kernel for vector dot product with parallel reduction
__global__ void vector_dot_kernel(const float* a, const float* b, float* partial_sums, int count) {
    __shared__ float shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element-wise product
    float temp = 0.0f;
    if (idx < count) {
        temp = a[idx] * b[idx];
    }
    shared_data[tid] = temp;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

// Host function to launch the kernel
extern "C" void launch_vector_dot_kernel(const float* a, const float* b, float* partial_sums, int count, int num_blocks) {
    vector_dot_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, partial_sums, count);
}
