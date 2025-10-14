#include <cuda_runtime.h>

#define TILE_SIZE 16

// CUDA kernel for matrix multiplication with shared memory
__global__ void matrix_multiply_kernel(const float* a, const float* b, float* c, int M, int N, int P) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from matrix A
        if (row < M && t * TILE_SIZE + threadIdx.x < N) {
            tile_a[threadIdx.y][threadIdx.x] = a[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from matrix B
        if (t * TILE_SIZE + threadIdx.y < N && col < P) {
            tile_b[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * P + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < P) {
        c[row * P + col] = sum;
    }
}

// Host function to launch the kernel
extern "C" void launch_matrix_multiply_kernel(const float* a, const float* b, float* c, int M, int N, int P) {
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks((P + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matrix_multiply_kernel<<<num_blocks, threads_per_block>>>(a, b, c, M, N, P);
}
