#include <cuda_runtime.h>

#define TILE_DIM 16
#define BLOCK_ROWS 16

// CUDA kernel for matrix transpose with shared memory to avoid memory bank conflicts
__global__ void matrix_transpose_kernel(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from input matrix (coalesced reads)
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }

    __syncthreads();

    // Transpose block indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed tile to output matrix (coalesced writes)
    if (x < rows && y < cols) {
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Host function to launch the kernel
extern "C" void launch_matrix_transpose_kernel(const float* in, float* out, int rows, int cols) {
    dim3 threads_per_block(TILE_DIM, TILE_DIM);
    dim3 num_blocks((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    matrix_transpose_kernel<<<num_blocks, threads_per_block>>>(in, out, rows, cols);
}
