#include "cuda_backend.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <algorithm>

// RuntimeError exception class
class RuntimeError : public std::exception {
public:
    std::string message;
    RuntimeError(const std::string& msg) : message(msg) {}
    const char* what() const noexcept override { return message.c_str(); }
};

#ifdef ENABLE_CUDA
// Forward declarations of CUDA kernel launchers (defined in .cu files)
extern "C" {
    void launch_vector_add_kernel(const float* a, const float* b, float* out, int count);
    void launch_vector_sub_kernel(const float* a, const float* b, float* out, int count);
    void launch_scalar_mul_kernel(const float* vec, float* out, float scalar, int count);
    void launch_matrix_multiply_kernel(const float* a, const float* b, float* c, int M, int N, int P);
    void launch_vector_dot_kernel(const float* a, const float* b, float* partial_sums, int count, int num_blocks);
    void launch_matrix_transpose_kernel(const float* in, float* out, int rows, int cols);
}
#endif

CudaBackend::CudaBackend() {
    register_functions();
}

CudaBackend::~CudaBackend() {
    cleanup();
}

void CudaBackend::register_functions() {
    functions_["matmul"] = [this](const std::vector<Value>& args) { return matmul(args); };
    functions_["add"] = [this](const std::vector<Value>& args) { return add(args); };
    functions_["sub"] = [this](const std::vector<Value>& args) { return sub(args); };
    functions_["mul"] = [this](const std::vector<Value>& args) { return mul(args); };
    functions_["dot"] = [this](const std::vector<Value>& args) { return dot(args); };
    functions_["transpose"] = [this](const std::vector<Value>& args) { return transpose(args); };
}

bool CudaBackend::is_available() const {
#ifdef ENABLE_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

bool CudaBackend::initialize() {
#ifdef ENABLE_CUDA
    if (initialized_) {
        return true;
    }

    try {
        // Check for CUDA devices
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess || device_count == 0) {
            std::cerr << "No CUDA-capable devices found" << std::endl;
            return false;
        }

        // Select device 0 (can be made configurable)
        device_id_ = 0;
        error = cudaSetDevice(device_id_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
            return false;
        }

        // Get device properties
        error = cudaGetDeviceProperties(&device_properties_, device_id_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
            return false;
        }

        initialized_ = true;
        std::cout << "CUDA backend initialized successfully on device: "
                  << device_properties_.name << std::endl;
        std::cout << "  Compute Capability: " << device_properties_.major << "."
                  << device_properties_.minor << std::endl;
        std::cout << "  Total Global Memory: " << (device_properties_.totalGlobalMem / (1024 * 1024))
                  << " MB" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception during CUDA initialization: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "CUDA support not compiled. Rebuild with -DENABLE_CUDA=ON" << std::endl;
    return false;
#endif
}

void CudaBackend::cleanup() {
#ifdef ENABLE_CUDA
    if (!initialized_) {
        return;
    }

    cudaDeviceReset();
    initialized_ = false;
#endif
}

Value CudaBackend::call_function(const std::string& func_name, const std::vector<Value>& args) {
    if (!initialized_) {
        throw RuntimeError("CUDA backend not initialized");
    }

    auto it = functions_.find(func_name);
    if (it == functions_.end()) {
        throw RuntimeError("CUDA backend: function '" + func_name + "' not found");
    }
    return it->second(args);
}

std::vector<std::string> CudaBackend::available_functions() const {
    std::vector<std::string> result;
    for (const auto& pair : functions_) {
        result.push_back(pair.first);
    }
    return result;
}

bool CudaBackend::has_function(const std::string& func_name) const {
    return functions_.find(func_name) != functions_.end();
}

std::string CudaBackend::get_device_info() const {
#ifdef ENABLE_CUDA
    if (!initialized_) {
        return "CUDA backend not initialized";
    }
    return std::string("CUDA Device: ") + device_properties_.name;
#else
    return "CUDA support not compiled";
#endif
}

// ============================================================================
// GPU Memory Management
// ============================================================================

#ifdef ENABLE_CUDA

bool CudaBackend::allocate_device_memory(size_t size, void** dev_ptr) {
    cudaError_t error = cudaMalloc(dev_ptr, size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

void CudaBackend::free_device_memory(void* dev_ptr) {
    if (dev_ptr != nullptr) {
        cudaFree(dev_ptr);
    }
}

bool CudaBackend::copy_to_device(void* dev_ptr, const void* host_ptr, size_t size) {
    cudaError_t error = cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy to device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

bool CudaBackend::copy_from_device(void* host_ptr, const void* dev_ptr, size_t size) {
    cudaError_t error = cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy from device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

// ============================================================================
// Helper Functions
// ============================================================================

std::vector<float> CudaBackend::value_to_float_array(const Value& v) {
    std::vector<float> result;

    if (v.is_numeric()) {
        result.push_back(static_cast<float>(v.as_number()));
    } else if (v.is_array()) {
        const auto& arr = std::get<std::vector<Value>>(v.data);
        for (const auto& elem : arr) {
            result.push_back(static_cast<float>(elem.as_number()));
        }
    } else if (v.is_matrix()) {
        const auto& mat = std::get<std::vector<std::vector<Value>>>(v.data);
        for (const auto& row : mat) {
            for (const auto& elem : row) {
                result.push_back(static_cast<float>(elem.as_number()));
            }
        }
    }

    return result;
}

Value CudaBackend::float_array_to_value(const std::vector<float>& data, const Value& shape_reference) {
    if (shape_reference.is_numeric()) {
        return Value(static_cast<double>(data[0]));
    } else if (shape_reference.is_array()) {
        std::vector<Value> result;
        for (float f : data) {
            result.push_back(Value(static_cast<double>(f)));
        }
        return Value(result);
    } else if (shape_reference.is_matrix()) {
        const auto& mat = std::get<std::vector<std::vector<Value>>>(shape_reference.data);
        size_t rows = mat.size();
        size_t cols = mat.empty() ? 0 : mat[0].size();

        std::vector<std::vector<Value>> result(rows, std::vector<Value>(cols));
        size_t idx = 0;
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result[i][j] = Value(static_cast<double>(data[idx++]));
            }
        }
        return Value(result);
    }

    return Value(0.0);
}

#endif // ENABLE_CUDA

// ============================================================================
// CUDA Operation Implementations
// ============================================================================

Value CudaBackend::add(const std::vector<Value>& args) {
#ifdef ENABLE_CUDA
    std::cerr << "CUDA Backend: add() function called" << std::endl;

    if (args.size() != 2) {
        throw RuntimeError("add requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // GPU-accelerated vector addition
    if (a.is_array() && b.is_array()) {
        std::cerr << "CUDA Backend: Processing array addition" << std::endl;
        std::vector<float> data_a = value_to_float_array(a);
        std::vector<float> data_b = value_to_float_array(b);

        if (data_a.size() != data_b.size()) {
            throw RuntimeError("add: arrays must have same size");
        }

        int count = data_a.size();
        std::cerr << "CUDA Backend: Array size = " << count << std::endl;
        size_t buffer_size = count * sizeof(float);

        // Allocate device memory
        float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;

        if (!allocate_device_memory(buffer_size, (void**)&d_a)) {
            throw RuntimeError("Failed to allocate device memory for input A");
        }

        if (!allocate_device_memory(buffer_size, (void**)&d_b)) {
            free_device_memory(d_a);
            throw RuntimeError("Failed to allocate device memory for input B");
        }

        if (!allocate_device_memory(buffer_size, (void**)&d_out)) {
            free_device_memory(d_a);
            free_device_memory(d_b);
            throw RuntimeError("Failed to allocate device memory for output");
        }

        // Copy data to device
        copy_to_device(d_a, data_a.data(), buffer_size);
        copy_to_device(d_b, data_b.data(), buffer_size);

        // Launch kernel
        launch_vector_add_kernel(d_a, d_b, d_out, count);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            free_device_memory(d_a);
            free_device_memory(d_b);
            free_device_memory(d_out);
            throw RuntimeError(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
        }

        // Wait for completion
        cudaDeviceSynchronize();

        // Copy result back
        std::vector<float> result_data(count);
        copy_from_device(result_data.data(), d_out, buffer_size);

        // Cleanup
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_out);

        return float_array_to_value(result_data, a);
    }

    // GPU-accelerated matrix addition (element-wise)
    if (a.is_matrix() && b.is_matrix()) {
        const auto& mat_a = std::get<std::vector<std::vector<Value>>>(a.data);
        const auto& mat_b = std::get<std::vector<std::vector<Value>>>(b.data);

        if (mat_a.empty() || mat_b.empty()) {
            throw RuntimeError("add: empty matrices");
        }

        if (mat_a.size() != mat_b.size() || mat_a[0].size() != mat_b[0].size()) {
            throw RuntimeError("add: matrices must have same dimensions");
        }

        // Flatten matrices to 1D arrays
        std::vector<float> data_a = value_to_float_array(a);
        std::vector<float> data_b = value_to_float_array(b);

        int count = data_a.size();
        size_t buffer_size = count * sizeof(float);

        // Allocate device memory
        float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;

        if (!allocate_device_memory(buffer_size, (void**)&d_a)) {
            throw RuntimeError("Failed to allocate device memory for matrix A");
        }

        if (!allocate_device_memory(buffer_size, (void**)&d_b)) {
            free_device_memory(d_a);
            throw RuntimeError("Failed to allocate device memory for matrix B");
        }

        if (!allocate_device_memory(buffer_size, (void**)&d_out)) {
            free_device_memory(d_a);
            free_device_memory(d_b);
            throw RuntimeError("Failed to allocate device memory for output");
        }

        // Copy data to device
        copy_to_device(d_a, data_a.data(), buffer_size);
        copy_to_device(d_b, data_b.data(), buffer_size);

        // Launch kernel (same as vector addition)
        launch_vector_add_kernel(d_a, d_b, d_out, count);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            free_device_memory(d_a);
            free_device_memory(d_b);
            free_device_memory(d_out);
            throw RuntimeError(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
        }

        // Wait for completion
        cudaDeviceSynchronize();

        // Copy result back
        std::vector<float> result_data(count);
        copy_from_device(result_data.data(), d_out, buffer_size);

        // Cleanup
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_out);

        return float_array_to_value(result_data, a);
    }

    // Fallback to CPU for scalars
    if (a.is_numeric() && b.is_numeric()) {
        return Value(a.as_number() + b.as_number());
    }

    throw RuntimeError("add requires numeric, array, or matrix arguments");
#else
    throw RuntimeError("CUDA support not compiled");
#endif
}

Value CudaBackend::sub(const std::vector<Value>& args) {
#ifdef ENABLE_CUDA
    if (args.size() != 2) {
        throw RuntimeError("sub requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // GPU-accelerated vector subtraction
    if (a.is_array() && b.is_array()) {
        std::vector<float> data_a = value_to_float_array(a);
        std::vector<float> data_b = value_to_float_array(b);

        if (data_a.size() != data_b.size()) {
            throw RuntimeError("sub: arrays must have same size");
        }

        int count = data_a.size();
        size_t buffer_size = count * sizeof(float);

        // Allocate device memory
        float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;

        if (!allocate_device_memory(buffer_size, (void**)&d_a)) {
            throw RuntimeError("Failed to allocate device memory for input A");
        }

        if (!allocate_device_memory(buffer_size, (void**)&d_b)) {
            free_device_memory(d_a);
            throw RuntimeError("Failed to allocate device memory for input B");
        }

        if (!allocate_device_memory(buffer_size, (void**)&d_out)) {
            free_device_memory(d_a);
            free_device_memory(d_b);
            throw RuntimeError("Failed to allocate device memory for output");
        }

        // Copy data to device
        copy_to_device(d_a, data_a.data(), buffer_size);
        copy_to_device(d_b, data_b.data(), buffer_size);

        // Launch kernel
        launch_vector_sub_kernel(d_a, d_b, d_out, count);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            free_device_memory(d_a);
            free_device_memory(d_b);
            free_device_memory(d_out);
            throw RuntimeError(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
        }

        // Wait for completion
        cudaDeviceSynchronize();

        // Copy result back
        std::vector<float> result_data(count);
        copy_from_device(result_data.data(), d_out, buffer_size);

        // Cleanup
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_out);

        return float_array_to_value(result_data, a);
    }

    // GPU-accelerated matrix subtraction (element-wise)
    if (a.is_matrix() && b.is_matrix()) {
        const auto& mat_a = std::get<std::vector<std::vector<Value>>>(a.data);
        const auto& mat_b = std::get<std::vector<std::vector<Value>>>(b.data);

        if (mat_a.empty() || mat_b.empty()) {
            throw RuntimeError("sub: empty matrices");
        }

        if (mat_a.size() != mat_b.size() || mat_a[0].size() != mat_b[0].size()) {
            throw RuntimeError("sub: matrices must have same dimensions");
        }

        // Flatten matrices to 1D arrays
        std::vector<float> data_a = value_to_float_array(a);
        std::vector<float> data_b = value_to_float_array(b);

        int count = data_a.size();
        size_t buffer_size = count * sizeof(float);

        // Allocate device memory
        float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;

        if (!allocate_device_memory(buffer_size, (void**)&d_a)) {
            throw RuntimeError("Failed to allocate device memory for matrix A");
        }

        if (!allocate_device_memory(buffer_size, (void**)&d_b)) {
            free_device_memory(d_a);
            throw RuntimeError("Failed to allocate device memory for matrix B");
        }

        if (!allocate_device_memory(buffer_size, (void**)&d_out)) {
            free_device_memory(d_a);
            free_device_memory(d_b);
            throw RuntimeError("Failed to allocate device memory for output");
        }

        // Copy data to device
        copy_to_device(d_a, data_a.data(), buffer_size);
        copy_to_device(d_b, data_b.data(), buffer_size);

        // Launch kernel (same as vector subtraction)
        launch_vector_sub_kernel(d_a, d_b, d_out, count);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            free_device_memory(d_a);
            free_device_memory(d_b);
            free_device_memory(d_out);
            throw RuntimeError(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
        }

        // Wait for completion
        cudaDeviceSynchronize();

        // Copy result back
        std::vector<float> result_data(count);
        copy_from_device(result_data.data(), d_out, buffer_size);

        // Cleanup
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_out);

        return float_array_to_value(result_data, a);
    }

    // Fallback to CPU for scalars
    if (a.is_numeric() && b.is_numeric()) {
        return Value(a.as_number() - b.as_number());
    }

    throw RuntimeError("sub requires numeric, array, or matrix arguments");
#else
    throw RuntimeError("CUDA support not compiled");
#endif
}

Value CudaBackend::mul(const std::vector<Value>& args) {
#ifdef ENABLE_CUDA
    if (args.size() != 2) {
        throw RuntimeError("mul requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // GPU-accelerated scalar * vector multiplication
    if ((a.is_numeric() && b.is_array()) || (a.is_array() && b.is_numeric())) {
        const Value& vec = a.is_array() ? a : b;
        float scalar = a.is_numeric() ? static_cast<float>(a.as_number()) : static_cast<float>(b.as_number());

        std::vector<float> data_vec = value_to_float_array(vec);
        int count = data_vec.size();
        size_t buffer_size = count * sizeof(float);

        // Allocate device memory
        float *d_in = nullptr, *d_out = nullptr;

        if (!allocate_device_memory(buffer_size, (void**)&d_in)) {
            throw RuntimeError("Failed to allocate device memory for input");
        }

        if (!allocate_device_memory(buffer_size, (void**)&d_out)) {
            free_device_memory(d_in);
            throw RuntimeError("Failed to allocate device memory for output");
        }

        // Copy data to device
        copy_to_device(d_in, data_vec.data(), buffer_size);

        // Launch kernel
        launch_scalar_mul_kernel(d_in, d_out, scalar, count);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            free_device_memory(d_in);
            free_device_memory(d_out);
            throw RuntimeError(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
        }

        // Wait for completion
        cudaDeviceSynchronize();

        // Copy result back
        std::vector<float> result_data(count);
        copy_from_device(result_data.data(), d_out, buffer_size);

        // Cleanup
        free_device_memory(d_in);
        free_device_memory(d_out);

        return float_array_to_value(result_data, vec);
    }

    // GPU-accelerated scalar * matrix multiplication
    if ((a.is_numeric() && b.is_matrix()) || (a.is_matrix() && b.is_numeric())) {
        const Value& mat = a.is_matrix() ? a : b;
        float scalar = a.is_numeric() ? static_cast<float>(a.as_number()) : static_cast<float>(b.as_number());

        std::vector<float> data_mat = value_to_float_array(mat);
        int count = data_mat.size();
        size_t buffer_size = count * sizeof(float);

        // Allocate device memory
        float *d_in = nullptr, *d_out = nullptr;

        if (!allocate_device_memory(buffer_size, (void**)&d_in)) {
            throw RuntimeError("Failed to allocate device memory for input");
        }

        if (!allocate_device_memory(buffer_size, (void**)&d_out)) {
            free_device_memory(d_in);
            throw RuntimeError("Failed to allocate device memory for output");
        }

        // Copy data to device
        copy_to_device(d_in, data_mat.data(), buffer_size);

        // Launch kernel
        launch_scalar_mul_kernel(d_in, d_out, scalar, count);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            free_device_memory(d_in);
            free_device_memory(d_out);
            throw RuntimeError(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
        }

        // Wait for completion
        cudaDeviceSynchronize();

        // Copy result back
        std::vector<float> result_data(count);
        copy_from_device(result_data.data(), d_out, buffer_size);

        // Cleanup
        free_device_memory(d_in);
        free_device_memory(d_out);

        return float_array_to_value(result_data, mat);
    }

    // Scalar * Scalar
    if (a.is_numeric() && b.is_numeric()) {
        return Value(a.as_number() * b.as_number());
    }

    throw RuntimeError("mul requires numeric or array arguments");
#else
    throw RuntimeError("CUDA support not compiled");
#endif
}

Value CudaBackend::matmul(const std::vector<Value>& args) {
#ifdef ENABLE_CUDA
    if (args.size() != 2) {
        throw RuntimeError("matmul requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // GPU-accelerated matrix multiplication
    if (a.is_matrix() && b.is_matrix()) {
        const auto& mat_a = std::get<std::vector<std::vector<Value>>>(a.data);
        const auto& mat_b = std::get<std::vector<std::vector<Value>>>(b.data);

        if (mat_a.empty() || mat_b.empty()) {
            throw RuntimeError("matmul: empty matrices");
        }

        int M = mat_a.size();          // rows of A
        int N = mat_a[0].size();       // cols of A / rows of B
        int P = mat_b[0].size();       // cols of B

        if (mat_b.size() != N) {
            throw RuntimeError("matmul: incompatible matrix dimensions");
        }

        // Convert matrices to flat float arrays (row-major order)
        std::vector<float> data_a = value_to_float_array(a);
        std::vector<float> data_b = value_to_float_array(b);

        size_t size_a = M * N * sizeof(float);
        size_t size_b = N * P * sizeof(float);
        size_t size_c = M * P * sizeof(float);

        // Allocate device memory
        float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

        if (!allocate_device_memory(size_a, (void**)&d_a)) {
            throw RuntimeError("Failed to allocate device memory for matrix A");
        }

        if (!allocate_device_memory(size_b, (void**)&d_b)) {
            free_device_memory(d_a);
            throw RuntimeError("Failed to allocate device memory for matrix B");
        }

        if (!allocate_device_memory(size_c, (void**)&d_c)) {
            free_device_memory(d_a);
            free_device_memory(d_b);
            throw RuntimeError("Failed to allocate device memory for matrix C");
        }

        // Copy data to device
        copy_to_device(d_a, data_a.data(), size_a);
        copy_to_device(d_b, data_b.data(), size_b);

        // Launch kernel
        launch_matrix_multiply_kernel(d_a, d_b, d_c, M, N, P);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            free_device_memory(d_a);
            free_device_memory(d_b);
            free_device_memory(d_c);
            throw RuntimeError(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
        }

        // Wait for completion
        cudaDeviceSynchronize();

        // Copy result back
        std::vector<float> result_data(M * P);
        copy_from_device(result_data.data(), d_c, size_c);

        // Cleanup
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_c);

        // Convert back to Value matrix
        std::vector<std::vector<Value>> result_matrix(M, std::vector<Value>(P));
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++) {
                result_matrix[i][j] = Value(static_cast<double>(result_data[i * P + j]));
            }
        }

        return Value(result_matrix);
    }

    throw RuntimeError("matmul requires matrix arguments");
#else
    throw RuntimeError("CUDA support not compiled");
#endif
}

Value CudaBackend::dot(const std::vector<Value>& args) {
#ifdef ENABLE_CUDA
    if (args.size() != 2) {
        throw RuntimeError("dot requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    if (!a.is_array() || !b.is_array()) {
        throw RuntimeError("dot requires array arguments");
    }

    // GPU-accelerated dot product
    std::vector<float> data_a = value_to_float_array(a);
    std::vector<float> data_b = value_to_float_array(b);

    if (data_a.size() != data_b.size()) {
        throw RuntimeError("dot: arrays must have same size");
    }

    int count = data_a.size();
    int num_blocks = (count + 255) / 256;

    size_t input_buffer_size = count * sizeof(float);
    size_t partial_buffer_size = num_blocks * sizeof(float);

    // Allocate device memory
    float *d_a = nullptr, *d_b = nullptr, *d_partial = nullptr;

    if (!allocate_device_memory(input_buffer_size, (void**)&d_a)) {
        throw RuntimeError("Failed to allocate device memory for input A");
    }

    if (!allocate_device_memory(input_buffer_size, (void**)&d_b)) {
        free_device_memory(d_a);
        throw RuntimeError("Failed to allocate device memory for input B");
    }

    if (!allocate_device_memory(partial_buffer_size, (void**)&d_partial)) {
        free_device_memory(d_a);
        free_device_memory(d_b);
        throw RuntimeError("Failed to allocate device memory for partial sums");
    }

    // Copy data to device
    copy_to_device(d_a, data_a.data(), input_buffer_size);
    copy_to_device(d_b, data_b.data(), input_buffer_size);

    // Launch kernel
    launch_vector_dot_kernel(d_a, d_b, d_partial, count, num_blocks);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_partial);
        throw RuntimeError(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
    }

    // Wait for completion
    cudaDeviceSynchronize();

    // Copy partial sums back
    std::vector<float> partial_sums(num_blocks);
    copy_from_device(partial_sums.data(), d_partial, partial_buffer_size);

    // Cleanup
    free_device_memory(d_a);
    free_device_memory(d_b);
    free_device_memory(d_partial);

    // Final reduction on CPU
    double final_sum = 0.0;
    for (float val : partial_sums) {
        final_sum += val;
    }

    return Value(final_sum);
#else
    throw RuntimeError("CUDA support not compiled");
#endif
}

Value CudaBackend::transpose(const std::vector<Value>& args) {
#ifdef ENABLE_CUDA
    if (args.size() != 1) {
        throw RuntimeError("transpose requires exactly 1 argument");
    }

    const Value& a = args[0];

    if (!a.is_matrix()) {
        throw RuntimeError("transpose requires a matrix argument");
    }

    const auto& mat = std::get<std::vector<std::vector<Value>>>(a.data);
    if (mat.empty()) {
        return a;
    }

    int rows = mat.size();
    int cols = mat[0].size();

    // GPU-accelerated matrix transpose
    std::vector<float> data_in = value_to_float_array(a);

    size_t input_size = rows * cols * sizeof(float);
    size_t output_size = rows * cols * sizeof(float);

    // Allocate device memory
    float *d_in = nullptr, *d_out = nullptr;

    if (!allocate_device_memory(input_size, (void**)&d_in)) {
        throw RuntimeError("Failed to allocate device memory for input");
    }

    if (!allocate_device_memory(output_size, (void**)&d_out)) {
        free_device_memory(d_in);
        throw RuntimeError("Failed to allocate device memory for output");
    }

    // Copy data to device
    copy_to_device(d_in, data_in.data(), input_size);

    // Launch kernel
    launch_matrix_transpose_kernel(d_in, d_out, rows, cols);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        free_device_memory(d_in);
        free_device_memory(d_out);
        throw RuntimeError(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
    }

    // Wait for completion
    cudaDeviceSynchronize();

    // Copy result back
    std::vector<float> result_data(rows * cols);
    copy_from_device(result_data.data(), d_out, output_size);

    // Cleanup
    free_device_memory(d_in);
    free_device_memory(d_out);

    // Convert back to Value matrix (note: transposed dimensions)
    std::vector<std::vector<Value>> result_matrix(cols, std::vector<Value>(rows));
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            result_matrix[i][j] = Value(static_cast<double>(result_data[i * rows + j]));
        }
    }

    return Value(result_matrix);
#else
    throw RuntimeError("CUDA support not compiled");
#endif
}
