#pragma once
#include "../backend_interface.hpp"
#include <memory>
#include <string>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

/**
 * @brief CUDA GPU compute backend
 *
 * This backend uses CUDA to accelerate operations on NVIDIA GPUs.
 */
class LAMINA_API CudaBackend : public ComputeBackend {
public:
    CudaBackend();
    ~CudaBackend() override;

    std::string name() const override { return "cuda"; }
    bool initialize() override;
    bool is_available() const override;
    Value call_function(const std::string& func_name, const std::vector<Value>& args) override;
    std::vector<std::string> available_functions() const override;
    bool has_function(const std::string& func_name) const override;

    /**
     * @brief Get information about the CUDA device
     */
    std::string get_device_info() const;

private:
    void register_functions();
    void cleanup();

    // CUDA operation implementations
    Value matmul(const std::vector<Value>& args);
    Value add(const std::vector<Value>& args);
    Value sub(const std::vector<Value>& args);
    Value mul(const std::vector<Value>& args);
    Value dot(const std::vector<Value>& args);
    Value transpose(const std::vector<Value>& args);

#ifdef ENABLE_CUDA
    // GPU memory management
    bool allocate_device_memory(size_t size, void** dev_ptr);
    void free_device_memory(void* dev_ptr);
    bool copy_to_device(void* dev_ptr, const void* host_ptr, size_t size);
    bool copy_from_device(void* host_ptr, const void* dev_ptr, size_t size);

    // Helper functions
    std::vector<float> value_to_float_array(const Value& v);
    Value float_array_to_value(const std::vector<float>& data, const Value& shape_reference);

    // Device properties
    int device_id_ = 0;
    cudaDeviceProp device_properties_;
#endif

    bool initialized_ = false;
};
