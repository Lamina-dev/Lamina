#pragma once
#include "../backend_interface.hpp"
#include <memory>
#include <string>

#ifdef ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

// Vulkan buffer structure
struct VulkanBuffer {
#ifdef ENABLE_VULKAN
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t size = 0;
#endif
};

// Vulkan compute pipeline structure
struct VulkanComputePipeline {
#ifdef ENABLE_VULKAN
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t buffer_count = 0;
#endif
};

// Forward declaration
struct VulkanContext;

/**
 * @brief Vulkan GPU compute backend
 *
 * This backend uses Vulkan compute shaders to accelerate operations on the GPU.
 */
class LAMINA_API VulkanBackend : public ComputeBackend {
public:
    VulkanBackend();
    ~VulkanBackend() override;

    std::string name() const override { return "vulkan"; }
    bool initialize() override;
    bool is_available() const override;
    Value call_function(const std::string& func_name, const std::vector<Value>& args) override;
    std::vector<std::string> available_functions() const override;
    bool has_function(const std::string& func_name) const override;

    /**
     * @brief Get information about the Vulkan device
     */
    std::string get_device_info() const;

private:
    void register_functions();
    void cleanup();

    // Vulkan operation implementations
    Value matmul(const std::vector<Value>& args);
    Value add(const std::vector<Value>& args);
    Value sub(const std::vector<Value>& args);
    Value mul(const std::vector<Value>& args);
    Value dot(const std::vector<Value>& args);
    Value transpose(const std::vector<Value>& args);

#ifdef ENABLE_VULKAN
    // GPU buffer management
    bool create_buffer(size_t size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VulkanBuffer& buffer);
    void destroy_buffer(VulkanBuffer& buffer);
    bool copy_to_buffer(VulkanBuffer& buffer, const void* data, size_t size);
    bool copy_from_buffer(VulkanBuffer& buffer, void* data, size_t size);
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);

    // Shader and pipeline management
    bool load_shader_module(const std::string& shader_path, VkShaderModule& shader_module);
    bool create_compute_pipeline(const std::string& shader_name,
                                 uint32_t buffer_count,
                                 VulkanComputePipeline& pipeline);
    void destroy_pipeline(VulkanComputePipeline& pipeline);

    // Compute execution
    bool execute_compute(VulkanComputePipeline& pipeline,
                        const std::vector<VulkanBuffer*>& buffers,
                        uint32_t push_constant_size,
                        const void* push_constant_data,
                        uint32_t group_count_x,
                        uint32_t group_count_y = 1,
                        uint32_t group_count_z = 1);

    // Helper functions
    std::vector<float> value_to_float_array(const Value& v);
    Value float_array_to_value(const std::vector<float>& data, const Value& shape_reference);

    // Cached pipelines
    VulkanComputePipeline pipeline_vector_add_;
    VulkanComputePipeline pipeline_vector_sub_;
    VulkanComputePipeline pipeline_scalar_mul_;
    VulkanComputePipeline pipeline_matrix_multiply_;
    VulkanComputePipeline pipeline_matrix_transpose_;
    VulkanComputePipeline pipeline_vector_dot_;
#endif

    std::unique_ptr<VulkanContext> context_;
    bool initialized_ = false;
};
