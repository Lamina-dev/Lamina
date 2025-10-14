#include "vulkan_backend.hpp"
#include <iostream>
#include <fstream>
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

// Vulkan context structure
struct VulkanContext {
#ifdef ENABLE_VULKAN
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_properties{};
    uint32_t queue_family_index = 0;
    std::string device_name;
#endif
};

VulkanBackend::VulkanBackend() : context_(std::make_unique<VulkanContext>()) {
    register_functions();
}

VulkanBackend::~VulkanBackend() {
    cleanup();
}

void VulkanBackend::register_functions() {
    functions_["matmul"] = [this](const std::vector<Value>& args) { return matmul(args); };
    functions_["add"] = [this](const std::vector<Value>& args) { return add(args); };
    functions_["sub"] = [this](const std::vector<Value>& args) { return sub(args); };
    functions_["mul"] = [this](const std::vector<Value>& args) { return mul(args); };
    functions_["dot"] = [this](const std::vector<Value>& args) { return dot(args); };
    functions_["transpose"] = [this](const std::vector<Value>& args) { return transpose(args); };
}

bool VulkanBackend::is_available() const {
#ifdef ENABLE_VULKAN
    return true;
#else
    return false;
#endif
}

bool VulkanBackend::initialize() {
#ifdef ENABLE_VULKAN
    if (initialized_) {
        return true;
    }

    try {
        // Create Vulkan instance
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "Lamina Compute";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "Lamina";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;

        if (vkCreateInstance(&create_info, nullptr, &context_->instance) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan instance" << std::endl;
            return false;
        }

        // Find physical device
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(context_->instance, &device_count, nullptr);

        if (device_count == 0) {
            std::cerr << "Failed to find GPUs with Vulkan support" << std::endl;
            return false;
        }

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(context_->instance, &device_count, devices.data());

        // Select first device with compute capability
        for (const auto& device : devices) {
            uint32_t queue_family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

            std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

            for (uint32_t i = 0; i < queue_family_count; i++) {
                if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    context_->physical_device = device;
                    context_->queue_family_index = i;

                    // Get device name and memory properties
                    VkPhysicalDeviceProperties device_properties;
                    vkGetPhysicalDeviceProperties(device, &device_properties);
                    context_->device_name = device_properties.deviceName;
                    vkGetPhysicalDeviceMemoryProperties(device, &context_->memory_properties);

                    break;
                }
            }
            if (context_->physical_device != VK_NULL_HANDLE) {
                break;
            }
        }

        if (context_->physical_device == VK_NULL_HANDLE) {
            std::cerr << "Failed to find suitable GPU" << std::endl;
            return false;
        }

        // Create logical device
        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_create_info{};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = context_->queue_family_index;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;

        VkDeviceCreateInfo device_create_info{};
        device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_create_info.pQueueCreateInfos = &queue_create_info;
        device_create_info.queueCreateInfoCount = 1;

        if (vkCreateDevice(context_->physical_device, &device_create_info, nullptr, &context_->device) != VK_SUCCESS) {
            std::cerr << "Failed to create logical device" << std::endl;
            return false;
        }

        vkGetDeviceQueue(context_->device, context_->queue_family_index, 0, &context_->compute_queue);

        // Create command pool
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = context_->queue_family_index;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(context_->device, &pool_info, nullptr, &context_->command_pool) != VK_SUCCESS) {
            std::cerr << "Failed to create command pool" << std::endl;
            return false;
        }

        initialized_ = true;
        std::cout << "Vulkan backend initialized successfully on device: " << context_->device_name << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception during Vulkan initialization: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "Vulkan support not compiled. Rebuild with -DENABLE_VULKAN=ON" << std::endl;
    return false;
#endif
}

void VulkanBackend::cleanup() {
#ifdef ENABLE_VULKAN
    if (!initialized_) {
        return;
    }

    if (context_->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(context_->device);
    }

    // Cleanup pipelines (check each individually)
    destroy_pipeline(pipeline_vector_add_);
    destroy_pipeline(pipeline_vector_sub_);
    destroy_pipeline(pipeline_scalar_mul_);
    destroy_pipeline(pipeline_matrix_multiply_);
    destroy_pipeline(pipeline_matrix_transpose_);
    destroy_pipeline(pipeline_vector_dot_);

    if (context_->command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(context_->device, context_->command_pool, nullptr);
    }

    if (context_->device != VK_NULL_HANDLE) {
        vkDestroyDevice(context_->device, nullptr);
    }

    if (context_->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(context_->instance, nullptr);
    }

    initialized_ = false;
#endif
}

Value VulkanBackend::call_function(const std::string& func_name, const std::vector<Value>& args) {
    if (!initialized_) {
        throw RuntimeError("Vulkan backend not initialized");
    }

    auto it = functions_.find(func_name);
    if (it == functions_.end()) {
        throw RuntimeError("Vulkan backend: function '" + func_name + "' not found");
    }
    return it->second(args);
}

std::vector<std::string> VulkanBackend::available_functions() const {
    std::vector<std::string> result;
    for (const auto& pair : functions_) {
        result.push_back(pair.first);
    }
    return result;
}

bool VulkanBackend::has_function(const std::string& func_name) const {
    return functions_.find(func_name) != functions_.end();
}

std::string VulkanBackend::get_device_info() const {
#ifdef ENABLE_VULKAN
    if (!initialized_) {
        return "Vulkan backend not initialized";
    }
    return "Vulkan Device: " + context_->device_name;
#else
    return "Vulkan support not compiled";
#endif
}

// ============================================================================
// GPU Buffer Management
// ============================================================================

#ifdef ENABLE_VULKAN

bool VulkanBackend::create_buffer(size_t size, VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags properties, VulkanBuffer& buffer) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context_->device, &buffer_info, nullptr, &buffer.buffer) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer" << std::endl;
        return false;
    }

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(context_->device, buffer.buffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(context_->device, &alloc_info, nullptr, &buffer.memory) != VK_SUCCESS) {
        vkDestroyBuffer(context_->device, buffer.buffer, nullptr);
        std::cerr << "Failed to allocate buffer memory" << std::endl;
        return false;
    }

    vkBindBufferMemory(context_->device, buffer.buffer, buffer.memory, 0);
    buffer.size = size;
    return true;
}

void VulkanBackend::destroy_buffer(VulkanBuffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(context_->device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(context_->device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    buffer.size = 0;
}

bool VulkanBackend::copy_to_buffer(VulkanBuffer& buffer, const void* data, size_t size) {
    void* mapped;
    if (vkMapMemory(context_->device, buffer.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
        std::cerr << "Failed to map buffer memory" << std::endl;
        return false;
    }
    memcpy(mapped, data, size);
    vkUnmapMemory(context_->device, buffer.memory);
    return true;
}

bool VulkanBackend::copy_from_buffer(VulkanBuffer& buffer, void* data, size_t size) {
    void* mapped;
    if (vkMapMemory(context_->device, buffer.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
        std::cerr << "Failed to map buffer memory" << std::endl;
        return false;
    }
    memcpy(data, mapped, size);
    vkUnmapMemory(context_->device, buffer.memory);
    return true;
}

uint32_t VulkanBackend::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < context_->memory_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (context_->memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw RuntimeError("Failed to find suitable memory type");
}

// ============================================================================
// Shader and Pipeline Management
// ============================================================================

bool VulkanBackend::load_shader_module(const std::string& shader_path, VkShaderModule& shader_module) {
    std::ifstream file(shader_path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << shader_path << std::endl;
        return false;
    }

    size_t file_size = static_cast<size_t>(file.tellg());
    std::vector<char> code(file_size);
    file.seekg(0);
    file.read(code.data(), file_size);
    file.close();

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    if (vkCreateShaderModule(context_->device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module" << std::endl;
        return false;
    }

    return true;
}

bool VulkanBackend::create_compute_pipeline(const std::string& shader_name,
                                           uint32_t buffer_count,
                                           VulkanComputePipeline& pipeline) {
    // Load shader
    std::string shader_path = "interpreter/compute_backend/vulkan/shaders/compiled/" + shader_name + ".spv";
    if (!load_shader_module(shader_path, pipeline.shader_module)) {
        return false;
    }

    pipeline.buffer_count = buffer_count;

    // Create descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> bindings(buffer_count);
    for (uint32_t i = 0; i < buffer_count; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = buffer_count;
    layout_info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(context_->device, &layout_info, nullptr,
                                    &pipeline.descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout" << std::endl;
        return false;
    }

    // Create descriptor pool
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = buffer_count * 1000;  // Total descriptors for all sets

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1000;  // Allow 1000 descriptor set allocations
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;  // Allow individual freeing

    if (vkCreateDescriptorPool(context_->device, &pool_info, nullptr,
                               &pipeline.descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool" << std::endl;
        return false;
    }

    // Create pipeline layout with push constants
    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = 16; // Max 16 bytes for push constants

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &pipeline.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;

    if (vkCreatePipelineLayout(context_->device, &pipeline_layout_info, nullptr,
                               &pipeline.pipeline_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        return false;
    }

    // Create compute pipeline
    VkComputePipelineCreateInfo compute_pipeline_info{};
    compute_pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute_pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute_pipeline_info.stage.module = pipeline.shader_module;
    compute_pipeline_info.stage.pName = "main";
    compute_pipeline_info.layout = pipeline.pipeline_layout;

    if (vkCreateComputePipelines(context_->device, VK_NULL_HANDLE, 1,
                                 &compute_pipeline_info, nullptr, &pipeline.pipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        return false;
    }

    return true;
}

void VulkanBackend::destroy_pipeline(VulkanComputePipeline& pipeline) {
    if (pipeline.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context_->device, pipeline.pipeline, nullptr);
        pipeline.pipeline = VK_NULL_HANDLE;
    }
    if (pipeline.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context_->device, pipeline.pipeline_layout, nullptr);
        pipeline.pipeline_layout = VK_NULL_HANDLE;
    }
    if (pipeline.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context_->device, pipeline.descriptor_pool, nullptr);
        pipeline.descriptor_pool = VK_NULL_HANDLE;
    }
    if (pipeline.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context_->device, pipeline.descriptor_set_layout, nullptr);
        pipeline.descriptor_set_layout = VK_NULL_HANDLE;
    }
    if (pipeline.shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(context_->device, pipeline.shader_module, nullptr);
        pipeline.shader_module = VK_NULL_HANDLE;
    }
}

// ============================================================================
// Compute Execution
// ============================================================================

bool VulkanBackend::execute_compute(VulkanComputePipeline& pipeline,
                                   const std::vector<VulkanBuffer*>& buffers,
                                   uint32_t push_constant_size,
                                   const void* push_constant_data,
                                   uint32_t group_count_x,
                                   uint32_t group_count_y,
                                   uint32_t group_count_z) {
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pipeline.descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &pipeline.descriptor_set_layout;

    VkDescriptorSet descriptor_set;
    if (vkAllocateDescriptorSets(context_->device, &alloc_info, &descriptor_set) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set" << std::endl;
        return false;
    }

    // Update descriptor set
    std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
    std::vector<VkWriteDescriptorSet> descriptor_writes(buffers.size());

    for (size_t i = 0; i < buffers.size(); i++) {
        buffer_infos[i].buffer = buffers[i]->buffer;
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = buffers[i]->size;

        descriptor_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[i].dstSet = descriptor_set;
        descriptor_writes[i].dstBinding = i;
        descriptor_writes[i].dstArrayElement = 0;
        descriptor_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_writes[i].descriptorCount = 1;
        descriptor_writes[i].pBufferInfo = &buffer_infos[i];
    }

    vkUpdateDescriptorSets(context_->device, descriptor_writes.size(),
                          descriptor_writes.data(), 0, nullptr);

    // Allocate command buffer
    VkCommandBufferAllocateInfo cmd_alloc_info{};
    cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc_info.commandPool = context_->command_pool;
    cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    if (vkAllocateCommandBuffers(context_->device, &cmd_alloc_info, &command_buffer) != VK_SUCCESS) {
        std::cerr << "Failed to allocate command buffer" << std::endl;
        return false;
    }

    // Record command buffer
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(command_buffer, &begin_info);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline.pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

    if (push_constant_size > 0 && push_constant_data != nullptr) {
        vkCmdPushConstants(command_buffer, pipeline.pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_size, push_constant_data);
    }

    vkCmdDispatch(command_buffer, group_count_x, group_count_y, group_count_z);
    vkEndCommandBuffer(command_buffer);

    // Submit command buffer
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    if (vkQueueSubmit(context_->compute_queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
        std::cerr << "Failed to submit compute command" << std::endl;
        vkFreeCommandBuffers(context_->device, context_->command_pool, 1, &command_buffer);
        return false;
    }

    // Wait for completion
    vkQueueWaitIdle(context_->compute_queue);

    // Cleanup
    vkFreeCommandBuffers(context_->device, context_->command_pool, 1, &command_buffer);
    vkFreeDescriptorSets(context_->device, pipeline.descriptor_pool, 1, &descriptor_set);

    return true;
}

// ============================================================================
// Helper Functions
// ============================================================================

std::vector<float> VulkanBackend::value_to_float_array(const Value& v) {
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

Value VulkanBackend::float_array_to_value(const std::vector<float>& data, const Value& shape_reference) {
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

#endif // ENABLE_VULKAN

// ============================================================================
// Vulkan Operation Implementations
// ============================================================================

Value VulkanBackend::add(const std::vector<Value>& args) {
#ifdef ENABLE_VULKAN
    if (args.size() != 2) {
        throw RuntimeError("add requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // Only accelerate vector/array operations
    if (a.is_array() && b.is_array()) {
        // Create pipeline on first use
        if (pipeline_vector_add_.pipeline == VK_NULL_HANDLE) {
            if (!create_compute_pipeline("vector_add", 3, pipeline_vector_add_)) {
                std::cerr << "Warning: Failed to create vector_add pipeline, using CPU fallback" << std::endl;
                return a.vector_add(b);
            }
        }

        // Convert to float arrays
        std::vector<float> data_a = value_to_float_array(a);
        std::vector<float> data_b = value_to_float_array(b);

        if (data_a.size() != data_b.size()) {
            throw RuntimeError("add: arrays must have same size");
        }

        uint32_t count = data_a.size();
        size_t buffer_size = count * sizeof(float);

        // Create GPU buffers
        VulkanBuffer buffer_a, buffer_b, buffer_out;

        if (!create_buffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buffer_a)) {
            throw RuntimeError("Failed to create input buffer A");
        }

        if (!create_buffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buffer_b)) {
            destroy_buffer(buffer_a);
            throw RuntimeError("Failed to create input buffer B");
        }

        if (!create_buffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buffer_out)) {
            destroy_buffer(buffer_a);
            destroy_buffer(buffer_b);
            throw RuntimeError("Failed to create output buffer");
        }

        // Upload data to GPU
        copy_to_buffer(buffer_a, data_a.data(), buffer_size);
        copy_to_buffer(buffer_b, data_b.data(), buffer_size);

        // Execute compute shader
        std::vector<VulkanBuffer*> buffers = {&buffer_a, &buffer_b, &buffer_out};
        if (!execute_compute(pipeline_vector_add_, buffers, sizeof(uint32_t), &count,
                            (count + 255) / 256, 1, 1)) {
            destroy_buffer(buffer_a);
            destroy_buffer(buffer_b);
            destroy_buffer(buffer_out);
            throw RuntimeError("Failed to execute compute shader");
        }

        // Download results
        std::vector<float> result_data(count);
        copy_from_buffer(buffer_out, result_data.data(), buffer_size);

        // Cleanup buffers
        destroy_buffer(buffer_a);
        destroy_buffer(buffer_b);
        destroy_buffer(buffer_out);

        // Convert back to Value
        return float_array_to_value(result_data, a);
    }

    // Fallback to CPU for scalars
    if (a.is_numeric() && b.is_numeric()) {
        return Value(a.as_number() + b.as_number());
    }

    throw RuntimeError("add requires numeric or array arguments");
#else
    throw RuntimeError("Vulkan support not compiled");
#endif
}

Value VulkanBackend::matmul(const std::vector<Value>& args) {
#ifdef ENABLE_VULKAN
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

        uint32_t M = mat_a.size();          // rows of A
        uint32_t N = mat_a[0].size();       // cols of A / rows of B
        uint32_t P = mat_b[0].size();       // cols of B

        if (mat_b.size() != N) {
            throw RuntimeError("matmul: incompatible matrix dimensions");
        }

        // Create pipeline on first use
        if (pipeline_matrix_multiply_.pipeline == VK_NULL_HANDLE) {
            if (!create_compute_pipeline("matrix_multiply", 3, pipeline_matrix_multiply_)) {
                std::cerr << "Warning: Failed to create matrix_multiply pipeline, using CPU fallback" << std::endl;
                return a.matrix_multiply(b);
            }
        }

        // Convert matrices to flat float arrays (row-major order)
        std::vector<float> data_a = value_to_float_array(a);
        std::vector<float> data_b = value_to_float_array(b);

        size_t size_a = M * N * sizeof(float);
        size_t size_b = N * P * sizeof(float);
        size_t size_c = M * P * sizeof(float);

        // Create GPU buffers
        VulkanBuffer buffer_a, buffer_b, buffer_c;

        if (!create_buffer(size_a, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buffer_a)) {
            throw RuntimeError("Failed to create matrix A buffer");
        }

        if (!create_buffer(size_b, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buffer_b)) {
            destroy_buffer(buffer_a);
            throw RuntimeError("Failed to create matrix B buffer");
        }

        if (!create_buffer(size_c, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buffer_c)) {
            destroy_buffer(buffer_a);
            destroy_buffer(buffer_b);
            throw RuntimeError("Failed to create matrix C buffer");
        }

        // Upload matrices to GPU
        copy_to_buffer(buffer_a, data_a.data(), size_a);
        copy_to_buffer(buffer_b, data_b.data(), size_b);

        // Prepare push constants (M, N, P)
        struct {
            uint32_t M;
            uint32_t N;
            uint32_t P;
        } push_constants = {M, N, P};

        // Execute compute shader
        // Work groups: (P+15)/16 x (M+15)/16 (16x16 local size)
        std::vector<VulkanBuffer*> buffers = {&buffer_a, &buffer_b, &buffer_c};
        if (!execute_compute(pipeline_matrix_multiply_, buffers,
                            sizeof(push_constants), &push_constants,
                            (P + 15) / 16, (M + 15) / 16, 1)) {
            destroy_buffer(buffer_a);
            destroy_buffer(buffer_b);
            destroy_buffer(buffer_c);
            throw RuntimeError("Failed to execute matrix multiply shader");
        }

        // Download result matrix
        std::vector<float> result_data(M * P);
        copy_from_buffer(buffer_c, result_data.data(), size_c);

        // Cleanup buffers
        destroy_buffer(buffer_a);
        destroy_buffer(buffer_b);
        destroy_buffer(buffer_c);

        // Convert back to Value matrix
        std::vector<std::vector<Value>> result_matrix(M, std::vector<Value>(P));
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < P; j++) {
                result_matrix[i][j] = Value(static_cast<double>(result_data[i * P + j]));
            }
        }

        return Value(result_matrix);
    }

    throw RuntimeError("matmul requires matrix arguments");
#else
    throw RuntimeError("Vulkan support not compiled");
#endif
}

Value VulkanBackend::sub(const std::vector<Value>& args) {
#ifdef ENABLE_VULKAN
    if (args.size() != 2) {
        throw RuntimeError("sub requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // GPU-accelerated vector subtraction
    if (a.is_array() && b.is_array()) {
        // Create pipeline on first use
        if (pipeline_vector_sub_.pipeline == VK_NULL_HANDLE) {
            if (!create_compute_pipeline("vector_sub", 3, pipeline_vector_sub_)) {
                std::cerr << "Warning: Failed to create vector_sub pipeline, using CPU fallback" << std::endl;
                return a.vector_minus(b);
            }
        }

        // Convert to float arrays
        std::vector<float> data_a = value_to_float_array(a);
        std::vector<float> data_b = value_to_float_array(b);

        if (data_a.size() != data_b.size()) {
            throw RuntimeError("sub: arrays must have same size");
        }

        uint32_t count = data_a.size();
        size_t buffer_size = count * sizeof(float);

        // Create GPU buffers
        VulkanBuffer buffer_a, buffer_b, buffer_out;

        if (!create_buffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buffer_a)) {
            throw RuntimeError("Failed to create input buffer A");
        }

        if (!create_buffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buffer_b)) {
            destroy_buffer(buffer_a);
            throw RuntimeError("Failed to create input buffer B");
        }

        if (!create_buffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buffer_out)) {
            destroy_buffer(buffer_a);
            destroy_buffer(buffer_b);
            throw RuntimeError("Failed to create output buffer");
        }

        // Upload data to GPU
        copy_to_buffer(buffer_a, data_a.data(), buffer_size);
        copy_to_buffer(buffer_b, data_b.data(), buffer_size);

        // Execute compute shader
        std::vector<VulkanBuffer*> buffers = {&buffer_a, &buffer_b, &buffer_out};
        if (!execute_compute(pipeline_vector_sub_, buffers, sizeof(uint32_t), &count,
                            (count + 255) / 256, 1, 1)) {
            destroy_buffer(buffer_a);
            destroy_buffer(buffer_b);
            destroy_buffer(buffer_out);
            throw RuntimeError("Failed to execute compute shader");
        }

        // Download results
        std::vector<float> result_data(count);
        copy_from_buffer(buffer_out, result_data.data(), buffer_size);

        // Cleanup buffers
        destroy_buffer(buffer_a);
        destroy_buffer(buffer_b);
        destroy_buffer(buffer_out);

        // Convert back to Value
        return float_array_to_value(result_data, a);
    }

    // Fallback to CPU for scalars
    if (a.is_numeric() && b.is_numeric()) {
        return Value(a.as_number() - b.as_number());
    }

    throw RuntimeError("sub requires numeric or array arguments");
#else
    throw RuntimeError("Vulkan support not compiled");
#endif
}

Value VulkanBackend::mul(const std::vector<Value>& args) {
#ifdef ENABLE_VULKAN
    if (args.size() != 2) {
        throw RuntimeError("mul requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // GPU-accelerated scalar * vector multiplication
    if ((a.is_numeric() && b.is_array()) || (a.is_array() && b.is_numeric())) {
        // Create pipeline on first use
        if (pipeline_scalar_mul_.pipeline == VK_NULL_HANDLE) {
            if (!create_compute_pipeline("scalar_mul", 2, pipeline_scalar_mul_)) {
                std::cerr << "Warning: Failed to create scalar_mul pipeline, using CPU fallback" << std::endl;
                if (a.is_numeric()) {
                    return b.scalar_multiply(a.as_number());
                } else {
                    return a.scalar_multiply(b.as_number());
                }
            }
        }

        const Value& vec = a.is_array() ? a : b;
        float scalar = a.is_numeric() ? static_cast<float>(a.as_number()) : static_cast<float>(b.as_number());

        // Convert to float array
        std::vector<float> data_vec = value_to_float_array(vec);
        uint32_t count = data_vec.size();
        size_t buffer_size = count * sizeof(float);

        // Create GPU buffers
        VulkanBuffer buffer_in, buffer_out;

        if (!create_buffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buffer_in)) {
            throw RuntimeError("Failed to create input buffer");
        }

        if (!create_buffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buffer_out)) {
            destroy_buffer(buffer_in);
            throw RuntimeError("Failed to create output buffer");
        }

        // Upload data to GPU
        copy_to_buffer(buffer_in, data_vec.data(), buffer_size);

        // Prepare push constants (count + scalar)
        struct {
            uint32_t count;
            float scalar;
        } push_constants = {count, scalar};

        // Execute compute shader
        std::vector<VulkanBuffer*> buffers = {&buffer_in, &buffer_out};
        if (!execute_compute(pipeline_scalar_mul_, buffers, sizeof(push_constants), &push_constants,
                            (count + 255) / 256, 1, 1)) {
            destroy_buffer(buffer_in);
            destroy_buffer(buffer_out);
            throw RuntimeError("Failed to execute compute shader");
        }

        // Download results
        std::vector<float> result_data(count);
        copy_from_buffer(buffer_out, result_data.data(), buffer_size);

        // Cleanup buffers
        destroy_buffer(buffer_in);
        destroy_buffer(buffer_out);

        // Convert back to Value
        return float_array_to_value(result_data, vec);
    }

    // Scalar * Scalar
    if (a.is_numeric() && b.is_numeric()) {
        return Value(a.as_number() * b.as_number());
    }

    throw RuntimeError("mul requires numeric or array arguments");
#else
    throw RuntimeError("Vulkan support not compiled");
#endif
}

Value VulkanBackend::dot(const std::vector<Value>& args) {
#ifdef ENABLE_VULKAN
    if (args.size() != 2) {
        throw RuntimeError("dot requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    if (!a.is_array() || !b.is_array()) {
        throw RuntimeError("dot requires array arguments");
    }

    // GPU-accelerated dot product
    // Create pipeline on first use
    if (pipeline_vector_dot_.pipeline == VK_NULL_HANDLE) {
        if (!create_compute_pipeline("vector_dot", 3, pipeline_vector_dot_)) {
            std::cerr << "Warning: Failed to create vector_dot pipeline, using CPU fallback" << std::endl;
            return a.dot_product(b);
        }
    }

    // Convert to float arrays
    std::vector<float> data_a = value_to_float_array(a);
    std::vector<float> data_b = value_to_float_array(b);

    if (data_a.size() != data_b.size()) {
        throw RuntimeError("dot: arrays must have same size");
    }

    uint32_t count = data_a.size();
    uint32_t num_work_groups = (count + 255) / 256;

    size_t input_buffer_size = count * sizeof(float);
    size_t partial_buffer_size = num_work_groups * sizeof(float);

    // Create GPU buffers
    VulkanBuffer buffer_a, buffer_b, buffer_partial;

    if (!create_buffer(input_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      buffer_a)) {
        throw RuntimeError("Failed to create input buffer A");
    }

    if (!create_buffer(input_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      buffer_b)) {
        destroy_buffer(buffer_a);
        throw RuntimeError("Failed to create input buffer B");
    }

    if (!create_buffer(partial_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      buffer_partial)) {
        destroy_buffer(buffer_a);
        destroy_buffer(buffer_b);
        throw RuntimeError("Failed to create partial sums buffer");
    }

    // Upload data to GPU
    copy_to_buffer(buffer_a, data_a.data(), input_buffer_size);
    copy_to_buffer(buffer_b, data_b.data(), input_buffer_size);

    // Execute compute shader
    std::vector<VulkanBuffer*> buffers = {&buffer_a, &buffer_b, &buffer_partial};
    if (!execute_compute(pipeline_vector_dot_, buffers, sizeof(uint32_t), &count,
                        num_work_groups, 1, 1)) {
        destroy_buffer(buffer_a);
        destroy_buffer(buffer_b);
        destroy_buffer(buffer_partial);
        throw RuntimeError("Failed to execute compute shader");
    }

    // Download partial sums
    std::vector<float> partial_sums(num_work_groups);
    copy_from_buffer(buffer_partial, partial_sums.data(), partial_buffer_size);

    // Cleanup buffers
    destroy_buffer(buffer_a);
    destroy_buffer(buffer_b);
    destroy_buffer(buffer_partial);

    // Final reduction on CPU
    double final_sum = 0.0;
    for (float val : partial_sums) {
        final_sum += val;
    }

    return Value(final_sum);
#else
    throw RuntimeError("Vulkan support not compiled");
#endif
}

Value VulkanBackend::transpose(const std::vector<Value>& args) {
#ifdef ENABLE_VULKAN
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

    uint32_t rows = mat.size();
    uint32_t cols = mat[0].size();

    // GPU-accelerated matrix transpose
    // Create pipeline on first use
    if (pipeline_matrix_transpose_.pipeline == VK_NULL_HANDLE) {
        if (!create_compute_pipeline("matrix_transpose", 2, pipeline_matrix_transpose_)) {
            std::cerr << "Warning: Failed to create matrix_transpose pipeline, using CPU fallback" << std::endl;
            // CPU fallback
            std::vector<std::vector<Value>> result(cols, std::vector<Value>(rows));
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result[j][i] = mat[i][j];
                }
            }
            return Value(result);
        }
    }

    // Convert matrix to flat float array (row-major order)
    std::vector<float> data_in = value_to_float_array(a);

    size_t input_size = rows * cols * sizeof(float);
    size_t output_size = rows * cols * sizeof(float);

    // Create GPU buffers
    VulkanBuffer buffer_in, buffer_out;

    if (!create_buffer(input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      buffer_in)) {
        throw RuntimeError("Failed to create input buffer");
    }

    if (!create_buffer(output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      buffer_out)) {
        destroy_buffer(buffer_in);
        throw RuntimeError("Failed to create output buffer");
    }

    // Upload data to GPU
    copy_to_buffer(buffer_in, data_in.data(), input_size);

    // Prepare push constants (rows, cols)
    struct {
        uint32_t rows;
        uint32_t cols;
    } push_constants = {rows, cols};

    // Execute compute shader
    // Work groups: (cols+15)/16 x (rows+15)/16 (16x16 local size)
    std::vector<VulkanBuffer*> buffers = {&buffer_in, &buffer_out};
    if (!execute_compute(pipeline_matrix_transpose_, buffers, sizeof(push_constants), &push_constants,
                        (cols + 15) / 16, (rows + 15) / 16, 1)) {
        destroy_buffer(buffer_in);
        destroy_buffer(buffer_out);
        throw RuntimeError("Failed to execute compute shader");
    }

    // Download result matrix
    std::vector<float> result_data(rows * cols);
    copy_from_buffer(buffer_out, result_data.data(), output_size);

    // Cleanup buffers
    destroy_buffer(buffer_in);
    destroy_buffer(buffer_out);

    // Convert back to Value matrix (note: transposed dimensions)
    std::vector<std::vector<Value>> result_matrix(cols, std::vector<Value>(rows));
    for (uint32_t i = 0; i < cols; i++) {
        for (uint32_t j = 0; j < rows; j++) {
            result_matrix[i][j] = Value(static_cast<double>(result_data[i * rows + j]));
        }
    }

    return Value(result_matrix);
#else
    throw RuntimeError("Vulkan support not compiled");
#endif
}
