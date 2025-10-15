// Compile Vulkan shaders to SPIR-V
// This script compiles all .comp files in the shaders directory

const std::string SHADER_DIR = "interpreter/shaders/";
const std::string OUTPUT_DIR = "interpreter/shaders/compiled/";

// Shader source codes embedded as strings (pre-compiled to SPIR-V in production)
namespace VulkanShaders {

// Vector addition shader (GLSL source embedded)
const char* vector_add_glsl = R"(
#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer InputBuffer1 { float data[]; } input1;
layout(binding = 1) buffer InputBuffer2 { float data[]; } input2;
layout(binding = 2) buffer OutputBuffer { float data[]; } output_data;
layout(push_constant) uniform PushConstants { uint count; } constants;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < constants.count) {
        output_data.data[idx] = input1.data[idx] + input2.data[idx];
    }
}
)";

// To compile shaders to SPIR-V, use:
// glslangValidator -V vector_add.comp -o vector_add.spv
// or use glslc from Vulkan SDK:
// glslc vector_add.comp -o vector_add.spv

// Note: In production, shaders should be pre-compiled to SPIR-V
// and embedded as binary data or loaded from .spv files

} // namespace VulkanShaders
