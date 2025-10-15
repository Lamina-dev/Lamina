#!/bin/bash
# 编译Vulkan Compute Shaders到SPIR-V

echo "Compiling Vulkan Compute Shaders..."
echo "================================"

# 检查glslc是否可用
if ! command -v glslc &> /dev/null; then
    echo "Error: glslc not found"
    echo "Please install Vulkan SDK: https://vulkan.lunarg.com/"
    exit 1
fi

SHADER_DIR="interpreter/compute_backend/vulkan/shaders"
OUTPUT_DIR="$SHADER_DIR/compiled"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 编译所有.comp文件
for shader in "$SHADER_DIR"/*.comp; do
    if [ -f "$shader" ]; then
        filename=$(basename "$shader" .comp)
        echo "Compiling: $filename.comp -> $filename.spv"
        glslc "$shader" -o "$OUTPUT_DIR/$filename.spv"

        if [ $? -eq 0 ]; then
            echo "  Success"
        else
            echo "  Failed"
            exit 1
        fi
    fi
done

echo ""
echo "================================"
echo "All shaders compiled successfully!"
echo "Output directory: $OUTPUT_DIR"
