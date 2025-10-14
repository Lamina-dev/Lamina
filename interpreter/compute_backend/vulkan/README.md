# Vulkan

## 支持的函数

| 函数        | 类型     | Shader                 | 说明                      |
| ----------- | -------- | ---------------------- | ------------------------- |  |
| `add`       | 向量运算 | `vector_add.spv`       | 向量/数组加法             |
| `sub`       | 向量运算 | `vector_sub.spv`       | 向量/数组减法             |
| `mul`       | 标量运算 | `scalar_mul.spv`       | 标量×向量乘法             |
| `matmul`    | 矩阵运算 | `matrix_multiply.spv`  | 矩阵乘法                  |
| `dot`       | 向量运算 | `vector_dot.spv`       | 向量点积（并行 reduction） |
| `transpose` | 矩阵运算 | `matrix_transpose.spv` | 矩阵转置                  |

## 使用方法

### 方式 1：显式调用

```lamina
# 向量加法
result = vulkan.add([1, 2, 3], [4, 5, 6])

# 向量减法
result = vulkan.sub([10, 20, 30], [1, 2, 3])

# 标量乘向量
result = vulkan.mul(5, [1, 2, 3, 4])

# 矩阵乘法
m1 = [[1, 2], [3, 4]]
m2 = [[5, 6], [7, 8]]
result = vulkan.matmul(m1, m2)

# 向量点积
result = vulkan.dot([1, 2, 3], [4, 5, 6])

# 矩阵转置
m = [[1, 2, 3], [4, 5, 6]]
result = vulkan.transpose(m)
```

### 方式 2：代码块语法

```lamina
@vulkan {
    v1 = [1, 2, 3, 4, 5]
    v2 = [10, 20, 30, 40, 50]

    # 在这个块内所有运算都使用 GPU 加速
    sum = add(v1, v2)
    diff = sub(v1, v2)
    scaled = mul(2.5, v1)

    print("GPU 加速结果：", sum)
}
```

## 技术实现

### Compute Shaders

加速功能通过 Vulkan Compute Shaders 实现：

1. **vector_add.comp** - 并行向量加法（256 工作组）
2. **vector_sub.comp** - 并行向量减法（256 工作组）
3. **scalar_mul.comp** - 标量广播乘法（256 工作组）
4. **matrix_multiply.comp** - 矩阵乘法（16×16 二维工作组）
5. **vector_dot.comp** - 点积+共享内存 reduction（256 工作组）
6. **matrix_transpose.comp** - 矩阵转置（16×16 二维工作组）

### GPU 管线

- **自动 pipeline 缓存**：首次调用时创建 pipeline，后续复用
- **自动降级**：pipeline 创建失败时自动 fallback 到 CPU

## 编译 shaders

```bash
# Linux/Mac
bash interpreter/compute_backend/vulkan/compile_shaders.sh

# Windows
interpreter\compute_backend\vulkan\compile_shaders.bat
```

编译后的 SPIR-V 文件位于：
```
interpreter/compute_backend/vulkan/shaders/compiled/
├── vector_add.spv
├── vector_sub.spv
├── scalar_mul.spv
├── matrix_multiply.spv
├── matrix_transpose.spv
└── vector_dot.spv
```

## 系统要求

### 编译时
- Vulkan SDK（包含 glslc 编译器）
- CMake 选项：`-DENABLE_VULKAN=ON`

### 运行时
- Vulkan 驱动（显卡驱动自带，无需 SDK）
- 支持 Compute Shader 的 GPU

## 文件结构

```
interpreter/compute_backend/
├── backend_interface.hpp          # 后端基类接口
├── backend_manager.cpp            # 后端管理器
├── cpu/
│   ├── cpu_backend.hpp           # CPU 后端头文件
│   └── cpu_backend.cpp           # CPU 后端实现
└── vulkan/
    ├── vulkan_backend.hpp        # Vulkan 后端头文件
    ├── vulkan_backend.cpp        # Vulkan 后端实现
    ├── compile_shaders.sh        # Linux/Mac 编译脚本
    ├── compile_shaders.bat       # Windows 编译脚本
    └── shaders/
        ├── vector_add.comp       # 向量加法 shader
        ├── vector_sub.comp       # 向量减法 shader
        ├── scalar_mul.comp       # 标量乘法 shader
        ├── matrix_multiply.comp  # 矩阵乘法 shader
        ├── matrix_transpose.comp # 矩阵转置 shader
        ├── vector_dot.comp       # 点积 shader
        └── compiled/             # 编译后的 SPIR-V
            └── *.spv
```
