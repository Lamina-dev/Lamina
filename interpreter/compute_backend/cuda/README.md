# CUDA

## 支持的函数

| 函数        | 类型     | Kernel                | 说明          |
| ----------- | -------- | --------------------- | ------------- |
| `add`       | 向量运算 | `vector_add.cu`       | 向量/数组加法 |
| `sub`       | 向量运算 | `vector_sub.cu`       | 向量/数组减法 |
| `mul`       | 标量运算 | `scalar_mul.cu`       | 标量×向量乘法 |
| `matmul`    | 矩阵运算 | `matrix_multiply.cu`  | 矩阵乘法      |
| `dot`       | 向量运算 | `vector_dot.cu`       | 向量点积      |
| `transpose` | 矩阵运算 | `matrix_transpose.cu` | 矩阵转置      |

## 使用方法

### 方式 1：显式调用

```lamina
// 向量加法
result = cuda.add([1, 2, 3], [4, 5, 6])

// 向量减法
result = cuda.sub([10, 20, 30], [1, 2, 3])

// 标量乘向量
result = cuda.mul(5, [1, 2, 3, 4])

// 矩阵乘法
m1 = [[1, 2], [3, 4]]
m2 = [[5, 6], [7, 8]]
result = cuda.matmul(m1, m2)

// 向量点积
result = cuda.dot([1, 2, 3], [4, 5, 6])

// 矩阵转置
m = [[1, 2, 3], [4, 5, 6]]
result = cuda.transpose(m)
```

### 方式 2：代码块语法

```lamina
@cuda {
    v1 = [1, 2, 3, 4, 5]
    v2 = [10, 20, 30, 40, 50]

    // 在这个块内所有运算都使用 GPU 加速
    sum = add(v1, v2)
    diff = sub(v1, v2)
    scaled = mul(2.5, v1)

    print("GPU 加速结果：", sum)
}
```

## 技术实现

### CUDA Kernels

加速功能通过 CUDA 内核实现：

1. **vector_add.cu** - 并行向量加法（256 线程/块）
2. **vector_sub.cu** - 并行向量减法（256 线程/块）
3. **scalar_mul.cu** - 标量广播乘法（256 线程/块）
4. **matrix_multiply.cu** - 矩阵乘法（16×16 共享内存分块）
5. **vector_dot.cu** - 点积 + 共享内存 reduction（256 线程/块）
6. **matrix_transpose.cu** - 矩阵转置（16×16 无 bank 冲突）

## 系统要求

### 编译时

- **NVIDIA CUDA Toolkit**（包含 nvcc 编译器）
- **CUDA 兼容的 GPU**（计算能力 5.0+）
- **CMake 选项**：`-DENABLE_CUDA=ON`

## 文件结构

```
interpreter/compute_backend/
├── backend_interface.hpp          # 后端基类接口
├── backend_manager.cpp            # 后端管理器
├── cpu/
│   ├── cpu_backend.hpp           # CPU 后端头文件
│   └── cpu_backend.cpp           # CPU 后端实现
└── cuda/
    ├── cuda_backend.hpp          # CUDA 后端头文件
    ├── cuda_backend.cpp          # CUDA 后端实现
    └── kernels/
        ├── vector_add.cu         # 向量加法 kernel
        ├── vector_sub.cu         # 向量减法 kernel
        ├── scalar_mul.cu         # 标量乘法 kernel
        ├── matrix_multiply.cu    # 矩阵乘法 kernel
        ├── vector_dot.cu         # 点积 kernel
        └── matrix_transpose.cu   # 矩阵转置 kernel
```

## 编译说明

### 基本编译

```bash
# 创建构建目录
mkdir build && cd build

# 配置项目（启用 CUDA）
cmake .. -DENABLE_CUDA=ON

# 编译
cmake --build . --config Release
```

### Windows (Visual Studio)

```bash
# 使用 Visual Studio 生成器
cmake .. -G "Visual Studio 17 2022" -DENABLE_CUDA=ON

# 编译
cmake --build . --config Release
```

### Linux/Mac

```bash
# 指定 CUDA 路径（如果需要）
cmake .. -DENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# 编译
make -j$(nproc)
```

## 许可证

CUDA 后端使用 NVIDIA CUDA Toolkit，需遵守 NVIDIA 软件许可协议。
