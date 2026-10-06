# Lamina

Lamina 是一个静态强类型、表达式导向的数学 DSL / 脚本语言，设计目标是"静态、模块化、数学"。
本仓库是 Lamina 语言的编译器前端与寄存器式虚拟机的参考实现（单仓库：`compiler/` + `runtime/`）。

语言规范由 [Lamina Standard Recommendation（LSR）](https://lsr.laminasys.org) 定义，当前核心规范为
[LSR 000 - Lamina 核心语言规范（草案）](https://lsr.laminasys.org/store/LSR-000.html)。
本实现目前覆盖 LSR 000 的核心子集，详细进度见文末 [LSR 实现进度](#lsr-实现进度)。

## 快速开始

依赖：CMake ≥ 3.26、C++23 编译器（GCC、Clang 或 AppleClang；Windows 使用 MinGW
或 GNU-driver Clang，MSVC ABI 前端包括 clang-cl 不受支持）。仓库使用递归 submodule
提供 `dyncall`、LMCAS、LMMC 与 LMMP。

支持 Windows x86_64、Linux x86_64，以及原生 macOS Apple Silicon (`arm64`) 和 Intel
(`x86_64`)。通用 Release 构建：

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

macOS 构建必须选择单一原生架构。Apple Silicon 使用 LMMP 的自动 ARM64 后端；Intel 使用
通用 C 后端：

```bash
# Apple Silicon
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DLMCAS_LMMP_ASM=AUTO

# Intel
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 -DLMCAS_LMMP_ASM=GENERIC
```

`LMX_ENABLE_LTO=ON` 在 Apple Release 构建中启用 ThinLTO；`strict-debug` preset 明确关闭
LTO。发布归档为 `lamina-linux-x86_64.tar.gz`、`lamina-macos-arm64.tar.gz`、
`lamina-macos-x86_64.tar.gz` 和 `lamina-windows-x64.zip`。每个归档包含 `bin/` 运行时、
`lib/` 静态库与 `include/lmx.h` 公共 C ABI 头。macOS 可执行文件以 `@loader_path` 查找
同目录动态库，LMCAS、LMMC 与 LMMP 使用 `@rpath` 安装名。

产物：

| 目标                         | 类型           | 说明                                   |
|------------------------------|----------------|----------------------------------------|
| `lamina`                     | 可执行文件     | CLI 入口，`./lamina <file.lm>`         |
| `liblamina`                  | 共享库         | 运行时 + 编译器，导出 C ABI（`lmx.h`） |
| `laminac` / `lamina_runtime` | OBJECT 库      | 编译器前端 / 虚拟机                    |
| `lmcas`                      | cas计算库      | 仓库LMCAS构建产物                      |
| `lmmp`                       | 核心数字计算库 | 仓库 LMMP 构建产物                     |
运行：

```bash
./build/lamina examples/fib.lm          # 斐波那契
./build/lamina examples/bernoulli.lm    # 伯努利数（分数精确计算）
./build/lamina examples/99.lm           # 99 乘法表
./build/lamina test.lm                  # 数组/函数/模块 import 冒烟测试
```

图形示例需要额外依赖（SDL3 与本地扩展库）：

```bash
cd examples
gcc -shared -fPIC snake_sdl.c -o snake_sdl
LD_LIBRARY_PATH=. ../build/lamina sdl.lm
LD_LIBRARY_PATH=. ../build/lamina snake.lm
```

## 当前功能快照（已实现）

> 与 LSR 000 完整规范的差异对照见下节。

- **词法**：关键字 `func return if else let var const unit module use and or loop break continue as
  while for in not import sym static`、数字（`_` 分隔）、字符串转义、`#` 行注释、运算符
  `+ - * / % ^ == != < <= > >= = |> -> => ! . ...`。
- **语法**：
  - 绑定：`let`（只读）/ `var`（可变），支持显式类型标注。
  - 函数：`func f(a int, b text, ...) -> frac { ... }`，末尾表达式作隐式返回值；
    可变参数 `...`；原生函数绑定 `func f(...) -> int = "symbol"`；动态库绑定 `static "lib"`。
  - 控制流：表达式化 `if / else if / else`、`loop`（可选循环次数）、`break`、`continue`、
    `return`、块表达式。
  - 表达式：调用、数组字面量 `[a, b, c]`、下标 `a[i]`、`.` 成员访问（模块导出）、
    管道 `|>`（语法糖）、一元 `-` `!` `not`、二元 `+ - * / % ^ == != < <= > >= and or`。
  - 模块：`import a.b.c`（`.lm` 源模块）、模块内符号通过 `a.b` 访问。
- **类型与数学值**：`int`、`bool`、`frac`、`real`、`complex`、`text`、`cptr`、`null`，
  以及数组、元组、集合、区间、向量、矩阵、量纲数值、`Expr` 和代数数据类型。注册的语言测试覆盖
  这些值的构造、运算、错误路径和跨模块调用。
- **数学模块**：`std.math`、`std.linalg`、`std.stats`、`std.random`、`std.units` 与 CAS
  模块提供数值计算、线性代数、统计、随机数、单位换算和符号计算接口；公开数学失败通过
  `Result` / `MathError` 返回。
- **运行时**：寄存器虚拟机、引用计数对象、函数/递归、模块对象、代数数据类型匹配，以及基于
  `dyncall` 的 FFI（含 C 变参，如 `printf`）。

## 与 LSR 000 的差距（尚未实现）

- `const` 编译期常量、`while`、`for ... in ...` 循环与推导式。
- `table`，以及广播运算符 `.* .+ .- ./ .^`、关系广播 `.== .< ...`、整除 `//`、
  转置 `'` 和 `\` 左除。
- 可空类型 `T?` 与 Lambda（LSR-006）。
- `===` 数学等价运算符；当前等价判定通过 CAS API 提供。
- 元组解构与完整的 LSR 标准库覆盖。

## 架构

### 编译流水线

```
源文件 (*.lm)
   │  Lexer（compiler/lexer.cpp）
   ▼
Token 流
   │  Parser（compiler/parser.cpp）
   ▼
AST（compiler/ast）
   │  TypeCkContext（compiler/hir/type_checker.cpp）
   ▼
HIR 类型检查 / 模块符号表
   │  MirBuilder（compiler/mir/mir_builder.cpp）
   ▼
MIR（compiler/mir/，定义见 docs/mir.md）
   │  Assembler（compiler/assembler.cpp）
   ▼
字节码模块 CodeModuleObj（格式见 docs/binary.md）
   │  LaminaVM::run（runtime/vm.cpp）
   ▼
执行（虚拟机 + dyncall FFI）
```

各阶段可通过 `Compiler`（`compiler/compiler.hpp`）状态机组合：
`lex → parse → sema → build → assemble`。`lmx.cpp` 以 C ABI（`include/lmx.h`）暴露
`lmx_doFile / lmx_doString / lmx_vmRunModule / lmx_moduleToFile` 等接口。

### 运行时

- `Value`（`runtime/object/value.hpp`）：带 kind 标签的值容器，承载标量、对象引用和
  FFI 调用数据。
- `Object` 体系（`runtime/object/`）：引用计数对象，覆盖字符串、数组、模块、代数数据类型、
  数学容器和符号表达式包装。
- `LaminaVM`（`runtime/vm.cpp`）：执行算术、容器、控制流、函数、模块和原生调用指令。
- 函数帧按字节码使用的局部槽和实参数量分配空间，退出时释放槽中的对象引用；FFI
  调用复用每个虚拟机的 `DCCallVM`。

## 目录结构

```
compiler/         编译器前端
  lexer.cpp       词法分析
  parser.cpp      语法分析
  ast/            AST 定义、TypePool（类型单例化）、AST 打印
  hir/            HIR 类型、类型检查器
  mir/            MIR 定义、MIR 生成器、MIR 打印
  cas/            CAS（符号计算）接入
  assembler.cpp   MIR → 字节码
  compiler.cpp    编译流水线封装
runtime/          运行时
  vm.cpp          寄存器虚拟机
  opcode.hpp      指令集
  binary.cpp      字节码读写
  gc.cpp          引用计数 GC
  object/         运行时对象和值包装
modules/std/      数学、线性代数、统计、随机数、单位与 CAS 标准模块
examples/         示例：fib / 99 / bernoulli / pipe / sdl / snake
docs/             设计文档：binary.md（字节码格式）、mir.md（MIR 定义）
include/lmx.h     运行时 C ABI（发行包中的公开头）
lmx.cpp           C ABI 实现，内建函数
main.cpp          lamina CLI
```

## 文档

- LSR 标准全集：<https://lsr.laminasys.org>
- LSR 000 核心语言规范（草案）：<https://lsr.laminasys.org/store/LSR-000.html>
- 字节码格式：`docs/binary.md`
- MIR 定义：`docs/mir.md`

---

## LSR 实现进度

> 状态列使用 LSR-001 定义的三态（Draft / Accepted / Applied）；"实现进度"为本仓库当前完成度。
> LSR 001 为流程规范，不涉及语言实现。

| LSR     | 标题                 | 状态    | 实现进度                                                                                                                                                           |
|---------|----------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LSR 000 | 核心语言规范（草案） | Draft   | **部分实现**：核心语言、模块、容器、数学值、量纲、集合、`Expr`、CAS 与 FFI 均有注册语言测试；剩余差距见上文 |
| LSR 001 | LSR 流程规范         | Applied | 不适用（流程文档，本仓库遵循其状态机约定） |
| LSR 002 | 标准常量             | Draft   | **部分实现**：数学与物理常量通过标准模块导出 |
| LSR 003 | C 扩展与插件         | Draft   | **部分实现**：`static "lib"`、原生函数绑定和公开 `lmx.h` C ABI 已实现；完整插件打包约定仍在演进 |
| LSR 004 | 标准库               | Draft   | **部分实现**：数学、线性代数、统计、随机数、单位和 CAS 模块已注册并由语言测试覆盖 |
| LSR 005 | 模式匹配             | Draft   | **部分实现**：代数数据类型构造器、通配分支、穷尽性与不可达分支检查已实现 |
| LSR 006 | Lambda 与类型推导    | Draft   | Lambda 尚未实现 |
| LSR 007 | `===` 数学等价判定   | Draft   | `Expr` 与 CAS 等价 API 已实现；`===` 运算符尚未实现 |
| LSR 008 | 量纲剥离             | Draft   | **部分实现**：量纲类型、单位声明、换算和剥离由正反语言测试覆盖 |
| LSR 009 | 集合与多结果返回     | Draft   | **部分实现**：集合运算、类型推导及 `Result` 返回已实现 |
| LSR 010 | 虚数单位与复数       | Draft   | **部分实现**：`Expr` 使用不可遮蔽的大写 `I`；runtime `complex` 值支持基础运算 |
| LSR 011 | 代数数据类型         | Draft   | **部分实现**：泛型 ADT、构造器与模式匹配已实现 |
| LSR 012 | 元组类型             | Draft   | **部分实现**：元组值和索引已实现；解构尚未实现 |
| LSR 013 | 集合类型             | Draft   | **部分实现**：集合字面量、运算、推导和 CAS 结果转换已实现 |
| LSR 015 | LMMP 接口            | Draft   | 不适用                                                                                                                                                             |
