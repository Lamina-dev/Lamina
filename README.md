# Lamina

Lamina 是一个静态强类型、表达式导向的数学 DSL / 脚本语言，设计目标是"静态、模块化、数学"。
本仓库是 Lamina 语言的编译器前端与寄存器式虚拟机的参考实现（单仓库：`compiler/` + `runtime/`）。

语言规范由 [Lamina Standard Recommendation（LSR）](https://lsr.laminasys.org) 定义，当前核心规范为
[LSR 000 - Lamina 核心语言规范（草案）](https://lsr.laminasys.org/store/LSR-000.html)。
本实现目前覆盖 LSR 000 的核心子集，详细进度见文末 [LSR 实现进度](#lsr-实现进度)。

## 快速开始

依赖：CMake ≥ 3.26、C++23 编译器（GCC 或 Clang，MSVC 不支持）。首次配置会自动通过
FetchContent 拉取 `dyncall`（FFI）与 `LmCAS`（符号计算）。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# 可按需开启汇编-DUSE_ASM=ON

cmake --build build --parallel
```

产物：

| 目标                         | 类型           | 说明                                   |
|------------------------------|----------------|----------------------------------------|
| `lamina`                     | 可执行文件     | CLI 入口，`./lamina <file.lm>`         |
| `laminaCore`                 | 共享库         | 运行时 + 编译器，导出 C ABI（`lmx.h`） |
| `laminac` / `lamina_runtime` | OBJECT 库      | 编译器前端 / 虚拟机                    |
| `lmcas`                      | cas计算库      | 仓库LMCAS构建产物                      |
| `Lammpore`                   | 核心数字计算库 | 仓库LAMMP构建产物                      |
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
- **类型**：`int`（int64）、`bool`、`frac`（有理数，分子/分母）、`text`、`cptr`、`null`、
  数组类型、函数类型、命名类型。运行时无浮点，小数与除法一律走有理数，无精度损失。
- **运行时**：寄存器虚拟机（256 寄存器）、对象 `StringObj` / `ArrayObj` / `CodeModuleObj` /
  `Fraction`、引用计数 GC、函数/递归、模块对象；FFI 基于 `dyncall`（含 C 变参，如 `printf`）。


## 与 LSR 000 的差距（尚未实现）

- `const` 编译期常量、`unit` 单位声明、`sym` 符号 / `Expr`、`use` 符号导入、
  `while`、`for ... in ...` 循环与推导式。
- 向量/矩阵 `vec[]` `mat[]`、`table`、`set{}`、`complex`、`Expr` 字面量与运算
  （`ObjectKind::Vector/Matrix/Table/Expr` 已声明，运行体未实现）。
- 量纲系统 `num<m>`、`as <unit>` 转换、量纲剥离 `as num / as scalar`（LSR-008）。
- 广播运算符 `.* .+ .- ./ .^`、关系广播 `.== .< ...`、整除 `//`、转置 `'`、`\` 左除。
- 可空类型 `T?`。
- 集合运算 `in / not in / subset / xor / | & -`；数值塔提升 `Z⊂Q⊂R⊂C⊂Expr`。
- `match` 模式匹配（LSR-005）、Lambda（LSR-006）、`===` 数学等价（LSR-007）。
- `as` 转换 todo。

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

- `Value`（`runtime/object/value.hpp`）：带 kind 标签的联合，覆盖
  `Null / C_Ptr / Obj / Int / Bool / Fraction / C_VaList`。
- `Object` 体系（`runtime/object/`）：引用计数对象，`Value::obj` 通过 `get()/release()` 维护引用。
- `LaminaVM`（`runtime/vm.cpp`）：、
  整数/分数运算指令、数组读写、函数创建与调用、模块加载、原生调用（`native_call`）。

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
  object/         运行时对象：StringObj / ArrayObj / CodeModuleObj；分数 Fraction 为内嵌值类型
modules/std/      标准库占位模块（LSR-004 尚未实现）
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
| LSR 000 | 核心语言规范（草案） | Draft   | **部分实现**，覆盖核心子集：变量/函数/控制流/模块/数组/整数与分数运算/FFI；向量矩阵、量纲、集合、`Expr` 等未实现（见上文"与 LSR 000 的差距"）                      |
| LSR 001 | LSR 流程规范         | Applied | 不适用（流程文档，本仓库遵循其状态机约定）                                                                                                                         |
| LSR 002 | 标准常量             | Draft   | 未实现（无 `std.constants` 导出）                                                                                                                                  |
| LSR 003 | C 扩展与插件         | Draft   | **部分实现**：`static "lib"` 绑定 + `func f(...) -> t = "sym"` FFI 机制已具备（见 `examples/sdl.lm`、`snake.lm`）；扩展标准布局、打包约定与 `lmx.h` 头分发尚未落地 |
| LSR 004 | 标准库               | Draft   | 未实现（`modules/std` 仅为占位，`std.math` / `std.linalg` / `std.stats` / `std.random` / `std.units` / `std.io` 模块均未建立）                                     |
| LSR 005 | 模式匹配             | Draft   | 未实现（无 `match` 语法；`=>` token 已词法化但未使用）                                                                                                             |
| LSR 006 | Lambda 与类型推导    | Draft   | 未实现（无 lambda；`->` 仅用于函数返回类型）                                                                                                                       |
| LSR 007 | `===` 数学等价判定   | Draft   | 未实现（无 `Expr`，无 CAS 化简流程）                                                                                                                               |
| LSR 008 | 量纲剥离             | Draft   | 未实现（量纲系统整体缺失，`as` 尚为 TODO）                                                                                                                         |
| LSR 009 | 集合与多结果返回     | Draft   | 未实现                                                                                                                                                             |
| LSR 010 | 虚数单位与复数       | Draft   | **部分实现**：`Expr` 使用不可遮蔽的大写 `I`，支持紧邻写法 `4I`；小写 `i` 是普通标识符。Runtime 已提供结构化 `complex` 值及基于 LMMC 的基础运算，复数函数覆盖仍待补齐       |
| LSR 011 | 代数数据类型         | Draft   | 未实现                                                                                                                                                             |
| LSR 012 | 元组类型             | Draft   | 部分实现，解构尚未实现                                                                                                                                             |
| LSR 013 | 集合类型             | Draft   | 未实现                                                                                                                                                             |
| LSR 015 | LAMMP 接口           | Draft   | 不适用                                                                                                                                                             |
