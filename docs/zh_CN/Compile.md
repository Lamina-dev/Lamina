## 编译方法

<div align="right">
  <a href="../zh_TW/Compile.md">繁體中文</a> | <strong>简体中文</strong> | <a href="../en_US/Compile.md">English</a>
</div>
<br>

Linux 平台:

```shell
mkdir build
cd build
cmake ..
make
```

Windows 平台（使用MinGW）:

```shell
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
mingw32-make
```

Windows 平台（使用msbuild(i.e. Visual Studio)）

```shell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
msbuild ALL_BUILD.vcxproj
```

使用 xmake 构建:

```shell
xmake
```

# 警告：如果在MacOS下使用cmake无法构建，尝试使用Unix Makefiles而不是Xcode

## 调试输出

在 Debug 配置构建时，符号化简器会输出额外调试信息，CMake 会在 Debug 模式下为目标定义 `_SYMBOLIC_DEBUG=1`。运行时也可以用环境变量 `LAMINA_SYMBOLIC_DEBUG` 覆盖：

- `LAMINA_SYMBOLIC_DEBUG=1` — 强制在运行时开启调试输出
- `LAMINA_SYMBOLIC_DEBUG=0` — 强制在运行时关闭调试输出

示例（PowerShell）：
```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Debug .
cmake --build build --config Debug --target lamina --parallel
$env:LAMINA_SYMBOLIC_DEBUG = '1'
.\build\bin\lamina.exe
```
