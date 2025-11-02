## Compile

<div align="right">
  <a href="../zh_TW/Compile.md">繁體中文</a> | <a href="../zh_CN/Compile.md">简体中文</a> | <strong>English</strong>
</div>
<br>

Linux:

```shell
mkdir build
cd build
cmake ..
make
```

Windows With MinGW:

```shell
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
mingw32-make
```

Windows With MSVC:

```shell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
msbuild ALL_BUILD.vcxproj
```

Use xmake

```shell
xmake
```

# WARNING: If you cannot compile it with cmake in MacOS, try to use Unix makefiles rather than Xcode
## Debug logging

When building in Debug configuration, the symbolic simplifier enables additional debug output by defining `_SYMBOLIC_DEBUG` for the targets. You can also control runtime debug logging with the environment variable `LAMINA_SYMBOLIC_DEBUG`:

- `LAMINA_SYMBOLIC_DEBUG=1` — force-enable debug output at runtime
- `LAMINA_SYMBOLIC_DEBUG=0` — force-disable debug output at runtime

Example (Windows PowerShell):
```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Debug .
cmake --build build --config Debug --target lamina --parallel
$env:LAMINA_SYMBOLIC_DEBUG = '1'
.\build\bin\lamina.exe
```
