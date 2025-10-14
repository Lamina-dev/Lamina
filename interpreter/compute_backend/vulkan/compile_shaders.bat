@echo off
REM 编译Vulkan Compute Shaders到SPIR-V (Windows)

echo Compiling Vulkan Compute Shaders...
echo ================================

REM 检查glslc是否可用
where glslc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: glslc not found
    echo Please install Vulkan SDK: https://vulkan.lunarg.com/
    exit /b 1
)

set SHADER_DIR=interpreter\compute_backend\vulkan\shaders
set OUTPUT_DIR=%SHADER_DIR%\compiled

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 编译所有.comp文件
for %%f in (%SHADER_DIR%\*.comp) do (
    echo Compiling: %%~nf.comp -> %%~nf.spv
    glslc "%%f" -o "%OUTPUT_DIR%\%%~nf.spv"

    if %ERRORLEVEL% EQU 0 (
        echo   Success
    ) else (
        echo   Failed
        exit /b 1
    )
)

echo.
echo ================================
echo All shaders compiled successfully!
echo Output directory: %OUTPUT_DIR%
