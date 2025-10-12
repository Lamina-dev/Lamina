cmake_minimum_required(VERSION 3.16.3)
project(lamina VERSION 1.1.1 LANGUAGES CXX)

include(GNUInstallDirs)
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR}/lamina)

# add sub directory
add_subdirectory(interpreter)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prioritize MSVC on Windows
if(WIN32)
    if(MSVC)
        # Set MSVC compile options - add exception handling, calling convention and parallel PDB write support
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MD /Gz /FS /utf-8")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MD /Gz /FS")
        
        # Set to multi-threaded dynamic link library
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
        
        # Add additional definitions to ensure consistent calling convention
        add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        
        message(STATUS "Using MSVC compiler (${CMAKE_CXX_COMPILER_VERSION})")
    else()
        # Settings for MinGW or other compilers
        message(STATUS "Using MinGW/GCC compiler. Note: Some features may require additional configuration.")
        # Add necessary link libraries for MinGW
        if(MINGW)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
        endif()
    endif()
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/interpreter
        ${CMAKE_CURRENT_BINARY_DIR} # version.hpp
)

include_directories(interpreter)

# Create lamina_core shared library
add_library(lamina_core SHARED
        interpreter/ast.hpp
        interpreter/bigint.hpp
        interpreter/interpreter.cpp
        interpreter/interpreter.hpp
        interpreter/irrational.hpp
        interpreter/lexer.cpp
        interpreter/lexer.hpp
        interpreter/parser.cpp
        interpreter/parser.hpp
        interpreter/rational.hpp
        interpreter/trackback.hpp
        interpreter/value.hpp
        interpreter/lamina.hpp
        interpreter/module_loader.cpp
        interpreter/module_loader.hpp
        interpreter/module_api.hpp
        interpreter/symbolic.hpp
        interpreter/symbolic.cpp
        extensions/standard/math.cpp
        extensions/standard/stdio.cpp
        extensions/standard/random.cpp
        extensions/standard/random.hpp
        extensions/standard/times.cpp
        extensions/standard/times.hpp
        extensions/standard/array.cpp
        extensions/standard/string.hpp
        extensions/standard/string.cpp
        extensions/standard/cas.hpp
        extensions/standard/cas.cpp
        extensions/standard/lstruct.hpp
        extensions/standard/lstruct.cpp
		extensions/standard/range.hpp
		extensions/standard/range.cpp
		extensions/standard/debugs.cpp
        interpreter/eval.cpp
)

# Add version header
target_sources(lamina_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/version.hpp)

# Set lamina_core properties
set_target_properties(lamina_core PROPERTIES 
    # 关键修改：禁用自动导出所有符号
    WINDOWS_EXPORT_ALL_SYMBOLS OFF  # ← 改为 OFF！
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    ENABLE_EXPORTS ON
    PREFIX ""  # Remove lib prefix on all platforms
)

# 针对 x86 架构的特殊处理
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(STATUS "Detected x86 (32-bit) build, applying compatibility fixes")
    
    # 添加链接器选项，忽略特定警告
    target_link_options(lamina_core PRIVATE 
        /IGNORE:4197  # 忽略多次导出警告
    )
    
    # 可选：如果仍有问题，取消下面的注释使用 .def 文件
    # if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lamina_core_x86.def")
    #     target_sources(lamina_core PRIVATE 
    #         "${CMAKE_CURRENT_SOURCE_DIR}/lamina_core_x86.def"
    #     )
    # endif()
endif()

# Define export macros
target_compile_definitions(lamina_core PRIVATE LAMINA_CORE_EXPORTS)

# Add imagehlp library link for Windows
if(WIN32)
    target_link_libraries(lamina_core PRIVATE imagehlp)
endif()

# Set compiler specific options
if(MSVC)
    target_compile_options(lamina_core PRIVATE
        /bigobj # 消除编译错误
        /W4
        /FS  # Add /FS flag to resolve PDB file conflicts during parallel compilation
        /wd4251  # 'class' needs to have dll-interface to be used by clients of class
        /wd4267  # 'var' : conversion from 'size_t' to 'type', possible loss of data
        /wd4996  # This function or variable may be unsafe
        /wd4702  # unreachable code
        /wd4100  # unreferenced formal parameter
        /wd4189  # local variable is initialized but not referenced
        /wd4459  # declaration hides global declaration
        /wd4244  # conversion from 'type1' to 'type2', possible loss of data
        /wd4005  # macro redefinition
        # 添加 x86 特有的警告抑制
        /wd4305  # 'initializing': truncation from 'unsigned __int64' to 'const size_t'
        /wd4309  # 'initializing': truncation of constant value
        /wd4456  # declaration hides previous local declaration
        /wd4101  # unreferenced local variable
        /wd4715  # not all control paths return a value
        /wd4701  # potentially uninitialized local variable used
    )
else()
    target_compile_options(lamina_core PRIVATE 
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-variable
    )
endif()

# Create main executable
add_executable(lamina interpreter/main.cpp interpreter/repl_input.cpp
               interpreter/module_loader.cpp
               interpreter/symbolic.cpp
        interpreter/console_ui.hpp
        interpreter/console_ui.cpp
        interpreter/color_style.hpp
        interpreter/eval.cpp
)

# Add version header
target_sources(lamina PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/version.hpp)

# Link to lamina_core
target_link_libraries(lamina PRIVATE lamina_core)

# Platform specific link libraries
if(UNIX)
    target_link_libraries(lamina PRIVATE dl)
endif()

# Copy lamina_core.dll on Windows
if(WIN32)
    add_custom_command(TARGET lamina POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:lamina_core>
            $<TARGET_FILE_DIR:lamina>
        COMMENT "Copying lamina_core.dll to output directory"
    )
endif()

# Extension/plugin system (supports Linux and Windows)
message(STATUS "Building extensions for ${CMAKE_SYSTEM_NAME}")

# 自动寻找并构建扩展
function(build_extension EXTENSION_PATH)
    get_filename_component(EXTENSION_NAME ${EXTENSION_PATH} NAME)
    get_filename_component(EXTENSION_DIR ${EXTENSION_PATH} DIRECTORY)
    
    # 检查是否有对应的 .cpp 文件
    set(EXTENSION_SOURCE "${EXTENSION_PATH}/${EXTENSION_NAME}.cpp")
    if(EXISTS ${EXTENSION_SOURCE})
        message(STATUS "Building extension: ${EXTENSION_NAME}")
        
        # 创建共享库
        add_library(${EXTENSION_NAME} SHARED ${EXTENSION_SOURCE})
        
        # 设置包含目录
        target_include_directories(${EXTENSION_NAME} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/interpreter
            ${CMAKE_CURRENT_SOURCE_DIR}/extensions
            ${EXTENSION_PATH}
        )
        
        # 设置输出目录和属性
        set_target_properties(${EXTENSION_NAME} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
            ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
            ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
            POSITION_INDEPENDENT_CODE ON
            PREFIX ""  # 去掉 lib 前缀
        )
        
        # 设置扩展的编译选项（抑制常见警告）
        if(MSVC)
            target_compile_options(${EXTENSION_NAME} PRIVATE 
                /W4
                /wd4251  # 'class' needs to have dll-interface
                /wd4267  # conversion from 'size_t' to 'type', possible loss of data
                /wd4996  # This function or variable may be unsafe
                /wd4702  # unreachable code
                /wd4100  # unreferenced formal parameter
                /wd4189  # local variable is initialized but not referenced
                /wd4459  # declaration hides global declaration
                /wd4244  # conversion possible loss of data
                /wd4005  # macro redefinition
            )
        else()
            target_compile_options(${EXTENSION_NAME} PRIVATE 
                -Wall -Wextra -Wpedantic
                -Wno-unused-parameter
                -Wno-unused-variable
            )
        endif()
        
        # Windows 下自动复制 DLL，支持 MSVC 和 MinGW
        if(WIN32)
            if(MSVC)
                # MSVC 构建，支持 Debug 配置的 PDB 文件
                add_custom_command(TARGET ${EXTENSION_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_FILE:${EXTENSION_NAME}>
                        $<TARGET_FILE_DIR:${EXTENSION_NAME}>
                    COMMENT "Copying ${EXTENSION_NAME}.dll to output directory"
                )
                # 在 Debug 配置下复制 PDB 文件
                if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                    add_custom_command(TARGET ${EXTENSION_NAME} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            $<TARGET_PDB_FILE:${EXTENSION_NAME}>
                            $<TARGET_FILE_DIR:${EXTENSION_NAME}>
                        COMMENT "Copying ${EXTENSION_NAME}.pdb to output directory"
                    )
                endif()
            else()
                # MinGW 构建，只复制 DLL
                add_custom_command(TARGET ${EXTENSION_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_FILE:${EXTENSION_NAME}>
                        $<TARGET_FILE_DIR:${EXTENSION_NAME}>
                    COMMENT "Copying ${EXTENSION_NAME}.dll to output directory"
                )
            endif()
        endif()
    else()
        message(STATUS "Skipping extension ${EXTENSION_NAME}: source file not found (${EXTENSION_SOURCE})")
    endif()
endfunction()

# 自动寻找扩展目录（排除 standard 目录）
file(GLOB EXTENSION_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/extensions/*")
set(FOUND_EXTENSIONS 0)
foreach(EXTENSION_DIR ${EXTENSION_DIRS})
    if(IS_DIRECTORY ${EXTENSION_DIR})
        get_filename_component(DIR_NAME ${EXTENSION_DIR} NAME)
        # 排除标准库扩展目录
        if(NOT DIR_NAME STREQUAL "standard")
            # 检查目录是否真实存在且包含源文件
            get_filename_component(EXTENSION_NAME ${EXTENSION_DIR} NAME)
            set(EXTENSION_SOURCE "${EXTENSION_DIR}/${EXTENSION_NAME}.cpp")
            if(EXISTS ${EXTENSION_SOURCE})
                build_extension(${EXTENSION_DIR})
                math(EXPR FOUND_EXTENSIONS "${FOUND_EXTENSIONS} + 1")
            endif()
        endif()
    endif()
endforeach()

if(FOUND_EXTENSIONS EQUAL 0)
    message(STATUS "No additional extensions found to build")
else()
    message(STATUS "Found and configured ${FOUND_EXTENSIONS} extension(s)")
endif()

# 确保可执行文件能找到库
set_target_properties(lamina PROPERTIES
    INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
    BUILD_WITH_INSTALL_RPATH True
)

# Installation rules
install(TARGETS lamina
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install header files (for plugin development)
install(DIRECTORY interpreter/
    DESTINATION include/lamina
    FILES_MATCHING PATTERN "*.hpp"
)