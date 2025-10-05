cmake_minimum_required(VERSION 3.16.3)
project(Lamina VERSION 1.1.1 LANGUAGES CXX)# Lamina version is here

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
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set libuv related variables
#set(LIBUV_LIBRARY uv_a)
#set(LIBUV_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/libuv)


# Check if libuv submodule exists
#if(NOT EXISTS ${LIBUV_ROOT}/CMakeLists.txt)
#    message(FATAL_ERROR "LibUV submodule not found. Please run 'git submodule update --init --recursive'")
#endif()

# Add libuv subdirectory
#add_subdirectory(${LIBUV_ROOT})
# Enable PIC only for libuv
#set_target_properties(uv uv_a PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set libuv libraries and include directories
#set(LIBUV_LIBRARIES uv_a)
#set(LIBUV_INCLUDE_DIRS ${LIBUV_ROOT}/include)
include_directories(
#        ${LIBUV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/interpreter
        ${CMAKE_CURRENT_BINARY_DIR} # version.hpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/external/exprtk  # 新增ExprTK头文件路径
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
      #  extensions/standard/sockets.hpp
      #  extensions/standard/sockets.cpp
      #  extensions/standard/latex.hpp
      #  extensions/standard/latex.cpp
        extensions/standard/string.hpp
        extensions/standard/string.cpp
        extensions/standard/cas.hpp
        extensions/standard/cas.cpp
        extensions/standard/lstruct.hpp
        extensions/standard/lstruct.cpp
		extensions/standard/range.hpp
		extensions/standard/range.cpp
        interpreter/eval.cpp
)

# Add version header
target_sources(lamina_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/version.hpp)

# Set lamina_core properties
set_target_properties(lamina_core PROPERTIES 
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    ENABLE_EXPORTS ON
    PREFIX ""  # Remove lib prefix on all platforms
)

# Define export macros
target_compile_definitions(lamina_core PRIVATE LAMINA_CORE_EXPORTS)
#target_compile_definitions(lamina_core PRIVATE USE_LIBUV)

# Link libuv
#target_link_libraries(lamina_core PRIVATE ${LIBUV_LIBRARY} uv)

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
    )
else()
    target_compile_options(lamina_core PRIVATE 
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-variable
    )
endif()

# Set lamina_core public header file directory
#target_include_directories(lamina_core PUBLIC
#        ${CMAKE_CURRENT_SOURCE_DIR}/interpreter
##        ${CMAKE_CURRENT_SOURCE_DIR}/external/exprtk  # 确保公共包含
#)

# Create main executable
add_executable(Lamina interpreter/main.cpp interpreter/repl_input.cpp
               # Directly include module_loader.cpp to resolve MinGW linking issues
               interpreter/module_loader.cpp
               # Include symbolic.cpp for symbolic expression support
               interpreter/symbolic.cpp
        interpreter/console_ui.hpp
        interpreter/console_ui.cpp
        interpreter/color_style.hpp
        interpreter/eval.cpp
)

# Add version header
target_sources(Lamina PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/version.hpp)

# Link to lamina_core
target_link_libraries(Lamina PRIVATE lamina_core)

# Platform specific link libraries
if(UNIX)
    target_link_libraries(Lamina PRIVATE dl)
endif()

# Automatically copy dependent dynamic libraries
#add_custom_command(TARGET Lamina POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:uv>
#        $<TARGET_FILE_DIR:Lamina>
#    COMMENT "Copying libuv.dll to output directory"
#)

# Copy lamina_core.dll on Windows
if(WIN32)
    add_custom_command(TARGET Lamina POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:lamina_core>
            $<TARGET_FILE_DIR:Lamina>
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
        
        # 链接到 lamina_core 和 libuv
#        target_link_libraries(${EXTENSION_NAME} PRIVATE lamina_core ${LIBUV_LIBRARY})
        
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

#[[ 
  This project supports MSVC (Visual Studio) and MinGW (GCC) compilation.
  - MSVC: Supports .pdb debug symbols, CMake automatically copies DLL and PDB.
  - MinGW: Only copies DLL, debug information is embedded in DLL/EXE.
  
  Extension build instructions:
  - Automatically searches all subdirectories under extensions/ (except standard/)
  - Each extension directory should contain a .cpp file with the same name as the directory
  - Extensions will be compiled as shared libraries and automatically copied to output directory
  
  For MSVC compilation, recommended commands:
    cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Debug .
    cmake --build build --config Debug --parallel
]]

# Installation rules
install(TARGETS Lamina lamina_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} #Utilized the CMAKE_INSTALL_LIBDIR variable to achieve platform-adaptive installation paths.
)

# Install header files (for plugin development)
install(DIRECTORY interpreter/
    DESTINATION include/lamina
    FILES_MATCHING PATTERN "*.hpp"
)
