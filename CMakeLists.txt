cmake_minimum_required(VERSION 3.16.3)
project(lamina VERSION 1.1.0 LANGUAGES CXX)# lamina version here

include(GNUInstallDirs)
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR}/lamina)

# add sub directory
add_subdirectory(interpreter)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    # 指定 macOS 最低部署版本（10.15+ 支持 _dyld_iterate_images 且兼容 ARM64）
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum macOS version" FORCE)
    # 确保架构正确（ARM64 或 X86_64）
    set(CMAKE_OSX_ARCHITECTURES "${ARCH}" CACHE STRING "macOS architecture" FORCE)
    
    # 添加编译器和链接器标志
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DARWIN_C_SOURCE")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-undefined,dynamic_lookup")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,dynamic_lookup")
endif()

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

# 设置 RPATH 相关选项
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@executable_path/../${CMAKE_INSTALL_LIBDIR}")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/interpreter
    ${CMAKE_CURRENT_BINARY_DIR} # version.hpp
)

include_directories(interpreter)

# Create lamina_core shared library
add_library(lamina_core SHARED
    interpreter/lamina_api/ast.hpp
    interpreter/lamina_api/bigint.hpp
    interpreter/lamina_api/irrational.hpp
    interpreter/lamina_api/rational.hpp
    interpreter/lamina_api/value.hpp
    interpreter/lamina_api/lamina.hpp
    interpreter/lamina_api/symbolic.hpp
    interpreter/lamina_api/symbolic.cpp

    extensions/standard/math.cpp
    extensions/standard/basic.cpp
    extensions/standard/random.cpp
    extensions/standard/times.cpp
    extensions/standard/array.cpp
    extensions/standard/string.cpp
    extensions/standard/cas.hpp
    extensions/standard/cas.cpp
    extensions/standard/lmStruct.hpp
    extensions/standard/lmStruct.cpp
    extensions/standard/standard.hpp
    extensions/standard/io.cpp

    interpreter/eval.cpp
    interpreter/interpreter.cpp
    interpreter/interpreter.hpp
    interpreter/lexer.cpp
    interpreter/lexer.hpp
    interpreter/parser.cpp
    interpreter/parser.hpp
    interpreter/parse_expr.cpp
    interpreter/parse_factor.cpp
    interpreter/parse_stmt.cpp
    interpreter/cpp_module_loader.hpp

    interpreter/utils/src_manger.cpp
    interpreter/cpp_module_loader.hpp
    interpreter/utils/properties_parser.hpp
    interpreter/utils/src_manger.hpp
)
if (APPLE)
    # 添加必要的框架和系统库
    target_link_libraries(lamina_core PRIVATE 
        "-framework CoreFoundation"
        "-framework CoreServices"
    )
    # 确保包含系统头文件
    target_include_directories(lamina_core PRIVATE 
        /usr/include
        /usr/include/mach-o
    )
    # 设置特定的编译选项
    target_compile_options(lamina_core PRIVATE
        -fPIC
        -D_DARWIN_C_SOURCE
    )
endif()

# Add version header
target_sources(lamina_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/version.hpp)

# Set lamina_core properties
set_target_properties(lamina_core PROPERTIES 
    # 关键修改：禁用自动导出所有符号
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    # 不使用版本号作为文件名，保持稳定的库名
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    ENABLE_EXPORTS ON
    PREFIX ""  # Remove lib prefix on all platforms
    OUTPUT_NAME "lamina_core"  # 固定输出名称（无版本号）
    # macOS 安装名设置为 @rpath/lamina_core.dylib
    INSTALL_NAME_DIR "@rpath"
    MACOSX_RPATH TRUE
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
add_executable(lamina 
    interpreter/main.cpp 
    interpreter/utils/repl_input.hpp
    interpreter/utils/repl_input.cpp
    interpreter/console_ui.hpp
    interpreter/console_ui.cpp
    interpreter/utils/color_style.hpp
    interpreter/utils/color_style.cpp
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

# Ensure `lamina` executable can find `lamina_core` at runtime (set rpath)
if(APPLE)
    set_target_properties(lamina PROPERTIES
        INSTALL_RPATH "@executable_path/../${CMAKE_INSTALL_LIBDIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
        MACOSX_RPATH TRUE
    )
else()
    set_target_properties(lamina PROPERTIES
        INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Post-build actions: place the shared library and executable in stable, non-versioned locations
add_custom_command(TARGET lamina_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    # On macOS, ensure the dylib has a stable install name (id) so executables can reference
    # a non-versioned filename. Then copy it to the stable location without version suffix.
    COMMAND ${CMAKE_COMMAND} -E echo "Post-build: preparing lamina_core for runtime"
    COMMAND ${CMAKE_COMMAND} -E env sh -c "if [ \"${CMAKE_SYSTEM_NAME}\" = \"Darwin\" ]; then \
        install_name_tool -id \"@rpath/lamina_core.dylib\" \"$<TARGET_FILE:lamina_core>\" || true; \
      fi"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:lamina_core> ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/lamina_core.dylib
    COMMENT "Copying lamina_core to ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/lamina_core.dylib"
)

add_custom_command(TARGET lamina POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:lamina> ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/lamina
        # On macOS, update the executable to point to the stable @rpath/lamina_core.dylib
        COMMAND ${CMAKE_COMMAND} -E env sh -c "if [ \"${CMAKE_SYSTEM_NAME}\" = \"Darwin\" ]; then \
                if [ -f \"${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/lamina\" ]; then \
                    install_name_tool -change \"$<TARGET_FILE_NAME:lamina_core>\" \"@rpath/lamina_core.dylib\" \"${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/lamina\" || true; \
                    install_name_tool -add_rpath \"@executable_path/../${CMAKE_INSTALL_LIBDIR}\" \"${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/lamina\" || true; \
                fi; \
            fi"
        COMMENT "Copying lamina executable to ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/lamina and fixing rpath"
)

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
