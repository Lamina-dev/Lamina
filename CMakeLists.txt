cmake_minimum_required(VERSION 3.26)
project(Lamina LANGUAGES CXX)

#set(FETCHCONTENT_FULLY_DISCONNECTED ON)
set(BUILD_TESTING OFF)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
if (MSVC AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    message(FATAL_ERROR "MSVC is not supported, use clang-cl or other compilers in Windows")
endif ()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(include)
include_directories(${CMAKE_BINARY_DIR}/_deps/dyncall-src)
include_directories(${CMAKE_BINARY_DIR}/_deps/lmcas-src/include)
add_subdirectory(runtime)
add_subdirectory(compiler)

option(LMX_BUILD_STATIC "Build laminac and lamina_runtime as standalone static libraries" OFF)

add_executable(lamina main.cpp)


include(FetchContent)

set(DYNCALL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DYNCALL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)


message(STATUS "downloading dyncall")
FetchContent_Declare(
        dyncall
        GIT_REPOSITORY https://github.com/ugexe/dyncall.git
        GIT_TAG r1.4
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL TRUE
        PATCH_COMMAND ${CMAKE_COMMAND} -DDYNCALL_SRC=<SOURCE_DIR> -P ${CMAKE_SOURCE_DIR}/scripts/patch_dyncall.cmake
        FIND_PACKAGE_ARGS 11 CONFIG
)
FetchContent_MakeAvailable(dyncall)



if (UNIX)
    set_target_properties(lamina PROPERTIES
            SKIP_BUILD_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
    )
    target_link_options(lamina PRIVATE "-Wl,-rpath,$ORIGIN")
endif ()

message(STATUS "Your System is '${CMAKE_SYSTEM_NAME}'")
add_library(laminaCore SHARED
        lmx.cpp
    $<TARGET_OBJECTS:laminac>
    $<TARGET_OBJECTS:lamina_runtime>
        utils/utils.cpp
)
target_link_libraries(laminaCore PRIVATE
        dyncall_s
        dynload_s
)
#message( ${CMAKE_BINARY_DIR}/_deps/dyncall-src)

message(STATUS "cloning LmCAS")
FetchContent_Declare(
        LmCAS
        GIT_REPOSITORY https://github.com/Lamina-dev/LMCAS.git
        GIT_TAG lmcas-2-current
        GIT_SHALLOW TRUE
        GIT_SUBMODULES "LMMC"
        #GIT_SUBMODULES_RECURSE ON
        #EXCLUDE_FROM_ALL TRUE

)
FetchContent_MakeAvailable(LmCAS)
target_link_libraries(laminaCore PRIVATE
    lmcas
)

if(TARGET LammpCore AND UNIX AND NOT APPLE)
    target_link_options(LammpCore PRIVATE -Wl,-Bsymbolic-functions)
endif()

target_compile_definitions(laminaCore PRIVATE LMX_DLL)
target_compile_definitions(laminaCore PRIVATE LMX_BUILD)
if(LMX_BUILD_STATIC)
    add_library(laminac_s STATIC $<TARGET_OBJECTS:laminac>)
    add_library(lamina_runtime_s STATIC $<TARGET_OBJECTS:lamina_runtime>)
    set_target_properties(laminac_s PROPERTIES OUTPUT_NAME laminac)
    set_target_properties(lamina_runtime_s PROPERTIES OUTPUT_NAME lamina_runtime)
endif()

set_target_properties(laminac PROPERTIES POSITION_INDEPENDENT_CODE ON)


target_link_libraries(lamina PRIVATE laminaCore)


function(disable_exceptions_rtti target)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-exceptions -fno-rtti>
    )
endfunction()



if (CMAKE_BUILD_TYPE STREQUAL "Release")


# 性能剖析优化(PGO), 两阶段, 仅用于 Release:
#   1) cmake -DCMAKE_BUILD_TYPE=Release -DLMX_PGO=GENERATE   -> build -> 运行工作负载脚本
#   2) cmake -DCMAKE_BUILD_TYPE=Release -DLMX_PGO=USE        -> build
# 也可直接执行 scripts/pgo_build.sh 一步完成两阶段构建
set(LMX_PGO "OFF" CACHE STRING "Profile-Guided Optimization: OFF | GENERATE | USE")
set_property(CACHE LMX_PGO PROPERTY STRINGS OFF GENERATE USE)
set(LMX_PGO_WORKLOAD "${CMAKE_SOURCE_DIR}/examples/fib.lm" CACHE FILEPATH "Lamina 脚本, 用于采集 PGO 剖析数据")
set(LMX_PGO_DIR "${CMAKE_BINARY_DIR}/pgo")

set(LMX_PGO_COMPILE_FLAGS "")
set(LMX_PGO_LINK_FLAGS "")
if(LMX_PGO STREQUAL "GENERATE")
    message(STATUS "PGO: 阶段1 GENERATE, 工作负载 = ${LMX_PGO_WORKLOAD}")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(LMX_PGO_COMPILE_FLAGS "-fprofile-generate" "-fprofile-dir=${LMX_PGO_DIR}")
        set(LMX_PGO_LINK_FLAGS   "-fprofile-generate" "-fprofile-dir=${LMX_PGO_DIR}")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        file(MAKE_DIRECTORY "${LMX_PGO_DIR}")
        set(LMX_PGO_COMPILE_FLAGS "-fprofile-instr-generate=${LMX_PGO_DIR}/default.profraw")
        set(LMX_PGO_LINK_FLAGS   "-fprofile-instr-generate=${LMX_PGO_DIR}/default.profraw")
    endif()
elseif(LMX_PGO STREQUAL "USE")
    message(STATUS "PGO: 阶段2 USE, 使用 ${LMX_PGO_DIR}")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(LMX_PGO_COMPILE_FLAGS "-fprofile-use" "-fprofile-dir=${LMX_PGO_DIR}" "-Wno-missing-profile")
        set(LMX_PGO_LINK_FLAGS   "-fprofile-use" "-fprofile-dir=${LMX_PGO_DIR}")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(LMX_PGO_COMPILE_FLAGS "-fprofile-instr-use=${LMX_PGO_DIR}/default.profdata")
        set(LMX_PGO_LINK_FLAGS   "-fprofile-instr-use=${LMX_PGO_DIR}/default.profdata")
    endif()
endif()

function(apply_pgo target)
    if(LMX_PGO_COMPILE_FLAGS)
        target_compile_options(${target} PRIVATE ${LMX_PGO_COMPILE_FLAGS})
    endif()
    if(LMX_PGO_LINK_FLAGS)
        target_link_options(${target} PRIVATE ${LMX_PGO_LINK_FLAGS})
    endif()
endfunction()

# set(CMAKE_CXX_FLAGS_  "${CMAKE_CXX_FLAGS} -g -pg -fno-omit-frame-pointer")

# 各编译器对应的性能优化参数, 通过 target_compile_options 应用到所有目标
# (含 laminac / lamina_runtime 两个 OBJECT 库, 否则虚拟机与编译器得不到任何优化)
set(LMX_PERF_FLAGS "")
set(LMX_LINK_FLAGS "")
if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # clang-cl
    message(STATUS "Use Clang-Cl toolchain compiling")
    set(LMX_PERF_FLAGS /GL /arch:AVX2 /Oi /fp:fast /guard:cf-)
    set(LMX_LINK_FLAGS /LTCG /OPT:REF /OPT:ICF)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Use GNU toolchain compiling")
    set(LMX_PERF_FLAGS
            -march=native -mtune=native
            -fno-plt -fvisibility=hidden
            -funroll-loops -fomit-frame-pointer
            -ffast-math -fno-math-errno
            -flto
            -fno-semantic-interposition
            -fmerge-all-constants
            -fpeel-loops -fsplit-loops -fpredictive-commoning
            -fipa-cp-clone -ftree-vectorize
            -fdata-sections -ffunction-sections
            -falign-loops=32 -falign-jumps=32 -falign-functions=32
            -frename-registers -fprefetch-loop-arrays
            -fipa-pta
            -fno-stack-protector -fno-stack-clash-protection
            -fcf-protection=none #-fzero-init-padding-bits=unions
            -U_FORTIFY_SOURCE
    )
    set(LMX_LINK_FLAGS -flto -s -Wl,--gc-sections -Wl,-O3 -Wl,--as-needed)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Use Clang toolchain compiling")
    set(LMX_PERF_FLAGS
            -march=native -mtune=native
            -fno-plt -fvisibility=hidden
            -funroll-loops -fomit-frame-pointer
            -ffast-math -fno-math-errno
            -flto=thin
            -fno-semantic-interposition
            -fmerge-all-constants
            -fdata-sections -ffunction-sections
            -fno-stack-protector -fno-stack-clash-protection
            -fcf-protection=none
            -U_FORTIFY_SOURCE
    )
    set(LMX_LINK_FLAGS -flto=thin -s -Wl,--gc-sections -Wl,-O3 -Wl,--as-needed)
endif()

set(LMX_ALL_TARGETS laminac lamina_runtime laminaCore lamina)
if(LMX_BUILD_STATIC)
    list(APPEND LMX_ALL_TARGETS laminac_s lamina_runtime_s)
endif()
foreach(tgt IN ITEMS ${LMX_ALL_TARGETS})
    disable_exceptions_rtti(${tgt})
    apply_pgo(${tgt})
    if(LMX_PERF_FLAGS)
        target_compile_options(${tgt} PRIVATE ${LMX_PERF_FLAGS})
    endif()
endforeach()
if(LMX_LINK_FLAGS)
    target_link_options(laminaCore PRIVATE ${LMX_LINK_FLAGS})
    target_link_options(lamina PRIVATE ${LMX_LINK_FLAGS})
endif()

# dyncall 用于所有 FFI 原生调用, 应用安全优化子集
# (不启用 -ffast-math / -flto, 避免影响浮点调用约定与汇编例程)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(LMX_DYNCALL_FLAGS
            -O3 -march=native -mtune=native
            -funroll-loops -fomit-frame-pointer
            -fno-semantic-interposition
            -fvisibility=hidden
            -fno-stack-protector -fno-stack-clash-protection
            -fcf-protection=none -U_FORTIFY_SOURCE
    )
    foreach(dc_tgt IN ITEMS dyncall_s dynload_s)
        if(TARGET ${dc_tgt})
            target_compile_options(${dc_tgt} PRIVATE ${LMX_DYNCALL_FLAGS})
        endif()
    endforeach()
endif()

endif ()