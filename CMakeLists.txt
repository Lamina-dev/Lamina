cmake_minimum_required(VERSION 3.35)
project(Lamina LANGUAGES CXX)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
if (MSVC)
    message(FATAL_ERROR "MSVC is not supported, use clang-cl or other compilers in Windows")
endif ()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(include)
include_directories(${CMAKE_BINARY_DIR}/_deps/dyncall-src)
add_subdirectory(runtime)
add_subdirectory(compiler)

option(LMX_BUILD_STATIC "Build laminac and lamina_runtime as standalone static libraries" OFF)

add_executable(lamina main.cpp)

include(FetchContent)

set(DYNCALL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DYNCALL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
        dyncall
        GIT_REPOSITORY https://github.com/ugexe/dyncall.git
        GIT_TAG r1.4
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL TRUE
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
add_library(laminaCore STATIC
        lmx.cpp
    $<TARGET_OBJECTS:laminac>
    $<TARGET_OBJECTS:lamina_runtime>
)
target_link_libraries(laminaCore PRIVATE
        dyncall_s
        dynload_s
)
message( ${CMAKE_BINARY_DIR}/_deps/dyncall-src)

target_compile_definitions(laminaCore PRIVATE LMX_DLL)
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

set(LMX_ALL_TARGETS laminac lamina_runtime laminaCore lamina)
if(LMX_BUILD_STATIC)
    list(APPEND LMX_ALL_TARGETS laminac_s lamina_runtime_s)
endif()
foreach(tgt IN ITEMS ${LMX_ALL_TARGETS})
    disable_exceptions_rtti(${tgt})
endforeach()

set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g -pg -fno-omit-frame-pointer")

if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # clang-cl
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL /arch:AVX2 /Oi /fp:fast")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -funroll-loops -ffast-math -flto -fomit-frame-pointer -fno-semantic-interposition -fvisibility=hidden -fmerge-all-constants -fpeel-loops -fsplit-loops -fpredictive-commoning -fipa-cp-clone -ftree-vectorize -fdata-sections -ffunction-sections")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto -s -Wl,--gc-sections")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -flto -s -Wl,--gc-sections")
endif()
