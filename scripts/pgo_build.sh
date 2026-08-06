#!/usr/bin/env bash
# 一键执行 Lamina PGO 两阶段构建 (Release)
# 用法: ./scripts/pgo_build.sh [build_dir] [workload.lm] [额外 CMake 参数...]
#   build_dir   默认 cmake-build-pgo
#   workload.lm 默认 examples/fib.lm, 用作剖析负载
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT}/cmake-build-pgo}"
WORKLOAD="${2:-${ROOT}/examples/fib.lm}"
shift 2 || true
EXTRA_ARGS=("$@")
JOBS="$(nproc 2>/dev/null || echo 4)"
mkdir -p "$BUILD_DIR"
PROBE_LOG="${BUILD_DIR}/.pgo_probe.log"

if ! cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "${EXTRA_ARGS[@]}" >"$PROBE_LOG" 2>&1; then
    cat "$PROBE_LOG"
    exit 1
fi
CXX_COMPILER="$(grep -m1 -E '^CMAKE_CXX_COMPILER:.*=' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2-)"
case "$CXX_COMPILER" in
    *clang*|*Clang*) IS_CLANG=1 ;;
    *) IS_CLANG=0 ;;
esac

echo "==> [1/4] Configure GENERATE (compiler: ${CXX_COMPILER:-auto})"
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DLMX_PGO=GENERATE "${EXTRA_ARGS[@]}"

echo "==> [2/4] Build instrumented binary"
cmake --build "$BUILD_DIR" -j"$JOBS"

echo "==> [3/4] Run workload: $WORKLOAD"
"$BUILD_DIR/lamina" "$WORKLOAD"

if [ "$IS_CLANG" = 1 ]; then
    echo "==> [3.5/4] Merge profile data (llvm-profdata)"
    PROF_RAW="${BUILD_DIR}/pgo/default.profraw"
    PROF_DATA="${BUILD_DIR}/pgo/default.profdata"
    mkdir -p "${BUILD_DIR}/pgo"
    llvm-profdata merge -o "$PROF_DATA" "$PROF_RAW"
fi

echo "==> [4/4] Configure USE + rebuild with profiles"
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DLMX_PGO=USE "${EXTRA_ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$JOBS"

echo "Done: ${BUILD_DIR}/lamina (PGO)"
