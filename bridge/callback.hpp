#pragma once

#include "runtime/vm.hpp"
#include "runtime/object/vector.hpp"
#include "lmmc/dense.h"
#include "lmmc/status.h"

#include <cstddef>
#include <string>
#include <vector>

namespace lmx::bridge {

struct CallbackContext {
    const lmx::runtime::FuncObj* function;
    std::string error;

    explicit CallbackContext(const lmx::runtime::FuncObj* function_) noexcept
        : function(function_) {}

    void fail(std::string message);
    [[nodiscard]] bool failed() const noexcept { return !error.empty(); }
};

struct ScalarCallbackContext final : CallbackContext {
    using CallbackContext::CallbackContext;
};

struct VectorCallbackContext : CallbackContext {
    lmx::runtime::VectorObj argument;
    std::size_t result_size;

    VectorCallbackContext(const lmx::runtime::FuncObj* function_,
                          std::size_t argument_size,
                          std::size_t result_size_);
};
struct VectorScalarCallbackContext final : VectorCallbackContext {
    VectorScalarCallbackContext(const lmx::runtime::FuncObj* function_,
                                std::size_t argument_size);
};


struct MatrixCallbackContext final : CallbackContext {
    lmx::runtime::VectorObj argument;
    std::size_t rows;
    std::size_t cols;

    MatrixCallbackContext(const lmx::runtime::FuncObj* function_,
                          std::size_t argument_size,
                          std::size_t rows_,
                          std::size_t cols_);
};

struct OdeCallbackContext final : CallbackContext {
    lmx::runtime::VectorObj argument;
    std::size_t dimension;

    OdeCallbackContext(const lmx::runtime::FuncObj* function_,
                       std::size_t dimension_);
};
struct OdeMatrixCallbackContext final : CallbackContext {
    lmx::runtime::VectorObj argument;
    std::size_t dimension;

    OdeMatrixCallbackContext(const lmx::runtime::FuncObj* function_,
                             std::size_t dimension_);
};


double scalar_callback_trampoline(double value, void* user_data) noexcept;
lmmc_status_t vector_callback_trampoline(
    const lmmc_vec_t* input, lmmc_vec_t* output, void* user_data) noexcept;
lmmc_status_t matrix_callback_trampoline(
    const lmmc_vec_t* input, lmmc_mat_t* output, void* user_data) noexcept;
double vector_scalar_callback_trampoline(
    const lmmc_vec_t* input, void* user_data) noexcept;
lmmc_status_t ode_callback_trampoline(
    double time, const double* state, double* derivative,
    std::size_t dimension, void* user_data) noexcept;
lmmc_status_t ode_matrix_callback_trampoline(
    double time, const double* state, double* jacobian,
    std::size_t dimension, void* user_data) noexcept;

} // namespace lmx::bridge
