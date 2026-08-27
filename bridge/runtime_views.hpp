#pragma once

#include "bridge/conversions.hpp"

#include "runtime/object/matrix.hpp"
#include "runtime/object/vector.hpp"

#include <lmmc/dense.h>

#include <string>
#include <vector>

namespace lmx::bridge {

using runtime::AdtObj;
using runtime::MatrixObj;
using runtime::ValueKind;
using runtime::VectorObj;

lmmc_vec_t vector_view(const VectorObj* value) noexcept;
lmmc_mat_t matrix_view(const MatrixObj* value) noexcept;
AdtObj* lmmc_vector_output(const char* name, lmmc_status_t status,
                           lmmc_vec_t& output);
AdtObj* lmmc_matrix_output(const char* name, lmmc_status_t status,
                           lmmc_mat_t& output);
MatrixObj* copy_lmmc_matrix(const lmmc_mat_t& matrix);
lmmc_vec_t vector_view(VectorObj* value) noexcept;
lmmc_mat_t matrix_view(MatrixObj* value) noexcept;
template <typename Operation>
AdtObj* vector_stat_result(const char* name, VectorObj* value,
                           Operation operation) {
    if (!value)
        return result_error(MathErrorCode::InvalidArgument, name,
                            "null vector");
    auto view = vector_view(value);
    lmmc_real_t result = 0.0;
    const auto status = operation(&view, &result);
    return lmmc_real_result(name, status, result);
}

template <typename Operation>
AdtObj* lmmc_unary_real_result(const char* name, const double value,
                               Operation operation) {
    lmmc_real_t result = 0.0;
    const auto status = operation(value, &result);
    return lmmc_real_result(name, status, result);
}

template <typename Operation>
AdtObj* lmmc_binary_real_result(const char* name, const double lhs,
                                const double rhs, Operation operation) {
    lmmc_real_t result = 0.0;
    const auto status = operation(lhs, rhs, &result);
    return lmmc_real_result(name, status, result);
}

template <typename Operation>
AdtObj* lmmc_ternary_real_result(const char* name, const double first,
                                 const double second, const double third,
                                 Operation operation) {
    lmmc_real_t result = 0.0;
    const auto status = operation(first, second, third, &result);
    return lmmc_real_result(name, status, result);
}

template <typename Operation>
AdtObj* lmmc_vector_binary_result(const char* name, VectorObj* lhs,
                                  VectorObj* rhs, Operation operation) {
    if (!lhs || !rhs)
        return result_error(MathErrorCode::InvalidArgument, name,
                            "null vector");
    auto left = vector_view(lhs);
    auto right = vector_view(rhs);
    lmmc_vec_t output{};
    const auto status = operation(&left, &right, &output);
    return lmmc_vector_output(name, status, output);
}

template <typename Operation>
AdtObj* lmmc_vector_scalar_result(const char* name, VectorObj* value,
                                  const double scalar, Operation operation) {
    if (!value)
        return result_error(MathErrorCode::InvalidArgument, name,
                            "null vector");
    auto input = vector_view(value);
    lmmc_vec_t output{};
    const auto status = operation(&input, scalar, &output);
    return lmmc_vector_output(name, status, output);
}

template <typename Operation>
AdtObj* lmmc_matrix_binary_result(const char* name, MatrixObj* lhs,
                                  MatrixObj* rhs, Operation operation) {
    if (!lhs || !rhs || !lhs->valid() || !rhs->valid()) {
        return result_error(MathErrorCode::InvalidArgument, name,
                            "invalid matrix");
    }
    auto left = matrix_view(lhs);
    auto right = matrix_view(rhs);
    lmmc_mat_t output{};
    const auto status = operation(&left, &right, &output);
    return lmmc_matrix_output(name, status, output);
}

template <typename Operation>
AdtObj* lmmc_matrix_scalar_result(const char* name, MatrixObj* value,
                                  const double scalar, Operation operation) {
    if (!value || !value->valid()) {
        return result_error(MathErrorCode::InvalidArgument, name,
                            "invalid matrix");
    }
    auto input = matrix_view(value);
    lmmc_mat_t output{};
    const auto status = operation(&input, scalar, &output);
    return lmmc_matrix_output(name, status, output);
}

template <typename Operation>
AdtObj* lmmc_matrix_unary_result(const char* name, MatrixObj* value,
                                 Operation operation) {
    if (!value || !value->valid()) {
        return result_error(MathErrorCode::InvalidArgument, name,
                            "invalid matrix");
    }
    auto input = matrix_view(value);
    lmmc_mat_t output{};
    const auto status = operation(&input, &output);
    return lmmc_matrix_output(name, status, output);
}

} // namespace lmx::bridge
