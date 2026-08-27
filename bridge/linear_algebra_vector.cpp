#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "lmmc/lsr_stdlib.h"

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_linear_algebra_vector_from_array(ArrayObj* values) {
    std::vector<double> data;
    std::string error;
    if (!array_numbers(values, data, error)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "vector: " + error);
    }
    if (data.empty()) return result_error(MathErrorCode::EmptyInput, __func__, "vector: empty vectors are not supported by LMMC");
    return result_ok(new VectorObj(std::move(data)), ValueKind::Vector);
}

extern "C" LM_API LmInt lmx_linear_algebra_vector_size(VectorObj* value) {
    return value ? static_cast<LmInt>(value->size()) : 0;
}

extern "C" LM_API AdtObj* lmx_linear_algebra_at(VectorObj* value, const LmInt index) {
    if (!value || index < 0 || static_cast<std::size_t>(index) >= value->size()) {
        return result_error(MathErrorCode::IndexOutOfBounds, __func__, "vector_at: index out of bounds");
    }
    return result_ok(value->data()[static_cast<std::size_t>(index)]);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_dot_product(VectorObj* lhs, VectorObj* rhs) {
    if (!lhs || !rhs) return result_error(MathErrorCode::InvalidArgument, __func__, "vector_dot: null vector");
    auto left = vector_view(lhs);
    auto right = vector_view(rhs);
    lmmc_real_t result = 0.0;
    const auto status = lmmc_lsr_linalg_dot(&left, &right, &result);
    return lmmc_real_result("vector_dot", status, result);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_norm(VectorObj* value) {
    return vector_stat_result("vector_norm", value, lmmc_lsr_linalg_norm);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_cross(VectorObj* lhs, VectorObj* rhs) {
    return lmmc_vector_binary_result("vector_cross", lhs, rhs,
                                     lmmc_lsr_linalg_cross);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_vector_add(VectorObj* lhs, VectorObj* rhs) {
    return lmmc_vector_binary_result("vector_add", lhs, rhs,
                                     lmmc_lsr_linalg_vec_add);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_vector_subtract(VectorObj* lhs, VectorObj* rhs) {
    return lmmc_vector_binary_result("vector_sub", lhs, rhs,
                                     lmmc_lsr_linalg_vec_sub);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_vector_multiply(VectorObj* lhs, VectorObj* rhs) {
    return lmmc_vector_binary_result("vector_mul", lhs, rhs,
                                     lmmc_lsr_linalg_vec_mul);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_divide(VectorObj* lhs, VectorObj* rhs) {
    return lmmc_vector_binary_result("vector_div", lhs, rhs,
                                     lmmc_lsr_linalg_vec_div);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_power(VectorObj* base, VectorObj* exponent) {
    return lmmc_vector_binary_result("vector_pow", base, exponent,
                                     lmmc_lsr_linalg_vec_pow);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_vector_add_scalar(VectorObj* value,
                                                  const double scalar) {
    return lmmc_vector_scalar_result("vector_add_scalar", value, scalar,
                                     lmmc_lsr_linalg_vec_add_scalar);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_vector_subtract_scalar(VectorObj* value,
                                                  const double scalar) {
    return lmmc_vector_scalar_result("vector_sub_scalar", value, scalar,
                                     lmmc_lsr_linalg_vec_sub_scalar);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_vector_multiply_scalar(VectorObj* value,
                                                  const double scalar) {
    return lmmc_vector_scalar_result("vector_mul_scalar", value, scalar,
                                     lmmc_lsr_linalg_vec_mul_scalar);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_vector_divide_scalar(VectorObj* value,
                                                  const double scalar) {
    return lmmc_vector_scalar_result("vector_div_scalar", value, scalar,
                                     lmmc_lsr_linalg_vec_div_scalar);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_vector_power_scalar(VectorObj* value,
                                                  const double scalar) {
    return lmmc_vector_scalar_result("vector_pow_scalar", value, scalar,
                                     lmmc_lsr_linalg_vec_pow_scalar);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_vector_scale(VectorObj* value,
                                             const double scalar) {
    return lmmc_vector_scalar_result("vector_scale", value, scalar,
                                     lmmc_lsr_linalg_vec_scale);
}
