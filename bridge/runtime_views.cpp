#include "bridge/runtime_views.hpp"

#include <utility>
#include <memory>
#include <cstdint>
#include <limits>

namespace lmx::bridge {

bool lmmc_vector_owner(const lmmc_vec_t& value) noexcept {
    if (value.owns_data != 1 || !value.data || value.size == 0 ||
        value.size > std::numeric_limits<std::size_t>::max() / sizeof(double))
        return false;
    return reinterpret_cast<std::uintptr_t>(value.data) <=
           std::numeric_limits<std::uintptr_t>::max() -
               value.size * sizeof(double);
}

bool lmmc_matrix_owner(const lmmc_mat_t& value) noexcept {
    if (value.owns_data != 1 || value.rows == 0 || value.cols == 0 ||
        value.stride != value.cols ||
        value.rows > std::numeric_limits<std::size_t>::max() / value.cols)
        return false;
    return lmmc_vector_owner(
        {value.rows * value.cols, value.data, value.owns_data});
}

lmmc_vec_t vector_view(VectorObj* value) noexcept {
    return {value ? value->size() : 0,
            value && !value->data().empty() ? value->data().data() : nullptr, 0};
}

lmmc_mat_t matrix_view(MatrixObj* value) noexcept {
    return {value ? value->rows() : 0, value ? value->cols() : 0,
            value ? value->cols() : 0,
            value && !value->data().empty() ? value->data().data() : nullptr, 0};
}

AdtObj* lmmc_vector_output(const char* name, const lmmc_status_t status,
                           lmmc_vec_t& output) {
    const auto cleanup =
        std::unique_ptr<lmmc_vec_t, decltype(&lmmc_vec_destroy)>(
            &output, &lmmc_vec_destroy);
    if (status != LMMC_STATUS_OK) {
        return result_error(status, name ? name : "LMMC");
    }
    if (!lmmc_vector_owner(output))
        return result_error(MathErrorCode::InvalidArgument,
                            name ? name : "LMMC", "invalid vector owner");
    return result_ok(new VectorObj(output), ValueKind::Vector);
}

AdtObj* lmmc_matrix_output(const char* name, const lmmc_status_t status,
                           lmmc_mat_t& output) {
    const auto cleanup =
        std::unique_ptr<lmmc_mat_t, decltype(&lmmc_mat_destroy)>(
            &output, &lmmc_mat_destroy);
    if (status != LMMC_STATUS_OK) {
        return result_error(status, name ? name : "LMMC");
    }
    if (!lmmc_matrix_owner(output))
        return result_error(MathErrorCode::InvalidArgument,
                            name ? name : "LMMC", "invalid matrix owner");
    return result_ok(new MatrixObj(output), ValueKind::Matrix);
}


} // namespace lmx::bridge
