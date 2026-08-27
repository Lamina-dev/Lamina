#include "bridge/runtime_views.hpp"

#include <utility>

namespace lmx::bridge {

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
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&output);
        return result_error(status, name ? name : "LMMC");
    }
    std::vector<double> data(output.data, output.data + output.size);
    lmmc_vec_destroy(&output);
    return result_ok(new VectorObj(std::move(data)), ValueKind::Vector);
}

AdtObj* lmmc_matrix_output(const char* name, const lmmc_status_t status,
                           lmmc_mat_t& output) {
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&output);
        return result_error(status, name ? name : "LMMC");
    }
    std::vector<double> data;
    data.reserve(output.rows * output.cols);
    for (std::size_t row = 0; row < output.rows; ++row) {
        data.insert(data.end(), output.data + row * output.stride,
                    output.data + row * output.stride + output.cols);
    }
    const auto rows = output.rows;
    const auto cols = output.cols;
    lmmc_mat_destroy(&output);
    return result_ok(new MatrixObj(rows, cols, std::move(data)),
                     ValueKind::Matrix);
}

MatrixObj* copy_lmmc_matrix(const lmmc_mat_t& matrix) {
    std::vector<double> data;
    data.reserve(matrix.rows * matrix.cols);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        data.insert(data.end(), matrix.data + row * matrix.stride,
                    matrix.data + row * matrix.stride + matrix.cols);
    }
    return new MatrixObj(matrix.rows, matrix.cols, std::move(data));
}

} // namespace lmx::bridge
