#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/linear_algebra_internal.hpp"
#include "lmmc/linear_algebra.h"
#include "lmmc/eigen.h"
#include <algorithm>
#include <numeric>

using namespace lmx::bridge;
using lmx::bridge::linear_algebra::copied_matrix;

namespace lmx::bridge::linear_algebra {

MatrixObj* copied_matrix(MatrixObj* input) {
    return input ? new MatrixObj(input->rows(), input->cols(), input->data())
                 : nullptr;
}

} // namespace lmx::bridge::linear_algebra

namespace {

ArrayObj* pivot_array(const std::vector<std::size_t>& pivots,
                      std::string& error) {
    auto* result = new ArrayObj();
    for (const auto pivot : pivots) {
        if (pivot > static_cast<std::size_t>(
                        std::numeric_limits<LmInt>::max())) {
            result->release();
            error = "linalg: pivot index exceeds Lamina int range";
            return nullptr;
        }
        result->append(Value(static_cast<LmInt>(pivot)));
    }
    return result;
}

bool factor_field(AdtObj* factor, const char* type_name,
                  const std::size_t index, const ValueKind kind,
                  const Value*& field, std::string& error) {
    if (!factor || factor->type_name() != type_name) {
        error = std::string("linalg: expected ") + type_name;
        return false;
    }
    field = factor->field(index);
    if (!field || field->kind != kind || !field->obj) {
        error = std::string("linalg: invalid ") + type_name;
        return false;
    }
    return true;
}

AdtObj* solve_with_factor(const char* name, MatrixObj* factor,
                          const std::size_t* pivots, const double* tau,
                          VectorObj* rhs, const int algorithm) {
    if (!factor || !factor->valid() || !rhs)
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string(name) + ": invalid argument");
    auto matrix = matrix_view(factor);
    auto input = vector_view(rhs);
    const std::size_t output_size =
        algorithm == 2 ? factor->cols() : factor->rows();
    std::vector<double> data(output_size);
    lmmc_vec_t output{output_size, data.data(), 0};
    lmmc_status_t status = LMMC_STATUS_INVALID_ARGUMENT;
    if (algorithm == 0)
        status = lmmc_lu_solve(&matrix, pivots, &input, &output);
    else if (algorithm == 1)
        status = lmmc_cholesky_solve(&matrix, &input, &output);
    else
        status = lmmc_qr_solve(&matrix, tau, &input, &output);
    if (status != LMMC_STATUS_OK) return result_error(status, name);
    return result_ok(new VectorObj(std::move(data)), ValueKind::Vector);
}
} // namespace

extern "C" LM_API AdtObj* lmx_linear_algebra_lower_upper_factorization(MatrixObj* value) {
    if (!value || !value->valid() || value->rows() != value->cols())
        return result_error(MathErrorCode::InvalidArgument, __func__, "linalg.lu_factor: expected a square matrix");
    auto* factor = copied_matrix(value);
    auto view = matrix_view(factor);
    std::vector<std::size_t> pivots(value->rows());
    const auto status =
        lmmc_lu_decompose_inplace(&view, pivots.data(), nullptr);
    if (status != LMMC_STATUS_OK) {
        factor->release();
        return result_error(status, "linalg.lu_factor");
    }
    std::string error;
    auto* pivot_values = pivot_array(pivots, error);
    if (!pivot_values) {
        factor->release();
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    }
    std::vector<Value> fields;
    fields.emplace_back(factor, ValueKind::Matrix);
    fields.emplace_back(pivot_values, ValueKind::Obj);
    return result_ok(
        new AdtObj("LUFactor", "LUFactor", std::move(fields)), ValueKind::Obj);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_solve_with_lower_upper_factorization(
    AdtObj* factor, VectorObj* rhs) {
    const Value* matrix_field = nullptr;
    const Value* pivot_field = nullptr;
    std::string error;
    if (!factor_field(factor, "LUFactor", 0, ValueKind::Matrix,
                      matrix_field, error) ||
        !factor_field(factor, "LUFactor", 1, ValueKind::Obj,
                      pivot_field, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* pivots = static_cast<ArrayObj*>(pivot_field->obj);
    std::vector<std::size_t> values;
    values.reserve(pivots->values().size());
    for (const auto& pivot : pivots->values()) {
        if (pivot.kind != ValueKind::Int || pivot.int_val < 0)
            return result_error(MathErrorCode::InvalidArgument, __func__, "linalg.lu_solve: invalid pivot");
        values.push_back(static_cast<std::size_t>(pivot.int_val));
    }
    return solve_with_factor(
        "linalg.lu_solve", static_cast<MatrixObj*>(matrix_field->obj),
        values.data(), nullptr, rhs, 0);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_cholesky_factor(MatrixObj* value) {
    if (!value || !value->valid() || value->rows() != value->cols())
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            "linalg.cholesky_factor: expected a square matrix");
    auto* factor = copied_matrix(value);
    auto view = matrix_view(factor);
    const auto status = lmmc_cholesky_decompose_inplace(&view);
    if (status != LMMC_STATUS_OK) {
        factor->release();
        return result_error(status, "linalg.cholesky_factor");
    }
    return result_ok(factor, ValueKind::Matrix);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_cholesky_solve(
    MatrixObj* factor, VectorObj* rhs) {
    return solve_with_factor(
        "linalg.cholesky_solve", factor, nullptr, nullptr, rhs, 1);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_orthogonal_triangular_factorization(MatrixObj* value) {
    if (!value || !value->valid() || value->rows() < value->cols())
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            "linalg.qr_factor: expected rows greater than or equal to columns");
    auto* factor = copied_matrix(value);
    auto view = matrix_view(factor);
    std::vector<double> tau(value->cols());
    const auto status =
        lmmc_qr_decompose_inplace(&view, tau.data(), tau.size());
    if (status != LMMC_STATUS_OK) {
        factor->release();
        return result_error(status, "linalg.qr_factor");
    }
    std::vector<Value> fields;
    fields.emplace_back(factor, ValueKind::Matrix);
    fields.emplace_back(new VectorObj(std::move(tau)), ValueKind::Vector);
    return result_ok(
        new AdtObj("QRFactor", "QRFactor", std::move(fields)), ValueKind::Obj);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_solve_with_orthogonal_triangular_factorization(
    AdtObj* factor, VectorObj* rhs) {
    const Value* matrix_field = nullptr;
    const Value* tau_field = nullptr;
    std::string error;
    if (!factor_field(factor, "QRFactor", 0, ValueKind::Matrix,
                      matrix_field, error) ||
        !factor_field(factor, "QRFactor", 1, ValueKind::Vector,
                      tau_field, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* tau = static_cast<VectorObj*>(tau_field->obj);
    return solve_with_factor(
        "linalg.qr_solve", static_cast<MatrixObj*>(matrix_field->obj),
        nullptr, tau->data().data(), rhs, 2);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_eigendecomposition(MatrixObj* value) {
    if (!value || !value->valid() || value->rows() != value->cols())
        return result_error(MathErrorCode::InvalidArgument, __func__, "linalg.eig: expected a square matrix");
    auto input = matrix_view(value);
    lmmc_eigen_gen_full_result_t output{};
    const auto status = lmmc_eigen_general_full(&input, &output);
    if (status != LMMC_STATUS_OK) {
        lmmc_eigen_gen_full_result_destroy(&output);
        return result_error(status, "linalg.eig");
    }
    const auto count = output.real_parts.size;
    std::vector<std::size_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](const auto lhs, const auto rhs) {
        if (output.real_parts.data[lhs] != output.real_parts.data[rhs])
            return output.real_parts.data[lhs] < output.real_parts.data[rhs];
        return output.imag_parts.data[lhs] < output.imag_parts.data[rhs];
    });
    std::vector<double> real_data(count);
    std::vector<double> imag_data(count);
    std::vector<double> real_vector_data(count * count);
    std::vector<double> imag_vector_data(count * count);
    for (std::size_t column = 0; column < count; ++column) {
        const auto source = order[column];
        real_data[column] = output.real_parts.data[source];
        imag_data[column] = output.imag_parts.data[source];
        for (std::size_t row = 0; row < count; ++row) {
            real_vector_data[row * count + column] =
                output.vectors_real.data[row * output.vectors_real.stride + source];
            imag_vector_data[row * count + column] =
                output.vectors_imag.data[row * output.vectors_imag.stride + source];
        }
    }
    auto* real_values =
        new MatrixObj(1, count, std::move(real_data));
    auto* imag_values =
        new MatrixObj(1, count, std::move(imag_data));
    auto* real_vectors =
        new MatrixObj(count, count, std::move(real_vector_data));
    auto* imag_vectors =
        new MatrixObj(count, count, std::move(imag_vector_data));
    lmmc_eigen_gen_full_result_destroy(&output);
    std::vector<Value> fields;
    fields.emplace_back(real_values, ValueKind::Matrix);
    fields.emplace_back(imag_values, ValueKind::Matrix);
    fields.emplace_back(real_vectors, ValueKind::Matrix);
    fields.emplace_back(imag_vectors, ValueKind::Matrix);
    return result_ok(
        new AdtObj("EigenResult", "EigenResult", std::move(fields)),
        ValueKind::Obj);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_singular_value_decomposition(MatrixObj* value) {
    if (!value || !value->valid())
        return result_error(MathErrorCode::InvalidArgument, __func__, "linalg.svd: invalid matrix");
    auto input = matrix_view(value);
    lmmc_svd_result_t output{};
    const auto status = lmmc_svd(&input, &output);
    if (status != LMMC_STATUS_OK) {
        lmmc_svd_result_destroy(&output);
        return result_error(status, "linalg.svd");
    }
    auto* u = copy_lmmc_matrix(output.U);
    std::vector<double> sigma(output.sigma.size * output.sigma.size, 0.0);
    for (std::size_t i = 0; i < output.sigma.size; ++i)
        sigma[i * output.sigma.size + i] = output.sigma.data[i];
    auto* s = new MatrixObj(
        output.sigma.size, output.sigma.size, std::move(sigma));
    auto* vt = copy_lmmc_matrix(output.Vt);
    lmmc_svd_result_destroy(&output);
    std::vector<Value> fields;
    fields.emplace_back(u, ValueKind::Matrix);
    fields.emplace_back(s, ValueKind::Matrix);
    fields.emplace_back(vt, ValueKind::Matrix);
    return result_ok(
        new AdtObj("SvdResult", "SvdResult", std::move(fields)),
        ValueKind::Obj);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_pseudoinverse(
    MatrixObj* value, const double tolerance) {
    if (!value || !value->valid() || tolerance < 0.0 ||
        !std::isfinite(tolerance))
        return result_error(MathErrorCode::InvalidArgument, __func__, "linalg.pinv: invalid argument");
    auto input = matrix_view(value);
    lmmc_mat_t output{};
    const auto create_status =
        lmmc_mat_create(value->cols(), value->rows(), &output);
    if (create_status != LMMC_STATUS_OK)
        return result_error(create_status, "linalg.pinv");
    const auto status = lmmc_pinv(&input, tolerance, &output);
    return lmmc_matrix_output("linalg.pinv", status, output);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_condition_number(MatrixObj* value) {
    if (!value || !value->valid())
        return result_error(MathErrorCode::InvalidArgument, __func__, "linalg.condition_number: invalid matrix");
    auto input = matrix_view(value);
    double output = 0.0;
    const auto status = lmmc_cond(&input, &output);
    return lmmc_real_result("linalg.condition_number", status, output);
}

namespace {
AdtObj* linalg_matrix_field(
    AdtObj* value, const char* type_name, const std::size_t index) {
    if (!value || value->type_name() != type_name)
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string("linalg: expected ") + type_name);
    const auto* field = value->field(index);
    if (!field || field->kind != ValueKind::Matrix || !field->obj)
        return result_error(MathErrorCode::InvalidArgument, __func__, "linalg: invalid decomposition field");
    return result_ok(field->obj->get(), ValueKind::Matrix);
}
} // namespace

extern "C" LM_API AdtObj* lmx_linear_algebra_eigen_real_values(AdtObj* value) {
    return linalg_matrix_field(value, "EigenResult", 0);
}
extern "C" LM_API AdtObj* lmx_linear_algebra_eigen_imag_values(AdtObj* value) {
    return linalg_matrix_field(value, "EigenResult", 1);
}
extern "C" LM_API AdtObj* lmx_linear_algebra_eigen_real_vectors(AdtObj* value) {
    return linalg_matrix_field(value, "EigenResult", 2);
}
extern "C" LM_API AdtObj* lmx_linear_algebra_eigen_imag_vectors(AdtObj* value) {
    return linalg_matrix_field(value, "EigenResult", 3);
}
extern "C" LM_API AdtObj* lmx_linear_algebra_svd_u(AdtObj* value) {
    return linalg_matrix_field(value, "SvdResult", 0);
}
extern "C" LM_API AdtObj* lmx_linear_algebra_svd_s(AdtObj* value) {
    return linalg_matrix_field(value, "SvdResult", 1);
}
extern "C" LM_API AdtObj* lmx_linear_algebra_svd_vt(AdtObj* value) {
    return linalg_matrix_field(value, "SvdResult", 2);
}
