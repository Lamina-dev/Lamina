#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include <optional>
#include "bridge/callback.hpp"
#include "runtime/object/sparse.hpp"
#include "lmmc/sparse.h"
#include "lmmc/itersolve.h"
#include "lmmc/precond.h"

using namespace lmx::bridge;

namespace {
using lmx::runtime::SparseMatrixObj;
enum class IterativeAlgorithm {
    ConjugateGradient,
    BiconjugateGradientStabilized,
    GeneralizedMinimalResidual,
    MinimumResidual,
    LeastSquaresOrthogonalTriangular,
};

AdtObj* iterative_options(const lmmc_itersolve_config_t& config,
                          const char* preconditioner,
                          const double ilut_drop_tolerance,
                          const LmInt ilut_max_fill) {
    if (config.max_iter > static_cast<std::size_t>(
                              std::numeric_limits<LmInt>::max()) ||
        config.restart > static_cast<std::size_t>(
                             std::numeric_limits<LmInt>::max()))
        return result_error(MathErrorCode::NumericalFailure, __func__, "sparse.default_options: integer overflow");
    std::vector<Value> fields;
    fields.emplace_back(config.abs_tol);
    fields.emplace_back(config.rel_tol);
    fields.emplace_back(static_cast<LmInt>(config.max_iter));
    fields.emplace_back(static_cast<LmInt>(config.restart));
    fields.emplace_back(take_object_value(
        make_owned_object<StringObj>(preconditioner), ValueKind::Obj));
    fields.emplace_back(ilut_drop_tolerance);
    fields.emplace_back(ilut_max_fill);
    return result_ok(
        new AdtObj("IterativeOptions", "IterativeOptions", std::move(fields)),
        ValueKind::Obj);
}

bool parse_iterative_options(
    AdtObj* options, const std::size_t problem_size,
    lmmc_itersolve_config_t& config, std::string& preconditioner,
    double& drop_tolerance, std::size_t& max_fill, std::string& error) {
    if (lmmc_itersolve_default_config(problem_size, &config) != LMMC_STATUS_OK) {
        error = "cannot initialize iterative solver options";
        return false;
    }
    preconditioner = "none";
    drop_tolerance = 1e-4;
    max_fill = 20;
    if (!options) return true;
    if (options->type_name() != "IterativeOptions" &&
        !options->type_name().ends_with("::IterativeOptions")) {
        error = "expected IterativeOptions";
        return false;
    }
    const auto* abs_tol = options->field(0);
    const auto* rel_tol = options->field(1);
    const auto* max_iter = options->field(2);
    const auto* restart = options->field(3);
    const auto* precond = options->field(4);
    const auto* drop = options->field(5);
    const auto* fill = options->field(6);
    if (!abs_tol || !rel_tol || !max_iter || !restart || !precond ||
        !drop || !fill || max_iter->kind != ValueKind::Int ||
        restart->kind != ValueKind::Int || fill->kind != ValueKind::Int ||
        max_iter->int_val <= 0 || restart->int_val <= 0 ||
        fill->int_val <= 0 || precond->kind != ValueKind::Obj ||
        !precond->obj ||
        precond->obj->get_kind() != lmx::runtime::ObjectKind::String) {
        error = "invalid IterativeOptions fields";
        return false;
    }
    if (!numeric_value(*abs_tol, config.abs_tol) ||
        !numeric_value(*rel_tol, config.rel_tol) ||
        !numeric_value(*drop, drop_tolerance) ||
        !std::isfinite(config.abs_tol) || !std::isfinite(config.rel_tol) ||
        !std::isfinite(drop_tolerance) || config.abs_tol < 0.0 ||
        config.rel_tol < 0.0 || drop_tolerance < 0.0) {
        error = "invalid iterative solver tolerances";
        return false;
    }
    config.max_iter = static_cast<std::size_t>(max_iter->int_val);
    config.restart = static_cast<std::size_t>(restart->int_val);
    preconditioner =
        static_cast<StringObj*>(precond->obj)->c_str();
    if (preconditioner != "none" && preconditioner != "jacobi" &&
        preconditioner != "ilu0" && preconditioner != "ilut") {
        error = "unknown preconditioner `" + preconditioner + "`";
        return false;
    }
    max_fill = static_cast<std::size_t>(fill->int_val);
    return true;
}

struct PreconditionerGuard {
    lmmc_precond_t value{};
    bool initialized = false;
    ~PreconditionerGuard() {
        if (initialized) lmmc_precond_destroy(&value);
    }
};

lmmc_status_t create_preconditioner(
    const std::string& name, const lmmc_sparse_mat_t* matrix,
    const std::size_t size, const double drop_tolerance,
    const std::size_t max_fill, PreconditionerGuard& output) {
    lmmc_status_t status = LMMC_STATUS_INVALID_ARGUMENT;
    if (name == "none")
        status = lmmc_precond_create_none(size, &output.value);
    else if (name == "jacobi")
        status = lmmc_precond_create_jacobi(matrix, &output.value);
    else if (name == "ilu0")
        status = lmmc_precond_create_ilu0(matrix, &output.value);
    else if (name == "ilut")
        status = lmmc_precond_create_ilut(
            matrix, drop_tolerance, max_fill, &output.value);
    output.initialized = status == LMMC_STATUS_OK;
    return status;
}

AdtObj* iterative_result(
    std::vector<double> solution, const lmmc_itersolve_result_t& result) {
    if (result.num_iter > static_cast<std::size_t>(
                              std::numeric_limits<LmInt>::max()))
        return result_error(MathErrorCode::NumericalFailure, __func__, "iterative solver iteration count overflow");
    std::vector<Value> fields;
    fields.emplace_back(take_object_value(
        make_owned_object<VectorObj>(std::move(solution)), ValueKind::Vector));
    fields.emplace_back(result.converged != 0);
    fields.emplace_back(static_cast<LmInt>(result.num_iter));
    fields.emplace_back(result.initial_residual_norm);
    fields.emplace_back(result.final_residual_norm);
    return result_ok(
        new AdtObj("IterativeResult", "IterativeResult", std::move(fields)),
        ValueKind::Obj);
}

struct IterativeOperatorCallbacks {
    std::optional<VectorCallbackContext> forward;
    std::optional<VectorCallbackContext> transpose;

    IterativeOperatorCallbacks(
        const lmx::runtime::FuncObj* forward_operation,
        const lmx::runtime::FuncObj* transpose_operation,
        const std::size_t input_size,
        const std::size_t output_size) {
        if (forward_operation)
            forward.emplace(forward_operation, input_size, output_size);
        if (transpose_operation)
            transpose.emplace(transpose_operation, output_size, input_size);
    }
};

lmmc_status_t iterative_forward_adapter(
    const lmmc_vec_t* input, lmmc_vec_t* output, void* user_data) noexcept {
    auto* callbacks = static_cast<IterativeOperatorCallbacks*>(user_data);
    if (!callbacks || !callbacks->forward)
        return LMMC_STATUS_INVALID_ARGUMENT;
    return vector_callback_trampoline(
        input, output, &*callbacks->forward);
}

lmmc_status_t iterative_transpose_adapter(
    const lmmc_vec_t* input, lmmc_vec_t* output, void* user_data) noexcept {
    auto* callbacks = static_cast<IterativeOperatorCallbacks*>(user_data);
    if (!callbacks || !callbacks->transpose)
        return LMMC_STATUS_INVALID_ARGUMENT;
    return vector_callback_trampoline(
        input, output, &*callbacks->transpose);
}

AdtObj* run_iterative_solver(
    const IterativeAlgorithm algorithm, SparseMatrixObj* matrix, VectorObj* rhs,
    AdtObj* options, const lmx::runtime::FuncObj* forward_operation,
    const lmx::runtime::FuncObj* transpose_operation,
    const std::size_t operator_solution_size = 0) {
    if (!rhs || rhs->size() == 0)
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse iterative solver: invalid rhs");
    if (!forward_operation && (!matrix || !matrix->valid()))
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse iterative solver: invalid matrix");
    if (algorithm == IterativeAlgorithm::LeastSquaresOrthogonalTriangular &&
        (!forward_operation || !transpose_operation))
        return result_error(MathErrorCode::InvalidArgument, __func__,
                            "matrix-free LSQR requires forward and transpose operations");
    const auto rhs_size = rhs->size();
    const auto solution_size =
        algorithm == IterativeAlgorithm::LeastSquaresOrthogonalTriangular
        ? (matrix ? matrix->matrix().cols : operator_solution_size)
        : rhs_size;
    if (solution_size == 0)
        return result_error(MathErrorCode::InvalidArgument, __func__,
                            "matrix-free LSQR requires a positive solution size");
    lmmc_itersolve_config_t config{};
    std::string preconditioner;
    double drop_tolerance = 0.0;
    std::size_t max_fill = 0;
    std::string error;
    if (!parse_iterative_options(
            options, solution_size, config, preconditioner,
            drop_tolerance, max_fill, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    if (forward_operation && preconditioner != "none")
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            "matrix-free solver only supports the `none` preconditioner");
    PreconditionerGuard preconditioner_guard;
    const lmmc_sparse_mat_t* sparse =
        matrix ? &matrix->matrix() : nullptr;
    const auto preconditioner_status = create_preconditioner(
        preconditioner, sparse, solution_size, drop_tolerance, max_fill,
        preconditioner_guard);
    if (preconditioner_status != LMMC_STATUS_OK)
        return result_error(preconditioner_status, "sparse preconditioner");

    IterativeOperatorCallbacks callbacks(
        forward_operation, transpose_operation, solution_size, rhs_size);
    if (forward_operation) {
        config.apply_op = iterative_forward_adapter;
        config.op_user_data = &callbacks;
    }
    if (transpose_operation) {
        config.apply_transpose_op = iterative_transpose_adapter;
        config.op_user_data = &callbacks;
    }
    auto b = vector_view(rhs);
    std::vector<double> solution(solution_size, 0.0);
    lmmc_vec_t x{solution_size, solution.data(), 0};
    lmmc_itersolve_result_t result{};
    const auto* precond =
        preconditioner == "none" ? nullptr : &preconditioner_guard.value;
    lmmc_status_t status = LMMC_STATUS_INVALID_ARGUMENT;
    switch (algorithm) {
        case IterativeAlgorithm::ConjugateGradient:
            status =
                lmmc_cg_solve(sparse, &b, precond, &config, &x, &result);
            break;
        case IterativeAlgorithm::BiconjugateGradientStabilized:
            status = lmmc_bicgstab_solve(
                sparse, &b, precond, &config, &x, &result);
            break;
        case IterativeAlgorithm::GeneralizedMinimalResidual:
            status = lmmc_gmres_solve(
                sparse, &b, precond, &config, &x, &result);
            break;
        case IterativeAlgorithm::MinimumResidual:
            for (int attempt = 0; attempt < 8; ++attempt) {
                status = lmmc_minres_solve(
                    sparse, &b, precond, &config, &x, &result);
                if (status != LMMC_STATUS_OK) break;
                if (forward_operation) {
                    std::vector<double> applied(rhs_size);
                    lmmc_vec_t applied_view{
                        rhs_size, applied.data(), 0};
                    status = iterative_forward_adapter(
                        &x, &applied_view, &callbacks);
                    if (status != LMMC_STATUS_OK) break;
                    double residual_squared = 0.0;
                    for (std::size_t index = 0; index < rhs_size; ++index) {
                        const double residual =
                            applied[index] - rhs->data()[index];
                        residual_squared += residual * residual;
                    }
                    result.final_residual_norm = std::sqrt(residual_squared);
                    result.converged =
                        result.final_residual_norm <=
                            config.abs_tol +
                                config.rel_tol * result.initial_residual_norm;
                }
                if (result.converged) break;
            }
            break;
        case IterativeAlgorithm::LeastSquaresOrthogonalTriangular:
            status = lmmc_lsqr_solve(
                sparse, &b, &config, &x, &result);
            break;
    }
    if (callbacks.forward && callbacks.forward->failed())
        return result_error(MathErrorCode::CallbackFailure,
                            "sparse iterative solver",
                            std::move(callbacks.forward->error));
    if (callbacks.transpose && callbacks.transpose->failed())
        return result_error(MathErrorCode::CallbackFailure,
                            "sparse iterative solver",
                            std::move(callbacks.transpose->error));
    if (status != LMMC_STATUS_OK)
        return result_error(status, "sparse iterative solver");
    return iterative_result(std::move(solution), result);
}
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_solve_with_lower_upper_factorization(
    SparseMatrixObj* matrix, VectorObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    if (!matrix || !matrix->valid() || !rhs)
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.lu_solve: invalid argument");
    lmmc_sparse_lu_t* factor = nullptr;
    auto status = lmmc_sparse_lu_symbolic(&matrix->matrix(), &factor);
    if (status == LMMC_STATUS_OK)
        status = lmmc_sparse_lu_numeric(&matrix->matrix(), factor);
    auto input = vector_view(rhs);
    std::vector<double> data(rhs->size());
    lmmc_vec_t output{data.size(), data.data(), 0};
    if (status == LMMC_STATUS_OK)
        status = lmmc_sparse_lu_solve(factor, &input, &output);
    lmmc_sparse_lu_destroy(factor);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "sparse.lu_solve");
    return result_ok(new VectorObj(std::move(data)), ValueKind::Vector);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_cholesky_solve(
    SparseMatrixObj* matrix, VectorObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    if (!matrix || !matrix->valid() || !rhs)
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.cholesky_solve: invalid argument");
    lmmc_sparse_chol_t* factor = nullptr;
    auto status = lmmc_sparse_chol_symbolic(&matrix->matrix(), &factor);
    if (status == LMMC_STATUS_OK)
        status = lmmc_sparse_chol_numeric(&matrix->matrix(), factor);
    auto input = vector_view(rhs);
    std::vector<double> data(rhs->size());
    lmmc_vec_t output{data.size(), data.data(), 0};
    if (status == LMMC_STATUS_OK)
        status = lmmc_sparse_chol_solve(factor, &input, &output);
    lmmc_sparse_chol_destroy(factor);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "sparse.cholesky_solve");
    return result_ok(new VectorObj(std::move(data)), ValueKind::Vector);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_default_options(const LmInt size) noexcept try {
    ensure_lmmc_runtime();
    if (size <= 0)
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.default_options: invalid size");
    lmmc_itersolve_config_t config{};
    const auto status = lmmc_itersolve_default_config(
        static_cast<std::size_t>(size), &config);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "sparse.default_options");
    return iterative_options(config, "none", 1e-4, 20);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_conjugate_gradient(
    SparseMatrixObj* matrix, VectorObj* rhs, AdtObj* options) noexcept try {
    ensure_lmmc_runtime();
    return run_iterative_solver(
        IterativeAlgorithm::ConjugateGradient,
        matrix, rhs, options, nullptr, nullptr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_biconjugate_gradient_stabilized(
    SparseMatrixObj* matrix, VectorObj* rhs, AdtObj* options) noexcept try {
    ensure_lmmc_runtime();
    return run_iterative_solver(
        IterativeAlgorithm::BiconjugateGradientStabilized,
        matrix, rhs, options, nullptr, nullptr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_generalized_minimal_residual(
    SparseMatrixObj* matrix, VectorObj* rhs, AdtObj* options) noexcept try {
    ensure_lmmc_runtime();
    return run_iterative_solver(
        IterativeAlgorithm::GeneralizedMinimalResidual,
        matrix, rhs, options, nullptr, nullptr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_minimum_residual(
    SparseMatrixObj* matrix, VectorObj* rhs, AdtObj* options) noexcept try {
    ensure_lmmc_runtime();
    return run_iterative_solver(
        IterativeAlgorithm::MinimumResidual,
        matrix, rhs, options, nullptr, nullptr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_least_squares_orthogonal_triangular(
    SparseMatrixObj* matrix, VectorObj* rhs, AdtObj* options) noexcept try {
    ensure_lmmc_runtime();
    return run_iterative_solver(
        IterativeAlgorithm::LeastSquaresOrthogonalTriangular,
        matrix, rhs, options, nullptr, nullptr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_minres_operator(
    const lmx::runtime::FuncObj* operation, VectorObj* rhs, AdtObj* options) noexcept try {
    ensure_lmmc_runtime();
    return run_iterative_solver(
        IterativeAlgorithm::MinimumResidual,
        nullptr, rhs, options, operation, nullptr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_lsqr_operator(
    const lmx::runtime::FuncObj* forward_operation,
    const lmx::runtime::FuncObj* transpose_operation,
    VectorObj* rhs, const LmInt solution_size,
    AdtObj* options) noexcept try {
    ensure_lmmc_runtime();
    if (solution_size <= 0)
        return result_error(MathErrorCode::InvalidArgument, __func__,
                            "matrix-free LSQR requires a positive solution size");
    return run_iterative_solver(
        IterativeAlgorithm::LeastSquaresOrthogonalTriangular,
        nullptr, rhs, options, forward_operation, transpose_operation,
        static_cast<std::size_t>(solution_size));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_iterative_solution(AdtObj* result) noexcept try {
    ensure_lmmc_runtime();
    if (!result || result->type_name() != "IterativeResult")
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse: expected IterativeResult");
    const auto* field = result->field(0);
    if (!field || field->kind != ValueKind::Vector || !field->obj)
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse: invalid IterativeResult");
    return result_ok(field->obj->get(), ValueKind::Vector);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API bool lmx_sparse_linear_algebra_iterative_converged(AdtObj* result) noexcept try {
    ensure_lmmc_runtime();
    const auto* field = result ? result->field(1) : nullptr;
    return field && field->kind == ValueKind::Bool && field->bool_val;
} catch (...) {
    return false;
}
