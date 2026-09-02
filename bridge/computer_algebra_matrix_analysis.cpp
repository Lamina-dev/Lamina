#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/math_internal.hpp"
#include "symbolic_matrix.hpp"
#include "matrix_decomposition.hpp"

using namespace lmx::bridge;
using lmx::bridge::math_internal::checked_expression_operation;
using lmx::bridge::math_internal::checked_expr_result;

namespace {
AdtObj* nested_expr_array_result(
    const std::vector<std::vector<lamina::lsr::ExprPtr>>& values) {
    auto outer = make_owned_object<ArrayObj>();
    for (const auto& vector : values) {
        auto inner = make_owned_object<ArrayObj>();
        for (const auto& expression : vector) {
            if (!expression) {
                return result_error(MathErrorCode::InternalError, __func__,
                    "CasError(InternalInvariant: null nested expression)");
            }
            inner->append(take_object_value(
                make_owned_object<ExprObj>(expression), ValueKind::Expr));
        }
        outer->append(take_object_value(std::move(inner), ValueKind::Obj));
    }
    return result_ok(outer.release(), ValueKind::Obj);
}
} // namespace

/** @brief Computes unordered symbolic matrix eigenvalues. @param value Borrowed matrix expression. @return Owning Result set or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_eigenvalues(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* matrix = checked_expr(value, error);
    if (!matrix) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::matrix_eigenvalues_checked(*matrix);
    if (!result) return result_error(result.error());
    return math_internal::unordered_expr_result(result.value());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes ordered symbolic matrix eigenvectors. @param value Borrowed matrix expression. @return Owning Result nested array or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_eigenvectors(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* matrix = checked_expr(value, error);
    if (!matrix) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::matrix_eigenvectors_checked(*matrix);
    if (!result) return result_error(result.error());
    return nested_expr_array_result(result.value());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Creates a symbolic rotation matrix. @param theta Angle in radians. @param dimension Matrix dimension. @return Owning Result Expr or error. @ownership Caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_rotation(double theta, LmInt dimension) noexcept try {
    ensure_lmmc_runtime();
    if (dimension <= 0 || dimension > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix.rotation: invalid dimension");
    return checked_expr_result(lamina::matrix_rotation_checked(
        theta, static_cast<int>(dimension)));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Creates a symbolic reflection matrix. @param angle Axis angle in radians. @param dimension Matrix dimension. @return Owning Result Expr or error. @ownership Caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_reflection(double angle, LmInt dimension) noexcept try {
    ensure_lmmc_runtime();
    if (dimension <= 0 || dimension > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix.reflection: invalid dimension");
    return checked_expr_result(lamina::matrix_reflection_checked(
        angle, static_cast<int>(dimension)));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Creates a symbolic scaling matrix. @param sx X scale. @param sy Y scale. @param dimension Matrix dimension. @return Owning Result Expr or error. @ownership Caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_scaling(
    double sx, double sy, LmInt dimension) noexcept try {
    ensure_lmmc_runtime();
    if (dimension <= 0 || dimension > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix.scaling: invalid dimension");
    return checked_expr_result(lamina::matrix_scaling_checked(
        sx, sy, static_cast<int>(dimension)));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes symbolic matrix trace. @param value Borrowed matrix. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_trace(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    return checked_expression_operation("matrix.trace", value,
        [](const auto& matrix) { return lamina::matrix_trace(matrix); });
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes symbolic matrix rank. @param value Borrowed matrix. @return Owning Result int or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_rank(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* matrix = checked_expr(value, error);
    if (!matrix) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto rank = lamina::matrix_rank_checked(*matrix);
    if (!rank) return result_error(rank.error());
    if (rank.value() >
        static_cast<std::size_t>(std::numeric_limits<LmInt>::max())) {
        return result_error(
            MathErrorCode::ResourceLimit, __func__,
            "matrix.rank: result exceeds language integer range");
    }
    return result_ok(static_cast<LmInt>(rank.value()));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes symbolic matrix exponential. @param value Borrowed matrix. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_exponential(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    return checked_expression_operation("matrix.exponential", value,
        [](const auto& matrix) { return lamina::matrix_exp(matrix); });
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes symbolic matrix logarithm. @param value Borrowed matrix. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_natural_logarithm(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    return checked_expression_operation("matrix.natural_logarithm", value,
        [](const auto& matrix) { return lamina::matrix_log(matrix); });
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes a symbolic Kronecker product. @param lhs Borrowed matrix. @param rhs Borrowed matrix. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_kronecker(ExprObj* lhs, ExprObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* a = checked_expr(lhs, error);
    const auto* b = checked_expr(rhs, error);
    if (!a || !b) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto result = lamina::kronecker(*a, *b);
    return result ? result_ok(new ExprObj(std::move(result)), ValueKind::Expr)
                  : result_error(MathErrorCode::InvalidArgument, __func__, "matrix.kronecker: invalid dimensions");
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Applies symbolic Gram-Schmidt. @param vectors Borrowed nested expression arrays. @param normalize Whether to normalize outputs. @return Owning Result nested array or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_gram_schmidt(
    ArrayObj* vectors, bool normalize) noexcept try {
    ensure_lmmc_runtime();
    std::vector<std::vector<lamina::lsr::ExprPtr>> values;
    std::string error;
    if (!math_internal::nested_expressions(vectors, values, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix.gram_schmidt: " + error);
    return nested_expr_array_result(lamina::gram_schmidt(values, normalize));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Classifies a symbolic quadratic-form matrix. @param value Borrowed matrix. @return Owning Result text or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_classify_quadratic_form(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* matrix = checked_expr(value, error);
    if (!matrix) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return result_ok(
        new StringObj(lamina::classify_quadratic_form(*matrix)),
        ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}
