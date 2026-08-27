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
    auto* outer = new ArrayObj();
    for (const auto& vector : values) {
        auto* inner = new ArrayObj();
        for (const auto& expression : vector) {
            if (!expression) {
                inner->release();
                outer->release();
                return result_error(MathErrorCode::InternalError, __func__, 
                    "CasError(InternalInvariant: null nested expression)");
            }
            inner->append(Value(new ExprObj(expression), ValueKind::Expr));
        }
        outer->append(Value(inner, ValueKind::Obj));
    }
    return result_ok(outer, ValueKind::Obj);
}
} // namespace

/** @brief Computes unordered symbolic matrix eigenvalues. @param value Borrowed matrix expression. @return Owning Result set or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_eigenvalues(ExprObj* value) {
    std::string error; const auto* matrix = checked_expr(value, error);
    if (!matrix) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::matrix_eigenvalues_checked(*matrix);
    if (!result) return result_error(result.error());
    return math_internal::unordered_expr_result(result.value());
}
/** @brief Computes ordered symbolic matrix eigenvectors. @param value Borrowed matrix expression. @return Owning Result nested array or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_eigenvectors(ExprObj* value) {
    std::string error; const auto* matrix = checked_expr(value, error);
    if (!matrix) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::matrix_eigenvectors_checked(*matrix);
    if (!result) return result_error(result.error());
    return nested_expr_array_result(result.value());
}
/** @brief Creates a symbolic rotation matrix. @param theta Angle in radians. @param dimension Matrix dimension. @return Owning Result Expr or error. @ownership Caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_rotation(double theta, LmInt dimension) {
    if (dimension <= 0 || dimension > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix.rotation: invalid dimension");
    return checked_expr_result(lamina::matrix_rotation_checked(
        theta, static_cast<int>(dimension)));
}
/** @brief Creates a symbolic reflection matrix. @param angle Axis angle in radians. @param dimension Matrix dimension. @return Owning Result Expr or error. @ownership Caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_reflection(double angle, LmInt dimension) {
    if (dimension <= 0 || dimension > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix.reflection: invalid dimension");
    return checked_expr_result(lamina::matrix_reflection_checked(
        angle, static_cast<int>(dimension)));
}
/** @brief Creates a symbolic scaling matrix. @param sx X scale. @param sy Y scale. @param dimension Matrix dimension. @return Owning Result Expr or error. @ownership Caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_scaling(
    double sx, double sy, LmInt dimension) {
    if (dimension <= 0 || dimension > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix.scaling: invalid dimension");
    return checked_expr_result(lamina::matrix_scaling_checked(
        sx, sy, static_cast<int>(dimension)));
}
/** @brief Computes symbolic matrix trace. @param value Borrowed matrix. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_trace(ExprObj* value) {
    return checked_expression_operation("matrix.trace", value,
        [](const auto& matrix) { return lamina::matrix_trace(matrix); });
}
/** @brief Computes symbolic matrix rank. @param value Borrowed matrix. @return Owning Result int or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_rank(ExprObj* value) {
    std::string error; const auto* matrix = checked_expr(value, error);
    if (!matrix) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    try {
        const int rank = lamina::matrix_rank(*matrix);
        return rank < 0 ? result_error(MathErrorCode::InvalidArgument, __func__, "matrix.rank: invalid matrix")
                        : result_ok(static_cast<LmInt>(rank));
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string("matrix.rank: ") + exception.what());
    }
}
/** @brief Computes symbolic matrix exponential. @param value Borrowed matrix. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_exponential(ExprObj* value) {
    return checked_expression_operation("matrix.exponential", value,
        [](const auto& matrix) { return lamina::matrix_exp(matrix); });
}
/** @brief Computes symbolic matrix logarithm. @param value Borrowed matrix. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_natural_logarithm(ExprObj* value) {
    return checked_expression_operation("matrix.natural_logarithm", value,
        [](const auto& matrix) { return lamina::matrix_log(matrix); });
}
/** @brief Computes a symbolic Kronecker product. @param lhs Borrowed matrix. @param rhs Borrowed matrix. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_kronecker(ExprObj* lhs, ExprObj* rhs) {
    std::string error; const auto* a = checked_expr(lhs, error);
    const auto* b = checked_expr(rhs, error);
    if (!a || !b) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    try {
        auto result = lamina::kronecker(*a, *b);
        return result ? result_ok(new ExprObj(std::move(result)), ValueKind::Expr)
                      : result_error(MathErrorCode::InvalidArgument, __func__, "matrix.kronecker: invalid dimensions");
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string("matrix.kronecker: ") + exception.what());
    }
}
/** @brief Applies symbolic Gram-Schmidt. @param vectors Borrowed nested expression arrays. @param normalize Whether to normalize outputs. @return Owning Result nested array or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_gram_schmidt(
    ArrayObj* vectors, bool normalize) {
    std::vector<std::vector<lamina::lsr::ExprPtr>> values;
    std::string error;
    if (!math_internal::nested_expressions(vectors, values, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix.gram_schmidt: " + error);
    try {
        return nested_expr_array_result(lamina::gram_schmidt(values, normalize));
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            std::string("matrix.gram_schmidt: ") + exception.what());
    }
}
/** @brief Classifies a symbolic quadratic-form matrix. @param value Borrowed matrix. @return Owning Result text or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_classify_quadratic_form(ExprObj* value) {
    std::string error; const auto* matrix = checked_expr(value, error);
    if (!matrix) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    try {
        return result_ok(
            new StringObj(lamina::classify_quadratic_form(*matrix)),
            ValueKind::Obj);
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            std::string("matrix.classify_quadratic_form: ") + exception.what());
    }
}
