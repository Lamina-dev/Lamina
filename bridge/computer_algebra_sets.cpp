#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_computer_algebra_set_contains(ArrayObj* set, ExprObj* element) {
    std::vector<lamina::lsr::ExprPtr> values;
    std::string error;
    const auto* expression = checked_expr(element, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    if (!array_expressions(set, values, error)) return result_error(MathErrorCode::InvalidArgument, __func__, error);
    const auto checked_set = lamina::lsr::expr_set(std::move(values));
    if (!checked_set) return result_error(checked_set.error());
    const auto result = lamina::lsr::expr_set_contains(checked_set.value(),
                                                       *expression);
    if (!result) return result_error(result.error());
    return result_ok(result.value());
}

/**
 * @brief Converts a Lamina `set<expr>` into an array in engine order.
 *
 * The source set is unordered; the produced array order is an implementation
 * detail of the expression set and must never be relied upon by callers.
 */
extern "C" LM_API ArrayObj* lmx_computer_algebra_set_to_array(
    lmx::runtime::LiteralObj* set) {
    auto* values = new ArrayObj();
    if (!set || set->literal_kind() != lmx::runtime::LiteralObj::Kind::Set) {
        return values;
    }
    for (const auto& element : set->elements()) {
        values->append(element);
    }
    return values;
}

extern "C" LM_API AdtObj* lmx_computer_algebra_set_subset(ArrayObj* lhs, ArrayObj* rhs) {
    std::vector<lamina::lsr::ExprPtr> left_values;
    std::vector<lamina::lsr::ExprPtr> right_values;
    std::string error;
    if (!array_expressions(lhs, left_values, error) ||
        !array_expressions(rhs, right_values, error)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, error);
    }
    const auto left = lamina::lsr::expr_set(std::move(left_values));
    const auto right = lamina::lsr::expr_set(std::move(right_values));
    if (!left) return result_error(left.error());
    if (!right) return result_error(right.error());
    const auto result = lamina::lsr::expr_set_subset(left.value(), right.value());
    if (!result) return result_error(result.error());
    return result_ok(result.value());
}

template <typename Operation>
AdtObj* set_binary_operation(ArrayObj* lhs, ArrayObj* rhs, Operation operation) {
    std::vector<lamina::lsr::ExprPtr> left_values;
    std::vector<lamina::lsr::ExprPtr> right_values;
    std::string error;
    if (!array_expressions(lhs, left_values, error) ||
        !array_expressions(rhs, right_values, error)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, error);
    }
    const auto left = lamina::lsr::expr_set(std::move(left_values));
    const auto right = lamina::lsr::expr_set(std::move(right_values));
    if (!left) return result_error(left.error());
    if (!right) return result_error(right.error());
    const auto combined = operation(left.value(), right.value());
    if (!combined) return result_error(combined.error());
    auto* values = new ArrayObj();
    for (const auto& element : combined.value().elements()) {
        values->append(Value(new ExprObj(element), ValueKind::Expr));
    }
    return result_ok(values, ValueKind::Obj);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_set_union(ArrayObj* lhs, ArrayObj* rhs) {
    return set_binary_operation(lhs, rhs, lamina::lsr::expr_set_union);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_set_intersection(ArrayObj* lhs, ArrayObj* rhs) {
    return set_binary_operation(lhs, rhs, lamina::lsr::expr_set_intersection);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_set_difference(ArrayObj* lhs, ArrayObj* rhs) {
    return set_binary_operation(lhs, rhs, lamina::lsr::expr_set_difference);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_set_symmetric_difference(ArrayObj* lhs,
                                                         ArrayObj* rhs) {
    return set_binary_operation(lhs, rhs,
                          lamina::lsr::expr_set_symmetric_difference);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_domain_contains(const char* domain,
                                                ExprObj* element) {
    const auto checked_domain = number_domain_for_name(domain);
    if (!checked_domain) return result_error(MathErrorCode::InvalidArgument, __func__, "unknown CAS number domain");
    std::string error;
    const auto* expression = checked_expr(element, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::lsr::domain_contains(*checked_domain, *expression);
    if (!result) return result_error(result.error());
    return result_ok(result.value());
}

extern "C" LM_API AdtObj* lmx_computer_algebra_domain_subset(const char* lhs,
                                              const char* rhs) {
    const auto left = number_domain_for_name(lhs);
    const auto right = number_domain_for_name(rhs);
    if (!left || !right) return result_error(MathErrorCode::InvalidArgument, __func__, "unknown CAS number domain");
    const auto result = lamina::lsr::domain_subset(*left, *right);
    if (!result) return result_error(result.error());
    return result_ok(result.value());
}

extern "C" LM_API AdtObj* lmx_computer_algebra_set_subset_domain(ArrayObj* set,
                                                  const char* domain) {
    const auto checked_domain = number_domain_for_name(domain);
    if (!checked_domain) return result_error(MathErrorCode::InvalidArgument, __func__, "unknown CAS number domain");
    std::vector<lamina::lsr::ExprPtr> values;
    std::string error;
    if (!array_expressions(set, values, error)) return result_error(MathErrorCode::InvalidArgument, __func__, error);
    const auto checked_set = lamina::lsr::expr_set(std::move(values));
    if (!checked_set) return result_error(checked_set.error());
    const auto result = lamina::lsr::expr_set_subset_domain(
        checked_set.value(), *checked_domain);
    if (!result) return result_error(result.error());
    return result_ok(result.value());
}
