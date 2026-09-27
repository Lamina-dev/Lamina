#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
using namespace lmx::bridge;

namespace {
AdtObj* equality_required() {
    return result_error(MathErrorCode::InvalidArgument, "LMCAS.solve_expr_set",
                        "solve requires an equality relation");
}
}

extern "C" LM_API AdtObj* lmx_cas_solve_by_name(ExprObj* equation,
                                      const char* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(equation, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    if (LMCAS::relation_op(*value) != LMCAS::RelationOp::EQ) return equality_required();
    return expression_set_literal_result(
        LMCAS::solve(*value, variable ? variable : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_roots_by_name(ExprObj* expression,
                                      const char* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expression_set_literal_result(
        LMCAS::roots(*value, variable ? variable : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

// equation 为等式，variable 为单个符号。确定的解集（含空集）通过 Result.Ok 返回，
// 求解失败通过 Result.Err(MathError) 返回。输入在当前 VM 线程内借用，
// 返回的 ADT 及其集合由调用方持有。
extern "C" LM_API AdtObj* lmx_cas_solve_by_symbol(ExprObj* equation, ExprObj* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string name;
    std::string error;
    if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* value = checked_expr(equation, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    if (LMCAS::relation_op(*value) != LMCAS::RelationOp::EQ) return equality_required();
    return expression_set_literal_result(LMCAS::solve(*value, name));
} catch (...) {
    return c_abi_current_exception(__func__);
}

// 将 expression 视为零点方程，variable 为单个符号。确定的根集（含空集）
// 通过 Result.Ok 返回，求根失败通过 Result.Err(MathError) 返回。
// 输入在当前 VM 线程内借用，返回的 ADT 及其集合由调用方持有。
extern "C" LM_API AdtObj* lmx_cas_roots_by_symbol(ExprObj* expression, ExprObj* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string name;
    std::string error;
    if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* value = checked_expr(expression, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expression_set_literal_result(LMCAS::roots(*value, name));
} catch (...) {
    return c_abi_current_exception(__func__);
}
