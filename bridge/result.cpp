#include "bridge/result.hpp"

#include <utility>

namespace lmx::bridge {

[[noreturn]] ExprObj* expression_internal_error(std::string message) {
    std::cerr << "liblamina internal expression error: " << message << '\n';
    std::terminate();
}

ExprObj* expr_from_result(const lamina::lsr::ExprResult& result) {
    if (!result) {
        return expression_internal_error(
            result.error().operation + ": " + result.error().message);
    }
    return new ExprObj(result.value());
}

AdtObj* expr_result_ok(const lamina::lsr::ExprResult& result) {
    if (!result) return result_error(result.error());
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
}

AdtObj* expr_pointer_result(lamina::lsr::ExprPtr value,
                                const char* operation) {
    if (!value) {
        return result_error(MathErrorCode::UnsupportedExpression,
                            operation ? operation : "computer_algebra",
                            "operation produced no expression");
    }
    return result_ok(new ExprObj(std::move(value)), ValueKind::Expr);
}

AdtObj* expression_set_literal_result(const lamina::lsr::ExprSetResult& result) {
    if (!result) return result_error(result.error());
    std::vector<Value> values;
    values.reserve(result.value().size());
    for (const auto& expression : result.value().elements()) {
        values.emplace_back(take_object_value(
            make_owned_object<ExprObj>(expression), ValueKind::Expr));
    }
    return result_ok(
        new lmx::runtime::LiteralObj(
            lmx::runtime::LiteralObj::Kind::Set, std::move(values)),
        ValueKind::Set);
}

AdtObj* transform_engine_result_value(const lamina::TransformEngineResult& result) {
    if (!result) return result_error(result.error());
    if (!result.value().value.expression) {
        return result_error(MathErrorCode::UnsupportedExpression, "transform",
                            "transform produced no expression");
    }
    return result_ok(
        new ExprObj(result.value().value.expression), ValueKind::Expr);
}

} // namespace lmx::bridge
