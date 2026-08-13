//
// Created by meian on 2026/8/8.
//

#pragma once
#include "object.hpp"
#include "lsr_expr.hpp"
#include "../error.hpp"

namespace lmx::runtime {

class ExprObj : public Object {
    lamina::lsr::ExprPtr expr;

public:
    explicit ExprObj(lamina::lsr::ExprPtr&& expr) noexcept
        : Object(ObjectKind::Expr), expr(std::move(expr)) {}
    explicit ExprObj(const char* sym) noexcept : Object(ObjectKind::Expr) {
        // todo!
        if (const auto res = lamina::lsr::sym(sym)) expr = nullptr;
        else VM_ERROR(RuntimeErrorType::Construct, "res.error().message");
    }
};

}
