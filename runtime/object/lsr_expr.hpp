//
// Created by meian on 2026/8/8.
//

#pragma once
#include "object.hpp"
#include "lsr_expr.hpp"

namespace lmx::runtime {

class ExprObj : public Object {
    lamina::lsr::Expr expr;

public:
    explicit ExprObj(lamina::lsr::Expr&& expr) noexcept
        : Object(ObjectKind::Expr), expr(std::move(expr)) {}

};

}
