//
// Created by meian on 2026/8/8.
//

#pragma once
#include "object.hpp"
#include "lsr_expr.hpp"

#include <string>
#include <utility>

namespace lmx::runtime {

class ExprObj : public Object {
    lamina::lsr::ExprPtr expr_;
    std::string error_;

public:
    explicit ExprObj(lamina::lsr::ExprPtr expr) noexcept
        : Object(ObjectKind::Expr), expr_(std::move(expr)) {}

    explicit ExprObj(std::string error) noexcept
        : Object(ObjectKind::Expr), error_(std::move(error)) {}

    [[nodiscard]] bool ok() const noexcept {
        return static_cast<bool>(expr_);
    }

    [[nodiscard]] const lamina::lsr::ExprPtr& expr() const noexcept {
        return expr_;
    }

    [[nodiscard]] const std::string& error() const noexcept {
        return error_;
    }

    [[nodiscard]] std::string to_string() const noexcept {
        if (!ok()) return error_;
        return expr_->to_string();
    }
};

}
