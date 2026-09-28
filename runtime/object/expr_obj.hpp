
#pragma once
#include "object.hpp"
#include "expr.hpp"

#include <optional>
#include <string>
#include <functional>
#include <utility>

namespace lmx::runtime {

class ExprObj : public Object {
    LMCAS::ExprPtr expr_;
    std::optional<LMCAS::CasError> error_;
public:
    explicit ExprObj(LMCAS::ExprPtr expr) noexcept
        : Object(ObjectKind::Expr), expr_(std::move(expr)) {}

    explicit ExprObj(LMCAS::CasError error) noexcept
        : Object(ObjectKind::Expr), error_(std::move(error)) {}

    [[nodiscard]] bool ok() const noexcept {
        return static_cast<bool>(expr_);
    }

    [[nodiscard]] const LMCAS::ExprPtr& expr() const noexcept {
        return expr_;
    }

    [[nodiscard]] const LMCAS::CasError& error() const noexcept {
        return *error_;
    }

    [[nodiscard]] std::string to_string() const noexcept {
        if (!ok()) return std::string(LMCAS::error_name(*error_)) + ": " +
                          error_->operation + ": " + error_->message;
        return expr_->to_string();
    }
    [[nodiscard]] bool equals(const ExprObj& other) const noexcept {
        if (ok() != other.ok()) return false;
        if (!ok()) return error_->code == other.error_->code &&
                          error_->operation == other.error_->operation &&
                          error_->message == other.error_->message;
        return expr_->compare(other.expr_) == 0;
    }

    [[nodiscard]] std::size_t hash() const noexcept {
        return std::hash<std::string>{}(to_string());
    }
};

}
