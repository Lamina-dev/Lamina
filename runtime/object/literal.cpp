#include "literal.hpp"
#include "adt.hpp"

#include <algorithm>
#include <functional>
#include <sstream>

using namespace lmx::runtime;

namespace {
void combine_hash(std::size_t& seed, const std::size_t value) noexcept {
    seed ^= value + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}

bool numeric_value(const Value& value, long double& result) noexcept {
    switch (value.kind) {
    case ValueKind::Int:
        result = value.int_val;
        return true;
    case ValueKind::Real:
        result = value.real_val;
        return true;
    case ValueKind::Fraction:
        result = static_cast<long double>(value.frac_val.num) / value.frac_val.den;
        return true;
    default:
        return false;
    }
}
}

LiteralObj::LiteralObj(const Kind kind, std::vector<Value> elements,
                       const bool lower_closed, const bool upper_closed) noexcept
    : Object(ObjectKind::Literal), kind_(kind), elements_(std::move(elements)),
      lower_closed_(lower_closed), upper_closed_(upper_closed) {
    if (kind_ == Kind::Set) {
        std::vector<Value> unique;
        unique.reserve(elements_.size());
        for (const auto& value : elements_) {
            if (std::find(unique.begin(), unique.end(), value) == unique.end()) unique.push_back(value);
        }
        elements_ = std::move(unique);
    }
}

bool LiteralObj::contains(const Value& value) const noexcept {
    if (kind_ == Kind::Set) {
        return std::find(elements_.begin(), elements_.end(), value) != elements_.end();
    }
    if (kind_ != Kind::Interval || elements_.size() != 2) return false;
    long double candidate{}, lower{}, upper{};
    if (!numeric_value(value, candidate) || !numeric_value(elements_[0], lower) ||
        !numeric_value(elements_[1], upper)) return false;
    const bool above = lower_closed_ ? candidate >= lower : candidate > lower;
    const bool below = upper_closed_ ? candidate <= upper : candidate < upper;
    return above && below;
}

bool LiteralObj::equals(const LiteralObj& other) const noexcept {
    if (kind_ != other.kind_ || lower_closed_ != other.lower_closed_ ||
        upper_closed_ != other.upper_closed_ || elements_.size() != other.elements_.size()) return false;
    if (kind_ != Kind::Set) return elements_ == other.elements_;
    return std::all_of(elements_.begin(), elements_.end(), [&](const Value& value) {
        return other.contains(value);
    });
}

std::size_t LiteralObj::hash() const noexcept {
    std::size_t result = std::hash<unsigned>{}(static_cast<unsigned>(kind_));
    if (kind_ == Kind::Set) {
        std::size_t unordered = 0;
        for (const auto& value : elements_) unordered ^= value.hash();
        combine_hash(result, unordered);
    } else {
        for (const auto& value : elements_) combine_hash(result, value.hash());
    }
    combine_hash(result, lower_closed_);
    combine_hash(result, upper_closed_);
    return result;
}

std::string LiteralObj::to_string() const noexcept {
    std::ostringstream out;
    const char open = kind_ == Kind::Set ? '{' : kind_ == Kind::Tuple ? '(' : (lower_closed_ ? '[' : '(');
    const char close = kind_ == Kind::Set ? '}' : kind_ == Kind::Tuple ? ')' : (upper_closed_ ? ']' : ')');
    out << open;
    for (std::size_t i = 0; i < elements_.size(); ++i) {
        if (i != 0) out << ", ";
        out << elements_[i].to_string();
    }
    out << close;
    return out.str();
}
