#include "unit.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <sstream>

namespace lmx {
namespace {

bool checked_multiply(const std::int64_t lhs, const std::int64_t rhs,
                      std::int64_t& result) noexcept {
    if (lhs == 0 || rhs == 0) {
        result = 0;
        return true;
    }
    if (lhs == -1 && rhs == std::numeric_limits<std::int64_t>::min()) return false;
    if (rhs == -1 && lhs == std::numeric_limits<std::int64_t>::min()) return false;
    const auto magnitude_lhs = lhs < 0 ? static_cast<std::uint64_t>(-(lhs + 1)) + 1
                                       : static_cast<std::uint64_t>(lhs);
    const auto magnitude_rhs = rhs < 0 ? static_cast<std::uint64_t>(-(rhs + 1)) + 1
                                       : static_cast<std::uint64_t>(rhs);
    const auto limit = (lhs < 0) != (rhs < 0)
        ? static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1
        : static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (magnitude_lhs > limit / magnitude_rhs) return false;
    result = lhs * rhs;
    return true;
}

void add_dimension(DimensionSignature& target, const DimensionSignature& source,
                   const int multiplier) {
    for (const auto& [name, exponent] : source.exponents) {
        const auto value = target.exponents[name] + exponent * multiplier;
        if (value == 0) target.exponents.erase(name);
        else target.exponents[name] = value;
    }
}

UnitDefinition base_unit(const std::string& dimension, const std::string& display,
                         const RationalScale scale = {}) {
    return UnitDefinition{{{{dimension, 1}}}, scale, display};
}

} // namespace

std::string UnitSpec::to_string() const {
    if (factors.empty()) return "1";
    std::ostringstream output;
    bool first = true;
    for (const auto& factor : factors) {
        if (!first) output << '*';
        output << factor.name;
        if (factor.exponent != 1) output << '^' << factor.exponent;
        first = false;
    }
    return output.str();
}

RationalScale::RationalScale(std::int64_t numerator_value,
                             std::int64_t denominator_value) noexcept
    : numerator(numerator_value), denominator(denominator_value) {
    if (denominator == 0) return;
    if (denominator < 0) {
        numerator = -numerator;
        denominator = -denominator;
    }
    const auto divisor = std::gcd(numerator, denominator);
    if (divisor != 0) {
        numerator /= divisor;
        denominator /= divisor;
    }
}

std::optional<RationalScale> RationalScale::from_decimal(
    const std::string& text) noexcept {
    bool negative = false;
    bool decimal = false;
    bool digit = false;
    std::int64_t numerator = 0;
    std::int64_t denominator = 1;
    for (const char character : text) {
        if (!digit && !decimal && (character == '+' || character == '-')) {
            negative = character == '-';
            continue;
        }
        if (character >= '0' && character <= '9') {
            digit = true;
            std::int64_t next = 0;
            if (!checked_multiply(numerator, 10, next) ||
                next > std::numeric_limits<std::int64_t>::max() - (character - '0')) {
                return std::nullopt;
            }
            numerator = next + (character - '0');
            if (decimal) {
                if (!checked_multiply(denominator, 10, next)) return std::nullopt;
                denominator = next;
            }
            continue;
        }
        if (character == '.' && !decimal) {
            decimal = true;
            continue;
        }
        return std::nullopt;
    }
    if (!digit) return std::nullopt;
    return RationalScale(negative ? -numerator : numerator, denominator);
}

std::optional<RationalScale> RationalScale::multiplied_by(
    const RationalScale& other) const noexcept {
    if (denominator == 0 || other.denominator == 0) return std::nullopt;
    auto left_numerator = numerator;
    auto left_denominator = denominator;
    auto right_numerator = other.numerator;
    auto right_denominator = other.denominator;
    const auto cross_left = std::gcd(left_numerator, right_denominator);
    left_numerator /= cross_left;
    right_denominator /= cross_left;
    const auto cross_right = std::gcd(right_numerator, left_denominator);
    right_numerator /= cross_right;
    left_denominator /= cross_right;
    std::int64_t result_numerator = 0;
    std::int64_t result_denominator = 0;
    if (!checked_multiply(left_numerator, right_numerator, result_numerator) ||
        !checked_multiply(left_denominator, right_denominator, result_denominator)) {
        return std::nullopt;
    }
    return RationalScale(result_numerator, result_denominator);
}

std::optional<RationalScale> RationalScale::added_to(
    const RationalScale& other) const noexcept {
    if (denominator == 0 || other.denominator == 0) return std::nullopt;
    const auto divisor = std::gcd(denominator, other.denominator);
    const auto left_factor = other.denominator / divisor;
    const auto right_factor = denominator / divisor;
    std::int64_t left = 0;
    std::int64_t right = 0;
    std::int64_t result_denominator = 0;
    if (!checked_multiply(numerator, left_factor, left) ||
        !checked_multiply(other.numerator, right_factor, right) ||
        !checked_multiply(denominator, left_factor, result_denominator)) {
        return std::nullopt;
    }
    if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
        return std::nullopt;
    }
    return RationalScale(left + right, result_denominator);
}

std::optional<RationalScale> RationalScale::subtracted_by(
    const RationalScale& other) const noexcept {
    if (other.numerator == std::numeric_limits<std::int64_t>::min()) return std::nullopt;
    return added_to(RationalScale(-other.numerator, other.denominator));
}

std::optional<RationalScale> RationalScale::divided_by(
    const RationalScale& other) const noexcept {
    if (other.numerator == 0) return std::nullopt;
    return multiplied_by(RationalScale(other.denominator, other.numerator));
}

std::optional<RationalScale> RationalScale::raised_to(const int exponent) const noexcept {
    if (exponent == 0) return RationalScale{};
    if (exponent < 0 && numerator == 0) return std::nullopt;
    RationalScale result;
    RationalScale factor = exponent < 0
        ? RationalScale(denominator, numerator) : *this;
    auto remaining = static_cast<unsigned int>(exponent < 0 ? -exponent : exponent);
    while (remaining != 0) {
        if ((remaining & 1U) != 0) {
            auto next = result.multiplied_by(factor);
            if (!next) return std::nullopt;
            result = *next;
        }
        remaining >>= 1U;
        if (remaining != 0) {
            auto next = factor.multiplied_by(factor);
            if (!next) return std::nullopt;
            factor = *next;
        }
    }
    return result;
}

std::string RationalScale::to_string() const {
    if (denominator == 1) return std::to_string(numerator);
    return std::to_string(numerator) + "/" + std::to_string(denominator);
}

DimensionSignature DimensionSignature::multiplied_by(
    const DimensionSignature& other) const {
    auto result = *this;
    add_dimension(result, other, 1);
    return result;
}

DimensionSignature DimensionSignature::divided_by(
    const DimensionSignature& other) const {
    auto result = *this;
    add_dimension(result, other, -1);
    return result;
}

DimensionSignature DimensionSignature::raised_to(const int exponent) const {
    DimensionSignature result;
    for (const auto& [name, value] : exponents) {
        if (value * exponent != 0) result.exponents[name] = value * exponent;
    }
    return result;
}

std::string DimensionSignature::to_string() const {
    if (exponents.empty()) return "1";
    std::ostringstream output;
    bool first = true;
    for (const auto& [name, exponent] : exponents) {
        if (!first) output << '*';
        output << name;
        if (exponent != 1) output << '^' << exponent;
        first = false;
    }
    return output.str();
}

UnitSystem::UnitSystem() {
    units_.emplace("1", UnitDefinition{});
    units_.emplace("m", base_unit("m", "m"));
    units_.emplace("km", base_unit("m", "km", RationalScale(1000)));
    units_.emplace("cm", base_unit("m", "cm", RationalScale(1, 100)));
    units_.emplace("mm", base_unit("m", "mm", RationalScale(1, 1000)));
    units_.emplace("s", base_unit("s", "s"));
    units_.emplace("min", base_unit("s", "min", RationalScale(60)));
    units_.emplace("h", base_unit("s", "h", RationalScale(3600)));
    units_.emplace("kg", base_unit("kg", "kg"));
    units_.emplace("g", base_unit("kg", "g", RationalScale(1, 1000)));
    units_.emplace("A", base_unit("A", "A"));
    units_.emplace("K", base_unit("K", "K"));
    units_.emplace("mol", base_unit("mol", "mol"));
    units_.emplace("cd", base_unit("cd", "cd"));

    const auto length = units_.at("m").dimension;
    const auto mass = units_.at("kg").dimension;
    const auto time = units_.at("s").dimension;
    const auto current = units_.at("A").dimension;
    units_.emplace("N", UnitDefinition{
        length.multiplied_by(mass).divided_by(time.raised_to(2)), {}, "N"});
    units_.emplace("Pa", UnitDefinition{
        mass.divided_by(length).divided_by(time.raised_to(2)), {}, "Pa"});
    units_.emplace("J", UnitDefinition{
        mass.multiplied_by(length.raised_to(2)).divided_by(time.raised_to(2)), {}, "J"});
    units_.emplace("C", UnitDefinition{current.multiplied_by(time), {}, "C"});
    units_.emplace("F", UnitDefinition{
        time.raised_to(4).multiplied_by(current.raised_to(2))
            .divided_by(mass).divided_by(length.raised_to(2)), {}, "F"});
    units_.emplace("H", UnitDefinition{
        mass.multiplied_by(length.raised_to(2))
            .divided_by(time.raised_to(2)).divided_by(current.raised_to(2)), {}, "H"});
}

bool UnitSystem::contains(const std::string& name) const noexcept {
    return units_.contains(name);
}

bool UnitSystem::declare_base(const std::string& name,
                              const std::string& dimension_name) {
    if (name.empty() || contains(name)) return false;
    const auto inserted = units_.emplace(name, base_unit(dimension_name, name)).second;
    if (inserted) explicit_declarations_.insert(name);
    return inserted;
}

bool UnitSystem::declare_derived(const std::string& name,
                                 UnitDefinition definition) {
    if (name.empty() || explicit_declarations_.contains(name) ||
        definition.scale_to_base.numerator <= 0) return false;
    definition.display_unit = name;
    if (const auto existing = units_.find(name); existing != units_.end()) {
        if (existing->second.dimension != definition.dimension ||
            existing->second.scale_to_base != definition.scale_to_base) return false;
        explicit_declarations_.insert(name);
        return true;
    }
    const auto inserted = units_.emplace(name, std::move(definition)).second;
    if (inserted) explicit_declarations_.insert(name);
    return inserted;
}

std::optional<UnitDefinition> UnitSystem::resolve(const std::string& name) const noexcept {
    const auto found = units_.find(name);
    if (found == units_.end()) return std::nullopt;
    return found->second;
}

std::optional<UnitDefinition> UnitSystem::resolve(const UnitSpec& spec) const noexcept {
    if (spec.factors.empty()) return UnitDefinition{};
    UnitDefinition result;
    result.display_unit = spec.to_string();
    for (const auto& factor : spec.factors) {
        const auto definition = resolve(factor.name);
        if (!definition || factor.exponent < -32 || factor.exponent > 32) return std::nullopt;
        result.dimension = result.dimension.multiplied_by(
            definition->dimension.raised_to(factor.exponent));
        const auto scale = definition->scale_to_base.raised_to(factor.exponent);
        if (!scale) return std::nullopt;
        const auto combined = result.scale_to_base.multiplied_by(*scale);
        if (!combined) return std::nullopt;
        result.scale_to_base = *combined;
    }
    return result;
}

void UnitSystem::import_unit(const std::string& qualified_name,
                             UnitDefinition definition) {
    definition.display_unit = qualified_name;
    units_.emplace(qualified_name, std::move(definition));
}

} // namespace lmx
