#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lmx {

struct UnitFactor {
    std::string name;
    int exponent{1};
};

struct UnitSpec {
    std::vector<UnitFactor> factors;

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] bool empty() const noexcept { return factors.empty(); }
};

struct RationalScale {
    std::int64_t numerator{1};
    std::int64_t denominator{1};

    RationalScale() = default;
    RationalScale(std::int64_t numerator, std::int64_t denominator = 1) noexcept;

    [[nodiscard]] static std::optional<RationalScale> from_decimal(const std::string& text) noexcept;
    [[nodiscard]] std::optional<RationalScale> added_to(const RationalScale& other) const noexcept;
    [[nodiscard]] std::optional<RationalScale> subtracted_by(const RationalScale& other) const noexcept;
    [[nodiscard]] std::optional<RationalScale> multiplied_by(const RationalScale& other) const noexcept;
    [[nodiscard]] std::optional<RationalScale> divided_by(const RationalScale& other) const noexcept;
    [[nodiscard]] std::optional<RationalScale> raised_to(int exponent) const noexcept;
    [[nodiscard]] std::string to_string() const;

    bool operator==(const RationalScale&) const noexcept = default;
};

struct DimensionSignature {
    std::map<std::string, int> exponents;

    [[nodiscard]] bool is_dimensionless() const noexcept { return exponents.empty(); }
    [[nodiscard]] DimensionSignature multiplied_by(const DimensionSignature& other) const;
    [[nodiscard]] DimensionSignature divided_by(const DimensionSignature& other) const;
    [[nodiscard]] DimensionSignature raised_to(int exponent) const;
    [[nodiscard]] std::string to_string() const;

    bool operator==(const DimensionSignature&) const noexcept = default;
};

struct UnitDefinition {
    DimensionSignature dimension;
    RationalScale scale_to_base;
    std::string display_unit{"1"};

    bool operator==(const UnitDefinition&) const noexcept = default;
};

class UnitSystem {
    std::unordered_map<std::string, UnitDefinition> units_;
    std::unordered_set<std::string> explicit_declarations_;

public:
    UnitSystem();

    [[nodiscard]] bool contains(const std::string& name) const noexcept;
    [[nodiscard]] bool declare_base(const std::string& name,
                                    const std::string& dimension_name);
    [[nodiscard]] bool declare_derived(const std::string& name,
                                       UnitDefinition definition);
    [[nodiscard]] std::optional<UnitDefinition> resolve(const UnitSpec& spec) const noexcept;
    [[nodiscard]] std::optional<UnitDefinition> resolve(const std::string& name) const noexcept;
    void import_unit(const std::string& qualified_name, UnitDefinition definition);
};

} // namespace lmx
