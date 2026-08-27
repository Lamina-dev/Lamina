#pragma once

#include "object.hpp"
#include "value.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace lmx::runtime {

class LiteralObj : public Object {
public:
    enum class Kind : uint8_t {
        Tuple,
        Set,
        Interval,
    };

    LiteralObj(Kind kind, std::vector<Value> elements,
               bool lower_closed = false, bool upper_closed = false) noexcept;

    [[nodiscard]] Kind literal_kind() const noexcept { return kind_; }
    [[nodiscard]] const std::vector<Value>& elements() const noexcept { return elements_; }
    [[nodiscard]] bool lower_closed() const noexcept { return lower_closed_; }
    [[nodiscard]] bool upper_closed() const noexcept { return upper_closed_; }
    [[nodiscard]] bool contains(const Value& value) const noexcept;
    [[nodiscard]] bool equals(const LiteralObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
    [[nodiscard]] std::vector<Value> union_elements(const LiteralObj& other) const;
    [[nodiscard]] std::vector<Value> intersection_elements(const LiteralObj& other) const;
    [[nodiscard]] std::vector<Value> difference_elements(const LiteralObj& other) const;
    [[nodiscard]] std::vector<Value> symmetric_difference_elements(const LiteralObj& other) const;
    [[nodiscard]] bool subset_of(const LiteralObj& other) const noexcept;

private:
    Kind kind_;
    std::vector<Value> elements_;
    bool lower_closed_;
    bool upper_closed_;
};

}
