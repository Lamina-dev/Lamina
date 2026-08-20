#pragma once

#include "object.hpp"
#include "value.hpp"

#include <string>
#include <utility>
#include <vector>

namespace lmx::runtime {

class TableObj final : public Object {
public:
    using Entry = std::pair<std::string, Value>;

private:
    std::vector<Entry> entries_;

public:
    explicit TableObj(std::vector<Entry> entries) noexcept;

    [[nodiscard]] const std::vector<Entry>& entries() const noexcept { return entries_; }
    [[nodiscard]] const Value* find(const std::string& key) const noexcept;
    [[nodiscard]] bool equals(const TableObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
};

}
