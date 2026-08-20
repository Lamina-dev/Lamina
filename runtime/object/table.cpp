#include "table.hpp"

#include <algorithm>
#include <functional>
#include <sstream>

namespace lmx::runtime {

namespace {
void combine_hash(std::size_t& seed, const std::size_t value) noexcept {
    seed ^= value + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}
}

TableObj::TableObj(std::vector<Entry> entries) noexcept
    : Object(ObjectKind::Table), entries_(std::move(entries)) {
    std::stable_sort(entries_.begin(), entries_.end(),
        [](const Entry& lhs, const Entry& rhs) { return lhs.first < rhs.first; });
    auto output = entries_.begin();
    for (auto input = entries_.begin(); input != entries_.end(); ++input) {
        if (output != entries_.begin() && std::prev(output)->first == input->first) {
            std::prev(output)->second = std::move(input->second);
        } else {
            if (output != input) *output = std::move(*input);
            ++output;
        }
    }
    entries_.erase(output, entries_.end());
}

const Value* TableObj::find(const std::string& key) const noexcept {
    const auto it = std::lower_bound(entries_.begin(), entries_.end(), key,
        [](const Entry& entry, const std::string& name) { return entry.first < name; });
    return it != entries_.end() && it->first == key ? &it->second : nullptr;
}

bool TableObj::equals(const TableObj& other) const noexcept {
    return entries_ == other.entries_;
}

std::size_t TableObj::hash() const noexcept {
    std::size_t result = std::hash<std::size_t>{}(entries_.size());
    for (const auto& [key, value] : entries_) {
        combine_hash(result, std::hash<std::string>{}(key));
        combine_hash(result, value.hash());
    }
    return result;
}

std::string TableObj::to_string() const noexcept {
    std::ostringstream out;
    out << '{';
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (i != 0) out << ", ";
        out << entries_[i].first << ": " << entries_[i].second.to_string();
    }
    out << '}';
    return out.str();
}

}
