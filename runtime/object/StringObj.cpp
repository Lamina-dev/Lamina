//
// Created by meian on 2026/3/28.
//

#include "StringObj.hpp"
#include <utility>

using namespace lmx::runtime;

StringObj::StringObj() noexcept: Object(ObjectKind::String) { }

StringObj::StringObj(const std::string& data) noexcept: Object(ObjectKind::String), data(data) { }

StringObj::StringObj(const char *data, const size_t size) noexcept: Object(ObjectKind::String), data(data, size) { }

StringObj::StringObj(std::string &&data) noexcept: Object(ObjectKind::String), data(std::move(data)) { }

StringObj::StringObj(const std::string &data, const size_t index) noexcept: Object(ObjectKind::String), data (data, index) { }

StringObj::StringObj(const char *data) noexcept: Object(ObjectKind::String), data (data) { }

StringObj::~StringObj() noexcept = default;


const char *StringObj::c_str() const noexcept {
    return data.c_str();
}

std::string StringObj::to_string() const noexcept {
    return data;
}

std::string StringObj::type_info() const noexcept {
    return "text";
}

bool StringObj::operator!=(const Object &other) const noexcept {
    return !equals(&other);
}

bool StringObj::operator==(const Object &other) const noexcept {
    return equals(&other);
}

bool StringObj::equals(const Object *other) const noexcept {
    if (this->get_kind() != other->get_kind()) return false;
    return this->data == reinterpret_cast<const StringObj*>(other)->data;
}

StringObj &StringObj::operator+=(const StringObj &other) noexcept {
    this->data += other.data;
    return *this;
}

StringObj StringObj::operator+(const StringObj &other) const noexcept {
    return StringObj(std::move(this->data + other.data));
}
