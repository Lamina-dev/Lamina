//
// Created by meian on 2026/3/28.
//

#include "string.hpp"
#include <utility>

using namespace lmx::runtime;

String::String() noexcept: Object(ObjectKind::String) { }

String::String(const std::string& data) noexcept: Object(ObjectKind::String), data(data) { }

String::String(const char *data, const size_t size) noexcept: Object(ObjectKind::String), data(data, size) { }

String::String(std::string &&data) noexcept: Object(ObjectKind::String), data(std::move(data)) { }

String::String(const std::string &data, const size_t index) noexcept: Object(ObjectKind::String), data (data, index) { }

String::String(const char *data) noexcept: Object(ObjectKind::String), data (data) { }

String::~String() noexcept = default;

Object *String::clone() const noexcept {
    return new String(data);
}

const char *String::c_str() const noexcept {
    return data.c_str();
}

std::string String::to_string() const noexcept {
    return data;
}

std::string String::type_info() const noexcept {
    return "text";
}

bool String::operator!=(const Object &other) const noexcept {
    return !equals(&other);
}

bool String::operator==(const Object &other) const noexcept {
    return equals(&other);
}

bool String::equals(const Object *other) const noexcept {
    if (this->get_kind() != other->get_kind()) return false;
    return this->data == reinterpret_cast<const String*>(other)->data;
}

String &String::operator+=(const String &other) noexcept {
    this->data += other.data;
    return *this;
}

String String::operator+(const String &other) const noexcept {
    return String(std::move(this->data + other.data));
}
