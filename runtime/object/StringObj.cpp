
#include "StringObj.hpp"

#include <functional>
#include <utility>

using namespace lmx::runtime;

StringObj::StringObj(): Object(ObjectKind::String) { }

StringObj::StringObj(const StringObjImpl& data): Object(ObjectKind::String), data(data) { }

StringObj::StringObj(const char *data, const size_t size): Object(ObjectKind::String), data(data, size) { }

StringObj::StringObj(StringObjImpl &&data) noexcept: Object(ObjectKind::String), data(std::move(data)) { }

StringObj::StringObj(const StringObjImpl &data, const size_t index): Object(ObjectKind::String), data (data, index) { }

StringObj::StringObj(const char *data): Object(ObjectKind::String), data (data) { }

StringObj::~StringObj() noexcept = default;


const char *StringObj::c_str() const noexcept {
    return data.c_str();
}

std::string StringObj::to_string() const {
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

std::size_t StringObj::hash() const noexcept {
    return std::hash<std::string>{}(data);
}

StringObj &StringObj::operator+=(const StringObj &other) {
    this->data += other.data;
    return *this;
}

StringObj StringObj::operator+(const StringObj &other) const {
    return StringObj(std::move(this->data + other.data));
}
