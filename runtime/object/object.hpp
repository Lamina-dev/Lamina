#pragma once
#include <cstdint>
#include <string>

namespace lmx::runtime {
class CodeModule;

namespace ObjectKind {
enum  {
    Object,
    Code  ,
    String,
    Table ,
    Vector,
    Matrix,
    Array ,
};
}
class Object {
    uint32_t kind { ObjectKind::Object };
    uint32_t rc { 1 };
public:
    explicit Object(uint32_t kind) noexcept;
    virtual ~Object() noexcept;

    [[nodiscard]] virtual Object*       clone       () const noexcept = 0;
    [[nodiscard]] virtual std::string   to_string   () const noexcept = 0;
    [[nodiscard]] virtual std::string   type_info   () const noexcept = 0;
    [[nodiscard]] uint32_t              get_kind    () const noexcept;
    [[nodiscard]] virtual bool          equals(const Object* other) const noexcept = 0;

    [[nodiscard]] virtual bool operator==(const Object& other) const noexcept = 0;
    [[nodiscard]] virtual bool operator!=(const Object& other) const noexcept = 0;

    [[nodiscard]] Object*       get() noexcept;
    void release() noexcept;
};
}
