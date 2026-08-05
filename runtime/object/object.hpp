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

    ~Object() noexcept;

    [[nodiscard]] uint32_t get_kind() const noexcept;

    // virtual ~Object() noexcept;

    [[nodiscard]] Object*       get() noexcept;
    void release() noexcept;


    [[nodiscard]] static std::string to_string(Object* obj) noexcept;
};
}
