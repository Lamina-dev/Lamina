#pragma once
#include <cstdint>
#include <string>

namespace lmx::runtime {
class CodeModuleObj;

namespace ObjectKind {
enum {
    Object,
    Code  ,
    String,
    Table ,
    Vector,
    Matrix,
    Array ,
    Expr,
    Tuple,
    Adt,
    Literal,
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

    [[nodiscard]] uint32_t get_rc() const noexcept {
        return rc;
    }
    [[nodiscard]] static std::string to_string(Object* obj) noexcept;
};
}
