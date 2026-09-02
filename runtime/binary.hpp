
#pragma once
#include <cstdint>
#include <vector>
namespace lmx::runtime {

enum class ConstantId : uint8_t {
    Int, Frac, Str, Arr, AdtConstructor
};
struct ConstantPoolInfo;
#pragma pack(push, 1)
struct FracInfo {
    int32_t num;
    int32_t den;
};
struct StringInfo {
    uint32_t length;

    [[nodiscard]] const char* data() const noexcept {
        return reinterpret_cast<const char*>(this) + sizeof(*this);
    }
};
struct AdtConstructorInfo {
    uint16_t type_name_length;
    uint16_t constructor_length;
    uint8_t field_count;

    [[nodiscard]] const char* data() const noexcept {
        return reinterpret_cast<const char*>(this) + sizeof(*this);
    }
};
enum class TypeTag : uint16_t {
    Func,
};
struct TypeInfo {

};
struct ArrayInfo;
struct ConstantPoolInfo {
    ConstantId id;
    union {
        const int64_t int_value;
        const FracInfo* frac_info;
        const StringInfo* str;
        const ArrayInfo* arr;
        const AdtConstructorInfo* adt_constructor;
    };

    explicit ConstantPoolInfo(decltype(int_value) int_value) noexcept;
    explicit ConstantPoolInfo(decltype(frac_info) frac_info) noexcept;
    explicit ConstantPoolInfo(decltype(str) str) noexcept;
    explicit ConstantPoolInfo(decltype(arr) arr) noexcept;
    explicit ConstantPoolInfo(decltype(adt_constructor) adt_constructor) noexcept;

};

struct ArrayInfo {
    uint32_t len;

    [[nodiscard]] ConstantPoolInfo* data() noexcept {
        return reinterpret_cast<ConstantPoolInfo*>(
            reinterpret_cast<char*>(this) + sizeof(*this));
    }
    [[nodiscard]] const ConstantPoolInfo* data() const noexcept {
        return reinterpret_cast<const ConstantPoolInfo*>(
            reinterpret_cast<const char*>(this) + sizeof(*this));
    }
};

#pragma pack(pop)
}
