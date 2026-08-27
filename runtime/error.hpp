#pragma once
#include <stdexcept>
#include <string>
#include "lmx.h"

namespace lmx::runtime {

enum class RuntimeErrorType {
    ModuleLoad,
    CanNotCalling,
    IndexOutOfRange,
    Construct,
    Runtime,
};
constexpr const char* error_str[] = {
    "ModuleLoaderError", "CanNotCalling", "IndexOutOfRange", "Construct",
    "Runtime"
};

class VmFault final : public std::runtime_error {
    RuntimeErrorType type_;
    std::string text_;

public:
    VmFault(const RuntimeErrorType type, std::string text)
        : std::runtime_error(std::string(error_str[static_cast<std::size_t>(type)]) +
                             "Error: " + text),
          type_(type),
          text_(std::move(text)) {}

    [[nodiscard]] RuntimeErrorType type() const noexcept { return type_; }
    [[nodiscard]] const std::string& text() const noexcept { return text_; }
};

[[noreturn]] LMX_INLINE void VM_ERROR(const RuntimeErrorType type,
                                      const std::string& message) {
    throw VmFault(type, message);
}

}
