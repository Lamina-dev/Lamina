//
// Created by meian on 2026/4/4.
//

#pragma once
#include <string>
#include <unordered_map>
#include <iostream>
#include "parser.hpp"

enum class ErrorType {
    Unknown,
    Parse,
    Analysis,
    Generate,
};
extern bool errd;
inline std::unordered_map<ErrorType, std::string> error_type_map = {
    {ErrorType::Unknown, "Unknown"},
    {ErrorType::Parse, "Parse"},
    {ErrorType::Analysis, "Analysis"},
    {ErrorType::Generate, "Generate"}
};
inline constexpr bool LMX_DEBUG =
#ifdef NDEBUG
false;
#else
true;
#endif
#define throw_error(error_type, message, line, col) do {\
    errd = true;\
if constexpr (LMX_DEBUG) {\
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "]";\
}\
    std::cerr << error_type_map[error_type] << "Error at \"" << lmx::cur_module->name << "\" " <<  line << ", " << col << ":\n\t" << message << std::endl;\
    } while(0);
