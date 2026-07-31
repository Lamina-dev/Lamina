//
// Created by meian on 2026/7/23.
//

#pragma once

#include "lmx.h"
#include <filesystem>
#include <vector>
#include <optional>

namespace lmx {
struct Module;


/*
 * current_path() = {toolchain_root}/bin
 * module_lath    = {toolchain_root}/modules
 */
LM_API extern const std::filesystem::path module_path;
LM_API extern std::filesystem::path current_module_path;
LM_API extern std::shared_ptr<Module> main_module;

inline constexpr auto file_suffix = ".lm";
inline constexpr auto file_suffix_binary = ".lmc";
inline constexpr auto file_default_mod = "module";

inline constexpr auto module_cache_fold = "_lm_cache";

LMX_INLINE std::optional<std::filesystem::path> find_module_path(std::filesystem::path real_path) {
    namespace fs = std::filesystem;
    if (fs::path named = (real_path.string() + file_suffix);
        fs::is_regular_file(named)) {
        return named;
    }
    if (fs::is_directory(real_path)) {
        real_path /= (std::string("module") + file_suffix);
        if (fs::is_regular_file(real_path)) return real_path;
    } else {
        return std::nullopt;
    }
    return std::nullopt;
}


/*
 * find_module_name
 *
 *
 *
 * Args:
 *     name  (const std::string&): 模块名或路径
 * Return:
 *     pair<path, path>: first是半路径， second是绝对路径
 *
 * Notes:
 *     path.to.name ---> {prefix}/path/to/name
 */
LMX_INLINE std::optional<std::pair<std::filesystem::path, std::filesystem::path>> find_module_name(const std::string& name) {
    using RetTy = std::pair<std::filesystem::path, std::filesystem::path>;
    namespace fs = std::filesystem;
    std::string real_name;
    for (const auto c: name) {
        if (c == '.') {
            real_name += '/';
        } else {
            real_name += c;
        }
    }
    if (auto result = find_module_path(current_module_path / real_name);
        result.has_value()
        ) {
        return std::make_pair(fs::path(real_name), *result);
    }

    if (auto result = find_module_path(module_path / real_name); result.has_value()
        ) {
        return std::make_pair(fs::path(real_name), *result);
    }
    return std::nullopt;
}

std::vector<uint8_t> ast_to_binary(const std::shared_ptr<Module>& mod) noexcept;

}
