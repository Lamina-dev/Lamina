//
// Created by meian on 2026/7/23.
//

#pragma once

#include "lmx.h"
#include <filesystem>
#include <memory>

namespace lmx {
struct Module;


/*
 * current_path() = {toolchain_root}/bin
 * module_lath    = {toolchain_root}/modules
 */
LM_API extern const std::filesystem::path module_path;
LM_API extern std::filesystem::path current_module_path;
LM_API extern std::filesystem::path toolchain_module_path;
LM_API extern std::shared_ptr<Module> main_module;

inline constexpr auto file_suffix = ".lm";
inline constexpr auto file_suffix_binary = ".lmc";
inline constexpr auto file_default_mod = "module";

inline constexpr auto module_cache_fold = "_lm_cache";

std::filesystem::path get_exe_dir() noexcept;
}
