//
// Created by meian on 2026/7/30.
//
#include "utils.hpp"
namespace lmx {

const std::filesystem::path module_path = (std::filesystem::current_path() / ".." / "modules").lexically_normal();
std::filesystem::path current_module_path{};
std::shared_ptr<Module> main_module = nullptr;
}