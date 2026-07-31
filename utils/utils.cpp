//
// Created by meian on 2026/7/30.
//
#include "utils.hpp"

#include "../compiler/assembler.hpp"
#include "../compiler/mir.hpp"
#include "../compiler/mir_builder.hpp"

namespace lmx {

const std::filesystem::path module_path = (std::filesystem::current_path() / ".." / "modules").lexically_normal();
std::filesystem::path current_module_path{};
std::shared_ptr<Module> main_module = nullptr;


std::vector<uint8_t> ast_to_binary(const std::shared_ptr<Module>& mod) noexcept {
    mir::MirModule mir = mir::MirBuilder::from_ast_module(mod);
    return Assembler().asm_module(&mir);
}
}
