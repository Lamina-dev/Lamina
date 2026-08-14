#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ast/ast.hpp"
#include "hir/type_checker.hpp"
#include "mir/mir.hpp"

namespace lmx {

enum class CompileStage {
    Semantic,
    Mir,
    Binary,
};

struct CompileResult {
    std::shared_ptr<Module> module;
    std::vector<hir::Scope::Var> exports;
    std::optional<mir::MirModule> mir;
    std::vector<uint8_t> binary;
};

// Owns one complete compilation session, including recursive module compilation.
class Compiler final : public hir::ModuleResolver {
    std::optional<std::string> initial_source;
    std::filesystem::path source_root;
    std::filesystem::path cache_root;
    std::unordered_map<std::string, hir::ResolvedModule> resolved_modules;
    std::unordered_set<std::string> modules_in_progress;

    bool compile_source_impl(std::string source, const std::string& name,
                             CompileStage stage, CompileResult& result) noexcept;
    bool compile_file_impl(const std::filesystem::path& path, CompileStage stage,
                           CompileResult& result) noexcept;
    void configure_root(const std::filesystem::path& source_name) noexcept;

    [[nodiscard]] std::optional<std::pair<std::filesystem::path, std::filesystem::path>>
    find_module(const std::string& name) const noexcept;
    [[nodiscard]] std::filesystem::path cache_path_for(
        const std::filesystem::path& logical_path,
        const std::filesystem::path& source_path) const noexcept;
    bool write_cache_artifact(const std::filesystem::path& path,
                              const std::vector<uint8_t>& binary) const noexcept;

public:
    Compiler() noexcept = default;
    explicit Compiler(std::string&& source) noexcept : initial_source(std::move(source)) {}
    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;
    Compiler(Compiler&&) = default;
    Compiler& operator=(Compiler&&) = delete;

    bool compile_source(const std::string& source, const std::string& name,
                        CompileResult& result,
                        CompileStage stage = CompileStage::Binary) noexcept;
    bool compile_file(const std::filesystem::path& path, CompileResult& result,
                      CompileStage stage = CompileStage::Binary,
                      bool is_main_module = false) noexcept;

    // Compatibility entry points now forward to the same session pipeline.
    bool compile(const std::string& name, std::vector<uint8_t>& result) noexcept;
    std::optional<std::vector<hir::Scope::Var>> compile_to_hir(
        const std::string& name) noexcept;

    std::optional<hir::ResolvedModule> resolve_module(
        const hir::ModuleRequest& request) noexcept override;
};

} // namespace lmx
