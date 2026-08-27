#include "compiler.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>

#include "assembler.hpp"
#include "error.hpp"
#include "lexer.hpp"
#include "mir/mir_builder.hpp"
#include "parser.hpp"
#include "../utils/utils.hpp"

namespace lmx {
namespace {

std::optional<std::filesystem::path> find_module_source(
    std::filesystem::path candidate) noexcept {
    namespace fs = std::filesystem;
    auto source_file = fs::path(candidate.string() + file_suffix);
    if (fs::is_regular_file(source_file)) return source_file;
    if (!fs::is_directory(candidate)) return std::nullopt;
    candidate /= std::string(file_default_mod) + file_suffix;
    if (fs::is_regular_file(candidate)) return candidate;
    return std::nullopt;
}

std::filesystem::path normalized_absolute(const std::filesystem::path& path) noexcept {
    std::error_code error;
    auto result = std::filesystem::weakly_canonical(path, error);
    if (!error) return result;
    return std::filesystem::absolute(path, error).lexically_normal();
}

} // namespace

void Compiler::configure_root(const std::filesystem::path& source_name) noexcept {
    if (!source_root.empty()) return;
    auto absolute_name = normalized_absolute(source_name);
    source_root = absolute_name.has_parent_path()
        ? absolute_name.parent_path() : std::filesystem::current_path();
    cache_root = source_root / module_cache_fold;
}

bool Compiler::compile_source_impl(std::string source, const std::string& name,
                                   const CompileStage stage,
                                   CompileResult& result) noexcept {
    result = {};
    auto tokens = Lexer(source, name).tokenize(source);
    if (errd) return false;

    result.module = Parser(tokens).parse_module(name);
    if (errd) return false;

    result.exports = hir::TypeCkContext(this).check_module(result.module);
    if (errd) return false;
    if (stage == CompileStage::Semantic) return true;

    result.mir = mir::MirBuilder::from_ast_module(result.module);
    if (errd) return false;
    if (stage == CompileStage::Mir) return true;

    result.binary = Assembler().asm_module(&*result.mir);
    if (errd) return false;
    result.binary.shrink_to_fit();
    return true;
}

bool Compiler::compile_file_impl(const std::filesystem::path& path,
                                 const CompileStage stage,
                                 CompileResult& result) noexcept {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) return false;
    std::string source{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
    source += '\n';
    return compile_source_impl(std::move(source), path.string(), stage, result);
}

bool Compiler::compile_source(const std::string& source, const std::string& name,
                              CompileResult& result,
                              const CompileStage stage) noexcept {
    errd = false;
    configure_root(name.empty() ? std::filesystem::path("<string>")
                                : std::filesystem::path(name));
    if (!compile_source_impl(source, name, stage, result)) return false;
    current_module_path = source_root;
    return true;
}

bool Compiler::compile_file(const std::filesystem::path& path, CompileResult& result,
                            const CompileStage stage,
                            const bool is_main_module) noexcept {
    errd = false;
    const auto source_path = normalized_absolute(path);
    configure_root(source_path);

    if (is_main_module && main_module) {
        std::cerr << "main module already exists: named \""
                  << main_module->name << "\"\n";
        return false;
    }
    const auto identity = source_path.string();
    modules_in_progress.insert(identity);
    const bool compiled = compile_file_impl(source_path, stage, result);
    modules_in_progress.erase(identity);
    if (!compiled) return false;
    current_module_path = source_root;
    if (is_main_module) {
        main_module = result.module;
    }
    return true;
}

bool Compiler::compile(const std::string& name,
                       std::vector<uint8_t>& result) noexcept {
    if (!initial_source) return false;
    CompileResult compilation;
    if (!compile_source(*initial_source, name, compilation)) return false;
    result = std::move(compilation.binary);
    return true;
}

std::optional<std::vector<hir::Scope::Var>> Compiler::compile_to_hir(
    const std::string& name) noexcept {
    if (!initial_source) return std::nullopt;
    CompileResult compilation;
    if (!compile_source(*initial_source, name, compilation, CompileStage::Semantic))
        return std::nullopt;
    return std::move(compilation.exports);
}

std::optional<std::pair<std::filesystem::path, std::filesystem::path>>
Compiler::find_module(const std::string& name) const noexcept {
    namespace fs = std::filesystem;
    fs::path logical_path;
    std::string component;
    for (const char c : name) {
        if (c == '.') {
            logical_path /= component;
            component.clear();
        } else {
            component += c;
        }
    }
    logical_path /= component;

    const std::filesystem::path search_roots[] = {
        source_root,
        module_path,
        fs::current_path() / "modules",
        toolchain_module_path,
    };
    for (const auto& root : search_roots) {
        if (auto found = find_module_source(root / logical_path))
            return std::pair{logical_path, normalized_absolute(*found)};
    }
    return std::nullopt;
}

std::filesystem::path Compiler::cache_path_for(
    const std::filesystem::path& logical_path,
    const std::filesystem::path& source_path) const noexcept {
    auto output = cache_root;
    if (source_path.filename() == std::string(file_default_mod) + file_suffix) {
        output /= logical_path;
        output /= std::string(file_default_mod) + file_suffix_binary;
    } else {
        output /= logical_path.string() + file_suffix_binary;
    }
    return output.lexically_normal();
}

bool Compiler::write_cache_artifact(const std::filesystem::path& path,
                                    const std::vector<uint8_t>& binary) const noexcept {
    std::ifstream existing(path, std::ios::binary);
    if (existing.is_open()) {
        const std::vector<uint8_t> cached{
            std::istreambuf_iterator<char>(existing),
            std::istreambuf_iterator<char>()
        };
        if (cached == binary) return true;
    }

    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) return false;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(binary.data()),
                 static_cast<std::streamsize>(binary.size()));
    return output.good();
}

std::optional<hir::ResolvedModule> Compiler::resolve_module(
    const hir::ModuleRequest& request) noexcept {
    const auto found = find_module(request.name);
    if (!found) {
        throw_error(ErrorType::Analysis,
                    "no module is called `" + request.name + "`",
                    request.line, request.col);
        return std::nullopt;
    }
    const auto& [logical_path, source_path] = *found;
    const auto identity = source_path.string();
    if (const auto cached = resolved_modules.find(identity);
        cached != resolved_modules.end()) return cached->second;

    if (!modules_in_progress.insert(identity).second) {
        throw_error(ErrorType::Analysis,
                    "circular import of `" + request.name + "` from `" +
                        request.importer + "`",
                    request.line, request.col);
        return std::nullopt;
    }

    CompileResult compilation;
    const bool compiled = compile_file_impl(source_path, CompileStage::Binary, compilation);
    modules_in_progress.erase(identity);
    if (!compiled || errd) {
        if (!errd)
            throw_error(ErrorType::Analysis, "cannot open `" + identity + "`", 0, 0);
        return std::nullopt;
    }

    const auto output_path = cache_path_for(logical_path, source_path);
    if (!write_cache_artifact(output_path, compilation.binary)) {
        throw_error(ErrorType::Generate,
                    "cannot write module cache `" + output_path.string() + "`", 0, 0);
        return std::nullopt;
    }

    auto load_path = output_path.lexically_relative(cache_root);
    if (load_path.empty()) load_path = output_path.filename();
    const auto binding_name = logical_path.filename().string();
    auto type = std::static_pointer_cast<ModuleType>(type_pool.module(
        output_path.string(), load_path.string(), binding_name,
        std::move(compilation.exports),
        std::move(compilation.module->function_slots),
        std::move(compilation.module->adt_exports),
        std::move(compilation.module->unit_exports)));
    hir::ResolvedModule result{
        identity,
        binding_name,
        std::move(type),
    };
    resolved_modules.emplace(identity, result);
    return result;
}

} // namespace lmx
