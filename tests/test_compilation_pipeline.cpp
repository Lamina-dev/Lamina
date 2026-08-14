#include "lmx.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "check failed at line " << __LINE__ << ": " #condition "\n"; \
        ++failures; \
    } \
} while (false)

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << text;
}

std::vector<char> read_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

bool compile_and_run_file(const std::filesystem::path& path, bool is_main) {
    auto* state = lmx_newState();
    const auto path_text = path.string();
    auto* module = lmx_doFile(state, path_text.c_str(), is_main);
    char program[] = "pipeline-test";
    char* argv[] = {program};
    auto* vm = module ? lmx_newLaminaVM(state, 1, argv) : nullptr;
    const bool ok = module && vm && lmx_vmRunModule(state, vm, module) == 0;
    lmx_deleteState(state);
    return ok;
}

bool compilation_fails(const std::filesystem::path& path) {
    auto* state = lmx_newState();
    const auto path_text = path.string();
    const bool failed = lmx_doFile(state, path_text.c_str(), false) == nullptr;
    lmx_deleteState(state);
    return failed;
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    // The standalone source entry point must use the same complete pipeline.
    {
        auto* state = lmx_newState();
        auto* module = lmx_doString(state, "let value = 1 + 2\n", "pipeline_source.lm");
        char program[] = "pipeline-test";
        char* argv[] = {program};
        auto* vm = module ? lmx_newLaminaVM(state, 1, argv) : nullptr;
        CHECK(module != nullptr);
        CHECK(vm != nullptr);
        CHECK(module && vm && lmx_vmRunModule(state, vm, module) == 0);
        lmx_deleteState(state);
    }

    const auto root = fs::current_path() / "compilation_pipeline_fixture";
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root);

    const auto c_source = root / "c.lm";
    const auto b_source = root / "b.lm";
    const auto main_source = root / "main.lm";
    const auto single_source = root / "single.lm";
    write_text(c_source, "func value() -> int { return 3 }\n");
    write_text(b_source,
               "import c\n"
               "func value() -> int { return c.value() + 1 }\n");
    write_text(main_source,
               "import b\n"
               "import b\n"
               "b.value()\n");
    write_text(single_source, "import c\nc.value()\n");

    // A non-main file must carry its own source root into runtime import loading.
    CHECK(compile_and_run_file(single_source, false));

    // Main-file compilation, nested imports, duplicate imports, and execution.
    CHECK(compile_and_run_file(main_source, true));

    const auto b_cache = root / "_lm_cache" / "b.lmc";
    const auto c_cache = root / "_lm_cache" / "c.lmc";
    CHECK(fs::is_regular_file(b_cache));
    CHECK(fs::is_regular_file(c_cache));

    // An identical rebuild must keep the existing artifact instead of rewriting it.
    const auto old_time = fs::file_time_type::clock::now() - std::chrono::hours(24);
    fs::last_write_time(c_cache, old_time);
    const auto cached_time = fs::last_write_time(c_cache);
    const auto before_hit = read_binary(c_cache);
    CHECK(compile_and_run_file(single_source, false));
    CHECK(fs::last_write_time(c_cache) == cached_time);
    CHECK(read_binary(c_cache) == before_hit);

    // A changed source must invalidate and replace the dependency artifact.
    write_text(c_source, "func value() -> int { return 4 }\n");
    CHECK(compile_and_run_file(single_source, false));
    CHECK(read_binary(c_cache) != before_hit);

    // Imported parse/type failures and missing modules must propagate to the caller.
    write_text(root / "missing.lm", "import module_that_does_not_exist\n");
    write_text(root / "bad_syntax_dep.lm", "let value =\n");
    write_text(root / "syntax_root.lm", "import bad_syntax_dep\n");
    write_text(root / "bad_type_dep.lm", "let value int = \"text\"\n");
    write_text(root / "type_root.lm", "import bad_type_dep\n");
    CHECK(compilation_fails(root / "missing.lm"));
    CHECK(compilation_fails(root / "syntax_root.lm"));
    CHECK(compilation_fails(root / "type_root.lm"));

    // Cycles now fail deterministically in the compilation coordinator.
    write_text(root / "cycle_a.lm", "import cycle_b\n");
    write_text(root / "cycle_b.lm", "import cycle_a\n");
    CHECK(compilation_fails(root / "cycle_a.lm"));

    fs::remove_all(root, error);
    return failures == 0 ? 0 : 1;
}
