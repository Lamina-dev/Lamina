#include "utils.hpp"

#if defined(__unix__)
#include <unistd.h>
#include <limits.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <limits.h>
#else
#include <windows.h>
#endif
namespace lmx {

std::filesystem::path get_exe_dir() noexcept {
    char* res;
#if defined(_WIN32) || defined(_WIN64)
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    res = buffer;
#elif defined(__unix__)
    char buffer[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);

    if (len != -1) {
        buffer[len] = 0;
    }
    res = buffer;
#else
    char buffer[PATH_MAX];
    uint32_t buf_size = PATH_MAX;
    if (_NSGetExecutablePath(buffer, &buf_size) == 0) {
        char resolve[PATH_MAX];
        if (realpath(buffer, resolve)) {
            res = resolve;
        }
    }
#endif
    return std::filesystem::path(res).parent_path();
}

const std::filesystem::path module_path = (std::filesystem::current_path() / ".." / "modules").lexically_normal();
std::filesystem::path toolchain_module_path = get_exe_dir().parent_path() / "modules";
std::filesystem::path current_module_path{};
std::shared_ptr<Module> main_module = nullptr;

}
