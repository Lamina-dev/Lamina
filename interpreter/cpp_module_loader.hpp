#pragma once
#include "lamina_api/value.hpp"
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>  // 新增：用于uintptr_t（32位地址转换需用）  // 关键修复1
#include <cstring>  // 用于 strstr/strcmp


// 平台宏适配：区分 Windows 与类 Unix 系统（Linux/macOS）
#ifdef _WIN32
#include <windows.h>
#include <imagehlp.h>  // Windows 符号表解析（PE 文件结构）
#include <tlhelp32.h>
// 链接 ImageHlp 库（Windows 必需，否则找不到符号解析函数）
#pragma comment(lib, "imagehlp.lib")

// Windows 动态库操作统一宏（隐藏平台差异）
#define DYLIB_HANDLE HMODULE
#define DYLIB_LOAD(path) LoadLibraryA(path.c_str())
#define DYLIB_GETSYM(handle, name) reinterpret_cast<void*>(GetProcAddress(handle, name))
#define DYLIB_UNLOAD(handle) FreeLibrary(handle)
#define DYLIB_ERROR() GetLastError()

#else
// Linux/macOS 通用动态库操作宏
#ifndef STT_DEBUG
#define STT_DEBUG 0x11  // 补充调试符号类型定义（部分系统可能未定义）
#endif

#define DYLIB_HANDLE void*
#define DYLIB_LOAD(path) dlopen(path.c_str(), RTLD_LAZY | RTLD_NOLOAD)  // 加载后不重复加载
#define DYLIB_GETSYM(handle, name) dlsym(handle, name)
#define DYLIB_UNLOAD(handle) dlclose(handle)
#define DYLIB_ERROR() dlerror()

// 类 Unix 系统通用头文件（Linux/macOS 均支持）
#include <dlfcn.h>  // 动态库加载核心接口（dlopen/dlsym 等）

// 按子平台区分专属头文件（Linux=ELF，macOS=Mach-O）
#if defined(__linux__)
// Linux 专属：ELF 动态库解析依赖（解析 ELF 符号表）
#include <elf.h>   // ELF 结构定义（Elf_Phdr/Elf_Sym 等）
#include <link.h>  // ELF 动态库遍历接口（dl_iterate_phdr）
#elif defined(__APPLE__)
// 修复1：恢复 macOS 必需的 Mach-O 头文件（缺一不可）
#include <mach-o/dyld.h>    // 定义 _dyld_iterate_images 函数
#include <mach-o/nlist.h>   // 定义 nlist_64/nlist 符号表结构
#include <mach-o/loader.h>  // 定义 mach_header_64/symtab_command 等结构
#endif

#endif

// 目标函数指针类型：定义动态库导出函数的统一签名
using ModuleFunc = Value (*)(const std::vector<Value>&);

#ifdef _WIN32
// Windows：获取动态库的所有导出符号名（基于 PE 文件结构解析）
inline std::vector<std::string> get_win_dylib_symbols(DYLIB_HANDLE hModule) {
    std::vector<std::string> symbols;
    if (!hModule) return symbols;

    // 步骤1：获取动态库的完整路径（用于后续映射文件）
    char module_path[MAX_PATH] = {0};
    GetModuleFileNameA(hModule, module_path, MAX_PATH);

    // 步骤2：映射动态库到内存（解析 PE 结构必需）
    LOADED_IMAGE loaded_img = {nullptr};
    if (!MapAndLoad(module_path, nullptr, &loaded_img, FALSE, TRUE)) {
        std::cerr << "Failed to map module: " << GetLastError() << std::endl;
        return symbols;
    }

    // 步骤3：定位 PE 文件的导出表（IMAGE_EXPORT_DIRECTORY）
    const PIMAGE_NT_HEADERS nt_headers = ImageNtHeader(loaded_img.MappedAddress);
    const auto export_dir =
        static_cast<PIMAGE_EXPORT_DIRECTORY>(
            ImageRvaToVa(nt_headers, loaded_img.MappedAddress, 
                        nt_headers->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress, 
                        nullptr)
        );
    if (!export_dir) {
        std::cerr << "No export directory found" << std::endl;
        UnMapAndLoad(&loaded_img);  // 释放映射资源
        return symbols;
    }

    // 步骤4：遍历导出符号名（AddressOfNames 是符号名的 RVA 数组）
    const auto* name_rvas = static_cast<DWORD*>(
        ImageRvaToVa(nt_headers, loaded_img.MappedAddress, export_dir->AddressOfNames, nullptr)
    );
    for (DWORD i = 0; i < export_dir->NumberOfNames; ++i) {
        auto sym_name = static_cast<const char*>(
            ImageRvaToVa(nt_headers, loaded_img.MappedAddress, name_rvas[i], nullptr)
        );
        if (sym_name && *sym_name != '\0') {
            symbols.emplace_back(sym_name);
        }
    }

    // 步骤5：释放内存映射资源
    UnMapAndLoad(&loaded_img);
    return symbols;
}
#endif

// 跨平台符号解析：对外统一接口（隐藏 Linux/macOS 底层差异）
std::vector<std::string> get_dylib_symbols(const std::string& lib_path);

#ifndef _WIN32
// -------------------------- Linux 平台：ELF 动态库符号解析 --------------------------
#if defined(__linux__)
// 静态全局变量：传递目标库路径给回调函数（回调无法直接访问 get_dylib_symbols 的局部参数）
static std::string s_linux_lib_path;

#if __SIZEOF_POINTER__ == 8  // 64位环境：指针8字节
using Elf_Phdr = Elf64_Phdr;
using Elf_Dyn = Elf64_Dyn;
using Elf_Sym = Elf64_Sym;
using Elf_Word = Elf64_Word;
#define ELF_ST_BIND ELF64_ST_BIND
#define ELF_ST_TYPE ELF64_ST_TYPE
#else  // 32位环境：指针4字节（包括x86_64-linux-gnux32）
using Elf_Phdr = Elf32_Phdr;
using Elf_Dyn = Elf32_Dyn;
using Elf_Sym = Elf32_Sym;
using Elf_Word = Elf32_Word;
#define ELF_ST_BIND ELF32_ST_BIND
#define ELF_ST_TYPE ELF32_ST_TYPE
#endif


// Linux：ELF 动态库遍历回调（由 dl_iterate_phdr 触发）
static int elf_symbol_callback(struct dl_phdr_info* info, size_t size, void* data) {
        std::vector<std::string>* symbols = reinterpret_cast<std::vector<std::string>*>(data);
    if (!symbols || !info->dlpi_name) return 1;

    // 只处理目标动态库
    if (std::string(info->dlpi_name) != s_linux_lib_path) return 0;

    void* handle = dlopen(info->dlpi_name, RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) return 0;

    // 遍历 ELF 动态段（PT_DYNAMIC）
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        // 现在 Elf_Phdr 类型与 info->dlpi_phdr 匹配（32位Elf32_Phdr，64位Elf64_Phdr）
        const Elf_Phdr* phdr = &info->dlpi_phdr[i];
        if (phdr->p_type != PT_DYNAMIC) continue;

        // 修复：动态段地址计算（32/64位通用，依赖Elf_Phdr的p_vaddr类型）
        Elf_Dyn* dyn = reinterpret_cast<Elf_Dyn*>(
            reinterpret_cast<uintptr_t>(info->dlpi_addr) + phdr->p_vaddr
        );
        Elf_Sym* symtab = nullptr;
        const char* strtab = nullptr;
        Elf_Word symcount = 0;

        // 提取符号表/字符串表（32/64位通过Elf_Dyn的d_un区分）
        for (; dyn->d_tag != DT_NULL; ++dyn) {
            switch (dyn->d_tag) {
                case DT_SYMTAB:
#if __SIZEOF_POINTER__ == 8
                    symtab = reinterpret_cast<Elf_Sym*>(dyn->d_un.d_ptr);
#else
                    symtab = reinterpret_cast<Elf_Sym*>(static_cast<uintptr_t>(dyn->d_un.d_val));
#endif
                    break;
                case DT_STRTAB:
#if __SIZEOF_POINTER__ == 8
                    strtab = reinterpret_cast<const char*>(dyn->d_un.d_ptr);
#else
                    strtab = reinterpret_cast<const char*>(static_cast<uintptr_t>(dyn->d_un.d_val));
#endif
                    break;
                case DT_SYMENT:
                    symcount = phdr->p_memsz / sizeof(Elf_Sym);
                    break;
            }
        }

        // 跳过符号表/字符串表缺失的情况
        if (!symtab || !strtab || symcount == 0) continue;

        // 遍历符号表：筛选有效导出符号
        for (Elf_Word j = 0; j < symcount; ++j) {
            const Elf_Sym* sym = &symtab[j];
            // 筛选条件：① 全局符号 ② 非调试符号 ③ 有符号名
            if (ELF_ST_BIND(sym->st_info) != STB_GLOBAL || 
                ELF_ST_TYPE(sym->st_info) == STT_DEBUG || 
                sym->st_name == 0) {
                continue;
            }
            const char* sym_name = strtab + sym->st_name;
            // 排除 ELF 内部符号（_init 初始化函数、_fini 清理函数）
            if (strstr(sym_name, "_init") || strstr(sym_name, "_fini")) continue;
            symbols->emplace_back(sym_name);  // 加入结果列表
        }
    }

    // 清理临时宏定义（避免污染其他代码）
    #undef ELF_ST_BIND
    #undef ELF_ST_TYPE
    dlclose(handle);  // 关闭动态库句柄
    return 0;
}

// Linux：实现跨平台统一接口（触发 ELF 符号解析）
std::vector<std::string> get_dylib_symbols(const std::string& lib_path) {
    std::vector<std::string> symbols;
    s_linux_lib_path = lib_path;  // 赋值全局变量，传递给回调
    // 遍历进程中已加载的 ELF 动态库，触发回调函数
    dl_iterate_phdr(elf_symbol_callback, &symbols);
    s_linux_lib_path.clear();     // 清空全局变量，避免污染
    return symbols;
}

// -------------------------- macOS 平台：Mach-O 动态库符号解析 --------------------------
#elif defined(__APPLE__)
// 静态全局变量：传递目标库路径和符号结果（适配 _dyld_iterate_imagees 回调）
static std::string s_target_lib_path;
static std::vector<std::string>* s_macho_symbols = nullptr;

// macOS：Mach-O 镜像遍历回调（由 _dyld_iterate_imagees 触发）
static bool macho_image_callback(const struct mach_header* mh, uintptr_t vmaddr_slide, 
                                 const char* image_path, void* data) {
    // 步骤1：只处理目标动态库（跳过系统库和路径不匹配的库）
    if (!image_path || std::string(image_path) != s_target_lib_path) {
        return false;  // 返回 false 继续遍历其他镜像
    }

    // 步骤2：打开目标动态库（验证有效性，获取符号访问权限）
    void* handle = dlopen(image_path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "macOS dlopen failed: " << dlerror() << std::endl;
        return false;
    }

    // 步骤3：解析 Mach-O 加载命令，找到符号表命令（LC_SYMTAB）
    const struct load_command* lc = nullptr;
    struct symtab_command* symtab_cmd = nullptr;
    uint32_t ncmds = 0;  // 加载命令总数

    // 适配 Mach-O 32/64 位结构（通过 magic 字段判断）
    uint32_t magic = mh->magic;
    if (magic == MH_MAGIC_64 || magic == MH_CIGAM_64) {
        // 64位 Mach-O 镜像
        const struct mach_header_64* mh64 = reinterpret_cast<const struct mach_header_64*>(mh);
        // 加载命令起始地址 = 镜像基地址 + mach_header_64 大小
        lc = reinterpret_cast<const struct load_command*>(
            reinterpret_cast<const uint8_t*>(mh64) + sizeof(struct mach_header_64)
        );
        ncmds = mh64->ncmds;  // 64位镜像的加载命令总数
    } else if (magic == MH_MAGIC || magic == MH_CIGAM) {
        // 32位 Mach-O 镜像（兼容旧版 macOS 镜像）
        lc = reinterpret_cast<const struct load_command*>(
            reinterpret_cast<const uint8_t*>(mh) + sizeof(struct mach_header)
        );
        ncmds = mh->ncmds;  // 32位镜像的加载命令总数
    } else {
        // 不支持的 Mach-O 格式（如 FAT 镜像）
        dlclose(handle);
        return false;
    }

    // 遍历加载命令：找到符号表命令（LC_SYMTAB）
    for (uint32_t i = 0; i < ncmds; ++i) {
        if (lc->cmd == LC_SYMTAB) {
            // 转换为符号表命令结构（需 const_cast，因为 lc 是 const）
            symtab_cmd = reinterpret_cast<struct symtab_command*>(const_cast<struct load_command*>(lc));
            break;
        }
        // 跳到下一个加载命令（加载命令大小由 cmdsize 字段指定）
        lc = reinterpret_cast<const struct load_command*>(
            reinterpret_cast<const uint8_t*>(lc) + lc->cmdsize
        );
    }

    // 无符号表命令，直接返回
    if (!symtab_cmd) {
        dlclose(handle);
        return false;
    }

    // 步骤4：提取符号表和字符串表（需加上 ASLR 偏移 vmaddr_slide，应对地址随机化）
    const char* strtab = reinterpret_cast<const char*>(vmaddr_slide + symtab_cmd->stroff);
    const uint32_t symcount = symtab_cmd->nsyms;  // 符号总数

    // 步骤5：遍历符号表，筛选有效符号（适配 32/64 位）
    if (magic == MH_MAGIC_64 || magic == MH_CIGAM_64) {
        // 64位 Mach-O 符号表（nlist_64 结构）
        const struct nlist_64* symtab = reinterpret_cast<const struct nlist_64*>(
            vmaddr_slide + symtab_cmd->symoff
        );
        for (uint32_t j = 0; j < symcount; ++j) {
            const struct nlist_64* sym = &symtab[j];
            // 筛选条件：① 有符号名 ② 全局符号（N_EXT=1）③ 非调试符号（N_STAB=0）
            if (sym->n_un.n_strx == 0 || !(sym->n_type & N_EXT) || (sym->n_type & N_STAB)) {
                continue;
            }
            const char* sym_name = strtab + sym->n_un.n_strx;
            // 处理 macOS 符号前缀：默认带 "_"，统一去掉（与 Linux 符号格式对齐）
            if (sym_name[0] == '_') sym_name++;
            // 排除 Mach-O 内部符号（__mh_ 镜像相关、_dyld_ 动态加载相关）
            if (strstr(sym_name, "__mh_") || strstr(sym_name, "_dyld_")) {
                continue;
            }
            s_macho_symbols->emplace_back(sym_name);
        }
    } else {
        // 32位 Mach-O 符号表（nlist 结构）
        const struct nlist* symtab = reinterpret_cast<const struct nlist*>(
            vmaddr_slide + symtab_cmd->symoff
        );
        for (uint32_t j = 0; j < symcount; ++j) {
            const struct nlist* sym = &symtab[j];
            if (sym->n_un.n_strx == 0 || !(sym->n_type & N_EXT) || (sym->n_type & N_STAB)) {
                continue;
            }
            const char* sym_name = strtab + sym->n_un.n_strx;
            if (sym_name[0] == '_') sym_name++;
            if (strstr(sym_name, "__mh_") || strstr(sym_name, "_dyld_")) {
                continue;
            }
            s_macho_symbols->emplace_back(sym_name);
        }
    }

    dlclose(handle);  // 关闭动态库句柄
    return true;  // 找到目标库，返回 true 停止遍历
}

// macOS：实现跨平台统一接口（触发 Mach-O 符号解析）
std::vector<std::string> get_dylib_symbols(const std::string& lib_path) {
    std::vector<std::string> symbols;
    // 初始化全局变量：传递目标路径和结果容器给回调
    s_target_lib_path = lib_path;
    s_macho_symbols = &symbols;

    // 遍历 macOS 进程中已加载的 Mach-O 镜像，触发回调
    _dyld_iterate_images(macho_image_callback, nullptr);

    // 清空全局变量，避免污染后续调用
    s_target_lib_path.clear();
    s_macho_symbols = nullptr;
    return symbols;
}
#endif
#endif

// 核心函数：加载 C++ 动态库，解析符号并封装为 Value 类型
inline std::unordered_map<std::string, Value> load_cppmodule(const std::string& path) {
    std::unordered_map<std::string, Value> result;  // 存储解析后的符号（名称->Value）
    DYLIB_HANDLE handle = nullptr;
    std::vector<std::string> all_symbols;  // 存储动态库的所有导出符号名

    // 步骤1：加载动态库
    handle = DYLIB_LOAD(path);
    if (!handle) {
        std::cerr << "Load module failed: " << DYLIB_ERROR() << std::endl;
        return result;
    }

    // 步骤2：获取动态库的所有导出符号名（调用跨平台接口）
#ifdef _WIN32
    all_symbols = get_win_dylib_symbols(handle);
#else
    all_symbols = get_dylib_symbols(path);  // 替换原 get_elf_dylib_symbols，支持 Linux/macOS
#endif

    // 步骤3：处理无符号的情况（释放句柄并返回）
    if (all_symbols.empty()) {
        std::cerr << "No exported symbols found in module" << std::endl;
        DYLIB_UNLOAD(handle);
        return result;
    }

    // 步骤4：遍历符号，区分变量和函数并封装为 Value
    for (const auto& sym_name : all_symbols) {
        // 跳过保护符号（如 lamina_protect_ 前缀的内部符号）
        if (sym_name.starts_with("lamina_protect_")) continue;

        // 尝试解析为 Value 类型变量（符号地址是 Value 实例指针）
        void* sym_addr = DYLIB_GETSYM(handle, sym_name.c_str());
        if (!sym_addr) continue;  // 符号地址无效，跳过

        // 转换为 Value 指针，验证是否为合法变量（非函数类型）
        const auto* var_ptr = static_cast<Value*>(sym_addr);
        bool is_valid_value_var = (!var_ptr->is_lmCppFunction());
        if (is_valid_value_var) {
            result[sym_name] = *var_ptr;  // 复制 Value 到结果
            std::cout << "Found Value var: " << sym_name << std::endl;
            continue;
        }

        // 尝试解析为 ModuleFunc 类型函数（符号地址是函数指针）
        auto func_ptr = reinterpret_cast<ModuleFunc>(sym_addr);
        if (func_ptr != nullptr) {
            // 封装为 LmCppFunction 智能指针，再转为 Value
            result[sym_name] = Value(std::make_shared<LmCppFunction>(func_ptr));
            std::cout << "Found ModuleFunc: " << sym_name << std::endl;
            continue;
        }

        // 未知类型符号（跳过并打印警告）
        std::cerr << "Skip unknown type symbol: " << sym_name << std::endl;
    }

    // 注意：动态库句柄需保留（后续调用函数时依赖），建议在 VM 生命周期结束时统一卸载
    // DYLIB_UNLOAD(handle); 

    return result;
}