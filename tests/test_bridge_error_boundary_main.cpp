#include "allocation_probe.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <new>

namespace {
struct AllocationProbeState {
    std::size_t ordinal = 0;
    std::size_t fail_at = 0;
    std::size_t outstanding = 0;
    bool enabled = false;
    bool failed = false;
    std::array<void*, 4096> pointers{};
};

thread_local AllocationProbeState allocation_probe;

void* allocate_with_probe(const std::size_t size) {
    if (allocation_probe.enabled &&
        ++allocation_probe.ordinal == allocation_probe.fail_at) {
        allocation_probe.enabled = false;
        allocation_probe.failed = true;
        throw std::bad_alloc{};
    }
    void* memory = std::malloc(size == 0 ? 1 : size);
    if (!memory) throw std::bad_alloc{};
    if (allocation_probe.enabled) {
        if (allocation_probe.outstanding == allocation_probe.pointers.size())
            std::abort();
        allocation_probe.pointers[allocation_probe.outstanding++] = memory;
    }
    return memory;
}

void free_with_probe(void* memory) noexcept {
    for (std::size_t i = 0; i < allocation_probe.outstanding; ++i) {
        if (allocation_probe.pointers[i] == memory) {
            allocation_probe.pointers[i] =
                allocation_probe.pointers[--allocation_probe.outstanding];
            break;
        }
    }
    std::free(memory);
}
} // namespace

namespace lmx::test {
void allocation_probe_start(const std::size_t fail_at) noexcept {
    if (allocation_probe.enabled || allocation_probe.outstanding != 0)
        std::abort();
    allocation_probe.ordinal = 0;
    allocation_probe.fail_at = fail_at;
    allocation_probe.failed = false;
    allocation_probe.enabled = true;
}

void allocation_probe_stop() noexcept { allocation_probe.enabled = false; }
std::size_t allocation_probe_count() noexcept {
    return allocation_probe.ordinal;
}
std::size_t allocation_probe_outstanding() noexcept {
    return allocation_probe.outstanding;
}
bool allocation_probe_failed() noexcept { return allocation_probe.failed; }
} // namespace lmx::test

void* operator new(const std::size_t size) { return allocate_with_probe(size); }
void* operator new[](const std::size_t size) { return allocate_with_probe(size); }
void* operator new(const std::size_t size, const std::nothrow_t&) noexcept {
    try { return allocate_with_probe(size); }
    catch (...) { return nullptr; }
}
void* operator new[](const std::size_t size, const std::nothrow_t&) noexcept {
    try { return allocate_with_probe(size); }
    catch (...) { return nullptr; }
}
void operator delete(void* memory) noexcept { free_with_probe(memory); }
void operator delete[](void* memory) noexcept { free_with_probe(memory); }
void operator delete(void* memory, std::size_t) noexcept {
    free_with_probe(memory);
}
void operator delete[](void* memory, std::size_t) noexcept {
    free_with_probe(memory);
}
void operator delete(void* memory, const std::nothrow_t&) noexcept {
    free_with_probe(memory);
}
void operator delete[](void* memory, const std::nothrow_t&) noexcept {
    free_with_probe(memory);
}

extern "C" int lmx_test_c_abi_exception_boundaries() noexcept;

int main() {
    const int result = lmx_test_c_abi_exception_boundaries();
    if (result == 0) return 0;
    std::cerr << "C ABI exception boundary probe failed: " << result << '\n';
    return 1;
}
