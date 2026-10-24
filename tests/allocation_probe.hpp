#pragma once

#include <cstddef>
#include <limits>

namespace lmx::test {

void allocation_probe_start(std::size_t fail_at) noexcept;
void allocation_probe_stop() noexcept;
std::size_t allocation_probe_count() noexcept;
std::size_t allocation_probe_outstanding() noexcept;
bool allocation_probe_failed() noexcept;

class AllocationProbeScope final {
public:
    explicit AllocationProbeScope(
        std::size_t fail_at = std::numeric_limits<std::size_t>::max()) noexcept {
        allocation_probe_start(fail_at);
    }
    ~AllocationProbeScope() { allocation_probe_stop(); }

    AllocationProbeScope(const AllocationProbeScope&) = delete;
    AllocationProbeScope& operator=(const AllocationProbeScope&) = delete;

    [[nodiscard]] std::size_t count() const noexcept {
        return allocation_probe_count();
    }
    [[nodiscard]] std::size_t outstanding() const noexcept {
        return allocation_probe_outstanding();
    }
    [[nodiscard]] bool failed() const noexcept {
        return allocation_probe_failed();
    }
};

} // namespace lmx::test
