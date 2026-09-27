#include <iostream>

extern "C" int lmx_test_c_abi_exception_boundaries() noexcept;

int main() {
    const int result = lmx_test_c_abi_exception_boundaries();
    if (result == 0) return 0;
    std::cerr << "C ABI exception boundary probe failed: " << result << '\n';
    return 1;
}
