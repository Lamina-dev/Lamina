//
// Created by meian on 2026/4/8.
//
#include <iostream>
#include <ostream>
#include <string>

#include "runtime/vm.hpp"
#include "lmx.h"

int main(int argc, char** argv) {
    const std::string code = R"(
func fib(n int) -> int {
    if (n <= 1) {n}
    else { fib(n - 1) + fib(n - 2) }
}
let v = fib(30)

)";
    auto state = lmx_newState();

    const auto module = lmx_doString(&state, code.c_str(), "test");

    const auto vm = lmx_newLaminaVM(&state, argc, argv);


    auto result = lmx_vmRunModule(&state, vm, module);

    std::cout << reinterpret_cast<lmx::runtime::LaminaVM*>(vm)->get_reg(1).int_val << std::endl;

    lmx_deleteState(&state);
}
