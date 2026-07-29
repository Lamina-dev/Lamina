//
// Created by meian on 2026/4/8.
//


#include "lmx.h"

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    auto state = lmx_newState();
    const auto module = lmx_doFile(&state, argv[1]);

    // lmx_printASTFromFile(&state, stdout, "../test.lm");
    const auto vm = lmx_newLaminaVM(&state, argc, argv);

    const auto result = lmx_vmRunModule(&state, vm, module);

    // std::cout << reinterpret_cast<lmx::runtime::LaminaVM*>(vm)->get_reg(1).int_val << std::endl;
    // while (true) {
    //     if (getchar() != 'c') {break;}
    // }
    // std::cout << reinterpret_cast<lmx::runtime::LaminaVM*>(vm)->get_reg(1).int_val << std::endl;
    lmx_deleteState(&state);
    return result;
}
