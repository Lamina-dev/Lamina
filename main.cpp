//
// Created by meian on 2026/4/8.
//


#include "lmx.h"

int main(int argc, char** argv) {
    if (argc < 2) return 1;

    auto* state = lmx_newState();
    //lmx_printASTFromFile(state, stdout, argv[1]);
    const auto module = lmx_doFile(state, argv[1], true);
    if (module == nullptr) {
        lmx_deleteState(state);
        return 1;
    }

    const auto vm = lmx_newLaminaVM(state, argc, argv);
    if (vm == nullptr) {
        lmx_deleteState(state);
        return 1;
    }

    const auto result = lmx_vmRunModule(state, vm, module);
    lmx_deleteState(state);
    return result;
}
