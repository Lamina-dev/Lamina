//
// Created by meian on 2026/7/20.
//

#pragma once
#include "mir.hpp"

namespace lmx::mir {

class MirBuilder {
public:
    static MirModule from_ast_module(const std::shared_ptr<Module>& ast);
};

}
