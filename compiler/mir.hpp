//
// Created by meian on 2026/7/16.
//

#pragma once
#include <string>

namespace lmx::mir {

struct MirNode {

};

struct MirExpr : MirNode {
    enum class Kind {
        Operate, Literal,
    };

    Kind kind;
};

struct MirStmt : MirNode {
    enum class Kind {
        Expr, TmpAssign, Assign,
    };

    Kind kind;
};

struct MirLiteral : MirExpr {
    enum class Kind {
        String, Int, Frac, Bool,
    };
    Kind kind;
    std::string data;
};

struct MirAddOp : MirExpr {
    std::string left, right;
};


}
