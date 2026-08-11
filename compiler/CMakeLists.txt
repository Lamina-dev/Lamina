add_library(laminac OBJECT
        ast/ast.cpp
        ast/ast_printer.cpp
    error.cpp
    lexer.cpp
    parser.cpp
        hir/type_checker.cpp
        mir/mir.cpp
        mir/mir_builder.cpp
        mir/mir_printer.cpp
    assembler.cpp
    compiler.cpp
    compiler.hpp
        hir/hir.cpp
        hir/hir.hpp
)
