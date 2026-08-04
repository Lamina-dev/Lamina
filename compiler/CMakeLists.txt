add_library(laminac OBJECT
    ast.cpp
    ast_printer.cpp
    error.cpp
    lexer.cpp
    parser.cpp
        type_checker.cpp
        mir/mir.cpp
        mir/mir_builder.cpp
        mir/mir_printer.cpp
    assembler.cpp
    compiler.cpp
    compiler.hpp
    cas/cas_parse.cpp
        hir/hir.cpp
        hir/hir.hpp
)