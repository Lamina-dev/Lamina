add_library(laminac OBJECT
    ast.cpp
    ast_printer.cpp
    error.cpp
    lexer.cpp
    parser.cpp
    hir.cpp
    mir.cpp
    mir_builder.cpp
    mir_printer.cpp
    assembler.cpp
)