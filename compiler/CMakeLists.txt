add_library(laminac OBJECT
    ast.cpp
    ast_printer.cpp
    error.cpp
    lexer.cpp
    parser.cpp
        type_checker.cpp
    mir.cpp
    mir_builder.cpp
    mir_printer.cpp
    assembler.cpp
    compiler.cpp
    compiler.hpp
    cas/cas_parse.cpp
)