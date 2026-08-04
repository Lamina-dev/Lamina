//
// Created by meian on 2026/7/31.
//

#pragma once
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "assembler.hpp"
#include "ast/ast.hpp"
#include "error.hpp"
#include "hir/type_checker.hpp"
#include "lexer.hpp"
#include "mir/mir.hpp"
#include "mir/mir_builder.hpp"
#include "parser.hpp"

namespace lmx {
enum class CompilerState {
    Src, Tok, Ast, FullAst, Mir, Bin, Err
};
class Compiler {
    using State = CompilerState;
    std::string src;
    std::vector<Token> tokens;
    std::shared_ptr<Module> module;
    mir::MirModule mir;
    std::vector<uint8_t> binary;
    State state{State::Src};

    static bool has_errd() noexcept {
        return errd;
    }
public:

    explicit Compiler(std::string&& src) noexcept : src(std::move(src)) {};
    explicit Compiler() noexcept = delete;
    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;
    Compiler(Compiler&&) = default;
    Compiler& operator=(Compiler&&) = delete;


    Compiler& lex() noexcept {
        if (state == State::Src) {
            tokens = Lexer(src).tokenize(src);
            if (!has_errd()) state = State::Tok;
            else state = State::Err;
        }
        return *this;
    }
    Compiler& get_tokens(decltype(tokens)& t) noexcept {
        if (state == State::Tok) {
            t = tokens;
        }
        return *this;
    }

    Compiler& parse(const std::string& name) noexcept {
        if (state == State::Tok) {
            module = Parser(tokens).parse_module(name);
            if (!has_errd()) state = State::Ast;
            else state = State::Err;
        }
        return *this;
    }
    Compiler& get_ast(decltype(module)& ast) noexcept {
        if (state == State::Ast) {
            ast = module;
        }
        return *this;
    }

    Compiler& sema() noexcept {
        if (state == State::Ast) {
            hir::TypeCkContext().check_module(module);
            if (!has_errd()) state = State::FullAst;
            else state = State::Err;
        }
        return *this;
    }

    Compiler& get_full_ast(decltype(module)& ast) noexcept {
        if (state == State::FullAst) {
            ast = module;
        }
        return *this;
    }

    Compiler& build() noexcept {
        if (state == State::FullAst) {
            mir = mir::MirBuilder::from_ast_module(module);
            if (!has_errd()) state = State::Mir;
            else state = State::Err;
        }
        return *this;
    }
    Compiler& get_mir(decltype(mir)& m) noexcept {
        if (state == State::Mir) {
            m = mir;
        }
        return *this;
    }

    Compiler& assemble() noexcept {
        if (state == State::Mir) {
            binary = Assembler().asm_module(&mir);
            if (!has_errd()) state = State::Bin;
            else state = State::Err;
        }
        return *this;
    }
    Compiler& get_bin(decltype(binary)& b) noexcept {
        if (state == State::Bin) {
            b = binary;
        }
        return *this;
    }

    template <typename F = void(*)(Compiler&)>
    Compiler& state_is(const State s, F&& f) noexcept {
        if (s == state) f(*this);
        return *this;
    }

    template <typename F = void(*)(Compiler&)>
    Compiler& state_not_is(const State s, F&& f) noexcept {
        if (s != state) f(*this);
        return *this;
    }
    template <typename F = void(*)(Compiler&)>
    Compiler& if_error(F&& f) noexcept {
        if (state == State::Err) {
            f(*this);
        }
        return *this;
    }


     bool compile(const std::string& name, std::vector<uint8_t>& result) noexcept {
        lex();
        if (has_errd()) return false;
        parse(name);
        if (has_errd()) return false;
        sema();
        if (has_errd()) return false;
        build();
        if (has_errd()) return false;
        assemble();
        if (has_errd()) return false;

        result = std::move(binary);

        return true;
    }

    std::optional<std::vector<hir::Scope::Var>> compile_to_hir(const std::string& name) noexcept {
        std::ifstream ifs(name);

        lex();
        if (has_errd()) return std::nullopt;
        parse(name);
        if (has_errd()) return std::nullopt;
        auto result = hir::TypeCkContext().check_module(module);
        if (!has_errd()) return result;
        return std::nullopt;
    }
};

} // lmx
