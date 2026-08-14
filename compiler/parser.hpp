//
// Created by meian on 2026/4/3.
//

#pragma once
#include "ast/ast.hpp"
#include "lexer.hpp"

namespace lmx {
extern std::shared_ptr<Module> cur_module;
class Parser {
    std::shared_ptr<ExprNode> parse_logical()       noexcept;   // or and
    std::shared_ptr<ExprNode> parse_equality()      noexcept;   // == !=
    std::shared_ptr<ExprNode> parse_relational()    noexcept;   // > < >= <=
    std::shared_ptr<ExprNode> parse_addition()      noexcept;   // + -
    std::shared_ptr<ExprNode> parse_multi()         noexcept;   // * / %
    std::shared_ptr<ExprNode> parse_exponent()      noexcept;   // ^
    std::shared_ptr<ExprNode> parse_term()          noexcept;   // -
    std::shared_ptr<ExprNode> parse_factor()        noexcept;   // !
    std::shared_ptr<ExprNode> parse_primary()       noexcept;   // num, (expr), ident, table, mat, set...()[].
    std::shared_ptr<ExprNode> parse_block()         noexcept;
    std::shared_ptr<ExprNode> parse_if()            noexcept;
    std::shared_ptr<ExprNode> parse_match()         noexcept;
    Pattern parse_pattern()                         noexcept;
    std::shared_ptr<Type> parse_type() noexcept;
    // std::shared_ptr<ExprStmtNode> parse_multi_naming()    noexcept;
    std::shared_ptr<ExprStmtNode> parse_param_name()      noexcept;


    std::shared_ptr<StmtNode> parse_func()              noexcept;

    std::shared_ptr<StmtNode> parse_var()               noexcept;
    std::shared_ptr<StmtNode> parse_var_decl()          noexcept;

    std::shared_ptr<StmtNode> parse_loop() noexcept;

    std::shared_ptr<StmtNode> parse_return() noexcept;
    std::shared_ptr<StmtNode> parse_import() noexcept;
    std::shared_ptr<StmtNode> parse_type_decl() noexcept;


    // void advance() noexcept;
    bool consume(TokenType token_type, const std::string &tk) noexcept;
    [[nodiscard]] bool match(TokenType t) const noexcept;

    [[nodiscard]] bool peek_match(TokenType t) const noexcept;

    std::shared_ptr<ExprNode> parse_pipe() noexcept;
    std::shared_ptr<ExprNode> parse_arrow() noexcept;
    std::shared_ptr<ExprNode> parse_set_literal() noexcept;
    std::shared_ptr<ExprNode> parse_interval_literal(bool lower_closed) noexcept;
    std::shared_ptr<ExprNode> parse_open_interval_or_group() noexcept;

    [[nodiscard]] Token& cur() const noexcept;
    [[nodiscard]] Token& peek() const noexcept;
    std::vector<Token>& tokens;
    size_t pos{0};
    size_t frame_count{0};

public:
    explicit Parser() = delete;
    std::shared_ptr<ExprNode> parse_expr() noexcept;
    std::shared_ptr<StmtNode> parse_stmt() noexcept;

    std::shared_ptr<StmtNode> parse_stmt(const std::vector<Token> &tokens) noexcept;

    std::shared_ptr<Module> parse_module(const std::string& name) noexcept;
    explicit Parser(std::vector<Token>& tokens) noexcept;
    void reset(const std::vector<Token>& new_tokens) noexcept;

};

}
