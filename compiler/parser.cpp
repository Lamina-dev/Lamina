//
// Created by meian on 2026/4/3.
//

#include "parser.hpp"
#include "error.hpp"

using namespace lmx;

static std::string cur_module_name;

Parser::Parser(std::vector<Token> &tokens) noexcept : tokens(tokens) {}

#define advance() \
    do {\
    if (pos >= tokens.size()) return nullptr;\
    pos++;\
    } while(0)


void Parser::reset(const std::vector<Token> &new_tokens) noexcept {
    pos = 0;
    frame_count = 0;
    tokens = new_tokens;
}

bool Parser::consume(const TokenType token_type, const std::string& tk) noexcept {
    if (pos >= tokens.size()) return false;
    if (match(token_type)) {
        pos++;
        return true;
    }
    throw_error(ErrorType::Parse, "expected `" + tk + "` but got `" + cur().text + "`", cur().line, cur().col);
    return false;
}

Token &Parser::cur() const noexcept {
    return tokens[pos];
}
Token& Parser::peek() const noexcept {
    return tokens[pos + 1];
}

bool Parser::match(const TokenType t) const noexcept {
    return cur().type == t;
}

bool Parser::peek_match(const TokenType t) const noexcept {
    return peek().type == t;
}
#define PARSER_BINOP(then, last, logic, ...) \
auto line = cur().line, col = cur().col;\
std::shared_ptr<ExprNode> node = last();   \
logic (__VA_ARGS__) {    \
auto op = cur().text;\
advance();\
node = std::make_shared<BinaryNode>(line, col, node, op, then());         \
line = cur().line, col = cur().col;\
}\
return node;

#define PARSER_BINOP_L(last, logic,  ...) PARSER_BINOP(last, last, logic, __VA_ARGS__)

std::shared_ptr<ExprNode> Parser::parse_assign() noexcept {
    PARSER_BINOP(parse_assign, parse_logical, if, cur().type == TokenType::ASSIGN)
}

std::shared_ptr<ExprNode> Parser::parse_logical() noexcept {
    PARSER_BINOP_L(parse_equality, while,
        cur().type == TokenType::KW_OR ||
        cur().type == TokenType::KW_AND
        )
}

std::shared_ptr<ExprNode> Parser::parse_equality() noexcept {
    PARSER_BINOP_L(parse_relational, while,
        cur().type == TokenType::EQ ||
        cur().type == TokenType::NE
        )
}

std::shared_ptr<ExprNode> Parser::parse_relational() noexcept {
    PARSER_BINOP_L(parse_addition, while,
        cur().type == TokenType::GT ||
        cur().type == TokenType::LT ||
        cur().type == TokenType::LE ||
        cur().type == TokenType::GE
        )
}

std::shared_ptr<ExprNode> Parser::parse_addition() noexcept {
    PARSER_BINOP_L(parse_multi, while,
        match(TokenType::OPER_PLUS) ||
        match(TokenType::OPER_MINUS)
        )
}

std::shared_ptr<ExprNode> Parser::parse_multi() noexcept {
    PARSER_BINOP_L(parse_exponent, while,
        cur().type == TokenType::OPER_MUL ||
        cur().type == TokenType::OPER_DIV ||
        cur().type == TokenType::OPER_MOD
        )
}

std::shared_ptr<ExprNode> Parser::parse_exponent() noexcept {
    PARSER_BINOP(parse_exponent, parse_term, if, cur().type == TokenType::OPER_POW)
}

std::shared_ptr<ExprNode> Parser::parse_term() noexcept {
    size_t line = cur().line, col = cur().col;

    switch (std::string op; cur().type) {
    case TokenType::OPER_MINUS:
    // case TokenType::OPER_MUL:
        {
        op = cur().text;
        advance();
        return std::make_shared<UnaryNode>(line, col, op, parse_factor());
    }
    default:return parse_factor();
    }

}

std::shared_ptr<ExprStmtNode> Parser::parse_multi_naming() noexcept {
    size_t line = cur().line, col = cur().col;
    std::shared_ptr<ExprNode> naming = std::make_shared<IdentifierNode>(line, col, cur().text);
    advance();
    while (match(TokenType::COL_COLON)) {
        auto op = cur().text;
        advance();
        naming = std::make_shared<BinaryNode>(line, col, naming, op, std::make_shared<IdentifierNode>(cur().line, cur().col, cur().text));
        advance();
    }
    return std::make_shared<ExprStmtNode>(line, col, naming);
}

std::shared_ptr<ExprStmtNode> Parser::parse_param_name() noexcept {
    size_t line = cur().line, col = cur().col;
    std::shared_ptr<ExprNode> id = std::make_shared<IdentifierNode>(line, col, cur().text);
    advance();
    return std::make_shared<ExprStmtNode>(line, col, id);
}

std::shared_ptr<ExprNode> Parser::parse_factor() noexcept {
    size_t line = cur().line, col = cur().col;
    auto primary = parse_primary();
    while (match(TokenType::DOT) || match(TokenType::LPAREN) || match(TokenType::LBRACK) || match(TokenType::COL_COLON)) {
        switch (cur().type) {
        case TokenType::DOT: case TokenType::COL_COLON: {
            auto op = cur().text;
            advance();
            primary = std::make_shared<BinaryNode>(line, col, primary, op, parse_primary());
            break;
        }
        case TokenType::LPAREN: {
            size_t pline = cur().line, pcol = cur().col;
            advance();
            decltype(ExprsNode::exprs) params;
            if (!match(TokenType::RPAREN)) {
                do {
                    params.push_back(parse_expr());

                    if (match(TokenType::RPAREN)) break;
                } while (consume(TokenType::COMMA, ","));
            }
            consume(TokenType::RPAREN, ")");
            primary = std::make_shared<SuffixParenNode>(line, col, primary, std::make_shared<ExprsNode>(pline, pcol, params));
            break;
        }
        case TokenType::LBRACK: {
            advance();
            auto e = parse_assign();
            consume(TokenType::RBRACK, "]");
            primary = std::make_shared<SuffixBracketNode>(line, col, primary, e);
            break;
        }
        default: {
            // 不会到达这里
            throw_error(ErrorType::Parse, "`" + cur().text + "` is wrong factor token", cur().line, cur().col);
            break;
        }
        }
    }
    return primary;
}
std::shared_ptr<ExprNode> Parser::parse_primary() noexcept {
    size_t line = cur().line, col = cur().col;
    switch (cur().type) {
    case TokenType::NUM_LITERAL: {
        auto num = cur().text;
        advance();
        if (match(TokenType::DOT) && peek_match(TokenType::NUM_LITERAL)) {
            advance();
            num += '.' + cur().text;
            advance();
            return std::make_shared<LiteralNode>(line, col, num, LiteralNode::Kind::Float);
        }
        return std::make_shared<LiteralNode>(line, col, num, LiteralNode::Kind::Integer);
    }
    case TokenType::LPAREN: {
        advance();
        auto e = parse_assign();
        consume(TokenType::RPAREN, ")");
        return e;
    }
    case TokenType::IDENTIFIER: {
        auto id = cur().text;
        advance();
        return std::make_shared<IdentifierNode>(line, col, id);
    }
    case TokenType::KW_IF: {
        advance();
        return parse_if();
    }
    case TokenType::END_OF_FILE: {
        return nullptr;
    }
    default: {
        throw_error(ErrorType::Parse, "`" + cur().text + "` is wrong primary token", cur().line, cur().col);
        if (pos <= tokens.size()) advance();
        return nullptr;
    }
    }
}

std::shared_ptr<ExprNode> Parser::parse_expr() noexcept {
    auto result = parse_assign();
    if (match(TokenType::KW_AS)) {
        advance();
        result->type = parse_type();
    }
    return result;
}

std::shared_ptr<StmtNode> Parser::parse_stmt() noexcept {
    size_t line = cur().line, col = cur().col;
    switch (cur().type) {
    case TokenType::KW_FUNC: {
        advance();
        return parse_func();
    }
    case TokenType::KW_VAR: case TokenType::KW_LET: {
        return std::static_pointer_cast<VarDeclNode>(parse_var());
    }
    default: {
        return std::make_shared<ExprStmtNode>(line, col, parse_expr());
    }
    }
}

std::shared_ptr<StmtNode> Parser::parse_var() noexcept {
    auto decl = std::static_pointer_cast<VarDeclNode>(parse_var_decl());

    consume(TokenType::ASSIGN, "=");

    decl->init_value = parse_expr();
    return decl;
}

std::shared_ptr<StmtNode> Parser::parse_var_decl() noexcept {
    bool is_mutable = false;
    if (match(TokenType::KW_VAR)) is_mutable = true;
    advance();
    size_t line = cur().line, col = cur().col;
    // auto id = parse_multi_naming();
    auto id = cur().text;
    consume(TokenType::IDENTIFIER, "identifier");
    // consume(TokenType::COLON, ":");
    std::shared_ptr<Type> type;
    if (!match(TokenType::ASSIGN)) type = parse_type();
    else type = std::make_shared<UnknownType>();

    return std::make_shared<VarDeclNode>(line, col, id, type, is_mutable);
}

std::shared_ptr<ExprNode> Parser::parse_if() noexcept {
    size_t line = cur().line, col = cur().col;
    auto cond = parse_expr();
    frame_count++;
    auto then = parse_block();
    std::shared_ptr<ASTNode> els = nullptr;
    if (match(TokenType::KW_ELSE)) {
        advance();
        if (match(TokenType::KW_IF)) {
            advance();
            els = parse_if();
        } else els = parse_block();
    }
    frame_count--;
    return std::make_shared<IfExprNode>(line, col, cond, then, els);
}

std::shared_ptr<ExprNode> Parser::parse_block() noexcept {
    size_t line = cur().line, col = cur().col;
    decltype(BlockExprNode::stmts) stmts;

    if (match(TokenType::LBRACE)) {
        advance();
    }

    while (pos < tokens.size() && !match(TokenType::RBRACE)) {
        const size_t last_pos = pos;
        auto stmt = parse_stmt();
        if (stmt == nullptr) {
            break;
        }
        if (pos == last_pos) {
            break;
        }
        stmts.push_back(stmt);
    }
    if (match(TokenType::RBRACE)) {
        advance();
    }
    if (stmts.back()->kind == ASTKind::ExprStmt) {
        const auto expr_stmt = std::static_pointer_cast<ExprStmtNode>(stmts.back());
        stmts.back() = std::make_shared<TailReturnNode>(expr_stmt->line, expr_stmt->col, expr_stmt->expr);
    }
    return std::make_shared<BlockExprNode>(line, col, stmts);
}

std::shared_ptr<StmtNode> Parser::parse_func() noexcept {
    auto line = cur().line, col = cur().col;
    // auto id = parse_multi_naming();
    auto id = cur().text;
    consume(TokenType::IDENTIFIER, "identifier");

    decltype(ParamsDeclNode::stmts) params;

    frame_count++;

    consume(TokenType::LPAREN, "(");
    auto psline = cur().line, pscol = cur().col;
    if (!match(TokenType::RPAREN)) {
        do {
            auto pid = cur().text;
            advance();
            params.emplace_back(pid, parse_type());
            if (match(TokenType::RPAREN)) { break; }
            if (!consume(TokenType::COMMA, ",")) break;
        } while (true);
    }
    consume(TokenType::RPAREN, ")");
    std::shared_ptr<Type> return_type;
    if (match(TokenType::ARROW))
        return_type = parse_type();
    else return_type = std::make_shared<UnknownType>();
    auto body = std::static_pointer_cast<BlockExprNode>(parse_block());
    frame_count--;
    return std::make_shared<FuncImplNode>(
        line, col, id,
        std::make_shared<ParamsDeclNode>(psline, pscol, params),
        return_type, body
        );
}

std::shared_ptr<Type> Parser::parse_type() noexcept {
    switch (cur().type) {
    case TokenType::IDENTIFIER: {
        auto id = cur().text;
        advance();
        static const std::unordered_map<std::string, runtime::ValueKind> basic_types = {
            {"int", runtime::ValueKind::Int}, {"bool", runtime::ValueKind::Bool}
        };
        if (const auto it = basic_types.find(id); it != basic_types.end())
            return std::make_shared<BasicType>(it->second);

        return std::make_shared<NamedType>(id);
    }
    default: {
        throw_error(ErrorType::Parse, "wrong type decl: `" + cur().text + "`", cur().line, cur().col);
        return nullptr;
    }
    }
}

std::shared_ptr<Module> Parser::parse_module(const std::string &name) noexcept {
    const auto save_cur_mod = cur_module_name;
    cur_module_name = name;
    decltype(Module::decls) decls;
    while (pos < tokens.size() && tokens[pos].type != TokenType::END_OF_FILE) {
        decls.push_back(parse_stmt());
    }
    cur_module_name = save_cur_mod;
    return std::make_shared<Module>(name, decls);
}

#undef PARSER_BINOP_R
#undef PARSER_BINOP_L
#undef advance
