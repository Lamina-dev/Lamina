//
// Created by meian on 2026/4/3.
//

#include "parser.hpp"

#include <filesystem>

#include "error.hpp"

namespace lmx {


std::shared_ptr<Module> cur_module = nullptr;

static BinaryNode::Op token_to_binary_op(const TokenType type) {
    switch (type) {
    case TokenType::KW_OR: return BinaryNode::Op::Or;
    case TokenType::KW_AND: return BinaryNode::Op::And;
    case TokenType::KW_IN: return BinaryNode::Op::In;
    case TokenType::DOUBLE_ARROW: return BinaryNode::Op::Bind;
    case TokenType::EQ: return BinaryNode::Op::Eq;
    case TokenType::NE: return BinaryNode::Op::Ne;
    case TokenType::GT: return BinaryNode::Op::Gt;
    case TokenType::LT: return BinaryNode::Op::Lt;
    case TokenType::LE: return BinaryNode::Op::Le;
    case TokenType::GE: return BinaryNode::Op::Ge;
    case TokenType::OPER_PLUS: return BinaryNode::Op::Add;
    case TokenType::OPER_MINUS: return BinaryNode::Op::Sub;
    case TokenType::OPER_MUL: return BinaryNode::Op::Mul;
    case TokenType::OPER_DIV: return BinaryNode::Op::Div;
    case TokenType::OPER_MOD: return BinaryNode::Op::Mod;
    case TokenType::OPER_POW: return BinaryNode::Op::Pow;
        // case TokenType::COL_COLON: return BinaryNode::Op::ColonColon;
    default: std::unreachable();
    }
}

static bool is_primary_start(const TokenType type) noexcept {
    switch (type) {
    case TokenType::NUM_LITERAL:
    case TokenType::STRING_LITERAL:
    case TokenType::TRUE_LITERAL:
    case TokenType::FALSE_LITERAL:
    case TokenType::IDENTIFIER:
    case TokenType::LPAREN:
        return true;
    default:
        return false;
    }
}

Parser::Parser(std::vector<Token> &tokens) noexcept : tokens(tokens) {}

std::shared_ptr<StmtNode> Parser::parse_stmt(const std::vector<Token> &t) noexcept {
    this->reset(t);
    return parse_stmt();
}

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

bool Parser::follows_unit_spec() const noexcept {
    if (!match(TokenType::LT) || pos + 1 >= tokens.size()) return false;
    auto cursor = pos + 1;
    if (tokens[cursor].type == TokenType::NUM_LITERAL) {
        return tokens[cursor].text == "1" && cursor + 1 < tokens.size() &&
               tokens[cursor + 1].type == TokenType::GT;
    }
    while (cursor < tokens.size()) {
        if (tokens[cursor].type != TokenType::IDENTIFIER) return false;
        ++cursor;
        while (cursor < tokens.size() && tokens[cursor].type == TokenType::DOT) {
            ++cursor;
            if (cursor >= tokens.size() ||
                tokens[cursor].type != TokenType::IDENTIFIER) return false;
            ++cursor;
        }
        if (cursor < tokens.size() && tokens[cursor].type == TokenType::OPER_POW) {
            ++cursor;
            if (cursor < tokens.size() &&
                (tokens[cursor].type == TokenType::OPER_PLUS ||
                 tokens[cursor].type == TokenType::OPER_MINUS)) ++cursor;
            if (cursor >= tokens.size() ||
                tokens[cursor].type != TokenType::NUM_LITERAL ||
                tokens[cursor].text.find('.') != std::string::npos) return false;
            ++cursor;
        }
        if (cursor >= tokens.size()) return false;
        if (tokens[cursor].type == TokenType::GT) return true;
        if (tokens[cursor].type != TokenType::OPER_MUL &&
            tokens[cursor].type != TokenType::OPER_DIV) return false;
        ++cursor;
    }
    return false;
}
#define PARSER_BINOP(then, last, logic, ...) \
auto line = cur().line, col = cur().col;\
std::shared_ptr<ExprNode> node = last();   \
logic (__VA_ARGS__) {    \
auto op = token_to_binary_op(cur().type);\
advance();\
node = std::make_shared<BinaryNode>(line, col, node, op, then());         \
line = cur().line, col = cur().col;\
}\
return node;

#define PARSER_BINOP_L(last, logic,  ...) PARSER_BINOP(last, last, logic, __VA_ARGS__)

std::shared_ptr<ExprNode> Parser::parse_pipe() noexcept {
    auto line = cur().line, col = cur().col;
    auto node = parse_logical();
    //printf("|awaa %s\n", match(TokenType::PIPE) ? "true" : "false");
    while (match(TokenType::PIPE)) {
        advance();
        auto rhs = parse_logical();
        node = std::make_shared<PipeExprNode>(line, col, node, rhs);
        line = cur().line, col = cur().col;
    }
    return node;
}

std::shared_ptr<ExprNode> Parser::parse_arrow() noexcept {
    auto line = cur().line, col = cur().col;
    auto node = parse_pipe();
    if (match(TokenType::DOUBLE_ARROW)) {
        auto op = token_to_binary_op(cur().type);
        advance();
        node = std::make_shared<BinaryNode>(line, col, node, op, parse_arrow());
    }
    return node;
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
    auto line = cur().line, col = cur().col;
    auto node = parse_addition();
    while (cur().type == TokenType::GT ||
           cur().type == TokenType::LT ||
           cur().type == TokenType::LE ||
           cur().type == TokenType::GE ||
           cur().type == TokenType::KW_IN ||
           (cur().type == TokenType::NOT && peek_match(TokenType::KW_IN))) {
        BinaryNode::Op op;
        if (cur().type == TokenType::NOT) {
            advance();
            op = BinaryNode::Op::NotIn;
        } else {
            op = token_to_binary_op(cur().type);
        }
        advance();
        node = std::make_shared<BinaryNode>(line, col, node, op, parse_addition());
        line = cur().line, col = cur().col;
    }
    return node;
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

    switch (cur().type) {
    case TokenType::OPER_MINUS:
        // case TokenType::OPER_MUL:
    {
        advance();
        return std::make_shared<UnaryNode>(line, col, UnaryNode::Op::Neg, parse_factor());
    }
    default:return parse_factor();
    }

}

// std::shared_ptr<ExprStmtNode> Parser::parse_multi_naming() noexcept {
//     size_t line = cur().line, col = cur().col;
//     std::shared_ptr<ExprNode> naming = std::make_shared<IdentifierNode>(line, col, cur().text);
//     advance();
//     while (match(TokenType::COL_COLON)) {
//         advance();
//         naming = std::make_shared<BinaryNode>(line, col, naming, BinaryNode::Op::ColonColon, std::make_shared<IdentifierNode>(cur().line, cur().col, cur().text));
//         advance();
//     }
//     return std::make_shared<ExprStmtNode>(line, col, naming);
// }

std::shared_ptr<ExprStmtNode> Parser::parse_param_name() noexcept {
    size_t line = cur().line, col = cur().col;
    std::shared_ptr<ExprNode> id = std::make_shared<IdentifierNode>(line, col, cur().text);
    advance();
    return std::make_shared<ExprStmtNode>(line, col, id);
}

std::shared_ptr<ExprNode> Parser::parse_factor() noexcept {
    size_t line = cur().line, col = cur().col;
    if (match(TokenType::NOT)) {
        advance();
        return std::make_shared<UnaryNode>(line, col, UnaryNode::Op::Not, parse_factor());
    }
    return parse_primary();
}

std::shared_ptr<ExprNode> Parser::parse_parenthesized_expression() noexcept {
    const size_t line = cur().line, col = cur().col;
    consume(TokenType::LPAREN, "(");
    auto first = parse_expr();
    if (!match(TokenType::COMMA)) {
        consume(TokenType::RPAREN, ")");
        return first;
    }

    std::vector<std::shared_ptr<ExprNode>> elements;
    elements.push_back(std::move(first));
    while (match(TokenType::COMMA)) {
        advance();
        if (match(TokenType::RPAREN)) {
            throw_error(ErrorType::Parse,
                        "TupleArityMismatch: tuple literals require at least two elements",
                        line, col);
            break;
        }
        elements.push_back(parse_expr());
    }
    consume(TokenType::RPAREN, ")");
    return std::make_shared<TupleLiteralNode>(line, col, std::move(elements));
}

std::shared_ptr<ExprNode> Parser::parse_set_literal() noexcept {
    const size_t line = cur().line, col = cur().col;
    consume(TokenType::LBRACE, "{");
    std::vector<std::shared_ptr<ExprNode>> elements;
    if (!match(TokenType::RBRACE)) {
        while (true) {
            elements.push_back(parse_expr());
            if (match(TokenType::RBRACE)) break;
            consume(TokenType::COMMA, ",");
        }
    }
    consume(TokenType::RBRACE, "}");
    return std::make_shared<LiteralPayloadNode>(
        line, col, LiteralPayloadNode::Kind::Set, std::move(elements));
}

std::shared_ptr<ExprNode> Parser::parse_primary() noexcept {
    size_t line = cur().line, col = cur().col;
    std::shared_ptr<ExprNode> primary;
    switch (cur().type) {
    case TokenType::NUM_LITERAL: {
        auto num = cur().text;
        const auto number_end_col = cur().col + cur().text.size();
        advance();
        if (match(TokenType::DOT) && peek_match(TokenType::NUM_LITERAL)) {
            advance();
            num += '.' + cur().text;
            const auto decimal_end_col = cur().col + cur().text.size();
            advance();
            primary = std::make_shared<LiteralNode>(line, col, num, LiteralNode::Kind::Float);
            if (match(TokenType::IDENTIFIER) && cur().text == "I" &&
                cur().line == line && cur().col == decimal_end_col) {
                const auto imaginary = std::make_shared<IdentifierNode>(cur().line, cur().col, "I");
                advance();
                primary = std::make_shared<BinaryNode>(line, col, primary,
                                                       BinaryNode::Op::Mul, imaginary);
            } else if (match(TokenType::LT) && cur().line == line &&
                       cur().col == decimal_end_col && follows_unit_spec()) {
                primary = std::make_shared<UnitAnnotatedExprNode>(
                    line, col, primary, parse_unit_spec());
            }
            break;
        }
        primary = std::make_shared<LiteralNode>(line, col, num, LiteralNode::Kind::Integer);
        if (match(TokenType::IDENTIFIER) && cur().text == "I" &&
            cur().line == line && cur().col == number_end_col) {
            const auto imaginary = std::make_shared<IdentifierNode>(cur().line, cur().col, "I");
            advance();
            primary = std::make_shared<BinaryNode>(line, col, primary,
                                                   BinaryNode::Op::Mul, imaginary);
        } else if (match(TokenType::LT) && cur().line == line &&
                   cur().col == number_end_col && follows_unit_spec()) {
            primary = std::make_shared<UnitAnnotatedExprNode>(
                line, col, primary, parse_unit_spec());
        }
        break;
    }
    case TokenType::NULL_LITERAL: {
        advance();
        primary = std::make_shared<LiteralNode>(line, col, "null", LiteralNode::Kind::Null);
        break;
    }
    case TokenType::STRING_LITERAL: {
        auto str = cur().text;
        advance();
        primary = std::make_shared<LiteralNode>(line, col, str, LiteralNode::Kind::String);
        break;
    }
    case TokenType::LPAREN: {
        primary = parse_parenthesized_expression();
        break;
    }
    case TokenType::IDENTIFIER: {
        auto id = cur().text;
        advance();
        primary = std::make_shared<IdentifierNode>(line, col, id);
        break;
    }
    case TokenType::TRUE_LITERAL:
    case TokenType::FALSE_LITERAL: {
        auto id = cur().text;
        advance();
        primary = std::make_shared<LiteralNode>(line, col, id, LiteralNode::Kind::Boolean);
        break;
    }
    case TokenType::KW_IF: {
        advance();
        primary = parse_if();
        break;
    }
    case TokenType::KW_MATCH: {
        primary = parse_match();
        break;
    }
    case TokenType::END_OF_FILE: {
        primary = nullptr;
        break;
    }
    case TokenType::LBRACK: {
        advance();
        std::vector<std::shared_ptr<ExprNode>> elements;
        if (!match(TokenType::RBRACK)) {
            while (true) {
                elements.push_back(parse_expr());
                if (match(TokenType::RBRACK)) break;
                consume(TokenType::COMMA, ",");
                if (match(TokenType::RBRACK)) break;
            }
        }
        consume(TokenType::RBRACK, "]");
        primary = std::make_shared<ArrayLiteralNode>(line, col, std::move(elements));
        break;
    }
    case TokenType::LBRACE: {
        size_t scan = pos + 1;
        size_t brace_depth = 1;
        size_t paren_depth = 0;
        size_t bracket_depth = 0;
        bool has_comma = false;
        while (scan < tokens.size() && brace_depth > 0) {
            if (tokens[scan].type == TokenType::LBRACE) brace_depth++;
            else if (tokens[scan].type == TokenType::RBRACE) brace_depth--;
            else if (tokens[scan].type == TokenType::LPAREN) paren_depth++;
            else if (tokens[scan].type == TokenType::RPAREN && paren_depth > 0) paren_depth--;
            else if (tokens[scan].type == TokenType::LBRACK) bracket_depth++;
            else if (tokens[scan].type == TokenType::RBRACK && bracket_depth > 0) bracket_depth--;
            else if (brace_depth == 1 && paren_depth == 0 && bracket_depth == 0 &&
                     tokens[scan].type == TokenType::COMMA) {
                has_comma = true;
                break;
            }
            scan++;
        }
        if (has_comma) {
            primary = parse_set_literal();
            break;
        }
        primary = parse_block();
        break;
    }
    default: {
        throw_error(ErrorType::Parse, "`" + cur().text + "` is wrong primary token", cur().line, cur().col);
        if (pos <= tokens.size()) advance();
        return nullptr;
    }
    }

    while (match(TokenType::LPAREN) || match(TokenType::LBRACK) || match(TokenType::DOT)) {
        switch (cur().type) {
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
            auto e = parse_expr();
            consume(TokenType::RBRACK, "]");
            primary = std::make_shared<SuffixBracketNode>(line, col, primary, e);
            break;
        }
        case TokenType::DOT: {
            advance();
            auto ident = cur();
            if (match(TokenType::NUM_LITERAL)) {
                unsigned long position = 0;
                try {
                    position = std::stoul(ident.text);
                } catch (...) {
                    position = 0;
                }
                advance();
                if (position == 0 || position > 256) {
                    throw_error(ErrorType::Parse,
                                "TupleIndexOutOfBounds: tuple positions start at 1",
                                ident.line, ident.col);
                    position = 1;
                }
                primary = std::make_shared<TupleGetExprNode>(
                    line, col, primary, static_cast<uint8_t>(position - 1));
            } else {
                consume(TokenType::IDENTIFIER, "identifier");
                primary = std::make_shared<DotExprNode>(line, col, primary, std::make_shared<IdentifierNode>(ident.line, ident.col, ident.text));
            }
            break;
        }
        default: {
            // 不会到达这里
            throw_error(ErrorType::Parse, "`" + cur().text + "` is wrong primary token", cur().line, cur().col);
            break;
        }
        }
    }
    if (cur().line == line && is_primary_start(cur().type)) {
        if (primary && primary->kind == ASTKind::Literal &&
            match(TokenType::IDENTIFIER) && cur().text == "I") {
            throw_error(ErrorType::Parse,
                        "ComplexLiteralSpacing: `I` must immediately follow the numeric literal",
                        cur().line, cur().col);
        } else {
            throw_error(ErrorType::Parse,
                        "implicit multiplication is not supported; insert `*`",
                        cur().line, cur().col);
        }
    }
    if (cur().line == line && cur().type == TokenType::UNKNOWN && cur().text == "**") {
        throw_error(ErrorType::Parse,
                    "operator `**` is not supported; use `^`",
                    cur().line, cur().col);
        advance();
    }
    return primary;
}

std::shared_ptr<ExprNode> Parser::parse_expr() noexcept {
    auto line = cur().line, col = cur().col;
    auto result = parse_arrow();
    if (match(TokenType::KW_AS)) {
        advance();
        if (match(TokenType::LT)) {
            result = std::make_shared<AsExprNode>(
                line, col, result, AsExprNode::Kind::Unit, parse_unit_spec());
        } else if (match(TokenType::IDENTIFIER) && cur().text == "num") {
            advance();
            if (match(TokenType::LT)) {
                throw_error(ErrorType::Parse,
                            "UnitStripLegacySyntax: use `as num` without a unit argument",
                            cur().line, cur().col);
                (void)parse_unit_spec();
            }
            result = std::make_shared<AsExprNode>(
                line, col, result, AsExprNode::Kind::Num);
        } else if (match(TokenType::IDENTIFIER) && cur().text == "scalar") {
            advance();
            result = std::make_shared<AsExprNode>(
                line, col, result, AsExprNode::Kind::Scalar);
        } else {
            auto cast_type = parse_type();
            result = std::make_shared<AsExprNode>(line, col, result, cast_type);
        }
    }
    return result;
}

std::shared_ptr<StmtNode> Parser::parse_stmt() noexcept {
    re_parse:

    switch (cur().type) {
    case TokenType::KW_FUNC: {
        advance();
        return parse_func();
    }
    case TokenType::KW_TYPE: {
        return parse_type_decl();
    }
    case TokenType::KW_UNIT: {
        advance();
        return parse_unit_decl();
    }
    case TokenType::KW_VAR: case TokenType::KW_LET: {
        return std::static_pointer_cast<VarDeclNode>(parse_var());
    }
    case TokenType::KW_RETURN: {
        // advance();
        return parse_return();
    }
    case TokenType::KW_LOOP: {
        advance();
        return parse_loop();
    }
    case TokenType::KW_STATIC: {
        advance();
        if (!cur_module->lib_name.empty()) {
            throw_error(ErrorType::Parse, "current module `" + cur_module->name + "` lib_name redefined", cur().line, cur().col);
        }
        const auto name = cur().text;
        consume(TokenType::STRING_LITERAL, "text");
        cur_module->lib_name = name;
        goto re_parse;
    }
    case TokenType::KW_BREAK: {
        size_t line = cur().line, col = cur().col;
        advance();
        return std::make_shared<BreakStmtNode>(line, col);
    }
    case TokenType::KW_CONTINUE: {
        size_t line = cur().line, col = cur().col;
        advance();
        return std::make_shared<ContinueStmtNode>(line, col);
    }
    case TokenType::KW_IMPORT: {
        advance();
        return parse_import();
    }
    case TokenType::KW_SYM: {
        const auto line = cur().line, col = cur().col;
        advance();
        std::vector<std::string> ids;
        do {
            auto name = cur().text;
            consume(TokenType::IDENTIFIER, "identifier");
            ids.push_back(std::move(name));
            if (!match(TokenType::COMMA)) break;
            advance();
        } while (true);
        return std::make_shared<SymDeclNode>(line, col, std::move(ids));
    }
    default: {
        size_t line = cur().line, col = cur().col;
        auto expr = parse_expr();
        if (!expr) return nullptr;
        if (match(TokenType::ASSIGN)) {
            advance();
            auto rhs = parse_expr();
            return std::make_shared<AssignStmtNode>(line, col, expr, rhs);
        }
        return std::make_shared<ExprStmtNode>(line, col, expr);
    }
    }
}

std::shared_ptr<StmtNode> Parser::parse_type_decl() noexcept {
    const auto line = cur().line, col = cur().col;
    consume(TokenType::KW_TYPE, "type");
    auto name = cur().text;
    consume(TokenType::IDENTIFIER, "type name");

    std::vector<std::string> type_params;
    if (match(TokenType::LT)) {
        advance();
        do {
            type_params.push_back(cur().text);
            consume(TokenType::IDENTIFIER, "type parameter");
            if (match(TokenType::GT)) break;
            consume(TokenType::COMMA, ",");
        } while (true);
        consume(TokenType::GT, ">");
    }
    consume(TokenType::ASSIGN, "=");

    std::vector<AdtConstructorDecl> constructors;
    do {
        if (match(TokenType::BAR)) advance();
        AdtConstructorDecl constructor;
        constructor.name = cur().text;
        consume(TokenType::IDENTIFIER, "constructor name");
        if (match(TokenType::LPAREN)) {
            advance();
            if (!match(TokenType::RPAREN)) {
                do {
                    constructor.fields.push_back(parse_type());
                    if (match(TokenType::RPAREN)) break;
                    consume(TokenType::COMMA, ",");
                } while (true);
            }
            consume(TokenType::RPAREN, ")");
        }
        constructors.push_back(std::move(constructor));
    } while (match(TokenType::BAR));

    return std::make_shared<TypeDeclNode>(line, col, std::move(name),
                                          std::move(type_params), std::move(constructors));
}

Pattern Parser::parse_pattern() noexcept {
    const auto line = cur().line, col = cur().col;
    if (match(TokenType::OPER_MINUS) && peek_match(TokenType::NUM_LITERAL)) {
        pos++;
        auto value = "-" + cur().text;
        pos++;
        auto literal_kind = LiteralNode::Kind::Integer;
        if (match(TokenType::DOT) && peek_match(TokenType::NUM_LITERAL)) {
            pos++;
            value += "." + cur().text;
            pos++;
            literal_kind = LiteralNode::Kind::Float;
        }
        Pattern pattern(Pattern::Kind::Literal, line, col);
        pattern.literal = std::make_shared<LiteralNode>(line, col, std::move(value), literal_kind);
        return pattern;
    }
    if (match(TokenType::NUM_LITERAL) || match(TokenType::STRING_LITERAL) ||
        match(TokenType::TRUE_LITERAL) || match(TokenType::FALSE_LITERAL) || match(TokenType::NULL_LITERAL)) {
        auto literal_kind = LiteralNode::Kind::Integer;
        auto value = cur().text;
        if (match(TokenType::NULL_LITERAL)) {
            literal_kind = LiteralNode::Kind::Null;
            pos++;
        } else if (match(TokenType::STRING_LITERAL)) {
            literal_kind = LiteralNode::Kind::String;
            pos++;
        } else if (match(TokenType::TRUE_LITERAL) || match(TokenType::FALSE_LITERAL)) {
            literal_kind = LiteralNode::Kind::Boolean;
            pos++;
        } else {
            pos++;
            if (match(TokenType::DOT) && peek_match(TokenType::NUM_LITERAL)) {
                pos++;
                value += "." + cur().text;
                pos++;
                literal_kind = LiteralNode::Kind::Float;
            }
        }
        Pattern pattern(Pattern::Kind::Literal, line, col);
        pattern.literal = std::make_shared<LiteralNode>(line, col, std::move(value), literal_kind);
        return pattern;
    }

    auto name = cur().text;
    consume(TokenType::IDENTIFIER, "pattern");
    if (match(TokenType::DOT)) {
        pos++;
        const auto constructor = cur().text;
        consume(TokenType::IDENTIFIER, "constructor name");
        Pattern pattern(Pattern::Kind::Constructor, line, col, constructor);
        pattern.adt_type_name = std::move(name);
        if (match(TokenType::LPAREN)) {
            pos++;
            if (!match(TokenType::RPAREN)) {
                do {
                    pattern.fields.push_back(parse_pattern());
                    if (match(TokenType::RPAREN)) break;
                    consume(TokenType::COMMA, ",");
                } while (true);
            }
            consume(TokenType::RPAREN, ")");
        }
        return pattern;
    }
    if (name == "_") return {Pattern::Kind::Wildcard, line, col};
    if (!match(TokenType::LPAREN)) return {Pattern::Kind::Binding, line, col, std::move(name)};

    Pattern pattern(Pattern::Kind::Constructor, line, col, std::move(name));
    pos++;
    if (!match(TokenType::RPAREN)) {
        do {
            pattern.fields.push_back(parse_pattern());
            if (match(TokenType::RPAREN)) break;
            consume(TokenType::COMMA, ",");
        } while (true);
    }
    consume(TokenType::RPAREN, ")");
    return pattern;
}

std::shared_ptr<ExprNode> Parser::parse_match() noexcept {
    const auto line = cur().line, col = cur().col;
    consume(TokenType::KW_MATCH, "match");
    auto target = parse_expr();
    consume(TokenType::LBRACE, "{");
    std::vector<MatchArm> arms;
    while (!match(TokenType::RBRACE) && !match(TokenType::END_OF_FILE)) {
        auto pattern = parse_pattern();
        std::shared_ptr<ExprNode> guard;
        if (match(TokenType::KW_IF)) {
            advance();
            guard = parse_pipe();
        }
        consume(TokenType::DOUBLE_ARROW, "=>");
        auto value = parse_expr();
        arms.push_back({std::move(pattern), std::move(guard), std::move(value)});
        if (match(TokenType::COMMA)) advance();
    }
    consume(TokenType::RBRACE, "}");
    return std::make_shared<MatchExprNode>(line, col, std::move(target), std::move(arms));
}

std::shared_ptr<StmtNode> Parser::parse_import() noexcept {
    auto line = cur().line, col = cur().col;
    auto name = cur().text;
    consume(TokenType::IDENTIFIER, "identifier");
    while (pos < tokens.size() && match(TokenType::DOT)) {
        advance();
        name += '.' + cur().text;
        consume(TokenType::IDENTIFIER, "identifier");
    }
    return std::make_shared<ImportStmtNode>(line, col, name);
}

std::shared_ptr<StmtNode> Parser::parse_loop() noexcept {
    const auto line = cur().line, col = cur().col;

    std::shared_ptr<ExprNode> expr = nullptr;
    if (!match(TokenType::LBRACE)) {
        expr = parse_expr();
    }
    decltype(LoopStmtNode::body) body;
    consume(TokenType::LBRACE, "{");
    while (pos < tokens.size() && !match(TokenType::RBRACE)) {
        const size_t last_pos = pos;
        auto stmt = parse_stmt();
        if (stmt == nullptr) {
            break;
        }
        if (pos == last_pos) {
            break;
        }
        body.push_back(stmt);
    }
    consume(TokenType::RBRACE, "}");
    return std::make_shared<LoopStmtNode>(line, col, expr, body);
}

std::shared_ptr<StmtNode> Parser::parse_return() noexcept {
    const auto old_line = cur().line;
    // auto old_col = cur().col;
    advance();
    std::shared_ptr<ExprNode> expr = nullptr;
    auto line = cur().line, col = cur().col;
    if (old_line == line) {
        expr = parse_expr();
    }
    return std::make_shared<ReturnNode>(line, col,  expr);
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
    else type = type_pool.unknown();

    return std::make_shared<VarDeclNode>(line, col, id, type, is_mutable);
}

std::shared_ptr<ExprNode> Parser::parse_if() noexcept {
    size_t line = cur().line, col = cur().col;
    auto cond = parse_expr();
    frame_count++;
    auto then = parse_block();
    std::shared_ptr<ExprNode> els = nullptr;
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
            if (cur().type == TokenType::DOT_DOT_DOT) {
                advance();
                params.emplace_back("#", type_pool.basic(runtime::ValueKind::C_VaList));
            } else {
                advance();
                params.emplace_back(pid, parse_type());
            }
            if (match(TokenType::RPAREN)) { break; }
            if (!consume(TokenType::COMMA, ",")) break;
        } while (true);
    }
    consume(TokenType::RPAREN, ")");
    std::shared_ptr<Type> return_type;
    if (match(TokenType::ARROW)) {
        advance();
        return_type = parse_type();
    }
    else return_type = type_pool.unknown();
    if (match(TokenType::ASSIGN)) {
        if (Type::is_null_type(return_type.get())) {
            throw_error(ErrorType::Parse, "native function declare must be have return_type declare", line, col);
        }
        advance();
        auto sym = cur().text;
        consume(TokenType::STRING_LITERAL, "lib symbol");
        cur_module->native_funcs.push_back(std::make_shared<NativeFuncDeclNode>(
        line, col, id,
        std::make_shared<ParamsDeclNode>(psline, pscol, params),
        return_type, sym
        ));
        return parse_stmt();
    }
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
    case TokenType::NULL_LITERAL:
        advance();
        return type_pool.basic(runtime::ValueKind::Null);
    case TokenType::IDENTIFIER: {
        auto id = cur().text;
        advance();
        if (id == "num") {
            if (match(TokenType::LT)) return type_pool.dimensioned(parse_unit_spec());
            return type_pool.basic(runtime::ValueKind::Fraction);
        }
        while (match(TokenType::DOT)) {
            advance();
            id += "." + cur().text;
            consume(TokenType::IDENTIFIER, "type name");
        }
        static const std::unordered_map<std::string, runtime::ValueKind> basic_types = {
            {"int", runtime::ValueKind::Int}, {"bool", runtime::ValueKind::Bool},
            {"null", runtime::ValueKind::Null}, {"frac", runtime::ValueKind::Fraction},
            {"real", runtime::ValueKind::Real}, {"expr", runtime::ValueKind::Expr},
            {"complex", runtime::ValueKind::Complex},
            {"vector", runtime::ValueKind::Vector},
            {"matrix", runtime::ValueKind::Matrix},
            {"table", runtime::ValueKind::Table},
            {"rng", runtime::ValueKind::Random},
            {"quantity", runtime::ValueKind::Quantity},
            {"cptr", runtime::ValueKind::C_Ptr}
        };
        std::shared_ptr<Type> type;
        if (const auto it = basic_types.find(id); it != basic_types.end()) {
            type = type_pool.basic(it->second);
        } else if (id == "text") {
            type = type_pool.string();
        }
        if (type) {
            if (match(TokenType::QUESTION)) {
                advance();
                return type_pool.nullable(std::move(type));
            }
            return type;
        }
        std::vector<std::shared_ptr<Type>> args;
        if (match(TokenType::LT)) {
            advance();
            do {
                args.push_back(parse_type());
                if (match(TokenType::GT)) break;
                consume(TokenType::COMMA, ",");
            } while (true);
            consume(TokenType::GT, ">");
        }
        if (id == "array") {
            if (args.size() != 1) {
                throw_error(ErrorType::Parse, "array type requires exactly one element type", cur().line, cur().col);
                return type_pool.array(type_pool.unknown());
            }
            auto array_type = type_pool.array(std::move(args.front()));
            if (match(TokenType::QUESTION)) {
                advance();
                return type_pool.nullable(std::move(array_type));
            }
            return array_type;
        }
        auto named_type = type_pool.named(id, std::move(args));
        if (match(TokenType::QUESTION)) {
            advance();
            return type_pool.nullable(std::move(named_type));
        }
        return named_type;
    }
    case TokenType::LPAREN: {
        const auto line = cur().line;
        const auto col = cur().col;
        advance();
        std::vector<std::shared_ptr<Type>> elements;
        elements.push_back(parse_type());
        if (!match(TokenType::COMMA)) {
            consume(TokenType::RPAREN, ")");
            throw_error(ErrorType::Parse,
                        "TupleArityMismatch: tuple types require at least two elements",
                        line, col);
            return type_pool.unknown();
        }
        while (match(TokenType::COMMA)) {
            advance();
            if (match(TokenType::RPAREN)) break;
            elements.push_back(parse_type());
        }
        consume(TokenType::RPAREN, ")");
        if (elements.size() < 2) {
            throw_error(ErrorType::Parse,
                        "TupleArityMismatch: tuple types require at least two elements",
                        line, col);
            return type_pool.unknown();
        }
        auto tuple_type = type_pool.tuple(std::move(elements));
        if (match(TokenType::QUESTION)) {
            advance();
            return type_pool.nullable(std::move(tuple_type));
        }
        return tuple_type;
    }
    default: {
        throw_error(ErrorType::Parse, "wrong type decl: `" + cur().text + "`", cur().line, cur().col);
        return nullptr;
    }
    }
}

UnitSpec Parser::parse_unit_spec() noexcept {
    UnitSpec result;
    consume(TokenType::LT, "<");
    if (match(TokenType::NUM_LITERAL) && cur().text == "1") {
        ++pos;
        consume(TokenType::GT, ">");
        return result;
    }
    int operation_sign = 1;
    while (pos < tokens.size() && !match(TokenType::GT) &&
           !match(TokenType::END_OF_FILE)) {
        const auto factor_line = cur().line;
        const auto factor_col = cur().col;
        std::string name = cur().text;
        consume(TokenType::IDENTIFIER, "unit name");
        while (match(TokenType::DOT)) {
            ++pos;
            name += "." + cur().text;
            consume(TokenType::IDENTIFIER, "unit name");
        }
        int exponent = 1;
        if (match(TokenType::OPER_POW)) {
            ++pos;
            int sign = 1;
            if (match(TokenType::OPER_MINUS) || match(TokenType::OPER_PLUS)) {
                if (match(TokenType::OPER_MINUS)) sign = -1;
                ++pos;
            }
            if (!match(TokenType::NUM_LITERAL) || cur().text.find('.') != std::string::npos) {
                throw_error(ErrorType::Parse, "unit exponent must be an integer", factor_line, factor_col);
                exponent = 0;
            } else {
                try {
                    exponent = sign * std::stoi(cur().text);
                } catch (...) {
                    throw_error(ErrorType::Parse, "unit exponent is out of range", factor_line, factor_col);
                    exponent = 0;
                }
                ++pos;
            }
        }
        result.factors.push_back({std::move(name), exponent * operation_sign});
        if (match(TokenType::GT)) break;
        if (match(TokenType::OPER_MUL)) operation_sign = 1;
        else if (match(TokenType::OPER_DIV)) operation_sign = -1;
        else {
            throw_error(ErrorType::Parse, "expected `*`, `/`, or `>` in unit expression",
                        cur().line, cur().col);
            break;
        }
        ++pos;
    }
    consume(TokenType::GT, ">");
    return result;
}

std::shared_ptr<StmtNode> Parser::parse_unit_decl() noexcept {
    const auto line = cur().line;
    const auto col = cur().col;
    const auto name = cur().text;
    consume(TokenType::IDENTIFIER, "unit name");
    std::shared_ptr<ExprNode> definition;
    if (match(TokenType::ASSIGN)) {
        advance();
        definition = parse_expr();
    }
    return std::make_shared<UnitDeclNode>(line, col, name, std::move(definition));
}

std::shared_ptr<Module> Parser::parse_module(const std::string &name) noexcept {
    const auto save_cur_mod = cur_module;
    cur_module = std::make_shared<Module>(name, decltype(Module::decls){});
    while (pos < tokens.size() && tokens[pos].type != TokenType::END_OF_FILE) {
        auto stmt = parse_stmt();
        if (stmt) cur_module->decls.push_back(std::move(stmt));
    }
    auto result = cur_module;
    cur_module = save_cur_mod;
    return result;
}


}
#undef PARSER_BINOP_R
#undef PARSER_BINOP_L
#undef advance
