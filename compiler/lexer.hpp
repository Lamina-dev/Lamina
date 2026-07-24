//
// Created by geguj on 2025/12/28.
//

#pragma once
#include <string>
#include <utility>
#include <vector>

#include "lmx.h"

#include <format>

namespace lmx {
enum class TokenType {
    END_OF_FILE,
    OPER_PLUS, OPER_MINUS, OPER_MUL, OPER_DIV, OPER_MOD, OPER_POW,

    ASSIGN, COLON, COL_COLON, COMMA, NOT,

    LPAREN, RPAREN, LBRACK, RBRACK, LBRACE, RBRACE,

    EQ, NE, LT, GT, LE, GE, PIPE, ARROW, DOUBLE_ARROW,

    NUM_LITERAL, STRING_LITERAL, TRUE_LITERAL, FALSE_LITERAL, IDENTIFIER,

    KW_FUNC, KW_RETURN,
    UNKNOWN, KW_IF, KW_ELSE, KW_LET,
    KW_MODULE,
    KW_USE,
    DOT,
    KW_LOOP,
    KW_BREAK,
    KW_CONTINUE,
    COMMENT,
    KW_AS, KW_VAR, KW_AND, KW_OR,
    KW_WHILE,
    KW_FOR,
    KW_IN,
    KW_CONST,
    KW_UNIT,
    KW_IMPORT,
    KW_SYM,
};

inline std::string to_string(const TokenType& type) {
    switch (type) {
        case TokenType::END_OF_FILE: return "END_OF_FILE";
        case TokenType::IDENTIFIER: return "IDENTIFIER";
        case TokenType::NUM_LITERAL: return "INT_LITERAL";
        case TokenType::STRING_LITERAL: return "STRING_LITERAL";
        case TokenType::COMMA: return "COMMA";
        case TokenType::TRUE_LITERAL: return "TRUE_LITERAL";
        case TokenType::FALSE_LITERAL: return "FALSE_LITERAL";
        case TokenType::OPER_PLUS: return "OPER_PLUS";
        case TokenType::OPER_MINUS: return "OPER_MINUS";
        case TokenType::OPER_MUL: return "OPER_MUL";
        case TokenType::OPER_DIV: return "OPER_DIV";
        case TokenType::OPER_MOD: return "OPER_MOD";
        case TokenType::EQ: return "EQ";
        case TokenType::GE: return "GE";
        case TokenType::GT: return "GT";
        case TokenType::LE: return "LE";
        case TokenType::LT: return "LT";
        case TokenType::COLON: return "COLON";
        case TokenType::COL_COLON: return "COL_COLON";
        case TokenType::OPER_POW: return "OPER_POW";
        case TokenType::ASSIGN: return "ASSIGN";
        case TokenType::NOT: return "NOT";
        case TokenType::NE: return "NE";
        case TokenType::LPAREN: return "LPAREN";
        case TokenType::RPAREN: return "RPAREN";
        case TokenType::LBRACK: return "LBRACK";
        case TokenType::RBRACK: return "RBRACK";
        case TokenType::LBRACE: return "LBRACE";
        case TokenType::RBRACE: return "RBRACE";
        case TokenType::UNKNOWN: return "UNKNOWN";
        case TokenType::KW_FUNC: return "KEYWORD_FUNC";
        case TokenType::KW_RETURN: return "KEYWORD_RETURN";
        default: return "_NOT_IMPLEMENTED";
    }
}

struct LM_API Token {
    TokenType type;
    std::string text;
    size_t line, col;

    friend std::ostream& operator<<(std::ostream& os, const Token& t);
};

class LM_API Lexer {
    size_t pos{0}, line{1}, col{1};
    std::string& content, filename;

    Token next();
    void advance();
    [[nodiscard]] bool valid_pos() const;

public:
    explicit Lexer(std::string& code, std::string filename = "<unknown>"): content(code), filename(std::move(filename)) {}
    // explicit Lexer(const char* code, std::string filename = "<unknown>"): content(code), filename(std::move(filename)) {}
    std::string error(const std::vector<Token>& tokens, size_t origin_lineno);
    std::vector<Token> tokenize(const std::string &code);

    bool has_err{false};
};

}

template<>
struct std::formatter<lmx::Token> {
    static constexpr auto parse(const format_parse_context& ctx) {
        return ctx.begin();
    }

    static auto format(const lmx::Token& token, format_context& ctx) {
        return format_to(
            ctx.out(),
            "Token({}, '{}', {}, {})",
            to_string(token.type),
            token.text,
            token.line,
            token.col
        );
    }
};