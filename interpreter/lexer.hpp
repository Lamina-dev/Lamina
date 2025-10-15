#pragma once
#include <string>
#include <vector>

#ifdef _WIN32
#ifdef LAMINA_CORE_EXPORTS
#define LAMINA_API __declspec(dllexport)
#else
#define LAMINA_API __declspec(dllimport)
#endif
#else
#define LAMINA_API
#endif

enum class TokenType {
    Var,
    Func,
    If,
    Else,
    While,
    For,
    Return,
    Include, // 新增
    Break,   // 新增
    Continue,// 新增
    Struct,  // 新增
    Define,  // define
    Bigint,  // bigint
    True,    // true
    False,   // false
    Null,    // null
    Input,   // input
    Identifier,
    Assign,
    Number,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,// [
    RBracket,// ]
    Comma,
    Dot,// 新增
    String,
    Semicolon,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    Bang,
    Equal,       // ==
    NotEqual,    // !=
    Less,        // <
    LessEqual,   // <=
    Greater,     // >
    GreaterEqual,// >=
    At,          // @
    EndOfFile,
    Unknown
};

struct Token {
    TokenType type;
    std::string text;
    size_t line;
    size_t column;
    Token(TokenType t, const std::string& txt, int l, int c)
        : type(t), text(txt), line(l), column(c) {}
};

class LAMINA_API Lexer {
public:
    static std::vector<Token> tokenize(const std::string& src);
};
