#pragma once
#include <string>
#include <utility>
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


enum class LexerTokenType {
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
    Loop,    // loop
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
    Lambda,
    LBracket, // [
    RBracket, // ]
    Comma,
    Dot,// 新增
    TripleDot, // ...
    String,
    Semicolon, // ;
    ExclamationMark, // !
    Plus,
    Minus,
    Star,
    Slash,       // /
    Backslash,   // \;
    Percent,     // %
    Caret,       // ^
    Bang,        // #
    Equal,       // ==
    NotEqual,    // !=
    Less,        // <
    LessEqual,   // <=
    Greater,     // >
    GreaterEqual,// >=
    DoubleColon, // ::
    Pipe,        // |
    FatArrow,    // =>
    ThinArrow,   // ->
	LogicalAnd,
	LogicalOr,
    EndOfFile,
    EndOfLine,
    Unknown
};

struct Token {
    LexerTokenType type;
    std::string text;
    size_t line;
    size_t column;
    Token(const LexerTokenType t, std::string  txt, const int l, const int c)
        : type(t), text(std::move(txt)), line(l), column(c) {}
};

class LAMINA_API Lexer {
public:
    static std::vector<Token> tokenize(const std::string& src);
};
