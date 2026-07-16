#include "lexer.hpp"

#include <algorithm>
#include <format>
#include <unordered_map>
#include <iostream>
#include <sstream>

namespace lmx {

std::ostream& operator<<(std::ostream& os, const Token& t) {
    os << "Token(" << to_string(t.type)
       << ", " << t.text << ", " << t.line << ", " << t.col << ')';
    LOG(ITIS(t.text) << ", " << ITIS(t.line) << ", " << ITIS(t.col));
    return os;
}

void Lexer::advance() {
    pos++;
    if (content[pos] == '\n') {
        line++;
        col = 1;
    } else col++;
}

bool Lexer::valid_pos() const {
    return pos < content.size();
}

Token Lexer::next() {
    while (isspace(content[pos])) {
        advance();
        LOG("advance!");
    }
    if (pos >= content.size()) {
        LOG("Directly EOF!");
        return {TokenType::END_OF_FILE,"", line, col};
    }

    switch (content[pos]) {
        case '+': {
            advance();
            return {TokenType::OPER_PLUS, "+", line, col - 1};
        }
        case '-': {
            advance();
            if (content[pos] == '>') {
                advance();
                return {TokenType::ARROW, "->", line, col - 2};
            }
            return {TokenType::OPER_MINUS, "-", line, col - 1};
        }
        case '*': {
            advance();
            return {TokenType::OPER_MUL, "*", line, col - 1};
        }
        case '/': {
            advance();
            return {TokenType::OPER_DIV, "/", line, col - 1};
        }
        case '%': {
            advance();
            return {TokenType::OPER_MOD, "%", line, col - 1};
        }
        case '=': {
            advance();
            if (content[pos] == '=') {
                advance();
                return {TokenType::EQ, "==", line, col - 1};
            }
            if (content[pos] == '>') {
                advance();
                return {TokenType::DOUBLE_ARROW, "=>", line, col - 2};
            }
            return {TokenType::ASSIGN, "=", line, col - 1};
        }
        case '>': {
            advance();
            if (content[pos] == '=') {
                advance();
                return {TokenType::GE, ">=", line, col - 1};
            }
            return {TokenType::GT, ">", line, col - 1};
        }
        case '<': {
            advance();
            if (content[pos] == '=') {
                advance();
                return {TokenType::LE, "<=", line, col - 1};
            }
            return {TokenType::LT, "<", line, col - 1};
        }
        case ':': {
            advance();
            if (content[pos] == ':') {
                advance();
                return {TokenType::COL_COLON, "::", line, col - 1};
            }
            return {TokenType::COLON, ":", line, col - 1};
        }
        case '^': {
            advance();
            return {TokenType::OPER_POW, "^", line, col - 1};
        }
        case '#': {
            while (pos <= content.size() && content[pos] != '\n' )
                advance();
            advance();
            return {TokenType::COMMENT, {}, line, col};
        }
        case '"': {
            advance();
            std::string str;
            if (!valid_pos()) return {TokenType::UNKNOWN, "", line, col - 1};

            while (valid_pos() && content[pos] != '"') {
                if (content[pos] == '\\') {
                    advance();
                    if (!valid_pos()) return {TokenType::UNKNOWN, "", line, col - 1};
                    switch (content[pos]) {
                        case 'n': str += '\n'; break;
                        case 't': str += '\t'; break;
                        case 'r': str += '\r'; break;
                        case 'b': str += '\b'; break;
                        case 'f': str += '\f'; break;
                        case 'v': str += '\v'; break;
                        case '0': str += '\0'; break;
                        default: str += content[pos]; break;
                    }
                    advance();
                    continue;
                }
                str += content[pos];
                advance();
            }
            if (!valid_pos() || content[pos] != '"')
                return {TokenType::UNKNOWN, str, line, col - str.size() - 1};

            advance();
            LOG("Content of the STRING_LITERAL: " << str);
            return {TokenType::STRING_LITERAL, str, line, col - str.size() - 1};
        }
        case '(': {
            advance();
            return {TokenType::LPAREN, "(", line, col - 1};
        }
        case ')': {
            advance();
            return {TokenType::RPAREN, ")", line, col - 1};
        }
        case '{': {
            advance();
            return {TokenType::LBRACE, "{", line, col - 1};
        }
        case '}': {
            advance();
            return {TokenType::RBRACE, "}", line, col - 1};
        }
        case '[': {
            advance();
            return {TokenType::LBRACK, "[", line, col - 1};
        }
        case ']': {
            advance();
            return {TokenType::RBRACK, "]", line, col - 1};
        }
        case ',': {
            advance();
            return {TokenType::COMMA, ", ", line, col - 1};
        }
        case '!': {
            advance();
            if (content[pos] == '=') {
                advance();
                return {TokenType::NE, "!=", line, col - 1};
            }
            return {TokenType::NOT, "!", line, col - 1};
        }
        case '|': {
            advance();
            if (content[pos] == '>') {
                advance();
                return {TokenType::PIPE, "|>", line, col - 1};
            }
            return {TokenType::UNKNOWN, std::string(1, content[pos]), line, col - 1};
        }
        case '.': {
            advance();
            return {TokenType::DOT, ".", line, col - 1};
        }
        default: {
            if (isdigit(content[pos])) {
                const auto cur_line = line, cur_col = col;
                std::string num;
                while (isdigit(content[pos]) || content[pos] == '_') {
                    if (content[pos] == '_') {
                        advance();
                        continue;
                    }
                    num += content[pos];
                    advance();
                }
                return {TokenType::NUM_LITERAL, num, cur_line, cur_col};
            }
            if (isalpha(content[pos]) || content[pos] == '_') {
                std::string id;
                const auto cur_line = line;
                const auto cur_col = col;
                while (isalnum(content[pos])|| content[pos] == '_') {
                    id += content[pos];
                    advance();
                }
                static const std::unordered_map<std::string, TokenType> keywords = {
                    {"func", TokenType::KW_FUNC},
                    {"return", TokenType::KW_RETURN},
                    {"if", TokenType::KW_IF},
                    {"else", TokenType::KW_ELSE},
                    {"let", TokenType::KW_LET},
                    {"var", TokenType::KW_VAR},
                    {"const", TokenType::KW_CONST},
                    {"unit", TokenType::KW_UNIT},
                    {"module", TokenType::KW_MODULE},
                    {"use", TokenType::KW_USE},
                    {"and", TokenType::KW_AND},
                    {"or", TokenType::KW_OR},
                    {"loop", TokenType::KW_LOOP},
                    {"break", TokenType::KW_BREAK},
                    {"continue", TokenType::KW_CONTINUE},
                    {"as", TokenType::KW_AS},
                    {"while", TokenType::KW_WHILE},
                    {"for", TokenType::KW_FOR},
                    {"in", TokenType::KW_IN},
                    {"not", TokenType::NOT},    // 让! 和 not等价
                    {"import", TokenType::KW_IMPORT},
                    {"sym", TokenType::KW_SYM},
                };
                if (const auto it = keywords.find(id); it != keywords.end()) {
                    return {it->second, id, cur_line, cur_col};
                }
                return {TokenType::IDENTIFIER, id, cur_line, cur_col};
            }
        }
    }

    auto token = Token{TokenType::UNKNOWN, std::string(1, content[pos]), line, col};
    advance();
    LOG("Will be UNKNOWN: Token: " << ITIS(token.col) << ", " << ITIS(token.line));
    return token;
}

std::vector<Token> Lexer::tokenize(const std::string& code) {
    content = code;
    has_err = false;
    pos = 0;
    LOG(ITIS(line));
    const auto orig_line = line;
    // line += [&]() -> size_t {
    //     if (content.empty()) return 0;
    //     return std::ranges::count(content, '\n') + 1;
    // }();
    LOG("now: " << ITIS(line) << ", " << ITIS(orig_line) << ", " << ITIS(col));
    col = 1;
    std::vector<Token> tokens;
    while (pos < content.size()) {
        tokens.push_back(next());
        LOG("Pushing...");
    }
    if (tokens.empty() || tokens.back().type != TokenType::END_OF_FILE) tokens.push_back({TokenType::END_OF_FILE, "", line, col});
    for ([[maybe_unused]] auto const &token : tokens) {
        LOG(ITIS(token));
    }
    const std::string res = error(tokens, orig_line);
    if (res.empty()) return tokens;
    LOG("Error!");
    has_err = true;
    std::cerr << res << std::endl;
    return {};
}

std::string Lexer::error(const std::vector<Token>& tokens, const size_t origin_lineno) {
    std::string k;

    std::unordered_map<size_t, std::vector<size_t>> line_errors;
    for (const auto& token : tokens) {
        if (token.type == TokenType::UNKNOWN) {
            line_errors[token.line].push_back(token.col);
        }
    }

    for (const auto& [lineno, cols] : line_errors) {
        std::string line_content = [&]{
            std::istringstream iss(content);
            std::string line_con;
            for (size_t i = 0; i < lineno - origin_lineno - 1; i++) {
                if (!std::getline(iss, line_con)) return std::string("");
            }
            std::getline(iss, line_con);
            return line_con;
        }();

        auto caret_line = [&]{
            if (cols.empty()) return std::string{};
            std::string str(cols.back() + 1, ' ');
            for (const size_t col : cols) str[col - 1] = '^';
            return str;
        }();

        k += std::format(
            "In line {}, file {}:\n>>> {}\n    {}\n",
            lineno,
            filename,
            line_content,
            caret_line
        );
    }

    return k;
}

}