#include "parser.hpp"

#include "lamina_api/lamina.hpp"
#include "lexer.hpp"
#include "utils/color_style.hpp"

Parser::Parser(const std::vector<Token>& tokens) : tokens_(tokens){};

std::string Parser::get_module_name() const {
    return module_name_;
}

std::string Parser::get_module_version() const {
    return module_version_;
}

Token Parser::skip_token(const std::string& want_skip) {
    if (curr_tok_idx_ < tokens_.size()) {
        auto& tok = tokens_[curr_tok_idx_];
        if (!want_skip.empty() and tok.text != want_skip) {
            std::cerr << ConClr::RED
            << "There should be '" << want_skip << "' , but you given '"
            << tok.text << "'" << ConClr::RESET << std::endl;
            throw StdLibException("");
        }
        curr_tok_idx_++;
        return tok;
    }
    return {LexerTokenType::EndOfFile, "", 0, 0};
}

Token Parser::curr_token() const {
    if (curr_tok_idx_ < tokens_.size()) {
        auto& tok = tokens_[curr_tok_idx_];
        return tok;
    }
    return {LexerTokenType::EndOfFile, "", 0, 0};
}

void Parser::skip_end_of_ln() {
    const Token tok = curr_token();
    if (tok.type == LexerTokenType::Semicolon) {
        skip_token(";");
        return;
    }
    if (tok.type == LexerTokenType::EndOfLine) {
        skip_token("\n");
        return;
    }
    if (tok.type == LexerTokenType::EndOfFile) {
        return;
    }
    std::cerr << ConClr::RED << "End of line must be ';', got '" << tok.text << "'" << ConClr::RESET << std::endl;
    throw StdLibException("");
}

void Parser::must_token(const std::string& text, const std::string& waring) const {
    if (const auto tok = this->curr_token();
        tok.text != text) {
        std::cerr << ConClr::RED << "The word'" << tok.text << "' cause error that : \n"
                  << waring
                  << ConClr::RESET << std::endl;
    }
}
std::vector<std::unique_ptr<Statement>> Parser::parse_program() {
    std::vector<std::unique_ptr<Statement>>
        stmts = {};
    while (curr_token().type != LexerTokenType::EndOfFile) {
        if (auto s = parse_stmt();
            s != nullptr
        ) {
            stmts.push_back(std::move(s));
        }
    }
    return stmts;
}

std::unique_ptr<Statement> Parser::parse_stmt() {
    auto tok = curr_token();

    if (tok.type == LexerTokenType::If) {
        skip_token("if");
        return parse_if();
    }
    if (tok.type == LexerTokenType::While) {
        skip_token("while");
        return parse_while();
    }
    if (tok.type == LexerTokenType::Func) {
        skip_token("func");
        return parse_func();
    }
    if (tok.type == LexerTokenType::Var) {
        skip_token("var");
        return parse_var();
    }
    if (tok.type == LexerTokenType::Struct) {
        skip_token("struct");
        return parse_struct();
    }

    if (tok.type == LexerTokenType::Return) {
        skip_token("return");
        auto expr = parse_expression();
        skip_end_of_ln();
        return std::make_unique<ReturnStmt>(std::move(expr));
    }
    if (tok.type == LexerTokenType::Break) {
        skip_token("break");
        skip_end_of_ln();
        return std::make_unique<BreakStmt>();
    }
    if (tok.type == LexerTokenType::Continue) {
        skip_token("continue");
        skip_end_of_ln();
        return std::make_unique<ContinueStmt>();
    }
    if (tok.type == LexerTokenType::Include) {
        skip_token("include");
        const auto path = skip_token().text;
        skip_end_of_ln();
        return std::make_unique<IncludeStmt>(path);
    }
    if (tok.type == LexerTokenType::Loop) {
        skip_token("loop");
        auto expr = std::make_unique<LiteralExpr>("true", Value::Type::Bool);
        skip_token("{");
        auto stmts = parse_block(true);
        skip_token("}");
        return std::make_unique<WhileStmt>(std::move(expr), std::move(stmts));
    }
    if (tok.type == LexerTokenType::Define) {
        skip_token("define");
        const auto name = skip_token().text;
        skip_token("=");
        auto value = parse_a_token();
        skip_end_of_ln();
        if (const auto string_val = dynamic_cast<LiteralExpr*>(value.get());
            name == "module_name" and string_val->type == Value::Type::String
        ) {
            module_name_ = string_val->value;
            return nullptr;
        }
        if (const auto string_val = dynamic_cast<LiteralExpr*>(value.get());
            name == "module_version" and string_val->type == Value::Type::String
        ) {
            module_version_ = string_val->value;
            return nullptr;
        }

        return std::make_unique<DefineStmt>(name, std::move(value));
    }
    if (tok.type == LexerTokenType::Bigint) {
        skip_token("bigint");
        const auto name = skip_token().text;
        skip_token("=");
        auto value = parse_expression();
        skip_end_of_ln();
        return std::make_unique<BigIntDeclStmt>(name, std::move(value));
    }
    if (tok.type == LexerTokenType::Identifier
        and curr_tok_idx_ + 1 < tokens_.size()
        and tokens_[curr_tok_idx_ + 1].type == LexerTokenType::Assign
    ) {
        const auto name = skip_token().text;
        skip_token("=");
        auto expr = parse_expression();
        skip_end_of_ln();
        return std::make_unique<AssignStmt>(name, std::move(expr));
    }
    auto expr = parse_expression();
    if (expr != nullptr and curr_token().text == "=") {
        if (dynamic_cast<GetMemberExpr*>(expr.get())) {
            skip_token("=");
            auto value = parse_expression();
            skip_end_of_ln();

            auto set_mem = std::make_unique<SetMemberExpr>(std::move(expr), std::move(value));
            return std::make_unique<ExprStmt>(std::move(set_mem));
        }
        //非成员访问表达式后不能跟 =
        throw StdLibException("invalid assignment target: expected member access");
    }
    if (expr != nullptr) {
        skip_end_of_ln();
        return std::make_unique<ExprStmt>(std::move(expr));
    }

    while (curr_tok_idx_ < tokens_.size() && curr_token().type == LexerTokenType::EndOfLine) {
        skip_token(); 
    }
    return nullptr;
}