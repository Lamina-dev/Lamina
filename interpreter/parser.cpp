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
			std::cerr << "[Parser] Errinfo: ";
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
	std::cerr << "[Debug output] token fetch got EOF\n";
    return {LexerTokenType::EndOfFile, "", 0, 0};
}

void Parser::skip_end_of_ln() {
    // 跳过所有连续的换行符
    while (curr_token().type == LexerTokenType::EndOfLine) {
        skip_token("\n");
    }

    // 若存在显式分号
    if (curr_token().type == LexerTokenType::Semicolon) {
        skip_token(";");
        // 跳过分号后的换行（如 ";\n" 场景）
        while (curr_token().type == LexerTokenType::EndOfLine) {
            skip_token("\n");
        }
        return;
    }

    // 仅在“非文件结尾”时报错
    if (curr_token().type != LexerTokenType::EndOfFile) {
        std::cerr << ConClr::RED 
                  << "Statement must end with ';' or newline, got '" << curr_token().text << "'" 
                  << ConClr::RESET << std::endl;
        throw StdLibException("invalid statement terminator");
    }
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
	while (curr_tok_idx_ < tokens_.size() && curr_token().type == LexerTokenType::EndOfLine) {
        skip_token(); 
    }
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
        // New syntax: loop <count-expression> { ... }
        // Parse an expression as the repeat count (allows variable or computation)
        skip_token("loop");
        // Disallow empty immediate '{' to remove infinite-loop syntax
        if (curr_token().text == "{") {
            std::cerr << "SyntaxError: infinite 'loop { }' is no longer supported; use 'loop <count-expression> { ... }'" << std::endl;
            throw StdLibException("invalid loop syntax: infinite loop removed");
        }
        auto count_expr = parse_expression();
        if (count_expr == nullptr) {
            std::cerr << "SyntaxError: expected count expression after 'loop'" << std::endl;
            throw StdLibException("invalid loop syntax: expected count expression");
        }
        skip_token("{");
        auto stmts = parse_block(true);
        skip_token("}");
        return std::make_unique<RepeatStmt>(std::move(count_expr), std::move(stmts));
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
		std::cerr << "[Debug output] regarded as assignment\n";
        skip_token("=");
        auto expr = parse_expression();
        skip_end_of_ln();
        return std::make_unique<AssignStmt>(name, std::move(expr));
    }
    auto expr = parse_expression();
    if (expr != nullptr and curr_token().text == "=") {
		std::cerr << "[Debug output] regarded as member assignment\n";
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