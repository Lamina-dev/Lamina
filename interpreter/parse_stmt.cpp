#include "lexer.hpp"
#include "parser.hpp"
std::unique_ptr<BlockStmt> Parser::parse_block(bool is_global) {
	// Is this boolean parameter really useful???
    std::vector<std::unique_ptr<Statement>> stmts;
    while (curr_token().type != LexerTokenType::RBrace) {
		//std::cerr << "[Debug output] parse through blocks! Current: " << curr_token().text << std::endl;
		auto pres = parse_stmt();
		if (pres)
			stmts.push_back(move(pres));
		//else std::cerr << "[Debug output] resolved empty statement!\n";
		//std::cerr << "[Debug output] statement parser end\n";
		// I think we should sooner or later make this an independent function:
		bool flag = false;
		do {
			flag = false;
			while (curr_token().type == LexerTokenType::Semicolon
					or curr_token().type == LexerTokenType::EndOfLine) {
				skip_token();	// Maybe shall be forced?
				flag = true;
			}
		} while (flag);
    }
    return std::make_unique<BlockStmt>(std::move(stmts));
}

std::unique_ptr<Statement> Parser::parse_if() {
    auto expr = parse_expression();
    skip_token("{");
    auto stmts = parse_block(true);
    skip_token("}");
    std::unique_ptr<BlockStmt> else_stmts = nullptr;
    if (curr_token().type == LexerTokenType::Else) {
        skip_token("else");
        if (curr_token().type == LexerTokenType::If) {
            skip_token("if");
            std::vector<std::unique_ptr<Statement>> if_branch = {};
            if_branch.emplace_back(std::move(parse_if()));
            else_stmts = std::make_unique<BlockStmt>(std::move(if_branch));
        } else {
            skip_token("{");
            else_stmts = parse_block(true);
            skip_token("}");
        }
    }
    return std::make_unique<IfStmt>(std::move(expr),std::move(stmts),std::move(else_stmts));
}

std::unique_ptr<Statement> Parser::parse_func() {
    auto name = skip_token().text;
    std::vector<std::string> params;

    if (curr_token().type == LexerTokenType::LParen) {
        skip_token("(");
        while (curr_token().type != LexerTokenType::RParen) {
            params.emplace_back(skip_token().text);
            if (curr_token().type == LexerTokenType::Comma) skip_token(",");
        }
        skip_token(")");
    }

    skip_token("{");
    auto stmts = parse_block(true);
    skip_token("}");
    return std::make_unique<FuncDefStmt>(name, std::move(params), std::move(stmts));
}

std::unique_ptr<Statement> Parser::parse_var() {
    auto name = skip_token().text;

    skip_token("=");
    auto expr = parse_expression();
    skip_end_of_ln();
    return std::make_unique<VarDeclStmt>(name, std::move(expr));
}

std::unique_ptr<Statement> Parser::parse_struct() {
    std::vector<std::unique_ptr<IdentifierExpr>> includes{};
    auto name = skip_token().text;
    std::vector<std::pair<std::string, std::unique_ptr<Expression>>> init_vec{};
    skip_token("{");
    while (curr_token().type != LexerTokenType::RBrace) {
        auto key = skip_token().text;
        skip_token("=");
        auto val = parse_expression();
        if (curr_token().type == LexerTokenType::Comma) skip_token(",");
        if (curr_token().type == LexerTokenType::Semicolon) skip_token(";");
        init_vec.emplace_back(std::move(key), std::move(val));
    }
    skip_token("}");
    skip_end_of_ln();
    return std::make_unique<StructDeclStmt>(name, std::move(init_vec),std::move(includes));
}

std::unique_ptr<Statement> Parser::parse_while() {
    auto expr = parse_expression();
    skip_token("{");
    auto stmts = parse_block(true);
    skip_token("}");
    return std::make_unique<WhileStmt>(std::move(expr), std::move(stmts));
}
