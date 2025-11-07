#include "lexer.hpp"
#include "parser.hpp"
std::unique_ptr<Expression> Parser::parse_a_token() {
    const auto tok = skip_token();
    if (tok.type == LexerTokenType::Number) {
        return std::make_unique<LiteralExpr>(tok.text, Value::Type::Int);
    }
    if (tok.type == LexerTokenType::String) {
        return std::make_unique<LiteralExpr>(tok.text, Value::Type::String);
    }
    if (tok.type == LexerTokenType::Null) {
        return std::make_unique<LiteralExpr>(tok.text, Value::Type::Null);
    }
    if (tok.type == LexerTokenType::True) {
        return std::make_unique<LiteralExpr>(tok.text, Value::Type::Bool);
    }
    if (tok.type == LexerTokenType::False) {
        return std::make_unique<LiteralExpr>(tok.text, Value::Type::Bool);
    }
    if (tok.type == LexerTokenType::Identifier) {
        return std::make_unique<IdentifierExpr>(tok.text);
    }
    if (tok.type == LexerTokenType::Lambda) {
        std::vector<std::string> params{};
        if (curr_token().type == LexerTokenType::Pipe) {
            skip_token("|");
            while (curr_token().type != LexerTokenType::Pipe) {
                params.emplace_back(skip_token().text);
                if (curr_token().type == LexerTokenType::Comma) skip_token(",");
            }
            skip_token("|");
        }

        skip_token("{");
		std::cerr << "[Debug output] pipe process begin\n";
        auto stmt = parse_block(true);
        skip_token("}");
        return std::make_unique<LambdaDeclExpr>("<lambda>", std::move(params),std::move(stmt));
    }
    if (tok.type == LexerTokenType::Pipe) {
        std::vector<std::string> params;
        while (curr_token().type != LexerTokenType::Pipe) {
            params.emplace_back(skip_token().text);
            if (curr_token().type == LexerTokenType::Comma) skip_token(",");
        }
        skip_token("|");
        auto expr = parse_expression();
        std::vector<std::unique_ptr<Statement>> stmts;
        stmts.emplace_back(std::make_unique<ReturnStmt>(std::move(expr)));

        return std::make_unique<LambdaDeclExpr>(
            "lambda",
            std::move(params),
            std::make_unique<BlockStmt>(std::move(stmts))
        );
    }
    if (tok.type == LexerTokenType::LBrace) {
        std::vector<std::pair<std::string, std::unique_ptr<Expression>>> init_vec{};
        while (curr_token().type != LexerTokenType::RBrace) {
            //auto key = skip_token().text;
			std::string key = "";
			do {
				key = skip_token().text;
			} while (key == "\n" || key == "\t" || key == " " || key == ";" || key == "\r" || key == "");
			if (key == "}") {
				break;
			}
			std::cerr << "[Debug output] member assignment: key: " << key << std::endl;
            skip_token("=");
            auto val = parse_expression();
            if (curr_token().type == LexerTokenType::Comma) skip_token(",");
            if (curr_token().type == LexerTokenType::Semicolon) skip_token(";");
            init_vec.emplace_back(std::move(key), std::move(val));
        }
        if (curr_token().type == LexerTokenType::RBrace) skip_token("}");
        return std::make_unique<LambdaStructDeclExpr>(std::move(init_vec));
    }
    if (tok.type == LexerTokenType::LBracket) {
        auto param = parse_params(LexerTokenType::RBracket);
        skip_token("]");
        return std::make_unique<ArrayExpr>(std::move(param));
    }
    if (tok.type == LexerTokenType::LParen) {
        auto expr = parse_expression();
        skip_token(")");
        return expr;
    }
    return nullptr;
}

std::unique_ptr<Expression> Parser::parse_func_call(std::unique_ptr<Expression> node) {
    skip_token("(");
    auto param = parse_params(LexerTokenType::RParen);
    skip_token(")");
    return std::make_unique<CallExpr>(std::move(node),std::move(param));
}

std::unique_ptr<GetMemberExpr> Parser::parse_get_member(std::unique_ptr<Expression> node) {
    skip_token(".");
    auto child = std::make_unique<IdentifierExpr>(skip_token().text);
    return std::make_unique<GetMemberExpr>(std::move(node),std::move(child));
}

std::unique_ptr<NameSpaceGetMemberExpr> Parser::parse_namespace_get_member(std::unique_ptr<Expression> node) {
    skip_token("::");
    auto child = std::make_unique<IdentifierExpr>(skip_token().text);
    return std::make_unique<NameSpaceGetMemberExpr>(std::move(node),std::move(child));
}

std::unique_ptr<GetItemExpr> Parser::parse_get_item(std::unique_ptr<Expression> node) {
    skip_token("[");
    auto param = parse_params(LexerTokenType::RBracket);
    skip_token("]");
    return std::make_unique<GetItemExpr>(std::move(node),std::move(param));
}

std::vector<std::unique_ptr<Expression>> Parser::parse_params(const LexerTokenType endswith){
    std::vector<std::unique_ptr<Expression>> params;
    while (curr_token().type != endswith) {
        params.emplace_back(parse_expression());
        if (curr_token().type == LexerTokenType::Comma) skip_token(",");
    }
    return params;
}