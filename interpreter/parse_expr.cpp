#include "lamina_api/lamina.hpp"
#include "lexer.hpp"
#include "parser.hpp"

std::unique_ptr<Expression> Parser::parse_expression() {
    // ToDo: add op support 'not in' 'in'
	auto node = parse_comparison();
	while (curr_token().type == LexerTokenType::LogicalAnd
		or curr_token().type == LexerTokenType::LogicalOr
	) {
		auto tok = curr_token();
        auto op = skip_token().text;
        auto right = parse_comparison();
        node = std::make_unique<BinaryExpr>(std::move(op), std::move(node), std::move(right));
	}
    return node;
}

std::unique_ptr<Expression> Parser::parse_comparison() {
    auto node = parse_add_sub();
    while (
        curr_token().type == LexerTokenType::Equal
        or curr_token().type == LexerTokenType::NotEqual
        or curr_token().type == LexerTokenType::Greater
        or curr_token().type == LexerTokenType::Less
        or curr_token().type == LexerTokenType::GreaterEqual
        or curr_token().type == LexerTokenType::LessEqual
    ) {
        auto tok = curr_token();
        auto op = skip_token().text;
        auto right = parse_add_sub();
        node = std::make_unique<BinaryExpr>(std::move(op), std::move(node), std::move(right));
    }
    return node;
}

std::unique_ptr<Expression> Parser::parse_add_sub() {
    auto node = parse_mul_div_mod();
    while (
        curr_token().type == LexerTokenType::Plus
        or curr_token().type == LexerTokenType::Minus
    ) {
        auto tok = curr_token();
        auto op = skip_token().text;
        auto right = parse_mul_div_mod();
        node = std::make_unique<BinaryExpr>(std::move(op), std::move(node), std::move(right));
    }
    return node;
}

std::unique_ptr<Expression> Parser::parse_mul_div_mod() {
    auto node = parse_power();
    while (
        curr_token().type == LexerTokenType::Star
        or curr_token().type == LexerTokenType::Slash
        or curr_token().type == LexerTokenType::Percent
    ) {
        auto tok = curr_token();
        auto op = skip_token().text;
        auto right = parse_power();
        node = std::make_unique<BinaryExpr>(std::move(op), std::move(node), std::move(right));
    }
    return node;
}

std::unique_ptr<Expression> Parser::parse_power() {
    auto node = parse_unary();
    if (curr_token().type == LexerTokenType::Caret) {
        auto tok = curr_token();
        auto op = skip_token().text;
        auto right = parse_power();  // 右结合
        node = std::make_unique<BinaryExpr>(std::move(op), std::move(node), std::move(right));
    }
    return node;
}

std::unique_ptr<Expression> Parser::parse_unary() {
    if (curr_token().type == LexerTokenType::Minus) {
        auto tok = curr_token();
        skip_token();
        auto operand = parse_unary();
        return std::make_unique<UnaryExpr>("-", std::move(operand));
    } else if (curr_token().type == LexerTokenType::LogicalNot) {
		skip_token("not");
		auto operand = parse_factor();
		return std::make_unique<UnaryExpr>("not", std::move(operand));
	}
    return parse_factor();
}

std::unique_ptr<Expression> Parser::parse_factor() {
    auto node = parse_a_token();

    while (true) {
        if (curr_token().type == LexerTokenType::ExclamationMark) {
            skip_token("!");
            node = std::make_unique<UnaryExpr>("!", std::move(node));
        }
        if (curr_token().type == LexerTokenType::Dot) {
            node = parse_get_member(std::move(node));
        }
        else if (curr_token().type == LexerTokenType::DoubleColon) {
            node = parse_namespace_get_member(std::move(node));
        }
        else if (curr_token().type == LexerTokenType::LBracket) {
            node = parse_get_item(std::move(node));
        }
        else if (curr_token().type == LexerTokenType::LParen) {
            node = parse_func_call(std::move(node));
        }
        else {
            break;
        }
    }
    return node;
}
