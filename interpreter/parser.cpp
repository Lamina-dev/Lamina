#include "parser.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>


// Debug output macro - no output if DEBUG is false
#define DEBUG_OUT \
    if (false) std::cerr

std::unique_ptr<Expression> Parser::parse_expression(const std::vector<Token>& tokens, size_t& i) {
    // Safety check
    if (i >= tokens.size() || tokens[i].type == TokenType::EndOfFile) {
        std::cerr << "Error: Attempting to parse expression at end of input" << std::endl;
        return nullptr;
    }

    size_t expr_start = i;
    auto expr = parse_comparison(tokens, i);
    size_t expr_end = i;
    if (expr) {
        std::string src;
        for (size_t j = expr_start; j < expr_end; ++j) src += tokens[j].text;
        expr->source = src;
    }
    return expr;
}

std::unique_ptr<Expression> Parser::parse_comparison(const std::vector<Token>& tokens, size_t& i) {
    auto left = parse_addition(tokens, i);
    while (i < tokens.size() && (tokens[i].type == TokenType::Equal || tokens[i].type == TokenType::NotEqual ||
                                 tokens[i].type == TokenType::Less || tokens[i].type == TokenType::LessEqual ||
                                 tokens[i].type == TokenType::Greater || tokens[i].type == TokenType::GreaterEqual)) {
        std::string op = tokens[i].text;
        ++i;
        auto right = parse_addition(tokens, i);
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }
    return left;
}

std::unique_ptr<Expression> Parser::parse_addition(const std::vector<Token>& tokens, size_t& i) {
    auto left = parse_term(tokens, i);
    while (i < tokens.size() && (tokens[i].type == TokenType::Plus || tokens[i].type == TokenType::Minus)) {
        std::string op = tokens[i].text;
        ++i;
        auto right = parse_term(tokens, i);
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }
    return left;
}

std::unique_ptr<Expression> Parser::parse_term(const std::vector<Token>& tokens, size_t& i) {
    auto left = parse_power(tokens, i);
    while (i < tokens.size() && (tokens[i].type == TokenType::Star || tokens[i].type == TokenType::Slash ||
                                 tokens[i].type == TokenType::Percent)) {
        std::string op = tokens[i].text;
        ++i;
        auto right = parse_power(tokens, i);
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }
    return left;
}

std::unique_ptr<Expression> Parser::parse_power(const std::vector<Token>& tokens, size_t& i) {
    auto left = parse_unary(tokens, i);
    while (i < tokens.size() && tokens[i].type == TokenType::Caret) {
        std::string op = tokens[i].text;
        ++i;
        auto right = parse_unary(tokens, i);
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }
    return left;
}

std::unique_ptr<Expression> Parser::parse_unary(const std::vector<Token>& tokens, size_t& i) {
    if (i >= tokens.size()) return nullptr;

    // Handle prefix unary operators
    if (tokens[i].type == TokenType::Minus) {
        ++i;
        auto operand = parse_unary(tokens, i);
        return std::make_unique<UnaryExpr>("-", std::move(operand));
    }

    // Parse primary expression first
    auto expr = parse_primary(tokens, i);

    // Handle postfix operators (factorial, array indexing)
    while (i < tokens.size()) {
        if (tokens[i].type == TokenType::Bang) {
            ++i;// consume '!'
            expr = std::make_unique<UnaryExpr>("!", std::move(expr));
        } else if (tokens[i].type == TokenType::LBracket) {
            // Array indexing: array[index]
            ++i;// consume '['
            auto index = parse_expression(tokens, i);
            if (!index) {
                std::cerr << "Error: Expected expression in array index" << std::endl;
                return nullptr;
            }
            if (i >= tokens.size() || tokens[i].type != TokenType::RBracket) {
                std::cerr << "Error: Expected ']' after array index" << std::endl;
                return nullptr;
            }
            ++i;// consume ']'
            expr = std::make_unique<IndexExpr>(std::move(expr), std::move(index));
        } else {
            break;// No more postfix operators
        }
    }

    return expr;
}

std::unique_ptr<Expression> Parser::parse_primary(const std::vector<Token>& tokens, size_t& i) {
    if (i >= tokens.size()) return nullptr;

    DEBUG_OUT << "Debug - parse_primary: token[" << i << "] = '" << tokens[i].text
              << "' (type=" << static_cast<int>(tokens[i].type) << ")" << std::endl;
    if (tokens[i].type == TokenType::Number) {
        return std::make_unique<LiteralExpr>(tokens[i++].text, Value::Type::Int);
    } else if (tokens[i].type == TokenType::String) {
        return std::make_unique<LiteralExpr>(tokens[i++].text, Value::Type::String);
    } else if (tokens[i].type == TokenType::True) {
        i++;
        return std::make_unique<LiteralExpr>("true", Value::Type::Bool);
    } else if (tokens[i].type == TokenType::False) {
        i++;
        return std::make_unique<LiteralExpr>("false", Value::Type::Bool);
    } else if (tokens[i].type == TokenType::Null) {
        i++;
        return std::make_unique<LiteralExpr>("null", Value::Type::Null);
    } else if (tokens[i].type == TokenType::LBracket) {
        // Parse array literal [expr1, expr2, ...]
        ++i;// Skip '['
        std::vector<std::unique_ptr<Expression>> elements;

        while (i < tokens.size() && tokens[i].type != TokenType::RBracket) {
            auto elem = parse_expression(tokens, i);
            elements.push_back(std::move(elem));

            if (tokens[i].type == TokenType::Comma) {
                ++i;// Skip ','
            } else if (tokens[i].type != TokenType::RBracket) {
                std::cerr << "Error: Expected ',' or ']' in array literal" << std::endl;
                return nullptr;
            }
        }

        if (i >= tokens.size() || tokens[i].type != TokenType::RBracket) {
            std::cerr << "Error: Unterminated array literal, expected ']'" << std::endl;
            return nullptr;
        }

        ++i;// Skip ']'
        return std::make_unique<ArrayExpr>(std::move(elements));
    } else if (tokens[i].type == TokenType::Identifier) {
        std::string name = tokens[i].text;
        // std::cerr << "DEBUG: Found identifier '" << name << "' at token " << i << std::endl;
        ++i;

        // Check for dot notation (namespace.function)
        if (i < tokens.size() && tokens[i].type == TokenType::Dot) {
            //   std::cerr << "DEBUG: Found dot at token " << i << std::endl;
            ++i;// Skip '.'
            if (i < tokens.size() && tokens[i].type == TokenType::Identifier) {
                //     std::cerr << "DEBUG: Found second identifier '" << tokens[i].text << "' at token " << i << std::endl;
                name += "." + tokens[i].text;// 保持点格式
                ++i;
                //   std::cerr << "DEBUG: Converted to namespace syntax: '" << name << "'" << std::endl;
            } else {
                std::cerr << "Error: Expected identifier after '.'" << std::endl;
                return nullptr;
            }
        } else {
            // std::cerr << "DEBUG: No dot found, next token type is " << (i < tokens.size() ? static_cast<int>(tokens[i].type) : -1) << std::endl;
        }

        // Function call
        if (i < tokens.size() && tokens[i].type == TokenType::LParen) {
            ++i;// Skip '('
            std::vector<std::unique_ptr<Expression>> args;

            while (i < tokens.size() && tokens[i].type != TokenType::RParen) {
                auto arg = parse_expression(tokens, i);
                args.push_back(std::move(arg));

                if (tokens[i].type == TokenType::Comma) {
                    ++i;// Skip ','
                } else if (tokens[i].type != TokenType::RParen) {
                    std::cerr << "Error: Expected ',' or ')' in function call" << std::endl;
                    return nullptr;
                }
            }

            if (i >= tokens.size() || tokens[i].type != TokenType::RParen) {
                std::cerr << "Error: Unterminated function call, expected ')'" << std::endl;
                return nullptr;
            }

            ++i;// Skip ')'
            return std::make_unique<CallExpr>(name, std::move(args));
        }
        // 否则作为普通变量处理
        return std::make_unique<VarExpr>(name);
    } else if (tokens[i].type == TokenType::LParen) {
        ++i;// Skip '('
        auto expr = parse_expression(tokens, i);

        if (i >= tokens.size() || tokens[i].type != TokenType::RParen) {
            std::cerr << "Error: Unterminated parenthesized expression, expected ')'" << std::endl;
            return nullptr;
        }

        ++i;// Skip ')'
        return expr;
    }

    std::cerr << "Error: Unexpected token in expression: " << tokens[i].text << std::endl;
    return nullptr;
}

// Helper function: print token context for error reporting
static void print_context(const std::vector<Token>& tokens, size_t pos, int context_size = 5) {
    std::cerr << "Context:" << std::endl;
    std::string context;
    for (const auto& t: tokens) {
        context += t.text + " ";
    }

    std::cerr << context << "" << std::endl;

    for (size_t i = 0; i < context.size(); ++i) {
        if (i == tokens[pos].column) std::cerr << "^";
        else
            std::cerr << "~";
    }
    std::cerr << std::endl;

    // Print line and column position indicators
    std::cerr << "Position: ";
    std::cerr << "\033[36m line " << tokens[pos].line << " | " << "col " << tokens[pos].column << "\033[0m ";
    std::cerr << std::endl;
}

std::unique_ptr<BlockStmt> Parser::parse_block(const std::vector<Token>& tokens, size_t& i, bool is_global) {
    auto block = std::make_unique<BlockStmt>();

    // Record current block start position for debugging
    int start_line = (i < tokens.size()) ? tokens[i].line : -1;
    int start_col = (i < tokens.size()) ? tokens[i].column : -1;

    // Use non-static variable to record nesting depth, avoid interference from multiple parsing
    thread_local int current_block_depth = 0;
    current_block_depth++;

    DEBUG_OUT << "Debug - Block parsing started: "
              << (is_global ? "global" : "local")
              << ", depth=" << current_block_depth
              << ", position=" << start_line << ":" << start_col
              << ", current token=" << (i < tokens.size() ? "'" + tokens[i].text + "'" : "EOF")
              << std::endl;

    // Record start token position for error reporting
    size_t block_start_index = i;

    while (i < tokens.size()) {
        // Check if we reached end of block
        if (!is_global && tokens[i].type == TokenType::RBrace) {
            // Local block ended with } brace
            DEBUG_OUT << "Debug - Block ended normally: depth=" << current_block_depth
                      << ", position=" << tokens[i].line << ":" << tokens[i].column
                      << ", statement_count=" << block->statements.size()
                      << std::endl;
            current_block_depth--;
            i++;// consume right brace
            return block;
        }

        // Check if we reached end of file
        if (tokens[i].type == TokenType::EndOfFile) {
            if (!is_global) {
                // Local block ended unexpectedly at EOF, provide detailed error info
                std::cerr << "\033[31mError: Missing closing brace '}' - block started at line " << start_line
                          << " col " << start_col
                          << ", not closed before end of file\033[0m" << std::endl;

                // Output context around the block start to help locate the issue
                std::cerr << "Block start context:" << std::endl;
                size_t context_start = block_start_index > 5 ? block_start_index - 5 : 0;
                size_t context_end = std::min(block_start_index + 5, tokens.size() - 1);
                for (size_t j = context_start; j <= context_end; j++) {
                    if (j == block_start_index) {
                        std::cerr << "[" << tokens[j].text << "] ";
                    } else {
                        std::cerr << tokens[j].text << " ";
                    }
                }
                std::cerr << std::endl;

                // Show block content summary
                std::cerr << "Block contains " << block->statements.size() << " statements" << std::endl;

                current_block_depth--;
                // Throw exception instead of direct exit, let caller recover
                throw std::runtime_error("Parse error: Unclosed block, missing closing brace");
            } else {
                // Global block reached EOF, normal termination
                DEBUG_OUT << "Debug - Global block ended at EOF: depth=" << current_block_depth
                          << ", statements=" << block->statements.size() << std::endl;
                current_block_depth--;
                return block;
            }
        }

        // Parse statements in block
        auto stmt = parse_statement(tokens, i);
        if (stmt) {
            block->statements.push_back(std::move(stmt));
        } else if (i < tokens.size() && tokens[i].type != TokenType::EndOfFile) {
            // Only report error if not at end of file and parsing failed
            std::cerr << "\033[31mError: Invalid or unexpected statement at token " << i
                      << " (line " << tokens[i].line
                      << " col " << tokens[i].column
                      << "): "
                      << tokens[i].text << "\033[0m" << std::endl;

            // Try to recover: skip current token and continue parsing
            i++;
            continue;
        } else {
            // Reached end of file, should be global block at this point
            if (!is_global) {
                std::cerr << "\033[31mError: Unexpected end of file, missing block closure, started at line " << start_line << "\033[0m" << std::endl;
                current_block_depth--;
                throw std::runtime_error("Parse error: Unclosed block, missing closing brace");
            }
            break;
        }

        // Check progress to avoid infinite loops
        if (i >= tokens.size()) {
            if (!is_global) {
                std::cerr << "\033[31mError: Unexpected end of file, missing block closure, started at line " << start_line << "\033[0m" << std::endl;
                current_block_depth--;
                throw std::runtime_error("Parse error: Unclosed block, missing closing brace");
            }
            break;
        }
    }

    DEBUG_OUT << "Debug - Block parsing completed: depth=" << current_block_depth
              << ", statements=" << block->statements.size() << std::endl;

    // Ensure depth count is reduced before returning
    current_block_depth--;
    return block;
}

// 2参数重载，兼容老代码
std::unique_ptr<BlockStmt> Parser::parse_block(const std::vector<Token>& tokens, size_t& i) {
    return parse_block(tokens, i, false);
}

// Parse struct decl
std::unique_ptr<StructDeclStmt> Parser::parse_struct(const std::vector<Token>& tokens, size_t& i) {
    if (!(i < tokens.size() && tokens[i].type == TokenType::Struct)) {
        return nullptr;
    }
    int struct_line = tokens[i].line;
    int struct_col = tokens[i].column;
    size_t struct_start_index = i;// Record struct statement start position

    DEBUG_OUT << "Debug - Starting while loop parsing, line " << struct_line << " col " << struct_col << std::endl;

    ++i;// Skip 'struct'
    if (i >= tokens.size()) {
        std::cerr << "\033[31mError: struct statement ended unexpectedly, missing the name\033[0m" << std::endl;
        print_context(tokens, struct_start_index);
        return nullptr;
    }

    const auto& struct_name = tokens[i].text;
    std::vector<std::pair<std::string, std::unique_ptr<Expression>>> init_vec;
    ++i;
    if (i >= tokens.size()) {
        std::cerr << "\033[31mError: struct statement ended unexpectedly, missing the body\033[0m" << std::endl;
        print_context(tokens, struct_start_index);
        return nullptr;
    }

    if (tokens[i].type == TokenType::LBrace) {
        ++i;// Consume opening brace
    } else {
        std::cerr << "\033[31mError: struct statement missing opening brace '{'\033[0m" << std::endl;
        print_context(tokens, i);
    }

    // Parse struct body
    DEBUG_OUT << "Debug - Parsing struct body" << std::endl;

    while (i < tokens.size() &&
           tokens[i].type != TokenType::RBrace) {
        if (!(i < tokens.size() && tokens[i].type == TokenType::Identifier)) {
            std::cerr << "\033[31mError: struct statement missing the sub name instead of '" << tokens[i].text << "'\033[0m" << std::endl;
            print_context(tokens, i);
            return nullptr;
        };
        std::string sub_name = tokens[i].text;
        ++i;

        if (!(i < tokens.size() && tokens[i].type == TokenType::Assign)) {
            std::cerr << "\033[31mError: struct member missing '=' after identifier '" << sub_name << "' (line " << tokens[i].line << ")\033[0m" << std::endl;
            print_context(tokens, i);
            return nullptr;
        }
        ++i;

        std::unique_ptr<Expression> sub_expr = parse_expression(tokens, i);
        while (i < tokens.size() && tokens[i].type != TokenType::Semicolon) ++i;
        if (i >= tokens.size() || tokens[i].type != TokenType::Semicolon) {
            std::cerr << "\033[31mError: struct sub item need ends with ';' (line " << tokens[i - 1].line << ")\033[0m" << std::endl;
            print_context(tokens, i);
            return nullptr;
        }
        ++i;

        if (!sub_expr) {
            std::cerr << "\033[31mError: struct sub item need init val" << std::endl;
            print_context(tokens, i);
            return nullptr;
        }

        init_vec.emplace_back(sub_name, std::move(sub_expr));
    }

    if (i >= tokens.size() || tokens[i].type != TokenType::RBrace) {
        std::cerr << "\033[31mError: struct body missing closing brace '}' (line " << tokens[i - 1].line << ")\033[0m" << std::endl;
        print_context(tokens, i);
        return nullptr;
    }
    ++i;

    DEBUG_OUT << "Debug - while loop parsing completed, defined at line " << struct_line
              << ", ended at line " << (i < tokens.size() ? tokens[i - 1].line : -1) << std::endl;

    return std::make_unique<StructDeclStmt>(struct_name, std::move(init_vec));
}

std::unique_ptr<Statement> Parser::parse_while(const std::vector<Token>& tokens, size_t& i) {
    if (!(i < tokens.size() && tokens[i].type == TokenType::While)) {
        return nullptr;
    }

    int while_line = tokens[i].line;
    int while_col = tokens[i].column;
    size_t while_start_index = i;// Record while statement start position

    DEBUG_OUT << "Debug - Starting while loop parsing, line " << while_line << " col " << while_col << std::endl;

    ++i;// Skip 'while'

    // Check for opening parenthesis
    if (i >= tokens.size() || tokens[i].type != TokenType::LParen) {
        std::cerr << "\033[31mError: Missing opening parenthesis '(' after while statement\033[0m" << std::endl;
        print_context(tokens, i > 0 ? i - 1 : 0);

        // Try to recover: look for opening parenthesis or opening brace
        while (i < tokens.size() &&
               tokens[i].type != TokenType::LParen &&
               tokens[i].type != TokenType::LBrace) {
            ++i;
        }

        if (i >= tokens.size() || tokens[i].type != TokenType::LParen) {
            return nullptr;// Can't find recoverable point
        }
    }

    ++i;// Skip opening parenthesis

    // Parse condition
    if (i >= tokens.size()) {
        std::cerr << "\033[31mError: while statement ended unexpectedly, missing condition expression\033[0m" << std::endl;
        print_context(tokens, while_start_index);
        return nullptr;
    }

    auto cond = parse_expression(tokens, i);
    if (!cond) {
        std::cerr << "\033[31mError: while statement missing valid condition expression\033[0m" << std::endl;
        print_context(tokens, i);

        // Try to recover: create a condition that's always true
        cond = std::make_unique<LiteralExpr>("true", Value::Type::Bool);

        // Try to find right parenthesis
        while (i < tokens.size() && tokens[i].type != TokenType::RParen) {
            ++i;
        }

        if (i >= tokens.size()) {
            return nullptr;// Can't find closing parenthesis, can't continue
        }
    }

    // Check for closing parenthesis
    if (i >= tokens.size()) {
        std::cerr << "\033[31mError: while statement condition ended unexpectedly, missing closing parenthesis\033[0m" << std::endl;
        print_context(tokens, while_start_index);
        return nullptr;
    }

    if (tokens[i].type == TokenType::RParen) {
        ++i;// Consume closing parenthesis
    } else {
        std::cerr << "\033[31mError: while statement condition missing closing parenthesis ')', after line " << tokens[i - 1].line << "\033[0m" << std::endl;
        print_context(tokens, i);

        // Try to recover: look for opening brace
        while (i < tokens.size() && tokens[i].type != TokenType::LBrace) {
            ++i;
        }

        if (i >= tokens.size()) {
            return nullptr;// Can't find opening brace, can't continue
        }
    }

    // Check for opening brace
    if (i >= tokens.size()) {
        std::cerr << "\033[31mError: while statement ended unexpectedly after closing parenthesis, missing loop body\033[0m" << std::endl;
        print_context(tokens, while_start_index);
        return nullptr;
    }

    if (tokens[i].type == TokenType::LBrace) {
        ++i;// Consume opening brace
    } else {
        std::cerr << "\033[31mError: while statement missing opening brace '{'\033[0m" << std::endl;
        print_context(tokens, i);

        // Try to recover: create an empty block and return
        auto emptyBody = std::make_unique<BlockStmt>();
        return std::make_unique<WhileStmt>(std::move(cond), std::move(emptyBody));
    }

    // Parse loop body
    DEBUG_OUT << "Debug - Parsing while loop body" << std::endl;
    std::unique_ptr<BlockStmt> body;

    try {
        body = parse_block(tokens, i, false);
        DEBUG_OUT << "Debug - while loop body parsing completed, contains " << body->statements.size() << " statements" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: Error parsing while loop body: " << e.what() << std::endl;
        // Create empty block as recovery measure
        body = std::make_unique<BlockStmt>();

        // Try to locate next statement start
        while (i < tokens.size() &&
               tokens[i].type != TokenType::Semicolon &&
               tokens[i].type != TokenType::RBrace) {
            ++i;
        }

        if (i < tokens.size() && tokens[i].type == TokenType::RBrace) {
            ++i;// Consume right brace
        }
    }

    DEBUG_OUT << "Debug - while loop parsing completed, defined at line " << while_line
              << ", ended at line " << (i < tokens.size() ? tokens[i - 1].line : -1) << std::endl;

    return std::make_unique<WhileStmt>(std::move(cond), std::move(body));
}

std::unique_ptr<Statement> Parser::parse_statement(const std::vector<Token>& tokens, size_t& i) {
    // std::cerr << "DEBUG: parse_statement starting at token " << i << std::endl;

    // Handle @backend { } statements first
    if (tokens[i].type == TokenType::At && i + 1 < tokens.size()) {
        ++i; // Skip '@'
        if (tokens[i].type != TokenType::Identifier) {
            std::cerr << "Error: Expected backend name after '@'" << std::endl;
            return nullptr;
        }
        std::string backend_name = tokens[i].text;
        ++i;

        if (i >= tokens.size() || tokens[i].type != TokenType::LBrace) {
            std::cerr << "Error: Expected '{' after backend name '@" << backend_name << "'" << std::endl;
            return nullptr;
        }
        ++i; // Skip '{'

        // Parse block content
        auto body = parse_block(tokens, i, false);
        if (!body) {
            std::cerr << "Error: Failed to parse backend block body" << std::endl;
            return nullptr;
        }

        return std::make_unique<BackendBlockStmt>(backend_name, std::move(body));
    }

    // Handle include statements first - only support quoted strings
    if (tokens[i].type == TokenType::Include && i + 1 < tokens.size()) {
        if (tokens[i + 1].type != TokenType::String) {
            std::cerr << "Error: Include statement requires a quoted string (e.g., include \"filename\";)" << std::endl;
            return nullptr;
        }
        std::string mod = tokens[i + 1].text;
        i += 2;
        if (i >= tokens.size() || tokens[i].type != TokenType::Semicolon) {
            std::cerr << "Error: Missing semicolon ';' after include statement" << std::endl;
            return nullptr;
        }
        ++i;
        return std::make_unique<IncludeStmt>(mod);
    }

    // Handle function definitions next, before other statement types
    if (tokens[i].type == TokenType::Func && i + 2 < tokens.size() &&
        tokens[i + 1].type == TokenType::Identifier && tokens[i + 2].type == TokenType::LParen) {
        std::string name = tokens[i + 1].text;
        int func_line = tokens[i].line;
        int func_col = tokens[i].column;
        DEBUG_OUT << "Debug - Starting function parsing: " << name << " at line " << func_line << " col " << func_col << std::endl;

        size_t func_start_index = i;// Record function start position for error reporting
        i += 3;                     // Skip 'func name('

        // Collect parameters
        std::vector<std::string> params;
        while (i < tokens.size() && tokens[i].type != TokenType::RParen) {
            if (tokens[i].type == TokenType::Identifier) {
                params.push_back(tokens[i].text);
                ++i;
                if (i < tokens.size() && tokens[i].type == TokenType::Comma) {
                    ++i;
                } else if (i < tokens.size() && tokens[i].type != TokenType::RParen) {
                    std::cerr << "\033[31mError: Function '" << name << "' parameter list missing comma, found: "
                              << tokens[i].text << " at line " << tokens[i].line << "\033[0m" << std::endl;
                    // Try to recover: look for closing parenthesis or comma
                    while (i < tokens.size() &&
                           tokens[i].type != TokenType::RParen &&
                           tokens[i].type != TokenType::Comma) {
                        ++i;
                    }
                    if (i < tokens.size() && tokens[i].type == TokenType::Comma) {
                        ++i;// Consume comma and continue
                    }
                }
            } else {
                std::cerr << "\033[31mError: Function '" << name << "' parameter list has invalid token: "
                          << tokens[i].text << " at line " << tokens[i].line << "\033[0m" << std::endl;

                // Try to recover: look for next comma or closing parenthesis
                while (i < tokens.size() &&
                       tokens[i].type != TokenType::RParen &&
                       tokens[i].type != TokenType::Comma) {
                    ++i;
                }
                if (i < tokens.size() && tokens[i].type == TokenType::Comma) {
                    ++i;// Consume comma and continue
                }
            }
        }

        if (i >= tokens.size()) {
            std::cerr << "\033[31mError: Function '" << name << "' ended unexpectedly while parsing parameters, started at line " << func_line << "\033[0m" << std::endl;
            // Output context around function start
            print_context(tokens, func_start_index);
            return nullptr;
        }

        if (tokens[i].type == TokenType::RParen) {
            ++i;// Consume closing parenthesis
        } else {
            std::cerr << "\033[31mError: Function '" << name << "' parameter list missing closing parenthesis, after line " << tokens[i - 1].line << " col " << tokens[i - 1].column << "\033[0m" << std::endl;
            print_context(tokens, i);
            // Try to recover: look for opening brace
            while (i < tokens.size() && tokens[i].type != TokenType::LBrace) {
                ++i;
            }
            if (i >= tokens.size()) {
                return nullptr;// Can't find opening brace, give up
            }
        }

        if (i >= tokens.size()) {
            std::cerr << "\033[31mError: Function '" << name << "' ended unexpectedly after closing parenthesis\033[0m" << std::endl;
            return nullptr;
        }

        if (tokens[i].type == TokenType::LBrace) {
            ++i;// Consume opening brace
            DEBUG_OUT << "Debug - Starting function '" << name << "' body parsing, line " << tokens[i - 1].line << " col " << tokens[i - 1].column << std::endl;
        } else {
            std::cerr << "\033[31mError: Function '" << name << "' definition missing opening brace '{', after line " << tokens[i - 1].line << "\033[0m" << std::endl;
            print_context(tokens, i);
            return nullptr;
        }

        // Parse function body
        try {
            auto body = parse_block(tokens, i, false);

            DEBUG_OUT << "Debug - Function '" << name << "' body parsing completed" << std::endl;

            // Note: parse_block now handles the closing brace, so no need to check again
            return std::make_unique<FuncDefStmt>(name, params, std::move(body));
        } catch (const std::exception& e) {
            std::cerr << "\033[31mError: Error parsing function '" << name << "' body: " << e.what() << "\033[0m" << std::endl;
            // Try to recover: look for closing brace or function definition end marker
            while (i < tokens.size() && tokens[i].type != TokenType::RBrace) {
                ++i;
            }
            if (i < tokens.size()) {
                ++i;// Skip closing brace
            }
            return nullptr;
        }
    }

    // Handle while statements
    if (tokens[i].type == TokenType::While) {
        return parse_while(tokens, i);
    }

    if (tokens[i].type == TokenType::Struct) {
        return parse_struct(tokens, i);
    }

    if (tokens[i].type == TokenType::Define && i + 2 < tokens.size() &&
        tokens[i + 1].type == TokenType::Identifier) {
        std::string name = tokens[i + 1].text;
        i += 2;
        auto expr = parse_expression(tokens, i);
        if (!expr) {
            std::cerr << "Error: Missing expression in define statement for '" << name << "'" << std::endl;
            return nullptr;
        }
        if (i >= tokens.size() || tokens[i].type != TokenType::Semicolon) {
            std::cerr << "Error: Missing semicolon ';' after define statement" << std::endl;
            return nullptr;
        }
        ++i;
        return std::make_unique<DefineStmt>(name, std::move(expr));
    } else if (tokens[i].type == TokenType::Bigint && i + 1 < tokens.size() &&
               tokens[i + 1].type == TokenType::Identifier) {
        std::string name = tokens[i + 1].text;
        i += 2;
        std::unique_ptr<Expression> init_value = nullptr;
        if (i < tokens.size() && tokens[i].type == TokenType::Assign) {
            ++i;
            init_value = parse_expression(tokens, i);
            if (!init_value) {
                std::cerr << "Error: Missing expression in bigint declaration for '" << name << "'" << std::endl;
                return nullptr;
            }
        }
        if (i >= tokens.size() || tokens[i].type != TokenType::Semicolon) {
            std::cerr << "Error: Missing semicolon ';' after bigint declaration" << std::endl;
            return nullptr;
        }
        ++i;
        return std::make_unique<BigIntDeclStmt>(name, std::move(init_value));
    } else if (tokens[i].type == TokenType::Var && tokens[i + 1].type == TokenType::Identifier && tokens[i + 2].type == TokenType::Assign) {
        std::string name = tokens[i + 1].text;
        DEBUG_OUT << "Debug - Parsing variable declaration: " << name << std::endl;
        i += 3;
        DEBUG_OUT << "Debug - About to parse expression for variable " << name << std::endl;
        auto expr = parse_expression(tokens, i);
        DEBUG_OUT << "Debug - Expression parsing result: " << (expr ? "success" : "failed") << std::endl;
        if (!expr) {
            std::cerr << "Error: Missing expression in variable declaration for '" << name << "'" << std::endl;
            return nullptr;
        }
        if (i >= tokens.size() || tokens[i].type != TokenType::Semicolon) {
            std::cerr << "Error: Missing semicolon ';' after variable declaration" << std::endl;
            return nullptr;
        }
        ++i;
        return std::make_unique<VarDeclStmt>(name, std::move(expr));
    } else if (tokens[i].type == TokenType::Identifier && tokens[i + 1].type == TokenType::Assign) {
        std::string name = tokens[i].text;
        i += 2;
        auto expr = parse_expression(tokens, i);
        if (!expr) {
            std::cerr << "Error: Missing expression in assignment to '" << name << "'" << std::endl;
            return nullptr;
        }
        if (i >= tokens.size() || tokens[i].type != TokenType::Semicolon) {
            std::cerr << "Error: Missing semicolon ';' after assignment" << std::endl;
            return nullptr;
        }
        ++i;
        return std::make_unique<AssignStmt>(name, std::move(expr));

    } else if (tokens[i].type == TokenType::Return) {
        ++i;
        auto expr = parse_expression(tokens, i);
        if (!expr) {
            std::cerr << "Error: Missing expression in return statement" << std::endl;
            return nullptr;
        }
        if (i >= tokens.size() || tokens[i].type != TokenType::Semicolon) {
            std::cerr << "Error: Missing semicolon ';' after return statement" << std::endl;
            return nullptr;
        }
        ++i;
        return std::make_unique<ReturnStmt>(std::move(expr));
    } else if (tokens[i].type == TokenType::Break) {
        ++i;
        if (i >= tokens.size() || tokens[i].type != TokenType::Semicolon) {
            std::cerr << "Error: Missing semicolon ';' after break statement" << std::endl;
            return nullptr;
        }
        ++i;
        return std::make_unique<BreakStmt>();
    } else if (tokens[i].type == TokenType::Continue) {
        ++i;
        if (i >= tokens.size() || tokens[i].type != TokenType::Semicolon) {
            std::cerr << "Error: Missing semicolon ';' after continue statement" << std::endl;
            return nullptr;
        }
        ++i;
        return std::make_unique<ContinueStmt>();
    } else if (tokens[i].type == TokenType::Semicolon) {
        ++i;// Empty statement
        return std::make_unique<NullStmt>();
    } else if (tokens[i].type == TokenType::If && i + 1 < tokens.size()) {
        // if statement
        int if_line = tokens[i].line;
        int if_col = tokens[i].column;
        size_t if_start_index = i;// Record if statement start position

        DEBUG_OUT << "Debug - Starting if statement parsing, line " << if_line << " column " << if_col << std::endl;

        ++i;// Skip 'if'

        // Check for left parenthesis
        if (i >= tokens.size() || tokens[i].type != TokenType::LParen) {
            std::cerr << "Error: Missing left parenthesis '(' after if statement" << std::endl;
            print_context(tokens, i > 0 ? i - 1 : 0);

            // Try to recover: look for left parenthesis or left brace
            while (i < tokens.size() &&
                   tokens[i].type != TokenType::LParen &&
                   tokens[i].type != TokenType::LBrace) {
                ++i;
            }

            if (i >= tokens.size() || tokens[i].type != TokenType::LParen) {
                return nullptr;// Cannot find recoverable point
            }
        }

        ++i;// Skip left parenthesis

        // Parse condition
        if (i >= tokens.size()) {
            std::cerr << "Error: if statement ended unexpectedly, missing condition expression" << std::endl;
            print_context(tokens, if_start_index);
            return nullptr;
        }

        auto cond = parse_expression(tokens, i);
        if (!cond) {
            std::cerr << "Error: if statement missing valid condition expression" << std::endl;
            print_context(tokens, i);

            // Try to recover: look for right parenthesis
            while (i < tokens.size() && tokens[i].type != TokenType::RParen) {
                ++i;
            }

            if (i >= tokens.size()) {
                return nullptr;// Cannot find right parenthesis, cannot continue
            }
        }

        // Check right parenthesis
        if (i >= tokens.size()) {
            std::cerr << "Error: if statement condition ended unexpectedly, missing right parenthesis" << std::endl;
            print_context(tokens, if_start_index);
            return nullptr;
        }

        if (tokens[i].type == TokenType::RParen) {
            ++i;// Consume right parenthesis
        } else {
            std::cerr << "Error: if statement condition missing right parenthesis ')', after line " << tokens[i - 1].line << std::endl;
            print_context(tokens, i);

            // Try to recover: look for left brace
            while (i < tokens.size() && tokens[i].type != TokenType::LBrace) {
                ++i;
            }

            if (i >= tokens.size()) {
                return nullptr;// Cannot find left brace, cannot continue
            }
        }

        // Check left brace
        if (i >= tokens.size()) {
            std::cerr << "Error: if statement ended unexpectedly after right parenthesis, missing body" << std::endl;
            print_context(tokens, if_start_index);
            return nullptr;
        }

        if (tokens[i].type == TokenType::LBrace) {
            ++i;// Consume left brace
        } else {
            std::cerr << "Error: if statement missing left brace '{'" << std::endl;
            print_context(tokens, i);

            // Try to recover: create an empty block and return
            auto emptyThenBlock = std::make_unique<BlockStmt>();
            return std::make_unique<IfStmt>(std::move(cond), std::move(emptyThenBlock), nullptr);
        }

        // Parse then block
        DEBUG_OUT << "Debug - Parsing if then block" << std::endl;
        std::unique_ptr<BlockStmt> thenBlock;

        try {
            thenBlock = parse_block(tokens, i, false);
            DEBUG_OUT << "Debug - if then block parsing completed, contains " << thenBlock->statements.size() << " statements" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: Error parsing if then block: " << e.what() << std::endl;
            // Create empty block as recovery measure
            thenBlock = std::make_unique<BlockStmt>();

            // Try to locate else keyword or next statement
            while (i < tokens.size() &&
                   tokens[i].type != TokenType::Else &&
                   tokens[i].type != TokenType::Semicolon &&
                   tokens[i].type != TokenType::RBrace) {
                ++i;
            }

            if (i >= tokens.size()) {
                return std::make_unique<IfStmt>(std::move(cond), std::move(thenBlock), nullptr);
            }
        }

        // Check for else block
        std::unique_ptr<BlockStmt> elseBlock;
        if (i < tokens.size() && tokens[i].type == TokenType::Else) {
            int else_line = tokens[i].line;
            ++i;// Consume else keyword
            DEBUG_OUT << "Debug - Starting else block parsing, line " << else_line << std::endl;

            if (i >= tokens.size()) {
                std::cerr << "Error: else keyword ended unexpectedly" << std::endl;
                return std::make_unique<IfStmt>(std::move(cond), std::move(thenBlock), nullptr);
            }

            // Check if it's else if
            if (tokens[i].type == TokenType::If) {
                DEBUG_OUT << "Debug - Detected else if, parsing recursively" << std::endl;
                // Recursively parse else if as a new if statement
                auto nestedIf = parse_statement(tokens, i);
                if (nestedIf) {
                    elseBlock = std::make_unique<BlockStmt>();
                    elseBlock->statements.push_back(std::move(nestedIf));
                    DEBUG_OUT << "Debug - else if parsing completed" << std::endl;
                } else {
                    std::cerr << "Error: else if parsing failed" << std::endl;
                    return std::make_unique<IfStmt>(std::move(cond), std::move(thenBlock), nullptr);
                }
            } else if (tokens[i].type == TokenType::LBrace) {
                // Regular else block
                ++i;// Consume left brace

                // Parse else block
                try {
                    elseBlock = parse_block(tokens, i, false);
                    DEBUG_OUT << "Debug - else block parsing completed, contains " << elseBlock->statements.size() << " statements" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Error: Error parsing else block: " << e.what() << std::endl;
                    // Create empty block as recovery measure
                    elseBlock = std::make_unique<BlockStmt>();
                }
            } else {
                std::cerr << "Error: else block missing left brace '{'" << std::endl;
                print_context(tokens, i);

                // Return if statement without else block
                return std::make_unique<IfStmt>(std::move(cond), std::move(thenBlock), nullptr);
            }
        }

        DEBUG_OUT << "Debug - if statement parsing completed, defined at line " << if_line
                  << ", ended at line " << (i < tokens.size() ? tokens[i - 1].line : -1) << std::endl;

        return std::make_unique<IfStmt>(std::move(cond), std::move(thenBlock), std::move(elseBlock));
    } else {
        // Expression statement (including function calls)
        if (i < tokens.size() && tokens[i].type != TokenType::EndOfFile) {
            auto expr = parse_expression(tokens, i);
            if (i >= tokens.size() || tokens[i].type != TokenType::Semicolon) {
                std::cerr << "Error: Missing semicolon ';' after expression statement" << std::endl;
                return nullptr;
            }
            ++i;
            if (expr) return std::make_unique<ExprStmt>(std::move(expr));
        }
    }

    // Add more detailed debug output and error handling at the end
    if (i < tokens.size()) {
        // Encountered an unrecognized token, provide detailed context
        std::cerr << "\033[1;31mError:\033[0m Unrecognized syntax element '" << tokens[i].text
                  << "' (type=" << static_cast<int>(tokens[i].type)
                  << ") at line " << tokens[i].line
                  << " column " << tokens[i].column << std::endl;

        // Print context
        std::cerr << "\033[1;33mContext:\033[0m ";
        size_t context_start = i > 5 ? i - 5 : 0;
        size_t context_end = std::min(i + 5, tokens.size() - 1);

        for (size_t j = context_start; j <= context_end; j++) {
            if (j == i) {
                std::cerr << "\033[1;31m[" << tokens[j].text << "]\033[0m ";
            } else {
                std::cerr << tokens[j].text << " ";
            }
        }
        std::cerr << std::endl;

        // Try to provide possible error causes
        if (tokens[i].type == TokenType::RBrace) {
            std::cerr << "Hint: Found extra closing brace '}', may be a block nesting issue" << std::endl;
        } else if (tokens[i].type == TokenType::RParen) {
            std::cerr << "Hint: Found extra closing parenthesis ')', check expressions or conditional statements" << std::endl;
        } else if (tokens[i].type == TokenType::Else) {
            std::cerr << "Hint: 'else' keyword missing complete if statement before it" << std::endl;
        }
    } else {
        std::cerr << "\033[31mError: Parser ended unexpectedly at end of file\033[0m" << std::endl;
    }

    // Return null pointer to indicate parsing failure
    return nullptr;
}

std::unique_ptr<ASTNode> Parser::parse(const std::vector<Token>& tokens) {
    size_t i = 0;
    try {
        DEBUG_OUT << "Debug - Starting file parsing, total tokens: " << tokens.size() << std::endl;
        auto result = parse_block(tokens, i, true);// Main block

        // Verify that all tokens were consumed
        if (i < tokens.size() && tokens[i].type != TokenType::EndOfFile) {
            std::cerr << "\033[33mWarning: Unprocessed tokens remain after parsing completion, starting from position " << i << "\033[0m" << std::endl;
            std::cerr << "First unprocessed token: " << tokens[i].text
                      << " (line " << tokens[i].line << ")" << std::endl;
        }

        return result;
    } catch (const std::exception& e) {
        std::cerr << "\033[31mParse error: " << e.what() << "\033[0m" << std::endl;
        return nullptr;
    }
}
