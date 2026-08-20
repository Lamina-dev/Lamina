#pragma once

#include <stdint.h>

/* Stable opcodes for the Lamina Runtime -> LMCAS construction boundary. */
typedef enum LmExprUnaryOp {
    LMX_EXPR_NEG = 0,
    LMX_EXPR_NOT = 1,
} LmExprUnaryOp;

typedef enum LmExprBinaryOp {
    LMX_EXPR_ADD = 0,
    LMX_EXPR_SUB = 1,
    LMX_EXPR_MUL = 2,
    LMX_EXPR_DIV = 3,
    LMX_EXPR_POW = 4,
    LMX_EXPR_EQ = 5,
    LMX_EXPR_NE = 6,
    LMX_EXPR_GT = 7,
    LMX_EXPR_GE = 8,
    LMX_EXPR_LT = 9,
    LMX_EXPR_LE = 10,
    LMX_EXPR_AND = 11,
    LMX_EXPR_OR = 12,
    LMX_EXPR_IN = 13,
    LMX_EXPR_NOT_IN = 14,
} LmExprBinaryOp;
