#include "../runtime/object/literal.hpp"
#include "../runtime/object/complex.hpp"
#include "../runtime/object/tuple.hpp"
#include "../runtime/object/vector.hpp"
#include "../runtime/object/matrix.hpp"
#include "../runtime/object/table.hpp"
#include "../runtime/object/quantity.hpp"
#include "../runtime/object/sparse.hpp"
#include "../runtime/object/tensor.hpp"
#include "../runtime/object/assumptions.hpp"
#include "lmmc/init.h"
#include "../runtime/object/value.hpp"

#include <exception>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <memory>

using lmx::runtime::LiteralObj;
using lmx::runtime::ComplexObj;
using lmx::runtime::TupleObj;
using lmx::runtime::Value;
using lmx::runtime::ValueKind;
using lmx::runtime::VectorObj;
using lmx::runtime::MatrixObj;
using lmx::runtime::TableObj;
using lmx::runtime::QuantityObj;
using lmx::runtime::SparseMatrixObj;
using lmx::runtime::TensorObj;
using lmx::runtime::AssumptionsObj;
using lmx::runtime::Fraction;

namespace {
bool require(const bool condition, const char* message) {
    if (condition) return true;
    std::cerr << message << '\n';
    return false;
}

template <typename Error, typename Operation>
bool require_throws(Operation&& operation, const char* message) {
    try {
        operation();
    } catch (const Error&) {
        return true;
    } catch (...) {
    }
    std::cerr << message << '\n';
    return false;
}
}

lmmc_sparse_mat_t make_sparse() {
    lmmc_sparse_builder_t* builder = nullptr;
    if (lmmc_sparse_builder_create(2, 2, 3, &builder) != LMMC_STATUS_OK)
        return {};
    (void)lmmc_sparse_builder_add(builder, 0, 0, 4.0);
    (void)lmmc_sparse_builder_add(builder, 0, 1, 1.0);
    (void)lmmc_sparse_builder_add(builder, 1, 1, 3.0);
    lmmc_sparse_mat_t result{};
    (void)lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, &result);
    lmmc_sparse_builder_destroy(builder);
    return result;
}

struct LmmcGuard {
    LmmcGuard() {
        if (lmmc_init() != LMMC_STATUS_OK) std::terminate();
    }
    ~LmmcGuard() {
        if (lmmc_deinit() != LMMC_STATUS_OK) std::terminate();
    }
};

int main() {
    const LmmcGuard lmmc_guard;
    const Value left_fraction(2, 4);
    const Value right_fraction(1, 2);
    if (!require(left_fraction == right_fraction,
                 "fraction equality must use normalized numeric components") ||
        !require(left_fraction.hash() == right_fraction.hash(),
                 "equal fractions must have equal structural hashes")) return 1;

    const Fraction normalized_sign(-2, -4);
    const Fraction normalized_zero(0, -7);
    const Fraction half(1, 2);
    const Fraction whole(8, 4);
    const Fraction rational_remainder = Fraction(5, 2) % Fraction(3, 2);
    if (!require(normalized_sign.numerator() == 1 &&
                     normalized_sign.denominator() == 2,
                 "fraction normalization must canonicalize denominator signs") ||
        !require(normalized_zero.numerator() == 0 &&
                     normalized_zero.denominator() == 1,
                 "fraction zero must use the canonical 0/1 representation") ||
        !require(half != 0 && half != 1,
                 "fraction-to-integer equality must not accept non-integers") ||
        !require(whole == 2,
                 "fraction-to-integer equality must accept normalized integers") ||
        !require(Fraction(std::numeric_limits<std::int32_t>::max(),
                          std::numeric_limits<std::int32_t>::max() - 1) >
                     Fraction(std::numeric_limits<std::int32_t>::max() - 1,
                              std::numeric_limits<std::int32_t>::max()),
                 "fraction comparison must use non-overflowing cross products") ||
        !require(rational_remainder == Fraction(1, 1),
                 "fraction remainder must preserve rational arithmetic") ||
        !require((half % 7) == 4,
                 "fraction modular reduction must use a modular inverse") ||
        !require_throws<std::domain_error>(
            [] { (void)Fraction(1, 0); },
            "zero denominators must be rejected") ||
        !require_throws<std::invalid_argument>(
            [] { (void)Fraction(std::string("1/2junk")); },
            "fraction parsing must consume the complete input") ||
        !require_throws<std::overflow_error>(
            [] {
                (void)(Fraction(std::numeric_limits<std::int32_t>::max(), 1) +
                       Fraction(1, 1));
            },
            "fraction arithmetic overflow must be reported") ||
        !require_throws<std::overflow_error>(
            [] {
                (void)-Fraction(std::numeric_limits<std::int32_t>::min(), 1);
            },
            "fraction negation overflow must be reported") ||
        !require_throws<std::domain_error>(
            [] { (void)(Fraction(1, 2) / Fraction(0, 1)); },
            "fraction division by zero must be rejected") ||
        !require_throws<std::domain_error>(
            [] { (void)(Fraction(1, 2) % 6); },
            "non-invertible modular denominators must be rejected") ||
        !require(!std::isfinite(std::numeric_limits<double>::quiet_NaN()),
                 "runtime compilation must preserve IEEE non-finite checks")) return 1;

    auto* left_tuple_object = new TupleObj(2);
    left_tuple_object->set(0, Value(static_cast<LmInt>(1)));
    left_tuple_object->set(1, Value(static_cast<LmInt>(2)));
    Value left_tuple(left_tuple_object, ValueKind::Tuple);

    auto* right_tuple_object = new TupleObj(2);
    right_tuple_object->set(0, Value(static_cast<LmInt>(1)));
    right_tuple_object->set(1, Value(static_cast<LmInt>(2)));
    Value right_tuple(right_tuple_object, ValueKind::Tuple);

    if (!require(left_tuple.kind == ValueKind::Tuple,
                 "tuple must retain its stable runtime kind") ||
        !require(left_tuple == right_tuple,
                 "tuple equality must be structural") ||
        !require(left_tuple.hash() == right_tuple.hash(),
                 "equal tuples must have equal hashes") ||
        !require(left_tuple.to_string() == "(1, 2)",
                 "tuple rendering must preserve its elements")) return 1;

    Value left_set(new LiteralObj(
        LiteralObj::Kind::Set,
        std::vector<Value>{Value(static_cast<LmInt>(1)),
                           Value(static_cast<LmInt>(2)),
                           Value(static_cast<LmInt>(1))}),
        ValueKind::Set);
    Value right_set(new LiteralObj(
        LiteralObj::Kind::Set,
        std::vector<Value>{Value(static_cast<LmInt>(2)),
                           Value(static_cast<LmInt>(1))}),
        ValueKind::Set);

    if (!require(left_set.kind == ValueKind::Set,
                 "set must retain its stable runtime kind") ||
        !require(left_set == right_set,
                 "set equality must ignore order and duplicates") ||
        !require(left_set.hash() == right_set.hash(),
                 "equal sets must have equal structural hashes")) return 1;

    Value left_interval(new LiteralObj(
        LiteralObj::Kind::Interval,
        std::vector<Value>{Value(static_cast<LmInt>(0)),
                           Value(static_cast<LmInt>(1))},
        true, false), ValueKind::Interval);
    Value right_interval(new LiteralObj(
        LiteralObj::Kind::Interval,
        std::vector<Value>{Value(static_cast<LmInt>(0)),
                           Value(static_cast<LmInt>(1))},
        true, false), ValueKind::Interval);

    if (!require(left_interval.kind == ValueKind::Interval,
                 "interval must retain its stable runtime kind") ||
        !require(left_interval == right_interval,
                 "interval equality must include bounds and endpoints") ||
        !require(left_interval.hash() == right_interval.hash(),
                 "equal intervals must have equal structural hashes") ||
        !require(left_interval.to_string() == "[0, 1)",
                 "interval rendering must preserve bound openness")) return 1;

    Value left_complex(new ComplexObj(3.0, 4.0), ValueKind::Complex);
    Value right_complex(new ComplexObj(3.0, 4.0), ValueKind::Complex);
    if (!require(left_complex.kind == ValueKind::Complex,
                 "complex must retain its stable runtime kind") ||
        !require(left_complex == right_complex,
                 "complex equality must compare both components") ||
        !require(left_complex.hash() == right_complex.hash(),
                 "equal complex values must have equal hashes") ||
        !require(left_complex.to_string() == "3 + 4I",
                 "complex rendering must use the uppercase imaginary unit")) return 1;

    Value left_vector(new VectorObj({1.0, 2.0, 3.0}), ValueKind::Vector);
    Value right_vector(new VectorObj({1.0, 2.0, 3.0}), ValueKind::Vector);
    if (!require(left_vector == right_vector, "vector equality must be structural") ||
        !require(left_vector.hash() == right_vector.hash(),
                 "equal vectors must have equal hashes") ||
        !require(left_vector.to_string() == "[1, 2, 3]",
                 "vector rendering must preserve order")) return 1;

    Value left_matrix(new MatrixObj(2, 2, {1.0, 2.0, 3.0, 4.0}), ValueKind::Matrix);
    Value right_matrix(new MatrixObj(2, 2, {1.0, 2.0, 3.0, 4.0}), ValueKind::Matrix);
    if (!require(left_matrix == right_matrix, "matrix equality must include shape and data") ||
        !require(left_matrix.hash() == right_matrix.hash(),
                 "equal matrices must have equal hashes")) return 1;

    Value left_table(new TableObj({{"b", Value(static_cast<LmInt>(2))},
                                   {"a", Value(static_cast<LmInt>(1))}}), ValueKind::Table);
    Value right_table(new TableObj({{"a", Value(static_cast<LmInt>(1))},
                                    {"b", Value(static_cast<LmInt>(2))}}), ValueKind::Table);
    if (!require(left_table == right_table, "table equality must ignore insertion order") ||
        !require(left_table.hash() == right_table.hash(),
                 "equal tables must have equal hashes") ||
        !require(left_table.to_string() == "{a: 1, b: 2}",
                 "table rendering must use deterministic key order")) return 1;

    Value kilometre(new QuantityObj(1000.0, "km"), ValueKind::Quantity);
    Value metres(new QuantityObj(1000.0, "m"), ValueKind::Quantity);
    if (!require(kilometre == metres,
                 "quantity equality must compare physical value and dimension") ||
        !require(kilometre.hash() == metres.hash(),
                 "equivalent quantities must have equal hashes") ||
        !require(kilometre.to_string() == "1 km",
                 "quantity rendering must convert from SI to its display unit")) return 1;

    Value left_sparse(new SparseMatrixObj(make_sparse()), ValueKind::Sparse);
    Value right_sparse(new SparseMatrixObj(make_sparse()), ValueKind::Sparse);
    if (!require(static_cast<SparseMatrixObj*>(left_sparse.obj)->valid(),
                 "sparse object must own canonical CSR") ||
        !require(left_sparse == right_sparse,
                 "sparse equality must compare canonical CSR structure") ||
        !require(left_sparse.hash() == right_sparse.hash(),
                 "equal sparse matrices must have equal hashes") ||
        !require(left_sparse.to_string() == "sparse(2x2, nnz=3)",
                 "sparse rendering must include shape and nnz")) return 1;

    const std::size_t tensor_dims[] = {2, 2, 2};
    lmmc_tensor_nd_t owning_tensor{};
    if (!require(lmmc_tensor_create(3, tensor_dims, &owning_tensor) ==
                     LMMC_STATUS_OK,
                 "tensor allocation must succeed")) return 1;
    for (std::size_t i = 0; i < 8; ++i) owning_tensor.data[i] = i + 1.0;
    auto* tensor_object = new TensorObj(std::move(owning_tensor));
    lmmc_tensor_nd_t reshaped{};
    const std::size_t reshaped_dims[] = {4, 2};
    if (!require(lmmc_tensor_nd_reshape_view(
                     &tensor_object->tensor(), 2, reshaped_dims, &reshaped) ==
                     LMMC_STATUS_OK,
                 "tensor reshape view must succeed")) return 1;
    Value tensor_source(tensor_object, ValueKind::Tensor);
    Value tensor_view(new TensorObj(tensor_object->storage(), reshaped),
                      ValueKind::Tensor);
    tensor_source = Value{};
    const auto* retained_view = static_cast<const TensorObj*>(tensor_view.obj);
    if (!require(retained_view->tensor().data[7] == 8.0,
                 "tensor view must retain shared storage after source release") ||
        !require(tensor_view.to_string() == "tensor(4x2)",
                 "tensor rendering must include its view shape")) return 1;

    lmmc_tensor_nd_t equal_tensor{};
    if (!require(lmmc_tensor_create(2, reshaped_dims, &equal_tensor) ==
                     LMMC_STATUS_OK,
                 "comparison tensor allocation must succeed")) return 1;
    for (std::size_t i = 0; i < 8; ++i) equal_tensor.data[i] = i + 1.0;
    Value tensor_equal(new TensorObj(std::move(equal_tensor)), ValueKind::Tensor);
    if (!require(tensor_view == tensor_equal,
                 "tensor equality must compare shape and logical values") ||
        !require(tensor_view.hash() == tensor_equal.hash(),
                 "equal tensor views must have equal hashes")) return 1;

    Value empty_assumptions(new AssumptionsObj(), ValueKind::Assumptions);
    Value copied_assumptions(
        static_cast<AssumptionsObj*>(empty_assumptions.obj)->copy(),
        ValueKind::Assumptions);
    const auto assumption_status =
        static_cast<AssumptionsObj*>(copied_assumptions.obj)
            ->context().assume_domain_checked("x", lamina::Domain::Real);
    if (!require(static_cast<bool>(assumption_status),
                 "checked assumption mutation must succeed") ||
        !require(empty_assumptions != copied_assumptions,
                 "assumption updates must not mutate the source context") ||
        !require(empty_assumptions.to_string() == "assumptions(depth=1)",
                 "assumption rendering must expose scope depth")) return 1;

    auto* retained = static_cast<AssumptionsObj*>(empty_assumptions.obj);
    {
        Value second_reference = empty_assumptions;
        if (!require(retained->get_rc() == 2,
                     "runtime object copy must retain assumptions")) return 1;
    }
    if (!require(retained->get_rc() == 1,
                 "runtime object destruction must release assumptions")) return 1;

    return 0;
}
