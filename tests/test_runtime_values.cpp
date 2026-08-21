#include "../runtime/object/literal.hpp"
#include "../runtime/object/complex.hpp"
#include "../runtime/object/tuple.hpp"
#include "../runtime/object/vector.hpp"
#include "../runtime/object/matrix.hpp"
#include "../runtime/object/table.hpp"
#include "../runtime/object/quantity.hpp"
#include "../runtime/object/value.hpp"

#include <iostream>
#include <vector>

using lmx::runtime::LiteralObj;
using lmx::runtime::ComplexObj;
using lmx::runtime::TupleObj;
using lmx::runtime::Value;
using lmx::runtime::ValueKind;
using lmx::runtime::VectorObj;
using lmx::runtime::MatrixObj;
using lmx::runtime::TableObj;
using lmx::runtime::QuantityObj;

namespace {
bool require(const bool condition, const char* message) {
    if (condition) return true;
    std::cerr << message << '\n';
    return false;
}
}

int main() {
    const Value left_fraction(2, 4);
    const Value right_fraction(1, 2);
    if (!require(left_fraction == right_fraction,
                 "fraction equality must use normalized numeric components") ||
        !require(left_fraction.hash() == right_fraction.hash(),
                 "equal fractions must have equal structural hashes")) return 1;

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

    return 0;
}
