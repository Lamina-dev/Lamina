#include "../runtime/object/literal.hpp"
#include "../runtime/object/complex.hpp"
#include "../runtime/object/tuple.hpp"
#include "../runtime/object/value.hpp"

#include <iostream>
#include <vector>

using lmx::runtime::LiteralObj;
using lmx::runtime::ComplexObj;
using lmx::runtime::TupleObj;
using lmx::runtime::Value;
using lmx::runtime::ValueKind;

namespace {
bool require(const bool condition, const char* message) {
    if (condition) return true;
    std::cerr << message << '\n';
    return false;
}
}

int main() {
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

    return 0;
}
