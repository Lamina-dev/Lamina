#ifndef RANGE_HPP
#define RANGE_HPP
#include "lamina.hpp"
#include "lstruct.hpp"
#include "cas.hpp"
#include <vector>
#include <limits>

/**
 Range is a kind of set. Must be at least SymbolicExpr to use range
	(or no comparison is available).
*/

#define PRE_FLAGS 0

#define PRE_FLAGS_PRE_SORT 1

using RangeValue = std::shared_ptr<SymbolicExpr>;

// Return if a > b.
bool is_greater(const RangeValue& a, const RangeValue& b);

// Return if a < b.
bool is_less(const RangeValue& a, const RangeValue& b);

// Return if a == b.
bool is_equal(const RangeValue& a, const RangeValue& b);

RangeValue from_lamina(const Value& src);

Value to_lamina(const RangeValue& src);

RangeValue mini(const RangeValue& a, const RangeValue& b);

RangeValue maxi(const RangeValue& a, const RangeValue& b);

struct Range {
	
	struct BasicRange {
		
		BasicRange()
			: l(SymbolicExpr::number(0)), r(SymbolicExpr::number(0)) {}
		BasicRange(RangeValue l, RangeValue r, bool l_incl = true, bool r_incl = true) 
			: l(l), r(r), l_incl(l_incl), r_incl(r_incl) {}
		
		RangeValue l, r;
		bool l_incl = false, r_incl = false;	// Empty set by default for better operation
		char flag = 0;							// Flag = 1: merge failure
		
		// For sorter only
		bool operator < (const BasicRange &other) const;
		bool is_empty() const;
		bool in_range(const RangeValue& val) const;
		BasicRange intersect(const BasicRange &other) const;
		bool can_merge(const BasicRange &other) const;
		
		// Warning: before calling this, must check weak interaction
		BasicRange try_merge(const BasicRange &other) const;
		std::string to_string() const;
		
	};
	
	Range() {
		
	}

	explicit Range(Value lamina_value);

	Value lamina() const;
	
	bool in_range(const RangeValue& val) const;
	
	std::string to_string() const;
	
	std::vector<BasicRange> segments;
};

Range intersect(const Range &a, const Range &b);
Range join(const Range &a, const Range &b);

// TODO: Reserve all conditions if comparision is not available.
// No extension/inheritance is given because they are actually different.
struct CASRange {
	// To be implemented ...
};

// TODO: Intersection and Merging of CAS Ranges.

// Lamina-compatible interfaces:
// (Infinity values are given here because inf are only used here)

/**
 * function: inf()
*/
Value lamina_inf(const std::vector<Value> &args);

/**
 * function: neginf()
 */
Value lamina_neginf(const std::vector<Value> &args);

/**
 * function: range(l, r)
 * generating [l, r] by default
 */
Value lamina_range(const std::vector<Value> &args);

/**
 * function: rangex(l, r, has_l, has_r)
 * (TODO: Probably we need range literals for a math-processing language!)
 */
Value lamina_rangex(const std::vector<Value> &args);

Value lamina_intersect(const std::vector<Value> &args);

Value lamina_join(const std::vector<Value> &args);

/**
 * function: in_range(range, value)
 */
Value lamina_range_test(const std::vector<Value> &args);


namespace Lamina {
	LAMINA_FUNC("inf", lamina_inf, 0);
	LAMINA_FUNC("neginf", lamina_neginf, 0);
	LAMINA_FUNC("ranges", lamina_range, 2);
	LAMINA_FUNC("rangex", lamina_rangex, 4);
	LAMINA_FUNC_WIT_ANY_ARGS("intersect", lamina_intersect);
	LAMINA_FUNC_WIT_ANY_ARGS("join", lamina_join);
	LAMINA_FUNC("in_range", lamina_range_test, 2);
}

#endif