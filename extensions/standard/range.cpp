#include "range.hpp"

// Return if a > b.
bool is_greater(const RangeValue& a, const RangeValue& b) {
	return a->to_double() > b->to_double();
}

// Return if a < b.
bool is_less(const RangeValue& a, const RangeValue& b) {
	return a->to_double() < b->to_double();
}

// Return if a == b.
bool is_equal(const RangeValue& a, const RangeValue& b) {
	return a->to_double() == b->to_double();
}

RangeValue from_lamina(const Value& src) {
	return src.as_symbolic();
}

Value to_lamina(const RangeValue& src) {
	return Value(src);
}

RangeValue mini(const RangeValue& a, const RangeValue& b) {
	if (is_less(a, b)) return a;
	else return b;
}

RangeValue maxi(const RangeValue& a, const RangeValue& b) {
	if (is_greater(a, b)) return a;
	else return b;
}

bool Range::BasicRange::operator < (const BasicRange &other) const {
	return is_less(this->l, other.l);
}
		
bool Range::BasicRange::is_empty() const {
	if (is_equal(l, r)) {
		return (!l_incl) || (!r_incl);
	} else return is_greater(l, r);
}
		
bool Range::BasicRange::in_range(const RangeValue& val) const {
	if (l_incl && is_equal(val, l)) return true;
	if (r_incl && is_equal(val, r)) return true;
	if (is_greater(val, l) && is_less(val, r)) return true;
	return false;
}
		
Range::BasicRange Range::BasicRange::intersect(const Range::BasicRange &other) const {
	Range::BasicRange result;
	if (is_greater(l, other.l)) {
		result.l = l;
		result.l_incl = l_incl;
	} else if (is_equal(l, other.l)) {
		result.l = l;
		result.l_incl = l_incl && other.l_incl;
	} else {
		result.l = other.l;
		result.l_incl = other.l_incl;
	}
	if (is_less(r, other.r)) {
		result.r = r;
		result.r_incl = r_incl;
	} else if (is_equal(r, other.r)) {
		result.r = r;
		result.r_incl = r_incl && other.r_incl;
	} else {
		result.r = other.r;
		result.r_incl = other.r_incl;
	}
	return result;
}
		
bool Range::BasicRange::can_merge(const Range::BasicRange &other) const {
	Range::BasicRange result;
	if (is_greater(l, other.l)) {
		result.l = l;
		result.l_incl = l_incl;
	} else if (is_equal(l, other.l)) {
		result.l = l;
		result.l_incl = l_incl || other.l_incl;
	} else {
		result.l = other.l;
		result.l_incl = other.l_incl;
	}
	if (is_less(r, other.r)) {
		result.r = r;
		result.r_incl = r_incl;
	} else if (is_equal(r, other.r)) {
		result.r = r;
		result.r_incl = r_incl || other.r_incl;
	} else {
		result.r = other.r;
		result.r_incl = other.r_incl;
	}
	return !result.is_empty();
}
		
Range::BasicRange Range::BasicRange::try_merge(const Range::BasicRange &other) const {
	Range::BasicRange result;
	if (is_greater(l, other.l)) {
		result.l = other.l;
		result.l_incl = other.l_incl;
	} else if (is_equal(l, other.l)) {
		result.l = l;
		result.l_incl = l_incl || other.l_incl;
	} else {
		result.l = l;
		result.l_incl = l_incl;
	}
	if (is_less(r, other.r)) {
		result.r = other.r;
		result.r_incl = other.r_incl;
	} else if (is_equal(r, other.r)) {
		result.r = r;
		result.r_incl = r_incl || other.r_incl;
	} else {
		result.r = r;
		result.r_incl = r_incl;
	}
	return result;
}
		
std::string Range::BasicRange::to_string() const {
	std::string lclose, rclose;
	if (l_incl) lclose = "[";
	else lclose = "(";
	if (r_incl) rclose = "]";
	else rclose = ")";
	return lclose + l->to_string() + "," + r->to_string() + rclose;
}

Range::Range(Value lamina_value) {
	if (!lamina_value.is_lstruct()) return;
	auto lamina_val = std::get<std::shared_ptr<lStruct> >(lamina_value.data);
	int sz = getattr_raw(lamina_val, "size").as_number();
	for (int i = 1; i <= sz; i++) {
		Value l = getattr_raw(lamina_val, "l_" + std::to_string(i));
		Value r = getattr_raw(lamina_val, "r_" + std::to_string(i));
		Value l_incl = getattr_raw(lamina_val, "l_inc_" + std::to_string(i));
		Value r_incl = getattr_raw(lamina_val, "r_inc_" + std::to_string(i));
		
		segments.push_back(BasicRange(from_lamina(l), from_lamina(r), l_incl.as_bool(), r_incl.as_bool()));
	}
	if (PRE_FLAGS & PRE_FLAGS_PRE_SORT) sort(segments.begin(), segments.end());
}

Value Range::lamina() const {
	auto ls = std::make_shared<lStruct>();
	Value tmp = Value((int)segments.size());
	ls->insert("size", tmp);
	int it = 1;
	for (const auto& i : segments) {
		tmp = to_lamina(i.l), ls->insert("l_" + std::to_string(it), tmp);
		tmp = to_lamina(i.r), ls->insert("r_" + std::to_string(it), tmp);
		tmp = Value(i.l_incl), ls->insert("l_inc_" + std::to_string(it), tmp);
		tmp = Value(i.r_incl), ls->insert("r_inc_" + std::to_string(it), tmp);
		it++;
	}
	return Value(ls);
}
	
bool Range::in_range(const RangeValue& val) const {
	for (const auto& i : segments) {
		if (is_greater(val, i.r)) continue;
		if (i.in_range(val)) return true;
		if (is_less(val, i.l)) break;
	}
	return false;
}
	
std::string Range::to_string() const {
	if (!segments.size()) return std::string("<empty range>");
	std::string result;
	for (const auto& i : segments) {
		result += i.to_string();
		result.push_back('u');	// Union symbol
	}
	result.pop_back();
	return result;
}


Range intersect(const Range &a, const Range &b) {
	int p1 = 0, p2 = 0;
	Range result;
	while (p1 < a.segments.size() && p2 < b.segments.size()) {
		while (p1 < a.segments.size() && p2 < b.segments.size()
			&& is_less(b.segments[p2].r, a.segments[p1].l)) {
				if (p1 < a.segments.size() && p2 < b.segments.size()
					&& (is_less(a.segments[p1].r, b.segments[p2].r)
					|| (is_equal(a.segments[p1].r, b.segments[p2].r) && (!a.segments[p1].r_incl) && (b.segments[p2].r_incl))))
					p1++;
				else
					p2++;
			}
		if (!(p1 < a.segments.size() && p2 < b.segments.size())) break;
		Range::BasicRange pre_result = a.segments[p1].intersect(b.segments[p2]);
		if (!pre_result.is_empty()) result.segments.push_back(pre_result);
		// Jump to next one to evaluate intersection:
		if (p1 < a.segments.size() && p2 < b.segments.size()
			&& (is_less(a.segments[p1].r, b.segments[p2].r)
			|| (is_equal(a.segments[p1].r, b.segments[p2].r) && (!a.segments[p1].r_incl) && (b.segments[p2].r_incl))))
			p1++;
		else
			p2++;
	}
	return result;
}

Range join(const Range &a, const Range &b) {
	if (!a.segments.size()) return b;
	int p1 = 0, p2 = 0;
	Range result;
	Range::BasicRange current_seg = a.segments[0];
	while (p1 < a.segments.size() && p2 < b.segments.size()) {
		bool flag = true;
		while (flag) {
			flag = false;
			while (p1 < a.segments.size() && current_seg.can_merge(a.segments[p1])) {
				current_seg = current_seg.try_merge(a.segments[p1]);
				flag = true;
				p1++;
			}
			while (p2 < b.segments.size() && current_seg.can_merge(b.segments[p2])) {
				current_seg = current_seg.try_merge(b.segments[p2]);
				flag = true;
				p2++;
			}
		}
		result.segments.push_back(current_seg);
		current_seg = Range::BasicRange();
		if (p1 < a.segments.size()) {
			current_seg = a.segments[p1];
			p1++;
		}
		else if (p2 < b.segments.size()) {
			current_seg = b.segments[p2];
			p2++;
		}
	}
	if (!current_seg.is_empty()) result.segments.push_back(current_seg);
	return result;
}

/**
 * function: inf()
*/
Value lamina_inf(const std::vector<Value> &args) {
	return Value(std::numeric_limits<double>::infinity());
}

/**
 * function: neginf()
 */
Value lamina_neginf(const std::vector<Value> &args) {
	return Value(-std::numeric_limits<double>::infinity());
}

/**
 * function: range(l, r)
 * generating [l, r] by default
 */
Value lamina_range(const std::vector<Value> &args) {
	Range tmp;
	auto s0 = args[0].as_symbolic();
	auto s1 = args[1].as_symbolic();
	tmp.segments.push_back(Range::BasicRange(mini(s0, s1), maxi(s0, s1)));
	return tmp.lamina();
}

/**
 * function: rangex(l, r, has_l, has_r)
 * (TODO: Probably we need range literals for a math-processing language!)
 */
Value lamina_rangex(const std::vector<Value> &args) {
	Range tmp;
	auto s0 = args[0].as_symbolic();
	auto s1 = args[1].as_symbolic();
	tmp.segments.push_back(Range::BasicRange(mini(s0, s1), maxi(s0, s1), args[2].as_bool(), args[3].as_bool()));
	return tmp.lamina();
}

Value lamina_intersect(const std::vector<Value> &args) {
	if (!args.size()) return Range().lamina();
	Range result = Range(args[0]);
	for (size_t i = 1; i < args.size(); i++) result = intersect(result, Range(args[i]));
	return result.lamina();
}

Value lamina_join(const std::vector<Value> &args) {
	if (!args.size()) return Range().lamina();
	Range result = Range(args[0]);
	for (size_t i = 1; i < args.size(); i++) result = join(result, Range(args[i]));
	return result.lamina();
}

/**
 * function: in_range(range, value)
 */
Value lamina_range_test(const std::vector<Value> &args) {
	return Value(Range(args[0]).in_range(args[1].as_symbolic()));
}
