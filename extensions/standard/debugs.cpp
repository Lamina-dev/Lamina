// Debug support, should be removed in release versions
#include "lamina.hpp"

Value __hash_symbolic(const std::vector<Value>& args) {
	auto hd = SymbolicExpr::HashData(args[0].as_symbolic());
	std::vector<Value> val = {Value(hd.k), Value(hd.ksqrt), Value(BigInt(hd.hash)), Value(hd.hash_obj), Value(BigInt(hd.to_single_hash()))}; //???
	return Value(val);
}

namespace Lamina {
	LAMINA_FUNC("__hash_symbolic", __hash_symbolic, 1);
}