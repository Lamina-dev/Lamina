#include "lmStruct.hpp"
#include "standard.hpp"
#include "interpreter.hpp"
#include <ranges>

Value range(const std::vector<Value>& args) {
    if (args.empty()) return LAMINA_NULL;

    const int start = args.size() > 1
                              ? std::get<int>(args[0].data)
                              : 0;
    const int end = args.size() > 1
                            ? std::get<int>(args[1].data)
                            : std::get<int>(args[0].data);
    const int sep = args.size() > 2
                            ? std::get<int>(args[2].data)
                            : 1;
    std::vector<Value> vec;
    for (auto i = start; i < end; i += sep) {
        vec.emplace_back(i);
    }
    return vec;
}

Value arr_at(const std::vector<Value>& args) {
    if (!args[0].is_array()) {
        L_ERR("First Arg Must Be A Array");
        return LAMINA_NULL;
    }

    const Value* current = &args[0];

    for (size_t i = 1; i < args.size(); ++i) {
        if (!args[i].is_int()) {
            L_ERR("Index argument must be an integer");
            return LAMINA_NULL;
        }
        int index = std::get<int>(args[i].data);

        if (!current->is_array()) {
            L_ERR("Cannot index non-array value at level " + std::to_string(i));
            return LAMINA_NULL;
        }

        const auto& arr = std::get<std::vector<Value>>(current->data);

        if (index < 0 || static_cast<size_t>(index) >= arr.size()) {
            L_ERR("Array Index Out Of Range at level " + std::to_string(i));
            return LAMINA_NULL;
        }

        current = &arr[index];
    }

    return *current;
}

Value arr_set(const std::vector<Value>& args) {
    check_cpp_function_argv(args, 3);
    if (!args[1].is_array() and !args[2].is_int()) {
        L_ERR("First Arg Must Be A Array, Second Arg Must Be a int");
        return LAMINA_NULL;
    }
    auto arr = std::get<std::vector<Value>>(args[0].data);
    const auto idx = std::get<int>(args[1].data);
    if (idx > arr.size()) {
        L_ERR("Array Index Out Of Range");
    }
    const auto& val = args[3];
    arr[idx] = val;

    return arr;
}

Value arr_index_of(const std::vector<Value>& args) {
    if (!args[0].is_array() || !args[1].is_string()) {
        L_ERR("Invalid arguments (expected array and string)");
        return LAMINA_NULL;
    }

    const std::string target_key = std::get<std::string>(args[1].data);
    const auto& arr = std::get<std::vector<Value>>(args[0].data);
    Value result = LAMINA_NULL;
    bool found = false;


    if (!found && arr.size() % 2 == 0) {
        for (size_t i = 0; i < arr.size(); i += 2) {
            if (i + 1 >= arr.size()) break;

            const auto& key_elem = arr[i];
            const auto& value_elem = arr[i + 1];

            if (!key_elem.is_string()) continue;

            const std::string current_key = std::get<std::string>(key_elem.data);
            if (current_key == target_key) {
                result = value_elem;
                found = true;
                break;
            }
        }
    }

    if (!found) {
        L_ERR("Key '" + target_key + "' not found in array");
        return LAMINA_NULL;
    }

    return result;
}

// 遍历容器：需2个参数（容器、遍历执行的函数），对容器每个元素执行函数并返回执行结果集
Value foreach(const std::vector<Value>& args){
    check_cpp_function_argv(args, 2);
    const auto func = std::get<std::shared_ptr<LambdaDeclExpr>>(args[1].data);

    if (args[0].is_array()) {
        const auto arr = std::get<std::vector<Value>>(args[0].data);
        int cnt = 0;
        for (const auto& value: arr) {
            Interpreter::call_function(func.get(), {cnt, value});
            ++cnt;
        }
        return LAMINA_NULL;
    }
    if (args[0].is_lstruct()) {
        const auto arr = std::get<std::shared_ptr<lmStruct>>(args[0].data)->to_vector();
        for (auto [key, value] : arr) {
            Interpreter::call_function(func.get(), {Value(key), value});
        }
        return LAMINA_NULL;
    }
    if (args[0].is_string()) {
        const auto arr = std::get<std::string>(args[0].data);
        int cnt = 0;
        for (const auto value: arr) {
            Interpreter::call_function(func.get(), {Value(cnt), Value(value)});
            ++cnt;
        }
        return LAMINA_NULL;
    }
    return LAMINA_NULL;
}

// 查找符合条件的元素：需2个参数（容器、判断函数），返回第一个满足条件的元素；无则返回空
Value find(const std::vector<Value>& args) {
    check_cpp_function_argv(args, 2);
    const auto arr = std::get<std::vector<Value>>(args[0].data);
    const auto func = std::get<std::shared_ptr<LambdaDeclExpr>>(args[1].data);
    for (const auto& value: arr) {
        auto ret = Interpreter::call_function(
            func.get(), {value});
        if (ret.as_bool() != false) {
            Value result;
            return result;
        }
    }
    return LAMINA_NULL;
}

// 映射转换：需2个参数（容器、转换函数），对容器每个元素执行函数，返回转换后的新容器
Value map(const std::vector<Value>& args){
    check_cpp_function_argv(args, 2);
    const auto arr = std::get<std::vector<Value>>(args[0].data);
    const auto func = std::get<std::shared_ptr<LambdaDeclExpr>>(args[1].data);
    std::vector<Value> result{};
    for (const auto& value: arr) {
        result.emplace_back(Interpreter::call_function(
            func.get(), {value}
        ));
    }
    return result;
}

// 替换内容：需2个参数（原容器、转换函数）
Value replace(const std::vector<Value>& args) {
    check_cpp_function_argv(args, 2);
    auto arr = std::get<std::vector<Value>>(args[0].data);
    const auto func = std::get<std::shared_ptr<LambdaDeclExpr>>(args[1].data);
    std::vector<Value> result{};
    for (const auto& value: arr) {
        result.emplace_back(Interpreter::call_function(
            func.get(), {value}
        ));
    }
    arr = result;
    return arr;
}


// 拼接给定的所有列表
Value concat(const std::vector<Value>& args) {
	std::vector<Value> result;
	for (const auto &i : args) {
		if (!i.is_array()) {
			L_ERR("Given parameter(s) have no-array element");
			return LAMINA_NULL;
		}
		for (const auto &j : std::get<std::vector<Value> >(i.data)) {
			result.push_back(j);
		}
	}
	return Value(result);
}

// 列表切片
Value slice(const std::vector<Value>& args) {
	check_cpp_function_argv_x(args, 3, 4);
	if (!args[0].is_array()) {
		L_ERR("slice() requires a list");
		return LAMINA_NULL;
	}
	std::vector<Value> val = std::get<std::vector<Value> >(args[0].data);
	std::vector<Value> result;
	int begin = (int) args[1].as_number(), end = (int) args[2].as_number(), step = 1;
	if (end < begin) step = -1;
	if (args.size() >= 4) {
		step = (int) args[3].as_number();
	}
	for (int i = begin; i != end; i += step) {
		int it = (i >= 0) ? i : (int(val.size()) + i);
		if (it < 0 || it >= val.size()) {
			break;
		}
		result.push_back(val[it]);
	}
	return Value(result);
}

// 排序给定的列表，第二个参数表示比较器
Value _sort(const std::vector<Value>& args) {
	check_cpp_function_argv_x(args, 1, 2);
	if (!args[0].is_array()) {
		L_ERR("sort() requires a list");
		return LAMINA_NULL;
	}
	std::vector<Value> val = std::get<std::vector<Value> >(args[0].data);
	for (auto &i : val) {
		if (!i.is_comparable()) {
			L_ERR("Array has uncomparable object");
			return args[0];	// A failure, not an error.
		}
	}
	std::function<bool(const Value &a, const Value &b)> comparer;
	if (args.size() >= 2) {
		if (!args[1].is_lambda()) {
			L_ERR("Comparer must be a lambda/function");
			return LAMINA_NULL;
		}
		const auto func = std::get<std::shared_ptr<LambdaDeclExpr>>(args[1].data);
		comparer = [&func](const Value &a, const Value &b) -> bool {
			return Interpreter::call_function(func.get(), {a, b}).as_bool();
		};
	} else {
		comparer = [](const Value &a, const Value &b) -> bool {
			return a < b;
		};
	}
	sort(val.begin(), val.end(), comparer);
	return Value(val);
}