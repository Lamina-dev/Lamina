#pragma once
#include "lamina_api/lamina.hpp"
#include "lamina_api/value.hpp"

// ====================== Math ======================

// 平方根：需1个数值参数（如数字、向量/矩阵的每个元素），返回平方根结果
Value sqrt_(const std::vector<Value>& args);

// 圆周率π：需0个参数，直接返回π的数值（如3.1415926535...）
Value pi(const std::vector<Value>& args);

// 自然常数e：需0个参数，直接返回e的数值（如2.7182818284...）
Value e(const std::vector<Value>& args);

// 绝对值：需1个数值参数（处理数字、向量/矩阵元素的正负），返回非负值
Value abs_(const std::vector<Value>& args);

// 正弦函数：需1个参数（弧度值，可处理单个数字或向量/矩阵元素），返回正弦值
Value sin_(const std::vector<Value>& args);

// 余弦函数：需1个参数（弧度值，可处理单个数字或向量/矩阵元素），返回余弦值
Value cos_(const std::vector<Value>& args);

// 正切函数：需1个参数（弧度值，可处理单个数字或向量/矩阵元素），返回正切值
Value tan_(const std::vector<Value>& args);

// 自然对数（ln）：需1个正数值参数（处理数字、向量/矩阵元素），返回对数结果
Value log_(const std::vector<Value>& args);

// 四舍五入：需1个数值参数，返回最接近的整数
Value round_(const std::vector<Value>& args);

// 向下取整（地板函数）：需1个数值参数，返回不大于该值的最大整数
Value floor_(const std::vector<Value>& args);

// 向上取整（天花板函数）：需1个数值参数，返回不小于该值的最小整数
Value ceil_(const std::vector<Value>& args);

// 点积：需2个同维度向量参数，返回标量结果
Value dot(const std::vector<Value>& args);

// 叉积：需2个3维向量参数，返回3维向量结果
Value cross(const std::vector<Value>& args);

// 范数（默认L2范数/欧几里得范数）：需1个向量/矩阵参数，返回标量
Value norm(const std::vector<Value>& args);

// 归一化：需1个向量参数，返回单位向量（各元素除以范数）
Value normalize(const std::vector<Value>& args);

// 行列式：需1个方阵参数（如2x2、3x3矩阵），返回标量结果
Value det(const std::vector<Value>& args);

// 大小/维度：需1个容器类参数（向量/矩阵），返回元素个数或维度信息（如{行,列}）
Value size(const std::vector<Value>& args);

// 整数除法：需2个整数参数（被除数、除数），返回商的整数部分（截断小数）
Value idiv(const std::vector<Value>& args);

// 分数部分：需1个数值参数，返回该数的小数部分（如5.7返回0.7，-3.2返回-0.2）
Value fraction(const std::vector<Value>& args);

// 小数转整数（或提取整数部分）：需1个数值参数，返回整数部分（如5.7返回5，-3.2返回-3）
Value decimal(const std::vector<Value>& args);

// 幂运算（base^exponent）：需2个参数（底数、指数），返回幂运算结果
Value pow_(const std::vector<Value>& args);

// 最大公约数：需2个正整数参数，返回二者的最大公约数
Value gcd(const std::vector<Value>& args);

// 最小公倍数：需2个正整数参数，返回二者的最小公倍数
Value lcm(const std::vector<Value>& args);

// 生成随机浮点数：需0个或2个参数（无参返回[0,1)随机数；有参返回[min,max)随机数）
Value random_(const std::vector<Value>& args);

// 生成随机整数：需2个参数（最小值、最大值），返回[min,max]范围内的随机整数
Value randint(const std::vector<Value>& args);

// 生成随机字符串：需1个参数（字符串长度），返回指定长度的随机字符组合（如字母+数字）
Value randstr(const std::vector<Value>& args);

// 获取当前时间：需0个参数，返回当前时间（如"HH:MM:SS"格式字符串或时间戳）
Value get_time(const std::vector<Value>& args);

// 获取当前日期：需0个参数，返回当前日期（如"YYYY-MM-DD"格式字符串）
Value get_date(const std::vector<Value>& args);

// 获取格式化日期时间：需1个参数（格式字符串，如"%Y-%m-%d %H:%M:%S"），返回对应格式的日期时间字符串
Value get_format_date(const std::vector<Value>& args);

// ====================== Basic ======================

// 生成数值范围：需2个或3个参数（start, end 生成[start,end)步长1的序列；start, end, step 生成指定步长序列）
Value range(const std::vector<Value>& args);

// 获取数组指定索引元素：需2个参数（数组、索引），返回该索引对应的元素
Value arr_at(const std::vector<Value>& args);

// 设置数组指定索引元素：需3个参数（数组、索引、新值），更新索引位置元素并返回null
Value arr_set(const std::vector<Value>& args);

// 查找元素在数组中的索引：需2个参数（数组、目标元素），返回第一个匹配元素的索引；无匹配返回-1
Value arr_index_of(const std::vector<Value>& args);

// 输入函数：需1个参数（输入提示信息），返回用户输入的内容
Value input(const std::vector<Value>& args);

// 打印函数：支持任意数量参数（待打印的变量/值），将内容输出到控制台
Value print(const std::vector<Value>& args);

// 执行系统命令：需1个参数（系统命令字符串），返回命令执行结果或退出状态
Value system_(const std::vector<Value>& args);

// 断言函数：需1个或2个参数（断言条件；可选断言失败提示信息），条件为假则触发断言失败
Value lm_assert(const std::vector<Value>& args);

// 类型判断：需1个参数（待判断类型的变量/值），返回其类型标识（如"int"、"string"、"array"）
Value typeof_(const std::vector<Value>& args);

// 获取对象属性：需2个参数（对象、属性名），返回该对象的指定属性值
Value getattr(const std::vector<Value>& args);

// 判断是否是同一个对象
Value is_same_thing(const std::vector<Value>& args);

// 设置对象属性：需3个参数（对象、属性名、属性值），为对象设置或更新指定属性
Value setattr(const std::vector<Value>& args);

// 复制结构体
Value copy_struct(const std::vector<Value>& args);

// 创建原型
Value new_struct_from(const std::vector<Value>& args);

// 更新容器/对象：需2个参数（原容器/对象、待更新的键值对/属性），合并内容并返回更新后的结果
Value update(const std::vector<Value>& args);

// 遍历容器：需2个参数（容器、遍历执行的函数），对容器每个元素执行函数并返回执行结果集
Value foreach(const std::vector<Value>& args);

// 查找符合条件的元素：需2个参数（容器、判断函数），返回第一个满足条件的元素；无则返回空
Value find(const std::vector<Value>& args);

// 映射转换：需2个参数（容器、转换函数），对容器每个元素执行函数，返回转换后的新容器
Value map(const std::vector<Value>& args);

// 替换内容：需3个参数（原字符串/容器、目标值、替换值），替换所有匹配的目标值并返回新结果
Value replace(const std::vector<Value>& args);

// 变量表
Value vars(const std::vector<Value>& args);

// 局部变量表
Value locals(const std::vector<Value>& args);

// 全局变量表
Value globals(const std::vector<Value>& args);

// 退出程序
Value exit_(const std::vector<Value>& args);

// 错误处理函数
Value xpcall(const std::vector<Value>& args);

// 拼接多个字符串，并返回一个新字符串
Value cat(const std::vector<Value>& args);

// 获取字符串指定位置的字符，以Int类型返回
Value char_at(const std::vector<Value>& args);

// 从指定位置开始查找子字符串，若查找成功，返回第一个符合结果的子字符串索引；失败返回-1（或指定标识）
Value str_find(const std::vector<Value>& args);

// 截取子字符串：需3个参数（原字符串、起始索引、截取长度），返回截取后的子串
Value sub_string(const std::vector<Value>& args);

Value to_string(const std::vector<Value>& args);

// ====================== CAS ======================

// CAS表达式解析：需1个参数（待解析的数学表达式字符串），返回解析后的CAS表达式对象
Value cas_parse(const std::vector<Value>& args);

// CAS表达式化简：需1个参数（已解析的CAS表达式），返回化简后的表达式
Value cas_simplify(const std::vector<Value>& args);

// CAS表达式求导：需2个参数（待求导的CAS表达式、求导变量），返回导函数表达式
Value cas_differentiate(const std::vector<Value>& args);

// CAS表达式求值：需1个参数（已解析的CAS表达式），返回计算后的数值或简化结果
Value cas_evaluate(const std::vector<Value>& args);

// CAS表达式存储：需2个参数（存储键名、待存储的CAS表达式），将表达式存入指定键
Value cas_store(const std::vector<Value>& args);

// CAS表达式加载：需1个参数（存储键名），返回该键对应的CAS表达式
Value cas_load(const std::vector<Value>& args);

// CAS表达式定点求值：需2个参数（CAS表达式、变量的指定值），返回表达式在该值下的结果
Value cas_evaluate_at(const std::vector<Value>& args);

// 线性方程求解：需1个参数（线性方程组的CAS表达式形式），返回方程的解集合
Value cas_solve_linear(const std::vector<Value>& args);

// CAS数值导数计算：需3个参数（CAS表达式、求导变量、计算点），返回该点的数值导数
Value cas_numerical_derivative(const std::vector<Value>& args);

// ====================== io ======================

Value fast_read(const std::vector<Value>& args);

Value fast_write(const std::vector<Value>& args);

Value exist_file(const std::vector<Value>& args);

Value create_file(const std::vector<Value>& args);

Value console_write(const std::vector<Value>& args);

Value console_getch(const std::vector<Value>& args);

Value console_getlines(const std::vector<Value>& args);

Value console_scanf(const std::vector<Value>& args);

Value console_clear(const std::vector<Value>& args);


inline std::unordered_map<std::string, Value> register_builtins =
    {
        // 字符串操作模块：封装字符串拼接、查找、截取等功能
        LAMINA_MODULE("string", LAMINA_VERSION, {
            LAMINA_FUNC("cat", cat),
            LAMINA_FUNC("at", char_at),
            LAMINA_FUNC("find", str_find),
            LAMINA_FUNC("sub", sub_string),
        }),

        // 数学计算模块：一级函数
        LAMINA_FUNC("sqrt", sqrt_),
        LAMINA_FUNC("pi", pi),
        LAMINA_FUNC("e", e),
        LAMINA_FUNC("abs", abs_),
        LAMINA_FUNC("sin", sin_),
        LAMINA_FUNC("cos", cos_),
        LAMINA_FUNC("tan", tan_),
        LAMINA_FUNC("log", log_),
        LAMINA_FUNC("round", round_),
        LAMINA_FUNC("floor", floor_),
        LAMINA_FUNC("ceil", ceil_),
        LAMINA_FUNC("dot", dot),
        LAMINA_FUNC("cross", cross),
        LAMINA_FUNC("norm", norm),
        LAMINA_FUNC("normalize", normalize),
        LAMINA_FUNC("det", det),
        LAMINA_FUNC("size", size),
        LAMINA_FUNC("idiv", idiv),
        LAMINA_FUNC("fraction", fraction),
        LAMINA_FUNC("decimal", decimal),
        LAMINA_FUNC("pow", pow_),
        LAMINA_FUNC("gcd", gcd),
        LAMINA_FUNC("lcm", lcm),
        LAMINA_FUNC("range", range),

        // 数组处理模块：封装数组元素访问、修改、查找功能
        LAMINA_MODULE("array", LAMINA_VERSION, {
            LAMINA_FUNC("at", arr_at),
            LAMINA_FUNC("set", arr_set),
            LAMINA_FUNC("index_of", arr_index_of)
        }),

        // 系统模块：一级函数
        LAMINA_FUNC("input", input),
        LAMINA_FUNC("print", print),
        LAMINA_FUNC("system", system_),
        LAMINA_FUNC("assert", lm_assert),

        // 操作模块：一级函数
        LAMINA_FUNC("typeof", typeof_),
        LAMINA_FUNC("getattr", getattr),
        LAMINA_FUNC("setattr", setattr),
        LAMINA_FUNC("update", update),
        LAMINA_FUNC("same", is_same_thing),
        LAMINA_FUNC("to_string", to_string),
        LAMINA_FUNC("copy_struct", copy_struct),
        LAMINA_FUNC("new", new_struct_from),
        LAMINA_FUNC("vars", vars),
        LAMINA_FUNC("locals", locals),
        LAMINA_FUNC("globals", globals),
        LAMINA_FUNC("exit", exit_),
        LAMINA_FUNC("xpcall", xpcall),

        // 容器遍历模块：一级函数
        LAMINA_FUNC("foreach", foreach),
        LAMINA_FUNC("find", find),
        LAMINA_FUNC("map", map),
        LAMINA_FUNC("replace", replace),

        // CAS数学模块：封装符号计算相关的解析、化简、求导等功能
        LAMINA_MODULE("cas", LAMINA_VERSION, {
            LAMINA_FUNC("cas_parse", cas_parse),
            LAMINA_FUNC("cas_simplify", cas_simplify),
            LAMINA_FUNC("cas_differentiate", cas_differentiate),
            LAMINA_FUNC("cas_evaluate", cas_evaluate),
            LAMINA_FUNC("cas_store", cas_store),
            LAMINA_FUNC("cas_load", cas_load),
            LAMINA_FUNC("cas_evaluate_at", cas_evaluate_at),
            LAMINA_FUNC("cas_solve_linear", cas_solve_linear),
            LAMINA_FUNC("cas_numerical_derivative", cas_numerical_derivative)
        })
    };


// 需要用户 include "lib_name" 导入
inline std::unordered_map<std::string, Value> register_std_libs = {
    // 随机模块：封装随机浮点数、整数、字符串生成功能
    LAMINA_MODULE("random", LAMINA_VERSION, {
        LAMINA_FUNC("random", random_),
        LAMINA_FUNC("randint", randint),
        LAMINA_FUNC("randstr", randstr)
    }),

    // 时间模块：封装当前时间、日期、格式化日期获取功能
    LAMINA_MODULE("time", LAMINA_VERSION, {
        LAMINA_FUNC("get_time", get_time),
        LAMINA_FUNC("get_date", get_date),
        LAMINA_FUNC("get_format_date", get_format_date)
    }),

    // IO模块
    // ToDo: not complete
    LAMINA_MODULE("io", LAMINA_VERSION, {
        LAMINA_FUNC("fast_read", fast_read),
        LAMINA_FUNC("fast_write", fast_write),
        LAMINA_FUNC("exist_file", exist_file),
        LAMINA_FUNC("create_file", create_file)
    }),

    // 控制台模块
    // ToDo: not complete
    LAMINA_MODULE("console", LAMINA_VERSION, {
        LAMINA_FUNC("console_write", console_write),
        LAMINA_FUNC("console_getch", console_getch),
        LAMINA_FUNC("console_getlines", console_getlines),
        LAMINA_FUNC("console_scanf", console_scanf),
        LAMINA_FUNC("console_clear", console_clear),
    }),
};
