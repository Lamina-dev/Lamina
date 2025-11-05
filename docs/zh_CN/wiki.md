# Lamina 语言 Wiki

## 优点

1. **精确数学计算**：从底层解决浮点数精度丢失问题，支持有理数（分数）和无理数（√、π、e）的符号化存储与运算，多次循环运算仍保持精确。
2. **语法简洁直观**：支持自动补充分号、省略if/while语句的圆括号、无参函数简写等，降低代码冗余，符合数学表达习惯。
3. **原生数学友好**：无需第三方库，直接支持向量、矩阵运算、大整数阶乘等数学操作，满足复杂数学问题需求。
4. **友好开发体验**：交互式REPL支持关键字高亮、自动补齐，提供完整错误栈追踪，便于调试；智能终端自动适配色彩，避免乱码。
5. **模块化设计**：通过`include`语句引入外部模块，支持`::`命名空间访问符，实现代码复用与隔离。
6. **灵活数据类型**：涵盖精确数值类型（rational/irrational）、复合类型（数组/矩阵/结构体）及匿名函数，适配多样开发场景。

## 基本语法

### 1. 变量声明

**模板**：

```lamina
// 普通变量声明
var var_name = expression;
// 大整数变量声明（任意精度）
bigint big_var_name = expression;
```

**实例**：

```lamina
var pi = 3.1415; // 声明浮点数变量pi
var cycle = 13.14545...; // 声明循环小数变量（自动变为分数）
var score = 95; // 声明整数变量score
var half = 1/2; // 声明有理数变量half（自动以分数形式存储）
bigint fact_25 = 25!; // 声明大整数变量，存储25的阶乘
```

### 2. 注释

**模板**：

```lamina
// 单行注释：注释内容
/* 
   块注释：
   支持多行文本
   适合长注释说明
*/
```

**实例**：

```lamina
// 这是单行注释，用于说明下一行代码功能
var radius = 5; // 声明圆的半径变量

/* 
   块注释示例：
   以下代码计算圆的面积
   依赖pi()函数获取精确圆周率
*/
var area = pi() * radius ^ 2;
```

### 3. 条件语句（if-else）

**模板**：

```lamina
if condition {
    // 条件为true时执行的代码
} else {
    // 条件为false时执行的代码
}
```

**实例**：

```lamina
var num = 8;
if num > 0 {
    print("num是正数"); // 条件成立，输出此内容
} else {
    print("num是非正数");
}
```

### 4. 循环语句

#### （1）while循环

**模板**：

```lamina
while condition {
    // 条件为true时重复执行的代码
}
```

**实例**：

```lamina
var count = 1;
while count <= 3 {
    print("当前计数：", count); // 依次输出1、2、3
    count = count + 1;
}
```

#### （2）loop 循环（计数循环）

注意：旧的无限 `loop {}` 语法已被移除。现在 `loop` 需要跟随一个表示次数的表达式。

**模板**：

```lamina
loop <count-expression> {
  // 循环体：按 count-expression 的值重复执行
}
```

说明：`<count-expression>` 可以是字面整数、变量、或任意求值为整数的表达式（例如 `loop (a + b) {}`）。允许的运行时类型包括：整数（int）、大整数（bigint，须在可表示范围内）、有理数（分母为1 即整数）以及数值型浮点数（当且仅当为整数值，如 `3.0`）。不接受无理数、符号表达式或字符串等类型。

错误行为：
- 如果 count-expression 求值为非整数（例如 2.5 或 1/2），运行时会抛出错误："Repeat count must be an integer"
- 如果 count-expression 是负数，会抛出错误："Repeat count must be non-negative"
- 如果数值超出可表示范围，会抛出错误："Repeat count too large" 或 "Repeat count out of range"

**实例**：

```lamina
var i = 0;
loop 3 { i = i + 1; };
print(i); // 输出 3

var n = 4;
var acc = 0;
loop n { acc = acc + 1; };
print(acc); // 输出 4

var a = 2; var b = 3;
loop (a + b) { print("hi"); } // 等同于 loop 5 { ... }
```

### 5. 函数定义

#### （1）无参函数

**模板**：

```lamina
func func_name {
    // 函数体代码
    return return_value; // 可选，无返回值可省略
}
```

**实例**：

```lamina
func say_hello {
    print("Hello, Lamina!"); // 无返回值，直接输出
}
say_hello(); // 调用函数，输出"Hello, Lamina!"
```

#### （2）有参函数

**模板**：

```lamina
func func_name(param1, param2) {
    // 基于参数的逻辑代码
    return result; // 返回计算结果
}
```

**实例**：

```lamina
func add(a, b) {
    return a + b; // 返回a与b的和
}
var sum = add(3, 5); // 调用函数，sum = 8
print("3 + 5 =", sum);
```

### 6. 匿名函数

**模板**：

```lamina
var func_var = do |param1, param2| {
    // 函数体代码
    return result;
};

// 只有单表达式时可以简写
var simple_func = |param1, param2| expression;
```

**实例**：

```lamina
var multiply = do |x, y| {
    return x * y; // 匿名函数实现乘法
};
var product = multiply(4, 6); // 调用匿名函数，product = 24
print("4 * 6 =", product);

var add = |a,b| a+b;
var result = add(1,33)
print("1 + 33 =", result)
```

### 7. 结构体声明

**模板**：

```lamina
var struct_name = {
    key1 = value1;
    key2 = value2;
    // 更多键值对...
};
```

**实例**：

```lamina
var student = {
    name = "Tom";
    age = 15;
    scores = [90, 85, 92]; // 结构体嵌套数组
};
```

### 8. 结构体成员访问

**模板**：

```lamina
struct_name.key; // 访问结构体的指定成员
```

**实例**：

```lamina
var student = { name = "Tom"; age = 15; };
print("学生姓名：", student.name); // 输出"学生姓名：Tom"
print("学生年龄：", student.age); // 输出"学生年龄：15"
student.age = 16
```

### 9. 模块引入

**模板**：

```lamina
include "module_path"; // 引入指定路径的模块，自动补全.lm扩展名
```

**实例**：

```lamina
include "math_utils"; // 引入数学工具模块
include "lib/vectors"; // 引入向量计算模块（相对路径）
var root = math::sqrt(2); // 通过命名空间访问模块函数
```

### 10. 续行符

**模板**：

```lamina
var long_expression = expression1 + expression2 + \
                      expression3 + expression4; // \ 用于拆分长表达式
```

**实例**：

```lamina
var total = 10 + 20 + 30 + \
            40 + 50; // 等价于var total = 10+20+30+40+50;
print("总和：", total); // 输出"总和：150"
```

## 基本类型

1. **int**：普通整数类型，支持正负整数的算术运算，无需显式声明类型。  
   示例：
   
   ```lamina
   var a = 42; // 正整数
   var b = -10; // 负整数
   var c = a + b; // c = 32（int类型）
   ```

2. **float**：浮点数类型，用于兼容传统浮点运算场景，存在精度限制。  
   示例：
   
   ```lamina
   var pi_approx = 3.14; // 浮点数
   var temp = -0.5; // 负浮点数
   ```

3. **rational**：精确有理数类型，自动以分数形式存储除法结果，避免精度丢失，支持自动化简。  
   示例：
   
   ```lamina
   var frac1 = 16/9; // 存储为16/9，非1.777...
   var frac2 = 4/6; // 自动化简为2/3
   var sum_frac = frac1 + frac2; // 精确计算，结果为34/9
   ```

4. **irrational**：精确无理数类型，以符号形式存储（如√、π、e），支持符号化运算与化简。  
   示例：
   
   ```lamina
   var root2 = sqrt(2); // 存储为√2
   var root8 = sqrt(8); // 自动化简为2√2
   var pi_val = pi(); // 存储为π
   var product = root2 * root2; // 结果为2（int类型）
   ```

5. **bool**：布尔类型，仅包含`true`（真）和`false`（假）两个值，用于条件判断。  
   示例：
   
   ```lamina
   var is_pass = true;
   var is_empty = false;
   if is_pass {
      print("考试通过");
   }
   ```

6. **string**：字符串类型，用双引号包裹文本内容，支持字符串相关函数操作。  
   示例：
   
   ```lamina
   var greeting = "Hello, Lamina!";
   var name = "Alice";
   ```

7. **null**：空值类型，仅表示`null`一个值，用于表示变量未赋值或无返回值。  
   示例：
   
   ```lamina
   var empty_var = null;
   func no_return() {
      print("无返回值");
      return null; // 显式返回空值
   }
   ```

8. **bigint**：任意精度大整数类型，需显式声明，支持超大整数（如阶乘、大数值运算）。  
   示例：
   
   ```lamina
   bigint large_num = 999999999999999; // 超大整数
   bigint fact_30 = 30!; // 30的阶乘（大整数结果）
   ```

9. **array**：数组类型，用方括号包裹元素，支持索引访问和数组相关函数。  
   示例：
   
   ```lamina
   var scores = [90, 85, 92]; // 一维数组
   var names = ["Tom", "Alice", "Bob"]; // 字符串数组
   ```

10. **matrix**：矩阵类型，用嵌套数组表示（二维数组），支持矩阵行列式、乘法等运算。  
    示例：
    
    ```lamina
    var mat2x2 = [[1, 2], [3, 4]]; // 2x2矩阵
    var mat3x1 = [[1], [2], [3]]; // 3x1列矩阵
    ```

11. **struct**：结构体类型，用大括号包裹键值对，支持自定义成员和成员访问。  
    示例：
    
    ```lamina
    var person = {
      name = "Bob";
      age = 20;
      is_student = true;
    };
    ```

12. **lambda**：匿名函数类型，用于表示未命名的函数，可赋值给变量或作为参数传递。  
    示例：
    
    ```lamina
    var subtract = |a, b| a - b ; // lambda类型变量
    ```

13. **module**：模块类型，通过`include`引入，包含模块内定义的函数、变量，支持`::`命名空间访问。  
    示例：
    
    ```lamina
    include random; // 引入模块，module类型
    var result = random::random(); // 访问模块函数
    ```

## 库

### 1. 数学函数

- **平方根函数**：用于计算数值的精确平方根，若为完全平方数返回int，否则返回irrational。
  
  ```lamina
  sqrt(x) -> int/irrational
  ```

- **圆周率函数**：返回精确的圆周率符号π，类型为irrational。
  
  ```lamina
  pi() -> irrational
  ```

- **自然常数函数**：返回精确的自然常数符号e，类型为irrational。
  
  ```lamina
  e() -> irrational
  ```

- **正弦函数**：计算角度的正弦值，支持精确数值输入，返回对应精度结果。
  
  ```lamina
  sin(x) -> rational/irrational/float
  ```

- **余弦函**数：计算角度的余弦值，支持精确数值输入，返回对应精度结果。
  
  ```lamina
  cos(x) -> rational/irrational/float
  ```

- **绝对值函数**：返回输入数值的绝对值，保持原类型不变。
  
  ```lamina
  abs(x) -> int/float/rational/irrational
  ```

- **自然对数函数**：计算数值的自然对数（以e为底），返回对应精度结果。
  
  ```lamina
  log(x) -> rational/irrational/float
  ```

- **阶乘函数**：计算非负整数的阶乘，支持int和bigint类型输入，返回对应整数类型。
  
  ```lamina
  factorial(n) -> int/bigint
  ```

### 2. 向量/矩阵函数

- **向量点积函数**：计算两个同维度向量的点积，返回数值类型（int/rational等）。
  
  ```lamina
  dot(v1: array, v2: array) -> int/rational/float
  ```

- **向量叉积函数**：计算两个三维向量的叉积，返回新的三维数组（矩阵）。
  
  ```lamina
  cross(v1: array, v2: array) -> array
  ```

- **向量模长函**数：计算向量的模长（长度），返回精确数值类型（irrational/rational等）。
  
  ```lamina
  norm(v: array) -> rational/irrational/float
  ```

- **矩阵行列式函数**：计算二维方阵的行列式，返回数值类型（int/rational等）。
  
  ```lamina
  det(mat: array) -> int/rational/float
  ```

### 3. 工具函数

- **打印函数**：向控制台输出一个或多个内容，结尾自动换行，无返回值。
  
  ```lamina
  print(...) -> null
  ```

- **输入函数**：在控制台显示提示文本，获取用户输入内容，返回字符串类型。
  
  ```lamina
  input(prompt: string) -> string/float
  ```

- **小数转分数函数**：将浮点数转换为精确的有理数（分数），自动化简。
  
  ```lamina
  fraction(x: float) -> rational
  ```

- **分数转小数函数**：将有理数（分数）转换为浮点数，支持按需保留精度。
  
  ```lamina
  decimal(x: rational) -> float
  ```

- **类型获取函数**：返回变量的类型名称，以字符串形式表示。
  
  ```lamina
  typeof(x) -> string
  ```

- **深拷贝函数**：对结构体、数组、匿名函数、模块等复合类型进行深拷贝，修改拷贝不影响原对象。
  
  ```lamina
  copy(x) -> any
  ```

- **大小获取函数**：返回数组的长度或结构体的成员数量，返回int类型。
  
  ```lamina
  size(x: array/struct) -> int
  ```

- **断言函数**：判断条件是否为true，若为false则抛出错误并显示自定义消息。
  
  ```lamina
  assert(condition: bool, msg: string = "") -> null
  ```

### 4. 数组函数

- **数组遍历函数**：遍历数组的每个元素，对元素执行指定函数，无返回值。
  
  ```lamina
  foreach(arr: array, func: lambda) -> null
  ```

- **数组映射函数**：遍历数组的每个元素，用指定函数处理元素，返回新的数组。
  
  ```lamina
  map(arr: array, func: lambda) -> array
  ```

- **数组查找函数**：在数组中查找首个满足条件的元素，返回元素值或null。
  
  ```lamina
  find(arr: array, func: lambda) -> any/null
  ```

- **数组替换函数**：替换数组中满足条件的元素，返回null。
  
  ```lamina
  replace(arr: array, func: lambda, new_val) -> null
  ```

### 5. 字符串函数

- **字符串拼接函数**：拼接多个字符串，返回拼接后的新字符串。
  
  ```lamina
  string::cat(...) -> string
  ```

- **字符获取函数**：获取指定索引的字符
  
  ```lamina
  string::at(index: int) -> int
  ```

- **子串截取函数**：从指定索引开始，截取指定长度的子串，返回新字符串。
  
  ```lamina
  string::sub(str: string, start_index: int, len: int) -> string
  ```
  
  ### 6. 随机函数

- **随机浮点数函数**：返回0（含）到1（不含）之间的随机浮点数。
  
  ```lamina
  random::random() -> float
  ```

- **随机整数函数**：返回`[start, end]`范围内的随机整数（包含边界值）。
  
  ```lamina
  random::randint(start: int, end: int) -> int
  ```

- **随机字符串函数**：从输入字符串中随机选取一个字符，返回该字符（string类型）。
  
  ```lamina
  random::randstr(chars: string) -> string
  ```

### 7. 时间函数

- **时间获取函数**：返回当前系统时间，格式为"HH:MM:SS"，字符串类型。
  
  ```lamina
  time::time() -> string
  ```

- **日期获取函数**：返回当前系统日期，格式为"YYYY-MM-DD"，字符串类型。
  
  ```lamina
  time::date() -> string
  ```

- **日期格式化函数**：将日期字符串按指定格式转换，返回格式化后的日期字符串。
  
  ```lamina
  time::format_date(date: string, format: string = "YYYY-MM-DD") -> string
  ```

### 8. CAS相关函数

- **CAS解析函数**：解析数学表达式，将其转换为CAS可处理的内部格式，返回解析后的CAS对象。
  
  ```lamina
  cas::parse(expr: string) -> cas_object
  ```

- **CAS化简函数**：对CAS对象表示的数学表达式进行化简，返回化简后的CAS对象。
  
  ```lamina
  cas::simplify(cas_obj: cas_object) -> cas_object
  ```

- **CAS求导函数**：对CAS对象表示的函数求导，返回求导后的CAS对象。
  
  ```lamina
  cas::differentiate(cas_obj: cas_object, var: string) -> cas_object
  ```

- **CAS计算函数**：对CAS对象表示的表达式进行数值或符号计算，返回计算结果（CAS对象或基础数据类型）。
  
  ```lamina
  cas::evaluate(cas_obj: cas_object) -> any
  ```

- **CAS存储函数**：将指定名称与CAS对象关联并存储，便于后续调用，无返回值或返回存储成功标识（布尔类型）。
  
  ```lamina
  cas::store(name: string, cas_obj: cas_object) -> bool/void
  ```

- **CAS加载函数**：根据指定名称加载已存储的CAS对象，返回对应的CAS对象（若不存在则返回空）。
  
  ```lamina
  cas::load(name: string) -> cas_object/null
  ```

- **CAS定点计算函数**：在指定变量值处计算CAS对象表示的表达式，返回计算结果（数值类型）。
  
  ```lamina
  cas::evaluate_at(cas_obj: cas_object, var: string, value: number) -> number
  ```

- **CAS线性求解函数**：求解线性方程或线性方程组（需以CAS对象形式传入），返回方程的解（CAS对象或解的集合）。
  
  ```lamina
  cas::solve_linear(cas_obj: cas_object) -> cas_object/array
  ```

- **CAS数值求导函数**：通过数值方法对函数（CAS对象表示）求导，返回数值导数结果（数值类型）。
  
  ```lamina
  cas::numerical_derivative(cas_obj: cas_object, var: string, value: number) -> number
  ```

### 9.IO函数

1. **阅读文件**：根据指定路径和匹配模式读取文件内容，返回读取到的字符串数据。
   
   ```lamina
   fast_io::read(path, pattern) -> string
   ```

2. **向文件写入**：向指定路径的文件中写入内容（注：需确保函数实际包含内容参数，此处按标准逻辑补充），返回写入成功与否的布尔值。
   
   ```lamina
   fast_io::write(path) -> bool
   ```

3. **创造文件**：在指定路径创建新文件，并写入初始内容，返回创建成功与否的布尔值。
   
   ```lamina
   fast_io::create(path, init_conten) -> bool
   ```

### 10. 程序控制函数

10. 程序控制函数
- **程序退出函数**：终止当前程序运行，可指定退出码（0表示正常退出，非0表示异常）。
  
  ```lamina
  exit(code: int = 0) -> null
  ```

- **类型转换函数**：将任意类型的变量转换为字符串类型，返回转换后的字符串。
  
  ```lamina
  to_string(x) -> string
  ```

- **系统执行函数**：执行系统指令
  
  ```
  system(cmd: string) -> null
  ```

### 11. 变量与函数查询函数

- **所有作用域变量查询函数**：返回当前所有全局变量的名称列表，数组类型。
  
  ```lamina
  vars() -> array
  ```

- **局部变量查询函数**：返回当前作用域内所有局部变量的名称列表，结构体类型。
  
  ```lamina
  locals() -> struct
  ```

- **全局变量查询函数**：返回全局作用域变量的名称列表，结构体类型。
  
  ```lamina
  locals() -> struct
  ```

## 开发中的库及类成员说明

### List类

**文件位置**：`interpreter/lamina_api/list.hpp`

**类说明**：继承了base_list<Value>

**注**：此类暂未投入解释器使用

### base_list类

**文件位置**：`interpreter/lamina_api/base_list.hpp`

**类说明**：一个单项链表模板类，在命名空间__LAMINA::ListType下

**类成员说明**：

**1.成员函数**

`class base_list_node`：

一个基本链表节点
包含`value`用于存储值，`next`用于存储下一个节点

**2.成员变量**

`base_list_node* root,end`：

两个链表指针
其中`root`用于存储第0个节点，不作为值存储，仅方便链表操作，指向第一个节点或`nullptr`

`end`用于指向最后一个节点，方便进行向末尾插入，在未存储值时为`root`

**注**：未使用安全指针请大佬有空出手相助

**3.成员函数**

`safe_new`：

用于检查传入的节点是否为空节点即检查`new`分配内存是否正常

`base_list`：

类初始化函数

其中`base_list()`为直接初始化将`root`分配新节点，`end`指向`root`

`base_list(const VT& val)`先调用直接初始化再调用`push_back()`插入`val`

`base_list(const base_list& other)`与`base_list(const std::initializer_list<VT>& initl)`均为赋值初始化调用赋值函数进行初始化

**注**：std::initializer_list是括号表达式的类用于接收类似于`{1, 2, 3}`的值进行初始化

`push_back`：

向链表节点末尾插入元素，不反回任何值

包含两个重载函数

`void push_back(const VT& val)`：在末尾插入单个值`val`

`void push_back(const base_list<VT>& other)`：在末尾插入另一个链表`other`的所有元素

`push_front`：

向链表节点开头插入元素，不反回任何值

包含两个重载函数

`void push_front(const VT& val)`：在开头插入单个值`val`

`void push_front(const base_list<VT>& other)`：在开头插入另一个链表`other`所有元素

`insert`：

在链表中插入元素，不反回任何值
包含三个重载函数

`void insert(size_t index,const VT& val)`：插入值`val`至第`index`项的位置

`void insert(size_t index, const base_list<VT>& other)`插入另一个链表`other`所有元素至第`index`项的位置

`void insert(size_t index, const base_list<VT>& other,size_t begini,size_t endi)`：插入插入另一个链表`other`的第`begini`项至`endi`项(不包含`endi`项)至第`index`项的位置

**示例**：

```c++
base_list<int> a({0, 1, 2});
base_list<int> b({3, 4, 5});
a.insert(1,b,1,3);
//将{4, 5}插入至1的位置
//执行结果：a为{0, 4, 5, 1, 2}
```

`pop_back`：

弹出链表最后一个元素，不反回任何值

包含一个重载函数

`void pop_back()`：弹出链表最后一个元素，如果链表为空抛出异常

`pop_front`：

弹出链表第一个元素，不反回任何值

包含一个重载函数

`void pop_front()`：弹出链表第一个元素，如果链表为空抛出异常

`pop`：

弹出元素，不反回任何值

包含两个重载函数

`void pop(size_t index)`：弹出第`index`项的值

`void pop(size_t begini,size_t endi)`:弹出第`begini`项至`endi`项(不包含`endi`项)的值

**示例**

```c++
base_list<int> a({0, 1, 2, 3, 4, 5});
a.pop(2,5);
//执行结果：a为{0, 1, 5}
```

`eraser`：与`pop`用法一致

`get_range`：

获取链表的一个区间，返回一个同类型的链表

包含一个重载函数

`base_list<VT> get_range(size_t begini, size_t endi) const`：获取链表的第`begini`项至`endi`项(不包含`endi`项)的值并存储在一个新链表中，返回这个新链表

`clear`：清空整个链表

**4.重载函数**

`operator+=`：与`push_back`用法一致

`operator+`:
拼接链表，返回一个同类型的链表

包含两个重载函数

`base_list<VT> operator+(const VT& val) const`：拼接`*this`与`val`至一个新链表并返回

`base_list<VT> operator+(const base_list<VT>& other) const`：拼接`*this`与`other`至一个新链表并返回

`operator[]`：

查找任意元素，返回元素的左值

包含一个重载函数

`VT& operator[](size_t index)`：查找到第`index`项的元素，并返回它的左值

`operator=`：

赋值函数，不反回任何值

包含两个重载函数

`void operator=(const base_list<VT>& other)`：同类型赋值

`void operator=(const std::initializer_list<VT>& initl)`：接受括号表达式的赋值

**注**：关于括号表达式请参阅初始化函数一栏

**5.保护的成员函数**

**警告：请在详细阅读代码后使用，以免造成安全性问题**

`void push_ones_back(base_list_node* one, const VT& val)`：将值`val`添加到节点`one`后面

`void push_ones_back(base_list_node* one, const base_list<VT>& other)`：将`other`中的所有元素添加到节点`one`后面

`void pop_ones_next(base_list_node* one)`：将节点`one`的下一个节点弹出

`base_list_node* get_last_node(size_t index)`：将第`index`个节点返回
