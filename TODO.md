# Lamina ToDo List

## 符号含义
⚠️(在第二个RC发布前一定要完成)

🟡建议早日完成(以完备基本的语法特性)

🟢实现时间较不重要

🟧 已经实现但功能不完善


<b> 注意 ToDo list 的每一项需要标注状态 (已完成) (未完成) (正在做)
</b>

## 库方面
- [ ] IO库（正在做）<br>
     备注：`已在extensions/standard/io.cpp`有预留

- [ ] bit库（正在做）<br>

- [ ] 🟡 数学图像库（未完成）<br>     备注：诸如展示函数图像等功能

- [ ] ⚠️ 集合库（正在做）<br>

- [ ] ⚠️线程库（未完成）<br>

- [ ] ⚠️console库（未完成）<br>
     备注：`已在extensions/standard/io.cpp`有预留

## 语法方面

- [ ] 🟡 列表项赋值语句 a[i] = v（未完成）<br>
     备注：需要让Value类储存shared_ptr<std::vector<Value>> 
     而不是 std::vector<Value>

- [ ] 🟡 三元表达式（未完成）<br>
     备注：
    方案一: `if cond : expr else expr`

    方案二: `cond if expr else expr`

    方案三:  `cond ? expr ! expr`

- [ ] 🟡 类型注释及类型系统（未完成）<br>
     备注：

- [ ] 🟡 match语句（未完成）<br>
     备注：

- [ ] 🟡 is not or in not int 运算符（未完成）<br>
     备注：完成interpreter/parse_expr.cpp的todo注释

- [ ] 🟡 管道运算符（未完成）<br>
     备注：

## 内置函数方面
- [ ] eval(code)（未完成）<br>
     备注：

- [ ] 🟡 help(key)（未完成）<br>
     备注：

- [ ] 🟡 open(name, pattern, encoding)（未完成）<br>
     备注：

- [ ] 🟧 ⚠️find(arr, lambda, times)（未完成）<br>
    备注：目前只能找到第一个符合条件的值, 希望有人能完成对times参数的处理

- [ ] 🟧 ⚠️replace(arr, lambda)（未完成）<br>
     备注：需要让Value类储存shared_ptr<std::vector<Value>> 
     而不是 std::vector<Value>

## 类型方面
- [ ] ⚠️ LmInt（未完成）<br>
     备注：

     优化现在Bigint效率(LmBigNum库在做了)，

     把Bigint改名为LmInt, 

     成为lamina唯一整数类型

     让Rational, Irrtional, Complex类型的整数部分都使用LmInt

     📍 该更新能解决到大部分有理数和无理数的精度问题


- [x] ⚠️ 分数转小数（已完成）<br>

     用户要从有理数转到安全小数，

     需要设置小数位数(以防止无限循环小数和不循环小数）

     备注：已添加有理数函数用于转化成小数字符串

- [ ] ⚠️ LmComplex（未完成）<br>
     备注：复数类型

- [ ] ⚠️ LmList（正在做）<br>
     备注：链表类型

     已完成基本实现，正在debug及完善

     无具体引用，欢迎后人补充，文件位置

     底层模板类：`interpreter/lamina_api/base_list.hpp`

     继承套用：`interpreter/lamina_api/list.hpp`

## 其他
- [ ] ⚠️ 添加检查循环导入的功能（未完成）<br>

- [ ] ⚠️ 测试新的c++ module loader（未完成）<br>
     备注：

- [ ] ⚠️ 修复repl input 的 bug（未完成）<br>
     备注：

- [ ] ⚠️ 报错系统使用 err_reporter 和 src_manger（未完成）<br>
     备注：
