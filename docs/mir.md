# Lamina MIR 核心定义

1. 临时值、绑定值
```aiignore
%name = int 2  # 临时储存，编译器会自己分配寄存器

$v1 = frac 4 7;  # 相当于定义变量
```
临时值只能被使用一次，使用之后就会被释放， 绑定值会跟随作用域结束销毁

2. 运算
```aiignore
# 加法运算，其他同理
%_1 = int 2
%_2 = int 6
%_3 = int add int %_1, int %_2
$v1 = %_3    # 8  

# 逻辑运算
%_4 = int 2
%_5 = int 5
%_6 = bool le int %_4, int %_5    # 2 <= 5   -> true
$v2 = %_6

#其他逻辑运算 lt ge gt eq ne and or 等
```
