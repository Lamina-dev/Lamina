# <img src="./assets/logo-icon.svg" width="10%"> This is Lamina version 1.1.0 (Release Candidate 0 )

<img src="./assets/logo.svg" width="100%">

<div align="right">
    <strong> 简体中文</strong> | <a href="/docs/en_US/README.md"> English</a> | <a href="/docs/zh_TW/README.md"> 繁体中文</a>
</div>
<br>
<code>
    __                    _
   / /   ____ _____ ___  (_)___  ____ _
  / /   / __ `/ __ `__ \/ / __ \/ __ `/
 / /___/ /_/ / / / / / / / / / / /_/ /
/_____/\__,_/_/ /_/ /_/_/_/ /_/\__,_/
</code>

### RC阶段停止接收新特性、语法、功能，仅作为debug阶段。

[![GitHub issues](https://img.shields.io/github/issues/lamina-dev/Lamina)](https://github.com/Lamina-dev/Lamina/issues)
[![GitHub stars](https://img.shields.io/github/stars/lamina-dev/Lamina?style=flat)](https://github.com/Lamina-dev/Lamina/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/lamina-dev/Lamina?style=flat)](https://github.com/Lamina-dev/Lamina/forks) 
[![GitHub contributors](https://img.shields.io/github/contributors/lamina-dev/Lamina?style=flat)](https://github.com/Lamina-dev/Lamina/graphs/contributors)
![GitHub last commit](https://img.shields.io/github/last-commit/lamina-dev/Lamina?style=flat)
[![License](https://img.shields.io/badge/license-LGPLv2.1-blue.svg)](LICENSE)
[![Language](https://img.shields.io/badge/language-C%2B%2B-orange.svg)](https://isocpp.org/)
[![Math](https://img.shields.io/badge/math-precise-green.svg)](#精确数学特性)
[![QQ](https://img.shields.io/badge/QQ-%E4%BA%A4%E6%B5%81%E7%BE%A4-red?logo=qq&logoColor=white)](https://qm.qq.com/q/QwPXCgsJea)


## 一种专注于精确数学计算的以面向过程为主体的编程语言

[语法指南](docs/zh_CN/wiki.md) • [示例代码](/examples) • [编译指南](/docs/zh_CN/Compile.md) • [贡献指南](/docs/zh_CN/CONTRIBUTING.md) • [Wiki](https://wiki.lm-lang.org) • [动态库插件开发](/docs/zh_CN/PLUGIN_GUIDE.md) • [ToDo list](TODO.md) • [LSR](https://github.com/Lamina-dev/LSR) • [官方论坛](https://forum.lm-lang.org/) • [第三方IDEA插件](https://github.com/LangYa466/FastLmPlugin)

## 精确数学特性
1. **精确数学计算**：从底层解决浮点数精度丢失问题，支持有理数（分数）和无理数（√、π、e）的符号化存储与运算，多次循环运算仍保持精确。
2. **语法简洁直观**：支持自动补充分号、省略if/while语句的圆括号、无参函数简写等，降低代码冗余，符合数学表达习惯。
3. **原生数学友好**：无需第三方库，直接支持向量、矩阵运算、大整数阶乘等数学操作，满足复杂数学问题需求。
4. **友好开发体验**：交互式REPL支持关键字高亮、自动补齐，提供完整错误栈追踪，便于调试；智能终端自动适配色彩，避免乱码。
5. **模块化设计**：通过`include`语句引入外部模块，支持`::`命名空间访问符，实现代码复用与隔离。
6. **灵活数据类型**：涵盖精确数值类型（rational/irrational）、复合类型（数组/矩阵/结构体）及匿名函数，适配多样开发场景。

