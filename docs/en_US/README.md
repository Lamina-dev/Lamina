# <img src="/assets/logo-icon.svg" width="10%"> This is Lamina version 1.1.0 (Release Candidate 0)

<img src="/assets/logo.svg" width="100%">

<div align="right">
    <a href="../zh_TW/README.md">繁體中文</a> | <a href="/README.md">简体中文</a> | <strong>English</strong>
</div>
<br>

### RC stage: no new features, syntax changes, or functionality will be accepted; this release candidate is intended for debugging only.

[![GitHub issues](https://img.shields.io/github/issues/lamina-dev/Lamina)](https://github.com/Lamina-dev/Lamina/issues)
[![GitHub stars](https://img.shields.io/github/stars/lamina-dev/Lamina?style=flat)](https://github.com/Lamina-dev/Lamina/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/lamina-dev/Lamina?style=flat)](https://github.com/Lamina-dev/Lamina/forks)
[![GitHub contributors](https://img.shields.io/github/contributors/lamina-dev/Lamina?style=flat)](https://github.com/Lamina-dev/Lamina/graphs/contributors)
![GitHub last commit](https://img.shields.io/github/last-commit/lamina-dev/Lamina?style=flat)
[![License](https://img.shields.io/badge/license-LGPLv2.1-blue.svg)](/LICENSE)
[![Language](https://img.shields.io/badge/language-C%2B%2B-orange.svg)](https://isocpp.org/)
[![Math](https://img.shields.io/badge/math-precise-green.svg)](#Precise-Math-Features)
[![QQ](https://img.shields.io/badge/QQ-%E4%BA%A4%E6%B5%81%E7%BE%A4-red?logo=qq&logoColor=white)](https://qm.qq.com/q/QwPXCgsJea)

---

## Overview

Lamina is a procedural-first programming language focused on exact mathematical computation.

[Syntax Guide](docs/en_US/wiki.md) • [Examples](/examples) • [Compile Guide](docs/en_US/Compile.md) • [Contribution Guide](docs/en_US/CONTRIBUTING.md) • [Wiki](https://wiki.lm-lang.org) • [Dynamic Library Plugin Development](docs/en_US/PLUGIN_GUIDE.md) • [ToDo list](TODO.md) • [What's New](docs/en_US/NewFeature.md) • [LSR](https://github.com/Lamina-dev/LSR) • [Official Forum](https://forum.lm-lang.org/)

## Precise Math Features
1. **Precise mathematical computation**: Solves floating-point precision loss at the low level, supports rational numbers (fractions) and symbolic storage/operations for irrationals (√, π, e), and preserves precision across iterative computations.
2. **Concise and intuitive syntax**: Supports automatic semicolon completion, omission of parentheses for if/while statements, shorthand for parameterless functions, reducing code verbosity and matching mathematical notation.
3. **Math-friendly by design**: No third-party libraries required; built-in support for vectors, matrix operations, big-integer factorials, and other math operations to satisfy complex mathematical requirements.
4. **Developer-friendly experience**: Interactive REPL with keyword highlighting and autocompletion, full error stack traces for easier debugging; terminal color auto-adaption to avoid garbled output.
5. **Modular design**: Import external modules via `include`, support `::` namespace access, enabling code reuse and isolation.
6. **Flexible data types**: Includes exact numeric types (rational/irrational), composite types (arrays/matrices/structs/modules), anonymous functions, and C++ functions to fit diverse development scenarios.

## Privacy Policy

Unless explicitly requested by the user or the person installing/operating this program, the core program will not transmit any information to other networked systems or third parties.

## Sponsors

<table>
        <tr>
                <td><img src="https://signpath.org/assets/logo.svg" alt="SignPath" width="200"></td>
                <td><a href="https://about.signpath.io/">SignPath.io</a> provides free code signing services; certificates are issued by the <a href="https://signpath.org/">SignPath Foundation</a>.</td>
        </tr>
        <tr>
                <td><img src="https://chuqiyun.com/static/images/logo2.png" alt="Chuqiyun" width="100"></td>
                <td><a href="https://chuqiyun.com/aff/HNHKAJUX">Chuqiyun</a> provides high-quality cloud services, powering Lamina's network services.</td>
        </tr>
</table>
