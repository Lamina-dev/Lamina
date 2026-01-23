# <img src="/assets/logo-icon.svg" width="10%"> This is Lamina version 1.1.0 (Release Candidate 0)

<img src="/assets/logo.svg" width="100%">

<div align="right">
    <strong>繁體中文</strong> | <a href="../zh_CN/README.md">简体中文</a> | <a href="../en_US/README.md">English</a>
</div>
<br>

### RC階段已停止接收新特性、語法或功能變更，僅作為除錯階段。

[![GitHub issues](https://img.shields.io/github/issues/lamina-dev/Lamina)](https://github.com/Lamina-dev/Lamina/issues)
[![GitHub stars](https://img.shields.io/github/stars/lamina-dev/Lamina?style=flat)](https://github.com/Lamina-dev/Lamina/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/lamina-dev/Lamina?style=flat)](https://github.com/Lamina-dev/Lamina/forks)
[![GitHub contributors](https://img.shields.io/github/contributors/lamina-dev/Lamina?style=flat)](https://github.com/Lamina-dev/Lamina/graphs/contributors)
![GitHub last commit](https://img.shields.io/github/last-commit/lamina-dev/Lamina?style=flat)
[![License](https://img.shields.io/badge/license-LGPLv2.1-blue.svg)](/LICENSE)
[![Language](https://img.shields.io/badge/language-C%2B%2B-orange.svg)](https://isocpp.org/)
[![Math](https://img.shields.io/badge/math-precise-green.svg)](#精確數學特性)
[![QQ](https://img.shields.io/badge/QQ-%E4%BA%A4%E6%B5%81%E7%BE%A4-red?logo=qq&logoColor=white)](https://qm.qq.com/q/QwPXCgsJea)

---

## 繁體中文

一種以面向過程為主、專注於精確數學計算的程式語言

[語法指南](docs/zh_TW/wiki.md) • [示例程式](/examples) • [編譯指南](docs/zh_TW/Compile.md) • [貢獻指南](docs/zh_TW/CONTRIBUTING.md) • [Wiki](https://wiki.lm-lang.org) • [動態庫外掛開發](docs/zh_TW/PLUGIN_GUIDE.md) • [待辦清單](TODO.md) • [更新內容](docs/zh_TW/NewFeature.md) • [LSR](https://github.com/Lamina-dev/LSR) • [官方論壇](https://forum.lm-lang.org/)

## 精確數學特性
1. **精確數學計算**：從底層解決浮點數精度遺失問題，支援有理數（分數）和無理數（√、π、e）的符號化儲存與運算，多次循環運算仍保持精確。
2. **語法簡潔直觀**：支援自動補充分號、省略 if/while 語句的圓括號、無參數函數簡寫等，降低程式碼冗贅，符合數學表達習慣。
3. **原生數學友好**：無需第三方函式庫，直接支援向量、矩陣運算、大整數階乘等數學運算，滿足複雜數學問題的需求。
4. **友好開發體驗**：互動式 REPL 支援關鍵字高亮、自動補齊，提供完整錯誤堆疊追蹤，便於除錯；智慧終端自動調整色彩以避免亂碼。
5. **模組化設計**：透過 `include` 語句引入外部模組，支援 `::` 命名空間存取符，實現程式碼重用與隔離。
6. **彈性資料型別**：涵蓋精確數值型別（rational/irrational）、複合型別（陣列/矩陣/結構/模組）及匿名函數與 C++ 函數，適配多樣開發情境。

### 隱私政策

除非使用者或安裝／操作該程式的人員明確要求，否則本核心程式不會將任何資訊傳輸到其他聯網系統或第三方。

### 讚助

<table>
        <tr>
                <td><img src="https://signpath.org/assets/logo.svg" alt="SignPath" width="200"></td>
                <td><a href="https://about.signpath.io/">SignPath.io</a> 提供免費程式碼簽章服務，憑證由 <a href="https://signpath.org/">SignPath 基金會</a> 頒發。</td>
        </tr>
        <tr>
                <td><img src="https://chuqiyun.com/static/images/logo2.png" alt="Chuqiyun" width="100"></td>
                <td><a href="https://chuqiyun.com/aff/HNHKAJUX">初七雲</a> 提供優質的雲端服務，為 Lamina 的網路服務注入強勁動力。</td>
        </tr>
</table>
