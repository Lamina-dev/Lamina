# Checked results and API migration

LMCAS computation APIs use the `LMCAS` namespace and treat `LMCAS::Result<T>` and `LMCAS::CasError` as the authoritative failure contract. Callers must inspect a result before reading `value()` and must propagate `error()` without converting every failure to `InvalidArgument`.

## Replace unchecked calls

Prefer the checked entry point whenever both forms exist:

| Previous call pattern | Checked call pattern |
| --- | --- |
| `Integrator::integrate(expr, variable)` | `Integrator::integrate_checked(expr, variable, context)` |
| `Integrator::integrate_def(...)` | `Integrator::integrate_def_checked(..., context)` |
| `IntervalUnion::from_intervals(...)` | `IntervalUnion::from_intervals_checked(..., context)` |
| `ExprSet::set_union`, `intersection`, `difference`, `symmetric_difference` | `expr_set_union`, `expr_set_intersection`, `expr_set_difference`, `expr_set_symmetric_difference` |
| assumption mutation/query helpers | the corresponding `*_checked` member |
| calculus, matrix, geometry, ODE, and complex-analysis helpers | the corresponding `*_checked(..., context)` overload |

The unchecked `ExprSet` members retain their legacy failure behavior: an
underlying failed result causes `Result::value()` to throw
`std::logic_error("Result does not contain a value")`. The checked `expr_set_*`
functions return the underlying `CasError` through `Result<ExprSet>`.

The unchecked `Value` compatibility methods were removed. Replace
`as_number`, `as_rational`, `as_irrational`, and `as_symbolic` with their
`*_checked` forms. Replace `vector_add`, `dot_product`, and `matrix_multiply`
with the corresponding `*_checked` methods. Propagate each `Result` error;
conversion errors retain `InvalidArgument` or `UnboundSymbol`, and shape
errors retain `DimensionMismatch`.

`SymbolicExpr` no longer exposes the broad `Type` enum or the deprecated
`get_type`, `get_operands`, `get_number_value`, and `get_identifier`
introspection methods. Use concrete predicates or `expr_match` for matching,
public transformations for expression traversal, `is_number` plus
`get_number` (or `evaluate_numeric`) for numeric values, and
`symbol_name(expression)` for a borrowed `optional<string_view>`. Recompile
C++ consumers after migrating these source-level API changes.

The overload without an explicit context remains suitable for a single bounded operation. Multi-step work should share one context so cancellation and resource accounting cover the complete computation.

```cpp
LMCAS::ComputationContext context({
    .max_steps = 100000,
    .max_recursion_depth = 256,
});
auto result = integrator.integrate_checked(expression, "x", context);
if (!result) {
    return handle(result.error().code,
                  result.error().operation,
                  result.error().message);
}
use(result.value());
```

`ComputationContext` is thread-confined. Create one context per operation or per request; do not share it across worker threads. A cancelled or exhausted context returns `CasErrc::Cancelled` or `CasErrc::ResourceLimit` instead of throwing.

## Error classification

Do not catch `std::exception` and relabel it as invalid input. Input validation reports `CasErrc::InvalidArgument`; domain, dimensional, unsupported, inconclusive, cancellation, resource, numeric, and invariant failures retain their specific codes. Unexpected C++ exceptions are internal failures.

The Lamina bridge converts checked failures directly to `Result.Err(MathError)` and preserves the error code, operation, and message. Every bridge export has a `noexcept` C ABI boundary. Result-returning exports classify exceptions as follows:

- `std::bad_alloc` becomes `MathErrorCode::ResourceLimit`;
- propagated `CasError` values keep their original classification;
- other standard and unknown exceptions become `MathErrorCode::InternalError`.

If constructing the error object itself cannot allocate, the boundary returns `nullptr`; no C++ exception crosses the C ABI. Legacy scalar and raw-object accessors return their documented sentinel (`nullptr`, `false`, `0`, `-1`, or NaN) when an unexpected exception reaches the boundary.

## Proof obligations and lexical scope

Callers using `ProvedNonzeroResidual` must migrate to
`ProvedNonIdentityResidual`. The result propositions, pointwise obligations,
and lexical binding rules are documented in
[CAS mathematical contracts](cas-contracts.md#proof-obligations-and-lexical-scope).

## Public bridge boundary

Lamina's bridge uses LMCAS public headers rather than AST node classes or the
LMCAS `src` include directory. `symbol_name(expression)` returns an optional
borrowed `string_view` for a single symbol; `relation_op(expression)` returns an
optional relation operator. Null or nonmatching expressions return `nullopt`.
The name view must not outlive the expression or survive reassignment of its
wrapper; copy it when longer ownership is needed.

The language `solve` entry points still require an equality relation, whereas
`roots` accepts an expression interpreted as zero. Public inspection replaces
private AST casts without removing this language-level validation.

## Lamina bridge 命名迁移

本次迁移移除了 143 个旧 C 符号，并以 142 个当前符号替代。两个旧的多项式
GCD 名称合并到同一个当前符号，因此当前符号数少一个。
`lmx_cas_assumptions_query_periodic` 是本次新增入口，不属于重命名。旧符号未保留
兼容别名；外部 C 调用方和缓存模块必须使用当前符号。

下表给出完整的族级清单。除后续例外表列出的项目外，`<op>` 后缀保持不变：

| 原符号族 | 旧→当前 | 当前符号族/规则 |
| --- | ---: | --- |
| `lmx_computer_algebra_algebra_<op>` | 9 → 8 | `lmx_cas_<op>` |
| `lmx_computer_algebra_assumptions_<op>` | 11 → 11 | `lmx_cas_assumptions_<op>` |
| `lmx_computer_algebra_domain_<op>` | 2 → 2 | `lmx_cas_domain_<op>` |
| `lmx_computer_algebra_equation_solving_<op>` | 26 → 26 | `lmx_cas_<op>` |
| `lmx_computer_algebra_expression_<op>` | 14 → 14 | `lmx_cas_expr_<op>` |
| `lmx_computer_algebra_set_<op>` | 8 → 8 | `lmx_cas_set_<op>` |
| CAS `symbol`、`parse`、`pi`、`euler_number`、`golden_ratio` | 5 → 5 | `lmx_computer_algebra_` → `lmx_cas_` |
| `lmx_mathematics_<op>` | 28 → 28 | `lmx_math_<op>` |
| `lmx_statistics_<op>` | 40 → 40 | `lmx_stats_<op>` |

后缀发生变化或多个旧符号合并时，采用以下精确映射：

| 原 C 符号 | 当前 C 符号 |
| --- | --- |
| `lmx_computer_algebra_algebra_multivariate_greatest_common_divisor` | `lmx_cas_polynomial_gcd` |
| `lmx_computer_algebra_algebra_polynomial_greatest_common_divisor` | `lmx_cas_polynomial_gcd` |
| `lmx_computer_algebra_equation_solving_polynomial_system_full_by_names` | `lmx_cas_polynomial_system_by_names` |
| `lmx_computer_algebra_equation_solving_polynomial_system_full_by_symbols` | `lmx_cas_polynomial_system_by_symbols` |
| `lmx_computer_algebra_equation_solving_symbol` | `lmx_cas_solve_by_symbol` |
| `lmx_computer_algebra_equation_solving_with_assumptions` | `lmx_cas_solve_with_assumptions` |
| `lmx_mathematics_absolute_value` | `lmx_math_abs` |
| `lmx_mathematics_binary_exponential` | `lmx_math_exp2` |
| `lmx_mathematics_binary_logarithm` | `lmx_math_log2` |
| `lmx_mathematics_common_logarithm` | `lmx_math_log10` |
| `lmx_mathematics_cosine` | `lmx_math_cos` |
| `lmx_mathematics_euler_number` | `lmx_math_e` |
| `lmx_mathematics_exponential` | `lmx_math_exp` |
| `lmx_mathematics_golden_ratio` | `lmx_math_phi` |
| `lmx_mathematics_hypotenuse` | `lmx_math_hypot` |
| `lmx_mathematics_imaginary_unit` | `lmx_math_i` |
| `lmx_mathematics_inverse_cosine` | `lmx_math_acos` |
| `lmx_mathematics_inverse_sine` | `lmx_math_asin` |
| `lmx_mathematics_inverse_tangent` | `lmx_math_atan` |
| `lmx_mathematics_logarithm` | `lmx_math_log_base` |
| `lmx_mathematics_natural_logarithm` | `lmx_math_ln` |
| `lmx_mathematics_natural_logarithm_legacy` | `lmx_math_log` |
| `lmx_mathematics_power` | `lmx_math_pow` |
| `lmx_mathematics_sine` | `lmx_math_sin` |
| `lmx_mathematics_square_root` | `lmx_math_sqrt` |
| `lmx_mathematics_tangent` | `lmx_math_tan` |
| `lmx_statistics_binomial_cumulative_distribution` | `lmx_stats_binomial_cdf` |
| `lmx_statistics_binomial_probability_mass` | `lmx_stats_binomial_pmf` |
| `lmx_statistics_chi_squared_cumulative_distribution` | `lmx_stats_chi2_cdf` |
| `lmx_statistics_chi_squared_probability_density` | `lmx_stats_chi2_pdf` |
| `lmx_statistics_fisher_f_cumulative_distribution` | `lmx_stats_f_cdf` |
| `lmx_statistics_fisher_f_probability_density` | `lmx_stats_f_pdf` |
| `lmx_statistics_normal_cumulative_distribution` | `lmx_stats_normal_cdf` |
| `lmx_statistics_normal_probability_density` | `lmx_stats_normal_pdf` |
| `lmx_statistics_poisson_cumulative_distribution` | `lmx_stats_poisson_cdf` |
| `lmx_statistics_poisson_probability_mass` | `lmx_stats_poisson_pmf` |
| `lmx_statistics_standard_deviation` | `lmx_stats_stddev` |
| `lmx_statistics_student_t_cumulative_distribution` | `lmx_stats_t_cdf` |
| `lmx_statistics_student_t_probability_density` | `lmx_stats_t_pdf` |

`by_name`、`by_symbol` 及其复数形式继续区分文本变量和符号表达式输入。

CAS 实现按职责归入 `bridge/cas/`；原 advanced 文件中的代数、方程组、
理想和不等式操作分别归入 `algebra.cpp`、`systems.cpp`、`ideals.cpp`
和 `inequalities.cpp`。数值与统计入口位于 `bridge/math.cpp` 和
`bridge/stats.cpp`。

`std.cas`、`std.math`、`std.stats` 的语言模块路径、函数名、参数和结果类型
保持稳定，FFI 绑定与编译器内置表达式调用同步使用新符号。C ABI 以新名称为准；
外部 C 调用方和动态加载方需要更新符号并重新链接。

已编译的 `.lmc` 保存原生符号名，升级后应从源文件重新编译相关模块及其
`_lm_cache` 产物。来源于原生函数名的 `MathError.operation` 同步采用新名称，
例如 `lmx_cas_solve_by_symbol`、`lmx_stats_factorial` 和 `lmx_math_log2`；
显式操作标识（如 `math.sqrt`）及依赖库提供的诊断继续沿用各自的契约。

## Mathematical contracts

- [CAS mathematical contracts](cas-contracts.md)
- [Numeric mathematical contracts](numeric-contracts.md)

## Ownership

Bridge object parameters are borrowed for the duration of a call. Returned runtime object pointers transfer one owning reference to the VM caller. Composite results now build payloads under RAII and release ownership only after the complete `Result` object has been constructed.

## Development checks

Use the checked presets from the repository root:

```text
cmake --preset strict-debug
cmake --build --preset strict-debug
ctest --preset strict-debug
```

Linux sanitizer presets are `linux-asan-ubsan` and `linux-tsan`. Standalone math-stack presets are in `external/LMCAS/CMakePresets.json`; framework versions, offline setup, test selection, and package commands are documented in [LMCAS development and tests](../external/LMCAS/README.md#开发与测试) and [LMMC development tests](../external/LMCAS/LMMC/README.md#development-tests).

Source-quality and package jobs export a Git-visible source tree, including the required LMMC tooling, then configure and build the exported tree.
For a local POSIX-shell check:

```sh
work=$(mktemp -d)
python external/LMCAS/LMMC/cmake/check_package.py \
  --project lmcas --source-root . --snapshot-dir "$work/source"
```

Configure the producer from `$work/source/external/LMCAS` (or its `LMMC`
subdirectory for a standalone LMMC build). Invoke the exported
`LMMC/cmake/check_package.py` with the matching `--project`, `--build-dir`,
`--work-dir`, and `--config` to build, install, relocate, and run the installed
consumer and standalone-header checks.

Snapshots include current tracked file contents and nonignored untracked files,
recursively including initialized dependency repositories. They exclude Git
metadata, ignored untracked helpers, and in-source build products, and reject
missing required tools. Snapshots make current working-tree changes testable without hidden local files. Before publishing, add new files, commit each changed dependency repository, and update its parent gitlink.
