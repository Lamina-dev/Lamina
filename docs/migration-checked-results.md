# Checked-result migration

LMCAS computation APIs now treat `lamina::Result<T>` and `lamina::CasError` as the authoritative failure contract. Callers must inspect a result before reading `value()` and must propagate `error()` without converting every failure to `InvalidArgument`.

## Replace unchecked calls

Prefer the checked entry point whenever both forms exist:

| Previous call pattern | Checked call pattern |
| --- | --- |
| `Integrator::integrate(expr, variable)` | `Integrator::integrate_checked(expr, variable, context)` |
| `Integrator::integrate_def(...)` | `Integrator::integrate_def_checked(..., context)` |
| `IntervalUnion::from_intervals(...)` | `IntervalUnion::from_intervals_checked(..., context)` |
| assumption mutation/query helpers | the corresponding `*_checked` member |
| calculus, matrix, geometry, ODE, and complex-analysis helpers | the corresponding `*_checked(..., context)` overload |

The overload without an explicit context remains suitable for a single bounded operation. Multi-step work should share one context so cancellation and resource accounting cover the complete computation.

```cpp
lamina::ComputationContext context({
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

## Ownership

Bridge object parameters are borrowed for the duration of a call. Returned runtime object pointers transfer one owning reference to the VM caller. Composite results now build payloads under RAII and release ownership only after the complete `Result` object has been constructed.

## Build verification

Use the checked presets from the repository root:

```text
cmake --preset strict-debug
cmake --build --preset strict-debug
ctest --preset strict-debug
```

Linux sanitizer presets are `linux-asan-ubsan` and `linux-tsan`. Full standalone LMCAS/LMMC equivalents live in `external/LMCAS/CMakePresets.json`. The `package` preset installs LMCAS for the consumer and standalone-header verification under `external/LMCAS/tests/package_consumer`.
