# Numeric mathematical contracts

This reference records the numerical contracts shared by LMMC and CAS numeric evaluation.

## Constants and special functions

- `lmmc_eps` returns binary64 machine epsilon. `LMMC_REAL_EPSILON` remains the
  existing empirical algorithm threshold, not an alias for machine precision.
- Lambert W success requires convergence; exhaustion propagates as a checked
  failure, and failed CAS constant folding retains the original function.
  Beta normalization is shared by the function and distribution kernels;
  valid underflow may return zero, but failed computations do not publish a
  successful approximate value.
- `extended_gcd(a, b, s, t)` accepts signed `int64_t` operands, returns the
  nonnegative gcd, and produces coefficients satisfying `a*s + b*t == gcd`.
  Inputs containing `INT64_MIN` raise `std::overflow_error` before modifying
  either output coefficient because that magnitude is outside `int64_t`.
- `Value::as_number_checked` preserves exact integers and rationals as finite
  binary64 values; a nonzero exact input must also remain nonzero after conversion.
  Overflow and underflow report `CasErrc::NumericFailure` with operation
  `Value::as_number_checked`.

## Integration and ODE

- Tanh-sinh convergence compares only completed quadrature layers. Exhausting
  the callback budget in an incomplete layer returns `CONVERGENCE_FAILED`,
  the last completed estimate (or zero), positive-infinite error, and the actual
  callback count. No rounded endpoint is sampled.
- SDIRK stage checks use state-dimensional user tolerances and never commit a
  failed stage; local tolerances are not a global-error guarantee.

## Statistics and sparse matrices

- LMMC correlation uses scaled centered moments; nonfinite or constant samples
  cannot produce a successful coefficient. Combinations avoid overflowing an
  intermediate when the final binary64 value is representable and report
  numerical failure for genuine overflow.
- Successful Beta samples are finite and lie in `[0,1]`, including rounded
  endpoints. Invalid arguments leave the RNG and output unchanged.
- Sparse builders and COO conversion merge duplicate coordinates, retain explicit
  zeros produced by duplicate-coordinate accumulation, and sort compressed indices.
  Borrowed CSR/CSC buffers must already be canonical and are never silently
  rewritten. Dense conversion, diagonal, norms, arithmetic, and solvers observe
  the same represented matrix.

## Optimization

- LMMC optimizers check convergence at the output point after the last permitted
  update. `num_iter` counts completed updates (zero for initial convergence);
  `final_residual` describes the output point, not the preceding iterate.
  If a failed callback prevents evaluating that norm, it is NaN.
- Levenberg–Marquardt can converge with nonzero residual when `||Jᵀr||₂`
  satisfies the absolute tolerance or the tolerance relative to its initial
  gradient norm. Its `final_residual` remains `||r||₂`. Both norms must be finite;
  residual-based convergence remains supported.
