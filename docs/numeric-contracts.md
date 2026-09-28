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
- `BigInt::is_prime_checked` returns a proved boolean result for machine-word
  inputs. For larger integers, a compositeness witness yields `false`; passing
  the tested witnesses yields `CasErrc::Inconclusive` rather than a primality
  claim.

## Integration and ODE

- Tanh-sinh convergence compares only completed quadrature layers. Exhausting
  the callback budget in an incomplete layer returns `CONVERGENCE_FAILED`,
  the last completed estimate (or zero), positive-infinite error, and the actual
  callback count. No rounded endpoint is sampled.
- SDIRK stage checks use state-dimensional user tolerances and never commit a
  failed stage; local tolerances are not a global-error guarantee.
- Characteristic quadratic roots retain distinct representable coefficients;
  root multiplicity follows a zero discriminant. ODE basis terms therefore
  preserve small nonintegral exponents and closely spaced roots.

## Statistics and sparse matrices

- LMMC correlation uses scaled centered moments; nonfinite or constant samples
  cannot produce a successful coefficient. Combinations avoid overflowing an
  intermediate when the final binary64 value is representable and report
  numerical failure for genuine overflow.
- Successful Beta samples are finite and lie in `[0,1]`, including rounded
  endpoints. Invalid arguments leave the RNG and output unchanged.
- Chi-square sampling accepts every finite positive degree of freedom,
  including values whose half-degree underflows; valid samples may round to
  zero. Student-t and F sampling report numerical failure when a sampled
  denominator vanishes or the result cannot be represented as finite.
- Chi-square CDF and Student-t PDF/CDF retain positive subnormal degrees of
  freedom when `df / 2` underflows. At the minimum positive binary64 df, a
  positive chi-square argument has CDF 1, finite t arguments have CDF 0.5,
  and the t density at zero is approximately `sqrt(df) / 2`.
- Chi-square PDF uses the small-shape Gamma limit when `df / 2` underflows;
  at `x = df = DBL_TRUE_MIN`, its density is 0.5. F CDF uses the small
  numerator-shape Beta limit for finite positive `x`; equal minimum positive
  numerator and denominator degrees yield probability 0.5.
- Sparse builders and COO conversion merge duplicate coordinates, retain explicit
  zeros produced by duplicate-coordinate accumulation, and sort compressed indices.
  Borrowed CSR/CSC buffers must already be canonical and are never silently
  rewritten. Dense conversion, diagonal, norms, arithmetic, and solvers observe
  the same represented matrix.

## Interpolation and linear algebra

- Bilinear interpolation computes fractions across finite endpoint spans,
  including spans whose subtraction overflows binary64. Lagrange interpolation
  normalizes barycentric weights and values so representable tiny node spacing
  and large constant ordinates remain evaluable.
- Vector and matrix division by a finite nonzero scalar computes each quotient
  directly and accepts representable results even when the scalar is subnormal.
- A 2×2 determinant retains cancellation between individually overflowing
  products when the resulting determinant fits binary64; singular finite
  matrices yield zero.

## Optimization

- LMMC optimizers check convergence at the output point after the last permitted
  update. `num_iter` counts completed updates (zero for initial convergence);
  `final_residual` describes the output point, not the preceding iterate.
  If a failed callback prevents evaluating that norm, it is NaN.
- Levenberg–Marquardt can converge with nonzero residual when `||Jᵀr||₂`
  satisfies the absolute tolerance or the tolerance relative to its initial
  gradient norm. Its `final_residual` remains `||r||₂`. Both norms must be finite;
  residual-based convergence remains supported.
