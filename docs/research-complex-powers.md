# Complex power semantics in LMCAS

For a nonzero base, the principal power is $z^w=\exp(w\log z)$. The principal
logarithm uses $\log z=\log|z|+i\arg z$; its cut follows the negative real axis.
Integer powers are single-valued products and reciprocals, so they use binary
exponentiation. A rational exponent such as $1/3$ follows the principal branch:
$(-8+i0)^{1/3}=1+\sqrt{3}i$. See [DLMF §4.2(i), (iv)](https://dlmf.nist.gov/4.2)
and the [C11 `cpow` contract, §7.3.8.2](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf).

The two sides of the cut satisfy $\log(-a\pm i0)=\log a\pm i\pi$ for $a>0$;
signed imaginary zero selects the side. See [DLMF 4.2.7](https://dlmf.nist.gov/4.2.E7).
[LMMC's power implementation](../external/LMCAS/LMMC/src/complex.c) provides
integer exponentiation and principal logarithm evaluation, with a documented
[zero-base convention](../external/LMCAS/LMMC/include/lmmc/complex.h): $0^0=1$,
$0^w=0$ when $\Re w>0$, and a domain error for negative real part or nonzero
pure imaginary exponent.

Exact exponent classification precedes binary64 evaluation. For example,
$9007199254740993/9007199254740992$ rounds to `1.0` in binary64 although
its exact value is fractional. [LMCAS `eval_complex`](../external/LMCAS/src/expr_complex_evaluation.cpp)
reports `UnsupportedExpression` when an exact fractional exponent rounds to an
integer; this preserves the branch choice. Finite noninteger results carry
uncertified absolute error bounds, and nonfinite intermediate results from the
LMMC logarithm/multiply/exponential path produce `NumericFailure`.
