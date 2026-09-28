# CAS mathematical contracts

This reference records the checked symbolic contracts that callers can rely on.

## Exact arithmetic

- `symbolic_to_poly<T>` returns `Result<Polynomial<T>>`; inspect success before
  accessing coefficients. A genuine zero polynomial is successful. Unsupported
  expressions and unrepresentable coefficients are `UnsupportedExpression`,
  excessive power expansion is `ResourceLimit`, and invalid inputs are
  `InvalidArgument`. The three `extract_coeff_value` specializations also return
  `Result`; integer extraction requires an exact integer value, including for
  doubles adjacent to one. Rational extraction preserves the exact binary64 value.
- Polynomial arithmetic rejects incompatible nonzero variable rings with
  `std::invalid_argument`, and equality is false across incompatible rings.
  Zero polynomials adopt the nonzero operand's ring. Monomial orderings may
  differ, but variable names and their positions must agree.
- `Irrational::sign_checked` and `abs_checked` use certified rational enclosures.
  Pure radicals refine until resolved or a resource limit; unresolved pi/e
  combinations at 4096 bits report `Inconclusive`. Unchecked `abs()` translates
  checked failure into an exception rather than guessing from a double value.
- `Rational::floor(n)` retains its truncation-toward-zero contract, including
  `n == 0`; successful integer truncation leaves a canonical denominator of one.
- `EqvBudget` is shared by the complete equivalence proof, including profile
  rewriting followed by the Core proof. Recursive rewrite traversal, candidate
  append operations, and proof/profile stages consume the step budget; active
  visitor depth and candidate-tree growth have independent limits. Exhaustion
  reports `CasErrc::ResourceLimit` from `LMCAS.equivalent_core`.

## Assumptions and inference

- Assumption caches observe committed store revisions, including mutation through
  a retained valid store reference. Empty scopes inherit parent facts; child
  declarations shadow the corresponding parent evidence, including dependent
  transitive relations. Store references still obey vector reallocation and
  scope-pop lifetime rules.
- Real logarithm, exponential, and power inference requires the original real
  domain. Checked queries report `DomainError` for proved undefined expressions
  and preserve `Unknown` when obligations are unproved; an integer assumption
  alone does not make its logarithm real. Ordered sign predicates imply realness,
  but `NonZero` alone does not.
- Product and quotient signs use all possible real signs, including zero.
  Weak signs are not silently strengthened, and zero factors cannot erase
  undefined or unproved factor domains.
- Simplification and expansion preserve unresolved domains and principal-branch
  restrictions, including cancellation, zero absorption, and nested zero powers.
  Checked context-aware transforms consume their supplied assumptions. A
  conditional rewrite requires every declared condition; missing context does
  not authorize an assumption-dependent rule.
- `InferenceEngine::propagate_bounds` preserves exact point values and original
  variable endpoint openness. Arithmetic and elementary-function bounds are
  certified outward enclosures, not necessarily tight; unavailable evidence or
  exhausted internal budgets produce `nullopt`, never a partial bound.
- Closed rational arithmetic in an exponent has the same domain as its exact
  value, even before normalization. Unknown integrality is not proof of a
  noninteger exponent; negative-base possibilities remain unresolved rather
  than being excluded by an invented positive-base requirement.
- Real-domain projection preserves a proved violation of a real function's
  argument restriction. When an explicit complex value occurs inside an
  otherwise defined expression, projection checks the complex result:
  unresolved real-valuedness is `Inconclusive` rather than a false empty set.
- Periodicity queries require an explicit independent variable:
  `query_periodic_checked(expression, variable)` and
  `get_period_checked(expression, variable)`. The latter returns no minimum
  period for constants even when periodicity is proved. Assumption serialization
  includes that variable, for example `PERIODIC f x 5`. Lamina raw and high-level
  `assumptions.query_periodic` / `assumptions.period` calls also take the variable;
  the old generic `"periodic"` query is rejected rather than guessing one.
- Declared periods are owned as `SymbolicExpr` values. Reassigning either the
  caller's wrapper or a value returned by `get_period` does not mutate the
  stored declaration; redeclaration is the only update path.
- Repeated bounded-interval declarations intersect with the existing bounds.
  Empty intersections, cancellation, and resource failures leave the store and
  revision unchanged; a successful declaration commits once. Serialized
  bounded declarations include their interval as
  `BOUNDED <symbol> Bounded <interval>`.

## Solving

- `solve_set` and `solve_equation` retain original domains, multiplicities,
  integer parameters, and unresolved parameter degeneracies. Exact polynomial
  solving includes complex roots. `allow_numeric == false` does not authorize
  approximate cubic roots; disabling `return_rootof` makes the whole solve
  `Inconclusive` when any root cannot be represented. `max_roots` limits numeric
  candidate searches, not complete exact sets or integer families.
- Real trigonometric inversion returns complete integer families, not samples
  from a search interval. Generated binders avoid names in the expression and
  all visible assumption scopes. Conditional solution predicates preserve
  zero-coefficient universal/empty branches instead of returning generic roots
  without their conditions.
- `factor_transcendental` and `tf_build_polynomial` return `Result` payloads.
  Unsupported polynomial conversion remains a strategy miss; resource,
  cancellation, and invariant errors propagate through factoring and the
  language bridge rather than returning the unchanged input as success.
- Typed transcendental solution sets remain typed through `solve_equation`.
  A proved `EmptySolutions` becomes an empty vector only at
  `solve_finite_checked`; conditional or non-finite sets cannot be projected
  into an unconditional finite vector.
- `solve_parametric_inequality_checked` returns exhaustive parameter-sign
  branches for certified affine and repeated-root quadratic cases. Fixed exact
  coefficients use the checked real-inequality solver; unresolved discriminants,
  root ordering, or degree above two yield `Inconclusive`.
- `factor_multivariate_checked` reconstructs its input exactly; `Complete`
  certifies terminal irreducibility. An unproved factorization retains the
  original product with `Inconclusive`; computation-budget exhaustion is
  `ResourceLimit`.
- Finite polynomial-system solving checks the reduced Gröbner basis for zero
  dimension before enumerating points. Positive-dimensional or unresolved
  parameter-degenerate systems yield `Inconclusive`; certified finite points
  still obey denominator exclusions.

## Calculus, geometry, and matrices

- Lambert W differentiation uses `exp(-W(x))/(1+W(x))` times the inner
  derivative, preserving the removable value at zero while retaining the
  genuine branch-point singularity.
- Finite-point `LimitDirection::Both` combines two independently certified
  one-sided results. `FiniteLimit` never contains unresolved poles, infinity
  products, or an unevaluated limit. Proved nonexistence and unsupported proofs
  remain distinct. A zero limiting value does not prove equality on a punctured
  neighborhood or justify selecting a Piecewise branch.
- Definite integration denotes the ordinary real improper integral, not a
  principal value. Original domains and interior boundaries are retained.
  Same-sign divergent contributions produce signed infinity; incompatible
  divergences produce `DomainError`. Unproved domain, primitive, or convergence
  obligations retain the complete original integral and bounds.
- Ordinary eigenvectors are bases of distinct eigenspaces, not positional
  partners of an eigenvalue list. Successful Jordan decomposition certifies
  `rank(P) == n` and `AP == PJ`; unsupported generalized chains return
  `Inconclusive`, never a singular successful transform. Induced matrix norms
  may retain exact `max` expressions. The zero quadratic form deterministically
  reports `positive_semidefinite`; unproved signs remain `unknown`.
- Real inflection candidates are projected from checked roots only when their
  real value and original curve/derivative domains are proved. These candidates
  satisfy the necessary second-derivative condition, not a concavity-change
  certificate or a general completeness guarantee.
- `atan2(y, x)` differentiation applies the chain rule to both arguments.
  Unsupported function derivatives now report `CasErrc::UnsupportedExpression`
  through checked differentiation instead of returning a false zero; the
  unchecked member throws `std::runtime_error`.
- Fixed positive numerical bases other than one support
  `d log(u,b)/dx = u'/(u ln b)` on the real logarithm domain; unsupported
  bases report `UnsupportedExpression`.
- Laurent classification recognizes `exp(1/(z-c)^m)` for positive integer
  `m` as essential independently of truncation. Its reported negative-power
  terms follow `1/(k!(z-c)^(mk))`; unresolved singularities stay
  `Inconclusive`. Ratio convergence uses an exact limit comparison with one.
- The canonical inverse Fourier pair `2/(1+omega^2)` and `exp(-abs(t))`
  carries an exact forward round-trip certificate. The unilateral
  `Z{n^3}=z(z^2+4z+1)/(z-1)^4` has ROC `|z|>1`.
- Checked explicit curvature, parametric curvature, and inflection points
  propagate unsupported first or second derivatives as `UnsupportedExpression`
  with the entry-point operation name. Each derivative uses the caller's context
  for cancellation and step-budget checks.
- Checked gradient, divergence, curl, Laplacian, directional derivative,
  Jacobian, and Hessian preserve `UnsupportedExpression` for unsupported
  derivatives, including second derivatives, with the entry-point operation.
  The later Hessian stage of checked extrema uses the same classification.
  Strict differentiation in integrals, extrema's first-derivative precheck,
  and Lagrange multipliers retains its `Inconclusive` contract.
- `matrix_reflection_checked(angle)` uses the reflection-axis angle in radians:
  `[[cos(2*angle), sin(2*angle)], [sin(2*angle), -cos(2*angle)]]`.
  Angle zero reflects across the x-axis. Finite angles use bounded sine/cosine
  products to construct the coefficients, including angles near the binary64
  limit. For dimension two, NaN and infinities return `InvalidArgument`;
  cancellation and step-budget checks retain priority.
- `integrate_multiple_checked` checks cancellation before work and before
  returning, and charges each integration step to the shared context, including
  constant definite-integral shortcuts.

## Expression values

- Binary64 expression text uses locale-independent `approx(<signed decimal>)`
  with round-trip precision, including `approx(-0)`. Ordinary expression
  decimals remain exact. Approximate literals reject nonfinite values,
  expressions, fractions, and unrepresentable nonzero underflow. Assumption and
  interval readers accept their legacy bare-decimal approximate syntax but
  serialize canonically. This is not a general AST serialization guarantee.
- `serialize_expr` and `parse_serialized_expr` exchange a versioned semantic
  encoding beginning with `LMCAS_EXPR/1\n`. Fields use decimal byte lengths
  (`length:bytes,`), so symbol names can contain arbitrary bytes. The encoding
  preserves exact integers and rationals, binary64 bit patterns (including
  negative zero), operators, binders, sets, physical quantities, and matrix
  values; a parsed expression reserializes to the same canonical value encoding.
  Mathematical constants such as `pi()` have distinct node identity from a
  same-named variable created with `SymbolicExpr::variable("pi")`. The ordinary
  expression parser still interprets `pi` as the mathematical constant, while
  the semantic decoder can reconstruct either kind. Unknown versions and
  malformed fields return `ParseError`; context limits and cancellation apply
  to both operations. Lamina exposes the same operations through
  `std.cas.serialize_expr` and `std.cas.parse_serialized_expr`.
- `evaluate_numeric` retains binary64 approximations. `ApproxReal::absolute_error`
  bounds the true real value's distance from `value`; `+infinity` means no
  finite bound has been certified. Binary64 literals and finite binary64
  bindings have zero error. BigInt/Rational conversion, arithmetic, RootOf,
  and function evaluation currently report `+infinity` even when the computed
  double happens to be exact; no tolerance can be inferred from its magnitude.
  A literal Infinity has its own `NumericStatus`; a finite mathematical
  operation overflowing binary64 reports `NumericFailure`. Cancellation and
  resource exhaustion remain `Cancelled` and `ResourceLimit`.
- `ExprSet::expression()` and `elements()` expose const-pointee expressions.
  Construction isolates mutable input wrappers, and element order comes from
  the canonical finite-set representation. Copy a wrapper explicitly when a
  mutable expression is needed; mutating a Lamina array returned by
  `set_to_array` does not mutate its source set.

## Proof obligations and lexical scope

Each checked result certifies the proposition listed below.

| Result or obligation | What it establishes | What it does not establish |
| --- | --- | --- |
| `ProvedZeroResidual` | Equality on the expressions' original common domain | Definedness at a particular parameter value |
| `ProvedNonIdentityResidual` | A residual is not identically zero | Nonzero value under the current assumptions |
| Pointwise comparison | Equality or inequality of defined values in the requested domain, using current assumptions | A result when parameter-dependent obligations remain unknown |
| Validated candidate | The candidate satisfies the original equation and its conditions | Completeness of the returned solution set |
| Complete solution set | All solutions in the supported domain, retaining necessary conditions and multiplicities | Permission to discard conditions when projecting to a simpler result type |

`ProvedNonIdentityResidual` is the current name for `ProvedNonzeroResidual`; callers must update the enumerator. For example, the polynomial `a` is not identically zero,
but can equal zero. The two-sided limit of `piecewise(a, x < 0, 0)` at zero
is therefore `Inconclusive` when only the realness of `a` is known. A proved
nonzero value of `a` permits `LimitDoesNotExist`; a proved zero permits the
finite zero limit. Continuity classification uses the same distinction.
Definedness, cancellation, and resource obligations are not bypassed by an
identity or a structurally identical pair of inputs.

Assumption-aware normalization respects lexical binding. Summations, products,
integrals, transforms, quantifiers, set builders, and limits hide outer facts
about their bound name only within their scoped body. Their bounds, domains,
transform targets, and limit points remain in the enclosing scope. Facts about
unrelated free symbols remain available, including facts derived from relations.
For example, an outer `i > 0` does not change `sum(abs(i), i, -1, 1)` from `2`
to `0`. Normalization does not infer additional body assumptions from a binder's
range; substitutions remain capture-avoiding.
