function(add_lamina_test name source)
    add_test(NAME ${name} COMMAND lamina "${CMAKE_CURRENT_SOURCE_DIR}/language/${source}")
    set_tests_properties(${name} PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endfunction()

function(add_lamina_error_test name source diagnostic)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND}
            "-DLAMINA=$<TARGET_FILE:lamina>"
            "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/language/${source}"
            "-DDIAGNOSTIC=${diagnostic}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_error.cmake"
    )
    set_tests_properties(${name} PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endfunction()

function(add_lamina_output_test name source expected)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND}
            "-DLAMINA=$<TARGET_FILE:lamina>"
            "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/language/${source}"
            "-DEXPECTED=${expected}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_output.cmake"
    )
    set_tests_properties(${name} PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endfunction()

add_executable(test_compilation_pipeline test_compilation_pipeline.cpp)
target_link_libraries(test_compilation_pipeline PRIVATE lamina_shared)
set_target_properties(test_compilation_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
add_test(NAME compilation_pipeline COMMAND test_compilation_pipeline)

add_executable(test_runtime_values
    test_runtime_values.cpp
    ../lmx.cpp
    ../utils/utils.cpp
    $<TARGET_OBJECTS:lamina_runtime>
    $<TARGET_OBJECTS:laminac>
)
target_link_libraries(test_runtime_values PRIVATE dyncall_s dynload_s lmcas)
set_target_properties(test_runtime_values PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
add_test(NAME runtime_values COMMAND test_runtime_values)

add_lamina_test(language_adt adt.lm)
add_lamina_test(language_adt_module adt_module.lm)
add_lamina_test(language_literal_payload literal_payload.lm)
add_lamina_output_test(adt_runtime adt_runtime.lm "7 0 9 1 1 1 3")

add_lamina_error_test(adt_uninferred adt_uninferred.lm "cannot infer ADT type arguments")
add_lamina_error_test(adt_uninferred_expr adt_uninferred_expr.lm "cannot infer ADT type arguments")
add_lamina_error_test(adt_missing_pattern adt_missing_pattern.lm "MissingWildcard")
add_lamina_error_test(adt_wrong_arity adt_wrong_arity.lm "constructor pattern field count mismatch")
add_lamina_error_test(adt_wrong_type_arity adt_wrong_type_arity.lm "expects 1 argument")
add_lamina_error_test(adt_unreachable adt_unreachable.lm "UnreachablePattern")
add_lamina_error_test(adt_ordering adt_ordering.lm "binary operation cannot applied to this type")
add_lamina_error_test(adt_nullable_wrong_value adt_nullable_wrong_value.lm "constructor field type mismatch")
add_lamina_error_test(adt_null_nonnullable adt_null_nonnullable.lm "constructor field type mismatch")
add_lamina_error_test(adt_nullable_option adt_nullable_option.lm "type mismatch")
add_lamina_error_test(adt_nominal_mismatch adt_nominal_mismatch.lm "type mismatch")
add_lamina_error_test(adt_duplicate_type_parameter adt_duplicate_type_parameter.lm "duplicate type parameter")
add_lamina_error_test(adt_duplicate_constructor adt_duplicate_constructor.lm "duplicate constructor")
add_lamina_error_test(adt_unknown_field_type adt_unknown_field_type.lm "unknown field type")
add_lamina_error_test(expr_bool_promotion expr_bool_promotion.lm "type mismatch")
add_lamina_error_test(set_mixed_elements set_mixed_elements.lm "SetElementTypeMismatch")
add_lamina_output_test(expr_direct_lowering expr_direct_lowering.lm "x + 3|x|f(x, 2)|g(2)|1|[0, x]|(0, 1)|4*(0 + 1*I)|3|2")
add_lamina_error_test(expr_condition expr_condition.lm "must be bool type")
add_lamina_error_test(expr_bool_logic expr_bool_logic.lm "binary operation cannot applied")
add_lamina_error_test(implicit_multiplication_spaced implicit_multiplication_spaced.lm "ComplexLiteralSpacing")
add_lamina_error_test(unexpected_uninterpreted_function unexpected_uninterpreted_function.lm "undefined var `f`")
add_lamina_output_test(lowercase_i_symbol lowercase_i_symbol.lm "i + 1")
add_lamina_output_test(lowercase_i_variable lowercase_i_variable.lm "2")
add_lamina_output_test(numeric_result numeric_result.lm "3.0|ok|err|err")
add_lamina_output_test(complex_runtime complex_runtime.lm "4.0,2.0|11.0,-2.0|2.0,6.0|3.0,-4.0|5.0|err|1|1")
add_lamina_output_test(vector_matrix_stats vector_matrix_stats.lm "3|2.0|1.0|3.7|2x3|3x2|34.9|1:2:2|2:5.0")
add_lamina_output_test(random_context random_context.lm "1|1|err")
add_lamina_output_test(quantity_units quantity_units.lm "1000.0|1|1.5|0.5 km*s^-1|500.0|1.0|1|1.0|4.0 s^2|err|invalid")
add_lamina_output_test(containers_linalg_extended containers_linalg_extended.lm "1.0,3.0,1|14.0|1|2x3,6.0,1|2x2,58.0,154.0|1|1|1|2,1,0,3,1")
add_lamina_output_test(stats_extended stats_extended.lm "2.500|1.667|1.250|1.291|1.118|2.500|1.750|3.333|1.000|4,3|1,1,1,1,1,1")
add_lamina_output_test(random_extended random_extended.lm "1,1,1:4,1|1|1,1,1,1,1")
add_lamina_output_test(quantity_extended quantity_extended.lm "1.500|1,1|6.0 N|3.0|1,1,1,1,1,1|2.0,1,1|7200.0")
add_lamina_output_test(stdlib_remaining stdlib_remaining.lm "x|1,1|5.0|8.0|5.0|3.0|CasError(InvalidArgument in lsr.differentiate: differentiate variable cannot be empty)")
add_lamina_output_test(computer_algebra_extended computer_algebra_extended.lm "x^2 + (2)*x + 1|(1/2)*(x^2)|2|(1/6)*(x^3) + (1/2)*(x^2) + x + 1|1.0|1.0,2.0|1|-1,1|-1,1")
add_lamina_output_test(computer_algebra_symbol_arguments computer_algebra_symbol_arguments.lm "2*x|(1/2)*(x^2)|0|ok|system")
add_lamina_output_test(computer_algebra_algebra_remaining computer_algebra_algebra_remaining.lm "1,1,1,1,1,1,1,1,1,1,1,1")
add_lamina_output_test(computer_algebra_symbolic_matrix_remaining computer_algebra_symbolic_matrix_remaining.lm "1,1,1,1,1,0,1,1,1,1,1,1")
add_lamina_output_test(computer_algebra_analysis_remaining computer_algebra_analysis_remaining.lm "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1")
add_lamina_output_test(computer_algebra_ode_geometry_remaining computer_algebra_ode_geometry_remaining.lm "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1")
add_lamina_output_test(lmmc_lsr_names lmmc_lsr_names.lm "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1")
add_lamina_output_test(lmmc_random_default lmmc_random_default.lm "1")
add_lamina_output_test(lmmc_sparse_remaining lmmc_sparse_remaining.lm "1,1,1,1")
add_lamina_output_test(math_scalar_distributions math_scalar_distributions.lm "3.142|0.0|3.0|3.0|0.500|0.375|0.368|err")
add_lamina_output_test(mathematics_error_contract mathematics_error_contract.lm "domain:DomainError:evaluate_numeric:zero cannot be raised to a non-positive power|matrix:DimensionMismatch:matrix_mul:dimension mismatch|parse:ParseError:lsr.parse_expr:expected expression at byte 1|callback:CallbackFailure:integrate.adaptive:IndexOutOfRangeError: array len is 1 but you index 4|")
add_lamina_output_test(math_linalg_advanced math_linalg_advanced.lm "6.0|9.0|9.0|10.0|10.0|2|0.6,0.4|30.0|1.0|3.0")
add_lamina_output_test(computer_algebra_sets_match_bindings computer_algebra_sets_match_bindings.lm "3|5|13.0|1|1|1,x,y|1|1|1|1,1,x|{1, 2}|[1, 2)")
add_lamina_output_test(computer_algebra_advanced_modules computer_algebra_advanced_modules.lm "0.3333|0.3333|(2)*x,(2)*y|4|(2)*x + (2)*y|(1)*(s^-1)|1|6|-1*(2)*(((2)*y)^-1)*x")
add_lamina_output_test(mathematics_solving_sets mathematics_solving_sets.lm "solve-x2m1:-1|1|\nroots-x2m1:-1|1|\nsolve-q:(1/2)*((-2)*(7^1/2) - 2)|(1/2)*((2)*(7^1/2) - 2)|\nroots-q:(1/2)*((-2)*(7^1/2) - 2)|(1/2)*((2)*(7^1/2) - 2)|\nsolve-neq:!solve requires an equality relation\nsolve-deg5:rootof(x^5 + (-1)*x - 1, x, 0)|rootof(x^5 + (-1)*x - 1, x, 1)|rootof(x^5 + (-1)*x - 1, x, 2)|rootof(x^5 + (-1)*x - 1, x, 3)|rootof(x^5 + (-1)*x - 1, x, 4)|\nsolve-complex:(0 + -1*I)|(0 + 1*I)|\nroots-repeated:1|\n")
add_lamina_output_test(math_constants_decompositions math_constants_decompositions.lm "9.80665|m*s^-2|EARTH_GRAVITY,ROOM_TEMPERATURE|0.0,1.0|2,2.0|2,3.0")
add_lamina_output_test(mathematics_api_unified mathematics_api_unified.lm "square_root(2) = 1.414214\ndeterminant = -14.0\nsparse cg converged=1 x[0] ~= 0.090909\nintegral x^2 = 0.333333 (ok)\nd/dx = 3*(x^2) + 2\nMathError{code=IndexOutOfBounds, operation=math.sqrt, message=out of range}\nsolve set {(1/2)*((-2)*(7^1/2) - 2), (1/2)*((2)*(7^1/2) - 2)}\nroots set {(1/2)*((-2)*(7^1/2) - 2), (1/2)*((2)*(7^1/2) - 2)}\nsets equal = 1")
add_lamina_output_test(core_language core_language.lm "120,25,36|1,8,3|8,6|negative,zero,positive|ok|1,1,2,32")
add_lamina_error_test(unresolved_math_modules unresolved_math_modules.lm "no module is called")
add_lamina_error_test(reserved_imaginary_variable reserved_imaginary_variable.lm "ImaginaryUnitReserved")
add_lamina_error_test(reserved_imaginary_symbol reserved_imaginary_symbol.lm "ImaginaryUnitReserved")
add_lamina_error_test(immutable_assignment immutable_assignment.lm "cannot assign to immutable var")
add_lamina_error_test(array_index_type array_index_type.lm "array index must be int")
add_lamina_error_test(if_branch_type_mismatch if_branch_type_mismatch.lm "if express then and else cannot type mismatch")
add_lamina_error_test(unsupported_double_star unsupported_double_star.lm "not supported")
add_lamina_output_test(dimension_units dimension_units.lm "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1")
add_lamina_output_test(compact_less_than compact_less_than.lm "1,0,0")
add_lamina_output_test(dimension_expr dimension_expr.lm "2<km>|2*(1000)<m>|2*(1000)|2|3*(100)<score>|5|2*(1000) + 1<m>|1 + 3<m>|0")
add_lamina_error_test(dimension_mismatch_add dimension_mismatch_add.lm "DimensionMismatch")
add_lamina_error_test(dimension_implicit_scale dimension_implicit_scale.lm "DimensionMismatch")
add_lamina_error_test(dimension_bad_conversion dimension_bad_conversion.lm "DimensionMismatch")
add_lamina_error_test(dimension_nonconstant_power dimension_nonconstant_power.lm "DimensionExponentMustBeConstantInteger")
add_lamina_error_test(dimension_strip_invalid dimension_strip_invalid.lm "UnitStripTypeMismatch")
add_lamina_error_test(dimension_legacy_strip dimension_legacy_strip.lm "UnitStripLegacySyntax")
add_lamina_output_test(dimension_module dimension_module.lm "1,1,1")
add_lamina_error_test(dimension_unit_redefined dimension_unit_redefined.lm "UnitRedefined")
add_lamina_error_test(dimension_unknown_unit dimension_unknown_unit.lm "UnitInvalid")
add_lamina_error_test(dimension_invalid_scale dimension_invalid_scale.lm "derived unit scale must be a positive")
add_lamina_error_test(dimension_function_arg_scale dimension_function_arg_scale.lm "type mismatch arg")
add_lamina_error_test(dimension_set_mismatch dimension_set_mismatch.lm "SetElementTypeMismatch")
add_lamina_error_test(dimension_strip_overflow dimension_strip_overflow.lm "UnitStripOverflow")
add_lamina_output_test(dimension_builtin_declaration dimension_builtin_declaration.lm "1,1")
add_lamina_output_test(dimension_expr_function_only dimension_expr_function_only.lm "2<m>|3<m>")
add_lamina_output_test(tuple_interval tuple_interval.lm "7,ok|1|2|1,1,0,1,1,1|1|1|1")
add_lamina_error_test(tuple_assignment tuple_assignment.lm "TupleAssignment")
add_lamina_error_test(tuple_index_zero tuple_index_zero.lm "TupleIndexOutOfBounds")
add_lamina_error_test(tuple_index_out_of_bounds tuple_index_out_of_bounds.lm "TupleIndexOutOfBounds")
add_lamina_error_test(tuple_type_arity tuple_type_arity.lm "TupleArityMismatch")
add_lamina_error_test(interval_arity interval_arity.lm "IntervalArityMismatch")
add_lamina_error_test(interval_bound_type interval_bound_type.lm "IntervalBoundTypeMismatch")
add_lamina_error_test(interval_bound_order interval_bound_order.lm "IntervalBoundNotOrdered")
add_lamina_error_test(interval_reversed interval_reversed.lm "IntervalBoundsReversed")
add_lamina_output_test(callback_quadrature callback_quadrature.lm "0.333333|ok|err|alive")
add_lamina_error_test(callback_wrong_signature callback_wrong_signature.lm "type mismatch arg")
add_lamina_output_test(lmmc_special_fft lmmc_special_fft.lm "24.000|0.000|1.000,0.000|1.000,2.000,3.000,4.000")
add_lamina_output_test(lmmc_linalg_decomp lmmc_linalg_decomp.lm "0.100,0.600|0.100,0.600|4.000,3.000|2.000,3.000|2x2,0.500")
add_lamina_output_test(lmmc_sparse_itersolve lmmc_sparse_itersolve.lm "0.091,0.636|0.091,0.636|0.091,0.636")
add_lamina_output_test(lmmc_tensor_interpolate lmmc_tensor_interpolate.lm "36|4x2:8|8|6|2.200")
add_lamina_output_test(lmmc_callback_algorithms lmmc_callback_algorithms.lm "1.4141|1.4142|2.71828|3.0000")
add_lamina_output_test(lmmc_stats_random lmmc_stats_random.lm "120|120|20|1,1,1,1,1|1,1,1,1,1")
add_lamina_output_test(computer_algebra_assumptions_solve computer_algebra_assumptions_solve.lm "U|P|U|1|2,1|(-1, 1)")
add_lamina_output_test(computer_algebra_algebra_matrix computer_algebra_algebra_matrix.lm "1|1|1|1|1|1,1,1,1,1")
add_lamina_output_test(computer_algebra_analysis_ode computer_algebra_analysis_ode.lm "1|1|1|1|1|0|1|err")
add_lamina_output_test(computer_algebra_geometry_full computer_algebra_geometry_full.lm "(2)*x,(2)*y|32|1,1,1|1|1|CasError")

add_lamina_output_test(function_overload function_overload.lm "1,2")
add_lamina_error_test(function_overload_ambiguous function_overload_ambiguous.lm "ambiguous overloaded function")
add_lamina_output_test(never_raise_value never_raise_value.lm "42")
add_lamina_error_test(raise_text raise_text.lm "RuntimeError: boom")
add_lamina_error_test(raise_math_error raise_math_error.lm "RuntimeError: MathError")
add_lamina_error_test(raise_redefinition raise_redefinition.lm "cannot redefine builtin `raise`")
add_lamina_output_test(cas_high_level cas_high_level.lm "2*x|-1,1|1|U|LU")
add_lamina_output_test(cas_raw_checked cas_raw_checked.lm "solve requires an equality relation")
add_lamina_error_test(cas_high_level_raise cas_high_level_raise.lm "solve requires an equality relation")
add_lamina_output_test(set_operations set_operations.lm "1,1,1,1|1,1,1|1,1|1")
add_lamina_output_test(set_inference set_inference.lm "1,1,1|1,1|1")
add_lamina_output_test(set_cas_results set_cas_results.lm "1,1,1|1")
add_lamina_error_test(set_operand_type_mismatch set_operand_type_mismatch.lm "SetOperandTypeMismatch")
add_lamina_error_test(set_element_not_hashable set_element_not_hashable.lm "SetElementNotHashable")
add_lamina_error_test(set_empty_uninferred set_empty_uninferred.lm "cannot infer ADT type arguments")
add_lamina_error_test(set_missing_closing_brace set_missing_closing_brace.lm "expected `}`")
