function(add_lamina_test name source)
    add_test(NAME ${name} COMMAND lamina "${CMAKE_CURRENT_SOURCE_DIR}/language/${source}")
endfunction()

function(add_lamina_error_test name source diagnostic)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND}
            "-DLAMINA=$<TARGET_FILE:lamina>"
            "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/language/${source}"
            "-DDIAGNOSTIC=${diagnostic}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_error.cmake"
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
endfunction()

add_executable(test_compilation_pipeline test_compilation_pipeline.cpp)
target_link_libraries(test_compilation_pipeline PRIVATE laminaCore)
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
add_lamina_error_test(set_mixed_elements set_mixed_elements.lm "set elements must have one type")
add_lamina_output_test(expr_direct_lowering expr_direct_lowering.lm "x + 3|x|f(x, 2)|g(2)|{2, x}|[0, x]|(0, 1)|4*(0 + 1*I)|3|2")
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
add_lamina_output_test(core_language core_language.lm "120,25,36|1,8,3|8,6|negative,zero,positive|ok|1,1,2,32")
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
add_lamina_error_test(dimension_set_mismatch dimension_set_mismatch.lm "set elements must have one type")
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
