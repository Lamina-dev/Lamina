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
add_lamina_output_test(expr_direct_lowering expr_direct_lowering.lm "x + 3|x|f(x, 2)|g(2)|{2, x}|[0, x]|4*(0 + 1*I)|3|2")
add_lamina_error_test(expr_condition expr_condition.lm "must be bool type")
add_lamina_error_test(expr_bool_logic expr_bool_logic.lm "binary operation cannot applied")
add_lamina_error_test(implicit_multiplication_spaced implicit_multiplication_spaced.lm "implicit multiplication is not supported")
add_lamina_error_test(unexpected_uninterpreted_function unexpected_uninterpreted_function.lm "undefined var `f`")
add_lamina_output_test(lowercase_i_symbol lowercase_i_symbol.lm "i + 1")
add_lamina_output_test(lowercase_i_variable lowercase_i_variable.lm "2")
add_lamina_output_test(numeric_result numeric_result.lm "3.0|ok|err|err")
add_lamina_output_test(complex_runtime complex_runtime.lm "4.0,2.0|11.0,-2.0|2.0,6.0|3.0,-4.0|5.0|err|1|1")
