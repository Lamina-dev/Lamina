#include "bridge/conversions.hpp"
#include "bridge/mathematics_error.hpp"
#include "runtime/object/StringObj.hpp"
#include "runtime/object/code_module.hpp"
#include "runtime/object/random.hpp"
#include "runtime/object/tensor.hpp"

#include <new>
#include <string_view>

namespace lmx::bridge {

extern "C" AdtObj* lmx_cas_system_by_symbols(
    ArrayObj* equations, ArrayObj* variables) noexcept;
extern "C" AdtObj* lmx_cas_polynomial_system_by_symbols(
    ArrayObj* equations, ArrayObj* variables) noexcept;
extern "C" AdtObj* lmx_cas_parametric_piecewise_by_symbols(
    ArrayObj* equations, ArrayObj* unknowns, ArrayObj* parameters) noexcept;
extern "C" AdtObj* lmx_cas_groebner_basis_by_symbols(
    ArrayObj* polynomials, ArrayObj* variables) noexcept;
extern "C" AdtObj* lmx_cas_reduced_groebner_basis_by_symbols(
    ArrayObj* polynomials, ArrayObj* variables) noexcept;
extern "C" AdtObj* lmx_cas_ideal_membership_by_symbols(
    ExprObj* polynomial, ArrayObj* basis, ArrayObj* variables) noexcept;
extern "C" AdtObj* lmx_cas_elimination_ideal_by_symbols(
    ArrayObj* basis, ArrayObj* variables, LmInt count) noexcept;
extern "C" AdtObj* lmx_cas_inequality(
    ExprObj* expression, const char* relation, const char* variable) noexcept;
extern "C" AdtObj* lmx_complex_numbers_exponential(
    ComplexObj* value) noexcept;
extern "C" AdtObj* lmx_fast_fourier_transform_forward(
    ArrayObj* values) noexcept;
extern "C" AdtObj* lmx_interpolation_linear(
    VectorObj* x, VectorObj* y, VectorObj* query) noexcept;
extern "C" AdtObj* lmx_random_gamma(
    runtime::RandomObj* value, double shape, double scale) noexcept;
extern "C" AdtObj* lmx_tensor_add(
    runtime::TensorObj* lhs, runtime::TensorObj* rhs) noexcept;
extern "C" AdtObj* lmx_stats_covariance_matrix_sample(
    MatrixObj* value) noexcept;
extern "C" AdtObj* lmx_ordinary_differential_equations_euler(
    const runtime::FuncObj* rhs, VectorObj* initial, double start, double end,
    double step, double abs_tol, double rel_tol, LmInt max_steps) noexcept;

namespace {

runtime::AdtObj* allocation_failure_probe() noexcept try {
    throw std::bad_alloc{};
} catch (...) {
    return c_abi_current_exception(__func__);
}

runtime::AdtObj* checked_failure_probe() noexcept try {
    throw LMCAS::CasError{
        LMCAS::CasErrc::DomainError,
        "checked failure",
        "checked.operation"};
} catch (...) {
    return c_abi_current_exception(__func__);
}

bool has_error(const runtime::AdtObj* result, const char* code,
               const char* operation, const char* message = nullptr) {
    if (!result || result->type_name() != "Result" ||
        result->constructor() != "Err") {
        return false;
    }
    const auto* error_field = result->field(0);
    const auto* error =
        error_field && error_field->kind == runtime::ValueKind::Obj &&
                error_field->obj &&
                error_field->obj->get_kind() == runtime::ObjectKind::Adt
            ? static_cast<const runtime::AdtObj*>(error_field->obj)
            : nullptr;
    if (!error || error->type_name() != "MathError") return false;
    const auto* code_field = error->field(0);
    const auto* code_value =
        code_field && code_field->kind == runtime::ValueKind::Obj &&
                code_field->obj &&
                code_field->obj->get_kind() == runtime::ObjectKind::Adt
            ? static_cast<const runtime::AdtObj*>(code_field->obj)
            : nullptr;
    const auto* operation_field = error->field(1);
    const auto* operation_value =
        operation_field && operation_field->kind == runtime::ValueKind::Obj &&
                operation_field->obj &&
                operation_field->obj->get_kind() == runtime::ObjectKind::String
            ? static_cast<const runtime::StringObj*>(operation_field->obj)
            : nullptr;
    const auto* message_field = error->field(2);
    const auto* message_value =
        message_field && message_field->kind == runtime::ValueKind::Obj &&
                message_field->obj &&
                message_field->obj->get_kind() == runtime::ObjectKind::String
            ? static_cast<const runtime::StringObj*>(message_field->obj)
            : nullptr;
    return code_value && code_value->constructor() == code &&
           operation_value &&
           std::string_view(operation_value->c_str()) == operation &&
           (!message ||
            (message_value &&
             std::string_view(message_value->c_str()) == message));
}

} // namespace

extern "C" int lmx_test_c_abi_exception_boundaries() noexcept {
    try {
        auto allocation = adopt_object(allocation_failure_probe());
        auto checked = adopt_object(checked_failure_probe());

        auto null_system =
            adopt_object(lmx_cas_system_by_symbols(nullptr, nullptr));

        auto integer_variables = make_owned_object<ArrayObj>();
        integer_variables->append(Value{static_cast<LmInt>(1)});
        auto non_expression_variable = adopt_object(
            lmx_cas_system_by_symbols(nullptr, integer_variables.get()));

        auto numeric_variables = make_owned_object<ArrayObj>();
        numeric_variables->append(take_object_value(
            make_owned_object<ExprObj>(LMCAS::SymbolicExpr::number(1)),
            ValueKind::Expr));
        auto non_symbol_variable = adopt_object(
            lmx_cas_system_by_symbols(nullptr, numeric_variables.get()));

        auto variable =
            make_owned_object<ExprObj>(LMCAS::SymbolicExpr::variable("x"));
        auto unknown_relation =
            adopt_object(lmx_cas_inequality(variable.get(), "=", "x"));
        auto null_relation =
            adopt_object(lmx_cas_inequality(variable.get(), nullptr, "x"));

        auto null_complex_exponential =
            adopt_object(lmx_complex_numbers_exponential(nullptr));
        auto null_fft =
            adopt_object(lmx_fast_fourier_transform_forward(nullptr));
        auto null_interpolation =
            adopt_object(lmx_interpolation_linear(nullptr, nullptr, nullptr));
        auto null_random_gamma =
            adopt_object(lmx_random_gamma(nullptr, 1.0, 1.0));
        auto null_tensor_add =
            adopt_object(lmx_tensor_add(nullptr, nullptr));
        auto null_matrix_stat =
            adopt_object(lmx_stats_covariance_matrix_sample(nullptr));
        auto null_ode = adopt_object(
            lmx_ordinary_differential_equations_euler(
                nullptr, nullptr, 0.0, 1.0, 0.1, 1e-9, 1e-9, 10));

        auto valid_symbols = make_owned_object<ArrayObj>();
        valid_symbols->append(take_object_value(
            make_owned_object<ExprObj>(LMCAS::SymbolicExpr::variable("x")),
            ValueKind::Expr));
        auto invalid_values = make_owned_object<ArrayObj>();
        invalid_values->append(Value{static_cast<LmInt>(1)});

        auto groebner_symbol_first = adopt_object(
            lmx_cas_groebner_basis_by_symbols(
                invalid_values.get(), invalid_values.get()));
        auto groebner_names_after_symbols = adopt_object(
            lmx_cas_groebner_basis_by_symbols(
                invalid_values.get(), valid_symbols.get()));
        auto reduced_symbol_first = adopt_object(
            lmx_cas_reduced_groebner_basis_by_symbols(
                invalid_values.get(), invalid_values.get()));
        auto reduced_names_after_symbols = adopt_object(
            lmx_cas_reduced_groebner_basis_by_symbols(
                invalid_values.get(), valid_symbols.get()));
        auto membership_symbol_first = adopt_object(
            lmx_cas_ideal_membership_by_symbols(
                nullptr, nullptr, invalid_values.get()));
        auto membership_names_after_symbols = adopt_object(
            lmx_cas_ideal_membership_by_symbols(
                nullptr, nullptr, valid_symbols.get()));
        auto elimination_symbol_first = adopt_object(
            lmx_cas_elimination_ideal_by_symbols(
                invalid_values.get(), invalid_values.get(), -1));
        auto elimination_names_after_symbols = adopt_object(
            lmx_cas_elimination_ideal_by_symbols(
                invalid_values.get(), valid_symbols.get(), -1));
        auto polynomial_system_symbol_first = adopt_object(
            lmx_cas_polynomial_system_by_symbols(
                invalid_values.get(), invalid_values.get()));
        auto polynomial_system_names_after_symbols = adopt_object(
            lmx_cas_polynomial_system_by_symbols(
                invalid_values.get(), valid_symbols.get()));
        auto piecewise_unknown_first = adopt_object(
            lmx_cas_parametric_piecewise_by_symbols(
                invalid_values.get(), invalid_values.get(), invalid_values.get()));
        auto piecewise_parameter_second = adopt_object(
            lmx_cas_parametric_piecewise_by_symbols(
                invalid_values.get(), valid_symbols.get(), invalid_values.get()));
        auto piecewise_names_after_symbols = adopt_object(
            lmx_cas_parametric_piecewise_by_symbols(
                invalid_values.get(), valid_symbols.get(), valid_symbols.get()));

        return has_error(allocation.get(), "ResourceLimit",
                         "allocation_failure_probe") &&
                       has_error(checked.get(), "DomainError",
                                 "checked.operation", "checked failure") &&
                       has_error(null_system.get(), "InvalidArgument",
                                 "lmx_cas_system_by_symbols") &&
                       has_error(non_expression_variable.get(),
                                 "InvalidArgument",
                                 "lmx_cas_system_by_symbols") &&
                       has_error(non_symbol_variable.get(), "InvalidArgument",
                                 "lmx_cas_system_by_symbols") &&
                       has_error(unknown_relation.get(), "InvalidArgument",
                                 "lmx_cas_inequality") &&
                       has_error(null_relation.get(), "InvalidArgument",
                                 "lmx_cas_inequality") &&
                       has_error(null_complex_exponential.get(),
                                 "InvalidArgument",
                                 "lmx_complex_numbers_exponential") &&
                       has_error(null_fft.get(), "EmptyInput",
                                 "lmx_fast_fourier_transform_forward") &&
                       has_error(null_interpolation.get(), "InvalidArgument",
                                 "lmx_interpolation_linear") &&
                       has_error(null_random_gamma.get(), "InvalidArgument",
                                 "lmx_random_gamma") &&
                       has_error(null_tensor_add.get(), "InvalidArgument",
                                 "lmx_tensor_add") &&
                       has_error(null_matrix_stat.get(), "InvalidArgument",
                                 "lmx_stats_covariance_matrix_sample") &&
                       has_error(
                           null_ode.get(), "InvalidArgument",
                           "lmx_ordinary_differential_equations_euler") &&
                       has_error(groebner_symbol_first.get(), "InvalidArgument",
                                 "lmx_cas_groebner_basis_by_symbols") &&
                       has_error(groebner_names_after_symbols.get(),
                                 "InvalidArgument",
                                 "lmx_cas_groebner_basis_by_names") &&
                       has_error(reduced_symbol_first.get(), "InvalidArgument",
                                 "lmx_cas_reduced_groebner_basis_by_symbols") &&
                       has_error(reduced_names_after_symbols.get(),
                                 "InvalidArgument",
                                 "lmx_cas_reduced_groebner_basis_by_names") &&
                       has_error(membership_symbol_first.get(), "InvalidArgument",
                                 "lmx_cas_ideal_membership_by_symbols") &&
                       has_error(membership_names_after_symbols.get(),
                                 "InvalidArgument",
                                 "lmx_cas_ideal_membership_by_names") &&
                       has_error(elimination_symbol_first.get(), "InvalidArgument",
                                 "lmx_cas_elimination_ideal_by_symbols") &&
                       has_error(elimination_names_after_symbols.get(),
                                 "InvalidArgument",
                                 "lmx_cas_elimination_ideal_by_names") &&
                       has_error(polynomial_system_symbol_first.get(),
                                 "InvalidArgument",
                                 "lmx_cas_polynomial_system_by_symbols") &&
                       has_error(polynomial_system_names_after_symbols.get(),
                                 "InvalidArgument",
                                 "lmx_cas_polynomial_system_by_names") &&
                       has_error(piecewise_unknown_first.get(), "InvalidArgument",
                                 "lmx_cas_parametric_piecewise_by_symbols") &&
                       has_error(piecewise_parameter_second.get(),
                                 "InvalidArgument",
                                 "lmx_cas_parametric_piecewise_by_symbols") &&
                       has_error(piecewise_names_after_symbols.get(),
                                 "InvalidArgument",
                                 "lmx_cas_parametric_piecewise_by_names")
            ? 0
            : 1;
    } catch (...) {
        return 2;
    }
}

} // namespace lmx::bridge
