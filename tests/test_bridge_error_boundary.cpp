#include "bridge/conversions.hpp"
#include "allocation_probe.hpp"
#include "bridge/callback.hpp"
#include "bridge/linear_algebra_internal.hpp"
#include "bridge/mathematics_error.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/result.hpp"
#include "assumption_context.hpp"
#include "runtime/object/tuple.hpp"
#include "runtime/object/StringObj.hpp"
#include "runtime/object/expr_obj.hpp"
#include "runtime/object/code_module.hpp"
#include "runtime/object/random.hpp"
#include "runtime/object/tensor.hpp"
#include "runtime/opcode.hpp"
#include "lmmc/eigen.h"
#include "lmmc/init.h"
#include "lmmc/linear_algebra.h"
#include "lmmc/random.h"

#include <new>
#include <iostream>
#include <limits>
#include <string_view>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace lmx::bridge {

extern "C" AdtObj* lmx_math_log2(ExprObj*) noexcept;
extern "C" AdtObj* lmx_math_exp2(ExprObj*) noexcept;
extern "C" AdtObj* lmx_math_hypot(ExprObj*, ExprObj*) noexcept;
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
extern "C" AdtObj* lmx_linear_algebra_eigendecomposition(MatrixObj*) noexcept;
extern "C" AdtObj* lmx_linear_algebra_singular_value_decomposition(
    MatrixObj*) noexcept;
extern "C" AdtObj* lmx_random_shuffle(
    runtime::RandomObj*, VectorObj*) noexcept;
extern "C" AdtObj* lmx_computer_algebra_integrate_simpson_by_name(
    ExprObj*, const char*, ExprObj*, ExprObj*, LmInt) noexcept;
extern "C" AdtObj* lmx_computer_algebra_integrate_gaussian_by_name(
    ExprObj*, const char*, ExprObj*, ExprObj*, LmInt) noexcept;
extern "C" AdtObj* lmx_computer_algebra_integrate_adaptive_by_name(
    ExprObj*, const char*, ExprObj*, ExprObj*, double, LmInt) noexcept;
extern "C" AdtObj* lmx_computer_algebra_integrate_simpson_by_symbol(
    ExprObj*, ExprObj*, ExprObj*, ExprObj*, LmInt) noexcept;
extern "C" AdtObj* lmx_computer_algebra_integrate_gaussian_by_symbol(
    ExprObj*, ExprObj*, ExprObj*, ExprObj*, LmInt) noexcept;
extern "C" AdtObj* lmx_computer_algebra_integrate_adaptive_by_symbol(
    ExprObj*, ExprObj*, ExprObj*, ExprObj*, double, LmInt) noexcept;
extern "C" AdtObj* lmx_computer_algebra_integral_transforms_laplace_by_names(
    ExprObj*, const char*, const char*) noexcept;
extern "C" AdtObj* lmx_computer_algebra_integral_transforms_laplace_by_symbols(
    ExprObj*, ExprObj*, ExprObj*) noexcept;

namespace {
bool has_real(const runtime::AdtObj* result, double expected) {
    const auto* value = result && result->type_name() == "Result" &&
                                result->constructor() == "Ok"
                            ? result->field(0) : nullptr;
    return value && value->kind == runtime::ValueKind::Real &&
           value->real_val == expected;
}

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

bool dense_output_adoption_probe() {
    ensure_lmmc_runtime();
    lmmc_vec_t vector{};
    if (lmmc_vec_create(3, &vector) != LMMC_STATUS_OK) return false;
    vector.data[0] = 1.0;
    vector.data[1] = 2.0;
    vector.data[2] = 3.0;
    const auto* vector_data = vector.data;
    auto vector_result = adopt_object(
        lmmc_vector_output("adopt.vector", LMMC_STATUS_OK, vector));
    const auto* vector_field = vector_result->field(0);
    const auto* vector_value =
        vector_result->constructor() == "Ok" && vector_field &&
                vector_field->kind == ValueKind::Vector
            ? static_cast<const VectorObj*>(vector_field->obj) : nullptr;
    const bool vector_ok = vector_value &&
        vector_value->data().data() == vector_data &&
        vector_value->size() == 3 && vector_value->data()[0] == 1.0 &&
        vector_value->data()[1] == 2.0 && vector_value->data()[2] == 3.0 &&
        vector.size == 0 && vector.data == nullptr && vector.owns_data == 0;
    if (!vector_ok) std::cerr << "vector output must adopt its MC buffer\n";

    lmmc_mat_t matrix{};
    if (lmmc_mat_create(2, 2, &matrix) != LMMC_STATUS_OK) return false;
    for (std::size_t i = 0; i < 4; ++i) matrix.data[i] = double(i + 1);
    const auto* matrix_data = matrix.data;
    auto matrix_result = adopt_object(
        lmmc_matrix_output("adopt.matrix", LMMC_STATUS_OK, matrix));
    const auto* matrix_field = matrix_result->field(0);
    const auto* matrix_value =
        matrix_result->constructor() == "Ok" && matrix_field &&
                matrix_field->kind == ValueKind::Matrix
            ? static_cast<const MatrixObj*>(matrix_field->obj) : nullptr;
    const bool matrix_ok = matrix_value &&
        matrix_value->data().data() == matrix_data &&
        matrix_value->rows() == 2 && matrix_value->cols() == 2 &&
        matrix_value->data()[0] == 1.0 && matrix_value->data()[1] == 2.0 &&
        matrix_value->data()[2] == 3.0 && matrix_value->data()[3] == 4.0 &&
        matrix.rows == 0 && matrix.cols == 0 && matrix.stride == 0 &&
        matrix.data == nullptr && matrix.owns_data == 0;
    if (!matrix_ok) std::cerr << "matrix output must adopt its MC buffer\n";
    return vector_ok && matrix_ok;
}

bool dense_output_preconditions_probe() {
    double data[] = {1.0, 2.0, 3.0, 4.0};
    lmmc_vec_t borrowed_vector{4, data, 0};
    auto vector_result = adopt_object(lmmc_vector_output(
        "adopt.borrowed_vector", LMMC_STATUS_OK, borrowed_vector));
    lmmc_mat_t borrowed_matrix{2, 2, 2, data, 0};
    auto matrix_result = adopt_object(lmmc_matrix_output(
        "adopt.borrowed_matrix", LMMC_STATUS_OK, borrowed_matrix));
    lmmc_vec_t invalid_vector{3, nullptr, 1};
    auto invalid_vector_result = adopt_object(lmmc_vector_output(
        "adopt.invalid_vector", LMMC_STATUS_OK, invalid_vector));
    lmmc_mat_t strided_matrix{};
    if (lmmc_mat_create(2, 3, &strided_matrix) != LMMC_STATUS_OK) return false;
    strided_matrix.cols = 2;
    auto strided_result = adopt_object(lmmc_matrix_output(
        "adopt.strided", LMMC_STATUS_OK, strided_matrix));
    lmmc_mat_t overflow_matrix{
        std::numeric_limits<std::size_t>::max(), 2, 2, nullptr, 1};
    auto overflow_result = adopt_object(lmmc_matrix_output(
        "adopt.overflow", LMMC_STATUS_OK, overflow_matrix));
    lmmc_vec_t noncanonical_vector{};
    if (lmmc_vec_create(2, &noncanonical_vector) != LMMC_STATUS_OK) return false;
    noncanonical_vector.owns_data = 2;
    auto noncanonical_vector_result = adopt_object(lmmc_vector_output(
        "adopt.noncanonical_vector", LMMC_STATUS_OK, noncanonical_vector));
    lmmc_mat_t noncanonical_matrix{};
    if (lmmc_mat_create(2, 2, &noncanonical_matrix) != LMMC_STATUS_OK) return false;
    noncanonical_matrix.owns_data = 2;
    auto noncanonical_matrix_result = adopt_object(lmmc_matrix_output(
        "adopt.noncanonical_matrix", LMMC_STATUS_OK, noncanonical_matrix));
    return has_error(vector_result.get(), "InvalidArgument",
                     "adopt.borrowed_vector") &&
           has_error(matrix_result.get(), "InvalidArgument",
                     "adopt.borrowed_matrix") &&
           has_error(invalid_vector_result.get(), "InvalidArgument",
                     "adopt.invalid_vector") &&
           has_error(strided_result.get(), "InvalidArgument", "adopt.strided") &&
           has_error(overflow_result.get(), "InvalidArgument", "adopt.overflow") &&
           has_error(noncanonical_vector_result.get(), "InvalidArgument",
                     "adopt.noncanonical_vector") &&
           has_error(noncanonical_matrix_result.get(), "InvalidArgument",
                     "adopt.noncanonical_matrix") &&
           data[0] == 1.0 && data[3] == 4.0;
}

const AdtObj* decomposition_value(const AdtObj* result, const char* type) {
    const auto* field = result && result->constructor() == "Ok"
                            ? result->field(0) : nullptr;
    const auto* value = field && field->obj && field->kind == ValueKind::Obj
                            ? static_cast<const AdtObj*>(field->obj) : nullptr;
    return value && value->type_name() == type ? value : nullptr;
}

const MatrixObj* decomposition_matrix(const AdtObj* value,
                                      const std::size_t index) {
    const auto* field = value ? value->field(index) : nullptr;
    return field && field->kind == ValueKind::Matrix && field->obj
               ? static_cast<const MatrixObj*>(field->obj) : nullptr;
}

bool eigen_equations(const AdtObj* result, const MatrixObj& input,
                     const std::span<const double> real,
                     const std::span<const double> imag) {
    const auto* value = decomposition_value(result, "EigenResult");
    const auto* real_values = decomposition_matrix(value, 0);
    const auto* imag_values = decomposition_matrix(value, 1);
    const auto* real_vectors = decomposition_matrix(value, 2);
    const auto* imag_vectors = decomposition_matrix(value, 3);
    const auto n = input.rows();
    if (!real_values || !imag_values || !real_vectors || !imag_vectors ||
        real_values->rows() != 1 || real_values->cols() != n ||
        imag_values->rows() != 1 || imag_values->cols() != n ||
        real_vectors->rows() != n || real_vectors->cols() != n ||
        imag_vectors->rows() != n || imag_vectors->cols() != n)
        return false;
    for (std::size_t column = 0; column < n; ++column) {
        const auto re = real_values->data()[column];
        const auto im = imag_values->data()[column];
        if (!std::isfinite(re) || !std::isfinite(im) ||
            std::abs(re - real[column]) > 1e-10 ||
            std::abs(im - imag[column]) > 1e-10) return false;
        double norm = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
            const auto vr = real_vectors->data()[row * n + column];
            const auto vi = imag_vectors->data()[row * n + column];
            if (!std::isfinite(vr) || !std::isfinite(vi)) return false;
            norm = std::hypot(norm, std::hypot(vr, vi));
            double ar = 0.0;
            double ai = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                ar += input.data()[row * n + k] *
                      real_vectors->data()[k * n + column];
                ai += input.data()[row * n + k] *
                      imag_vectors->data()[k * n + column];
            }
            if (!std::isfinite(ar) || !std::isfinite(ai) ||
                std::abs(ar - (re * vr - im * vi)) > 1e-10 ||
                std::abs(ai - (re * vi + im * vr)) > 1e-10) return false;
        }
        if (!std::isfinite(norm) || norm < 0.5) return false;
    }
    return true;
}

bool eigen_order_probe() {
    MatrixObj cycle(3, 3, {3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0});
    MatrixObj complex(3, 3, {4.0, 0.0, 0.0, 0.0, 0.0, -1.0,
                             0.0, 1.0, 0.0});
    MatrixObj repeated(3, 3, {2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0});
    const double cycle_real[] = {1.0, 2.0, 3.0};
    const double complex_real[] = {0.0, 0.0, 4.0};
    const double repeated_real[] = {1.0, 2.0, 2.0};
    const double zero_imag[] = {0.0, 0.0, 0.0};
    const double complex_imag[] = {-1.0, 1.0, 0.0};
    auto cycle_result = adopt_object(lmx_linear_algebra_eigendecomposition(&cycle));
    auto complex_result =
        adopt_object(lmx_linear_algebra_eigendecomposition(&complex));
    auto repeated_result =
        adopt_object(lmx_linear_algebra_eigendecomposition(&repeated));
    if (!eigen_equations(cycle_result.get(), cycle, cycle_real, zero_imag) ||
        !eigen_equations(complex_result.get(), complex, complex_real, complex_imag) ||
        !eigen_equations(repeated_result.get(), repeated, repeated_real, zero_imag))
        return false;

    auto input = matrix_view(&repeated);
    lmmc_eigen_gen_full_result_t raw{};
    const auto cleanup = std::unique_ptr<
        lmmc_eigen_gen_full_result_t, decltype(&lmmc_eigen_gen_full_result_destroy)>(
            &raw, &lmmc_eigen_gen_full_result_destroy);
    if (lmmc_eigen_general_full(&input, &raw) != LMMC_STATUS_OK) return false;
    std::array<std::size_t, 3> order{0, 1, 2};
    std::stable_sort(order.begin(), order.end(), [&](auto lhs, auto rhs) {
        if (raw.real_parts.data[lhs] != raw.real_parts.data[rhs])
            return raw.real_parts.data[lhs] < raw.real_parts.data[rhs];
        return raw.imag_parts.data[lhs] < raw.imag_parts.data[rhs];
    });
    const auto* value = decomposition_value(repeated_result.get(), "EigenResult");
    const auto* re = decomposition_matrix(value, 2);
    const auto* im = decomposition_matrix(value, 3);
    for (std::size_t column = 0; column < 3; ++column)
        for (std::size_t row = 0; row < 3; ++row)
            if (re->data()[row * 3 + column] !=
                    raw.vectors_real.data[row * raw.vectors_real.stride + order[column]] ||
                im->data()[row * 3 + column] !=
                    raw.vectors_imag.data[row * raw.vectors_imag.stride + order[column]])
                return false;
    return true;
}

bool svd_equations(const AdtObj* result, const MatrixObj& input) {
    const auto* value = decomposition_value(result, "SvdResult");
    const auto* u = decomposition_matrix(value, 0);
    const auto* s = decomposition_matrix(value, 1);
    const auto* vt = decomposition_matrix(value, 2);
    if (!u || !s || !vt || !u->valid() || !s->valid() || !vt->valid() ||
        u->rows() != input.rows() || vt->cols() != input.cols() ||
        u->cols() != s->rows() || s->cols() != vt->rows())
        return false;
    for (std::size_t row = 0; row < input.rows(); ++row)
        for (std::size_t column = 0; column < input.cols(); ++column) {
            double sum = 0.0;
            for (std::size_t k = 0; k < s->rows(); ++k)
                sum += u->data()[row * u->cols() + k] *
                       s->data()[k * s->cols() + k] *
                       vt->data()[k * vt->cols() + column];
            if (!std::isfinite(sum) ||
                std::abs(sum - input.data()[row * input.cols() + column]) >
                    1e-10) return false;
        }
    return true;
}

bool decomposition_allocation_probe() {
    MatrixObj input(3, 3, {3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0});
    const double real[] = {1.0, 2.0, 3.0};
    const double imag[] = {0.0, 0.0, 0.0};
    for (const bool eigen : {true, false}) {
        const auto operation = eigen ? lmx_linear_algebra_eigendecomposition
                                     : lmx_linear_algebra_singular_value_decomposition;
        std::size_t count = 0;
        {
            test::AllocationProbeScope probe;
            auto result = adopt_object(operation(&input));
            const bool correct = eigen
                ? eigen_equations(result.get(), input, real, imag)
                : svd_equations(result.get(), input);
            count = probe.count();
            result.reset();
            if (!correct || count == 0 || probe.outstanding() != 0) return false;
        }
        for (std::size_t ordinal = 1; ordinal <= count; ++ordinal) {
#ifdef LMMC_DEBUG_LEAKS
            const auto baseline = lmmc_debug_leaks_get_count();
#endif
            {
                test::AllocationProbeScope probe(ordinal);
                auto result = adopt_object(operation(&input));
                const bool correct = result && result->constructor() == "Ok"
                    ? (eigen ? eigen_equations(result.get(), input, real, imag)
                             : svd_equations(result.get(), input))
                    : has_error(result.get(), "ResourceLimit",
                                eigen ? "lmx_linear_algebra_eigendecomposition"
                                      : "lmx_linear_algebra_singular_value_decomposition");
                result.reset();
                if (!probe.failed() || !correct || probe.outstanding() != 0)
                    return false;
            }
#ifdef LMMC_DEBUG_LEAKS
            if (lmmc_debug_leaks_get_count() != baseline) return false;
#endif
        }
    }
    return true;
}

bool vector_assign_failure_probe() {
    lmmc_vec_t raw{};
    if (lmmc_vec_create(2, &raw) != LMMC_STATUS_OK) return false;
    raw.data[0] = 1.0;
    raw.data[1] = 2.0;
    const auto* data = raw.data;
    VectorObj value(raw);
    const double replacement[] = {3.0, 4.0, 5.0};
    test::AllocationProbeScope probe(1);
    bool rejected = false;
    try {
        value.assign(replacement);
    } catch (const std::bad_alloc&) {
        rejected = true;
    }
    return rejected && probe.failed() && probe.outstanding() == 0 &&
           value.data().data() == data && value.size() == 2 &&
           value.data()[0] == 1.0 && value.data()[1] == 2.0;
}

thread_local const runtime::FuncObj* resize_callback = nullptr;
thread_local bool resize_callback_ok = false;

LmInt resized_callback_entry() noexcept {
    try {
        VectorCallbackContext context(resize_callback, 1, 3);
        double data[] = {4.0, 5.0, 6.0};
        for (const std::size_t size : {3U, 2U, 3U}) {
            context.result_size = size;
            lmmc_vec_t input{size, data, 0};
            lmmc_vec_t output{};
            const auto cleanup =
                std::unique_ptr<lmmc_vec_t, decltype(&lmmc_vec_destroy)>(
                    &output, &lmmc_vec_destroy);
            if (vector_callback_trampoline(&input, &output, &context) !=
                    LMMC_STATUS_OK ||
                context.failed() || context.argument.size() != size ||
                !std::ranges::equal(context.argument.data(),
                                    std::span(data, size)) ||
                output.size != size ||
                !std::ranges::equal(std::span(output.data, output.size),
                                    std::span(data, size)))
            {
                std::cerr << "resized callback failed at size " << size
                          << ": " << context.error << '\n';
                return 0;
            }
        }
        resize_callback_ok = true;
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "resized callback exception: " << error.what() << '\n';
        return 0;
    } catch (...) {
        std::cerr << "resized callback raised an unknown exception\n";
        return 0;
    }
}

bool resized_callback_probe() {
    const auto state = std::unique_ptr<LmState, decltype(&lmx_deleteState)>(
        lmx_newState(), &lmx_deleteState);
    auto* module = reinterpret_cast<runtime::CodeModuleObj*>(lmx_doString(
        state.get(), "static \"liblamina\"\nfunc identity(value vector) -> vector { value }\n",
        "callback_resize.lm"));
    if (!module || module->funcs.empty()) {
        std::cerr << "resize callback fixture has "
                  << (module ? module->funcs.size() : 0) << " functions\n";
        return false;
    }
    const auto index = module->native_funcs.size();
    if (index > std::numeric_limits<std::uint16_t>::max()) return false;
    module->native_funcs.emplace_back(
        reinterpret_cast<const void*>(&resized_callback_entry), 0, nullptr,
        ValueKind::Int, "resized_callback_entry");
    const std::uint8_t code[] = {
        static_cast<std::uint8_t>(runtime::Opcode::CCall),
        static_cast<std::uint8_t>(index),
        static_cast<std::uint8_t>(index >> 8U), 0,
        static_cast<std::uint8_t>(runtime::Opcode::Halt), 0, 0, 0};
    module->code = code;
    module->code_len = sizeof(code);
    runtime::LaminaVM vm(0, nullptr);
    resize_callback = &module->funcs.front();
    resize_callback_ok = false;
    const auto status = vm.run(module);
    resize_callback = nullptr;
    return status == 0 && resize_callback_ok;
}

bool copied_inputs_probe() {
    MatrixObj matrix(2, 2, {1.0, 2.0, 3.0, 4.0});
    auto copy = adopt_object(linear_algebra::copied_matrix(&matrix));
    if (copy->data().data() == matrix.data().data()) return false;
    copy->data()[0] = -1.0;
    if (matrix.data()[0] != 1.0) return false;
    const double expected[] = {1.0, 2.0, 3.0, 4.0};
    for (const bool owner : {true, false}) {
        lmmc_vec_t raw{};
        if (owner && lmmc_vec_create(4, &raw) != LMMC_STATUS_OK) return false;
        if (owner) std::copy(std::begin(expected), std::end(expected), raw.data);
        auto input = owner
            ? make_owned_object<VectorObj>(raw)
            : make_owned_object<VectorObj>(std::vector<double>{1.0, 2.0, 3.0, 4.0});
        lmmc_rng_t* raw_rng = nullptr;
        if (lmmc_rng_create(&raw_rng) != LMMC_STATUS_OK) return false;
        auto rng = make_owned_object<runtime::RandomObj>(raw_rng);
        if (lmmc_rng_seed(raw_rng, 12345) != LMMC_STATUS_OK) return false;
        auto result = adopt_object(lmx_random_shuffle(rng.get(), input.get()));
        const auto* field = result && result->constructor() == "Ok"
                                ? result->field(0) : nullptr;
        const auto* shuffled = field && field->kind == ValueKind::Vector &&
                                       field->obj
                                   ? static_cast<const VectorObj*>(field->obj)
                                   : nullptr;
        if (!shuffled || shuffled->size() != 4 ||
            shuffled->data().data() == input->data().data() ||
            !std::ranges::equal(input->data(), expected)) return false;
        std::array<double, 4> sorted{};
        std::ranges::copy(shuffled->data(), sorted.begin());
        std::ranges::sort(sorted);
        if (!std::ranges::equal(sorted, expected)) return false;
    }
    return true;
}

bool dense_output_allocation_probe() {
    for (const bool matrix : {true, false}) {
        std::size_t count = 0;
        for (std::size_t ordinal = 0; ordinal <= count; ++ordinal) {
#ifdef LMMC_DEBUG_LEAKS
            const auto baseline = lmmc_debug_leaks_get_count();
#endif
            lmmc_mat_t mat{};
            lmmc_vec_t vec{};
            if ((matrix ? lmmc_mat_create(2, 2, &mat)
                        : lmmc_vec_create(4, &vec)) != LMMC_STATUS_OK)
                return false;
            const auto invoke = [&]() noexcept -> AdtObj* {
                try {
                    return matrix
                        ? lmmc_matrix_output("adopt.failure", LMMC_STATUS_OK, mat)
                        : lmmc_vector_output("adopt.failure", LMMC_STATUS_OK, vec);
                } catch (...) {
                    return c_abi_current_exception("adopt.failure");
                }
            };
            {
                test::AllocationProbeScope probe(
                    ordinal == 0 ? std::numeric_limits<std::size_t>::max()
                                 : ordinal);
                auto result = adopt_object(invoke());
                const bool correct = ordinal == 0
                    ? result && result->constructor() == "Ok"
                    : probe.failed() &&
                      has_error(result.get(), "ResourceLimit", "adopt.failure");
                if (ordinal == 0) count = probe.count();
                result.reset();
                if (!correct || count == 0 || probe.outstanding() != 0)
                    return false;
            }
            if (mat.data || mat.rows || mat.cols || mat.stride || mat.owns_data ||
                vec.data || vec.size || vec.owns_data) return false;
#ifdef LMMC_DEBUG_LEAKS
            if (lmmc_debug_leaks_get_count() != baseline) return false;
#endif
        }
    }
    return true;
}

bool real_math_probe() {
    auto eight = make_owned_object<ExprObj>(LMCAS::SymbolicExpr::number(8));
    auto three = make_owned_object<ExprObj>(LMCAS::SymbolicExpr::number(3));
    auto four = make_owned_object<ExprObj>(LMCAS::SymbolicExpr::number(4));
    auto symbol = make_owned_object<ExprObj>(LMCAS::SymbolicExpr::variable("x"));
    auto invalid = make_owned_object<ExprObj>(LMCAS::CasError{
        LMCAS::CasErrc::ParseError, "bad expression", "parse_expr"});

    auto log_value = adopt_object(lmx_math_log2(eight.get()));
    auto exp_value = adopt_object(lmx_math_exp2(three.get()));
    auto hypot_value = adopt_object(lmx_math_hypot(three.get(), four.get()));
    auto log_unbound = adopt_object(lmx_math_log2(symbol.get()));
    auto exp_unbound = adopt_object(lmx_math_exp2(symbol.get()));
    auto hypot_unbound = adopt_object(lmx_math_hypot(three.get(), symbol.get()));
    auto log_invalid = adopt_object(lmx_math_log2(invalid.get()));
    auto exp_invalid = adopt_object(lmx_math_exp2(invalid.get()));
    auto hypot_invalid = adopt_object(lmx_math_hypot(invalid.get(), four.get()));

    return has_real(log_value.get(), 3.0) &&
           has_real(exp_value.get(), 8.0) &&
           has_real(hypot_value.get(), 5.0) &&
           has_error(log_unbound.get(), "UnboundSymbol", "evaluate_numeric") &&
           has_error(exp_unbound.get(), "UnboundSymbol", "evaluate_numeric") &&
           has_error(hypot_unbound.get(), "UnboundSymbol", "evaluate_numeric") &&
           has_error(log_invalid.get(), "InvalidArgument", "lmx_math_log2") &&
           has_error(exp_invalid.get(), "InvalidArgument", "lmx_math_exp2") &&
           has_error(hypot_invalid.get(), "InvalidArgument", "lmx_math_hypot");
}

const runtime::TupleObj* result_tuple(const AdtObj* result,
                                      std::size_t size) {
    const auto* field = result && result->type_name() == "Result" &&
                                 result->constructor() == "Ok"
                            ? result->field(0) : nullptr;
    if (!field || field->kind != ValueKind::Tuple || !field->obj ||
        field->obj->get_kind() != runtime::ObjectKind::Tuple) return nullptr;
    const auto* tuple = static_cast<const runtime::TupleObj*>(field->obj);
    return tuple->size() == size ? tuple : nullptr;
}

bool quadrature_tuple(const AdtObj* result) {
    const auto* tuple = result_tuple(result, 2);
    if (!tuple) return false;
    const auto& value = tuple->get(0);
    const auto& error = tuple->get(1);
    return value.kind == ValueKind::Real && error.kind == ValueKind::Real &&
           std::isfinite(value.real_val) && std::isfinite(error.real_val) &&
           std::abs(value.real_val - 1.0 / 3.0) < 1e-12 &&
           error.real_val >= 0.0 && error.real_val < 1e-12;
}

bool quadrature_tuple_probe() {
    auto expression = make_owned_object<ExprObj>(
        LMCAS::parse_expr("x^2").value());
    auto variable = make_owned_object<ExprObj>(
        LMCAS::SymbolicExpr::variable("x"));
    auto lower = make_owned_object<ExprObj>(LMCAS::SymbolicExpr::number(0));
    auto upper = make_owned_object<ExprObj>(LMCAS::SymbolicExpr::number(1));
    for (const bool symbol : {false, true}) {
        auto simpson = adopt_object(symbol
            ? lmx_computer_algebra_integrate_simpson_by_symbol(
                expression.get(), variable.get(), lower.get(), upper.get(), 8)
            : lmx_computer_algebra_integrate_simpson_by_name(
                expression.get(), "x", lower.get(), upper.get(), 8));
        auto gaussian = adopt_object(symbol
            ? lmx_computer_algebra_integrate_gaussian_by_symbol(
                expression.get(), variable.get(), lower.get(), upper.get(), 3)
            : lmx_computer_algebra_integrate_gaussian_by_name(
                expression.get(), "x", lower.get(), upper.get(), 3));
        auto adaptive = adopt_object(symbol
            ? lmx_computer_algebra_integrate_adaptive_by_symbol(
                expression.get(), variable.get(), lower.get(), upper.get(), 1e-10, 12)
            : lmx_computer_algebra_integrate_adaptive_by_name(
                expression.get(), "x", lower.get(), upper.get(), 1e-10, 12));
        if (!quadrature_tuple(simpson.get()) ||
            !quadrature_tuple(gaussian.get()) ||
            !quadrature_tuple(adaptive.get())) return false;
    }
    auto fourth_power = make_owned_object<ExprObj>(
        LMCAS::parse_expr("x^4").value());
    auto simpson = adopt_object(lmx_computer_algebra_integrate_simpson_by_name(
        fourth_power.get(), "x", lower.get(), upper.get(), 4));
    const auto* simpson_tuple = result_tuple(simpson.get(), 2);
    auto gaussian = adopt_object(lmx_computer_algebra_integrate_gaussian_by_name(
        expression.get(), "x", lower.get(), upper.get(), 1));
    const auto* gaussian_tuple = result_tuple(gaussian.get(), 2);
    if (!simpson_tuple || !gaussian_tuple ||
        simpson_tuple->get(0).kind != ValueKind::Real ||
        simpson_tuple->get(1).kind != ValueKind::Real ||
        gaussian_tuple->get(0).kind != ValueKind::Real ||
        gaussian_tuple->get(1).kind != ValueKind::Real ||
        std::abs(simpson_tuple->get(0).real_val - 0.2) > 1e-12 ||
        std::abs(simpson_tuple->get(1).real_val - 1.0 / 30720.0) > 1e-12 ||
        std::abs(gaussian_tuple->get(0).real_val - 0.25) > 1e-12 ||
        std::abs(gaussian_tuple->get(1).real_val - 1.0 / 12.0) > 1e-12)
        return false;
    return true;
}

bool expression_value(const Value& value, const LMCAS::ExprPtr& expected) {
    if (value.kind != ValueKind::Expr || !value.obj ||
        value.obj->get_kind() != runtime::ObjectKind::Expr) return false;
    const auto* expression = static_cast<const ExprObj*>(value.obj);
    if (!expression->ok()) return false;
    LMCAS::ComputationContext context;
    const auto actual = LMCAS::simplify(expression->expr(), context);
    const auto target = LMCAS::simplify(expected, context);
    if (!actual || !target) return false;
    const auto proof = LMCAS::equivalent(*actual.value(), *target.value(), context);
    const bool matches = proof && proof.value();
    if (!matches)
        std::cerr << "transform expression: actual=" << expression->expr()->to_string()
                  << " expected=" << expected->to_string() << '\n';
    return matches;
}

bool transform_tuple(const AdtObj* result, const LMCAS::ExprPtr& expression,
                     const LMCAS::ExprPtr& condition,
                     const LMCAS::ExprPtr& roc) {
    const auto* tuple = result_tuple(result, 3);
    if (!tuple || !expression_value(tuple->get(0), expression)) return false;
    for (const auto index : {1, 2}) {
        const auto& value = tuple->get(static_cast<uint8_t>(index));
        if (value.kind != ValueKind::Obj || !value.obj ||
            value.obj->get_kind() != runtime::ObjectKind::Array) return false;
        auto* array = static_cast<ArrayObj*>(value.obj);
        const auto& expected = index == 1 ? condition : roc;
        if (array->len() != (expected ? 1 : 0)) {
            std::cerr << "transform tuple field " << index
                      << " has " << array->len() << " conditions\n";
            return false;
        }
        if (expected && !expression_value(array->at(0), expected)) return false;
    }
    return true;
}

AdtObj* transform_conversion_probe(
    const LMCAS::TransformEngineResult& result) noexcept try {
    return transform_engine_result(result);
} catch (...) {
    return c_abi_current_exception("transform.conversion");
}

LMCAS::TransformEngineResult conditional_fourier_result() {
    auto assumptions = std::make_shared<LMCAS::AssumptionContext>();
    const auto real = assumptions->assume_domain_checked("a", LMCAS::Domain::Real);
    if (!real) return LMCAS::TransformEngineResult::failure(real.error());
    const auto time = assumptions->assume_domain_checked("t", LMCAS::Domain::Real);
    if (!time) return LMCAS::TransformEngineResult::failure(time.error());
    LMCAS::ComputationContext context;
    const auto attached = context.set_assumptions(assumptions);
    if (!attached) return LMCAS::TransformEngineResult::failure(attached.error());
    // The inverse pair exposes builtin |t| through the installed matching API.
    const auto inverse = LMCAS::inverse_fourier_transform_checked(
        LMCAS::parse_expr("2/(1+omega^2)").value(), "omega", "t", context);
    if (!inverse) return LMCAS::TransformEngineResult::failure(inverse.error());
    const auto base = LMCAS::simplify(inverse.value().value.expression, context);
    if (!base) return LMCAS::TransformEngineResult::failure(base.error());
    const auto matched = LMCAS::expr_match(
        LMCAS::SymbolicExpr::exp(LMCAS::SymbolicExpr::variable("exponent")),
        base.value(), {"exponent"}, context);
    if (!matched) return LMCAS::TransformEngineResult::failure(matched.error());
    if (!matched.value().matched || matched.value().bindings.size() != 1)
        return LMCAS::TransformEngineResult::failure(
            LMCAS::CasErrc::InternalInvariant,
            "inverse pair did not expose an exponential", "bridge.fixture");
    const auto absolute = LMCAS::simplify(LMCAS::SymbolicExpr::multiply(
        LMCAS::parse_expr("-1").value(),
        matched.value().bindings.front().value), context);
    if (!absolute) return LMCAS::TransformEngineResult::failure(absolute.error());
    const auto input = LMCAS::SymbolicExpr::multiply(
        LMCAS::parse_expr("3").value(), LMCAS::SymbolicExpr::exp(
            LMCAS::SymbolicExpr::multiply(
                LMCAS::parse_expr("-2*a").value(), absolute.value())));
    return LMCAS::fourier_transform_checked(input, "t", "omega", context);
}

bool transform_tuple_probe() {
    auto one = make_owned_object<ExprObj>(LMCAS::SymbolicExpr::number(1));
    auto t = make_owned_object<ExprObj>(LMCAS::SymbolicExpr::variable("t"));
    auto s = make_owned_object<ExprObj>(LMCAS::SymbolicExpr::variable("s"));
    const auto reciprocal = LMCAS::simplify(LMCAS::parse_expr("1/s").value()).value();
    const auto roc = LMCAS::parse_expr("s > 0").value();
    auto names = adopt_object(
        lmx_computer_algebra_integral_transforms_laplace_by_names(one.get(), "t", "s"));
    auto symbols = adopt_object(
        lmx_computer_algebra_integral_transforms_laplace_by_symbols(
            one.get(), t.get(), s.get()));
    if (!transform_tuple(names.get(), reciprocal, nullptr, roc) ||
        !transform_tuple(symbols.get(), reciprocal, nullptr, roc)) return false;

    const auto conditional = conditional_fourier_result();
    if (!conditional) {
        std::cerr << "conditional Fourier result: "
                  << conditional.error().operation << ": "
                  << conditional.error().message << '\n';
        return false;
    }
    const auto expected = LMCAS::simplify(
        LMCAS::parse_expr("12*a/(4*a^2+omega^2)").value()).value();
    const auto condition = LMCAS::parse_expr("a > 0").value();
    auto fourier = adopt_object(transform_conversion_probe(conditional));
    if (!transform_tuple(fourier.get(), expected, condition, nullptr)) return false;

    const auto failed = LMCAS::TransformEngineResult::failure(
        LMCAS::CasErrc::Cancelled, "transform stopped", "transform.original");
    auto error = adopt_object(transform_conversion_probe(failed));
    auto missing = conditional;
    missing.value().value.expression.reset();
    auto empty = adopt_object(transform_conversion_probe(missing));
    return has_error(error.get(), "Cancelled", "transform.original", "transform stopped") &&
           has_error(empty.get(), "UnsupportedExpression", "transform");
}

bool tuple_allocation_probe() {
    auto expression = make_owned_object<ExprObj>(
        LMCAS::parse_expr("x^2").value());
    auto lower = make_owned_object<ExprObj>(LMCAS::SymbolicExpr::number(0));
    auto upper = make_owned_object<ExprObj>(LMCAS::SymbolicExpr::number(1));
    const auto laplace = LMCAS::laplace_transform_checked(
        LMCAS::SymbolicExpr::number(1), "t", "s");
    const auto fourier = conditional_fourier_result();
    if (!laplace || !fourier) return false;
    for (int operation = 0; operation != 5; ++operation) {
        const auto invoke = [&]() noexcept -> AdtObj* {
            switch (operation) {
            case 0: return lmx_computer_algebra_integrate_simpson_by_name(
                expression.get(), "x", lower.get(), upper.get(), 8);
            case 1: return lmx_computer_algebra_integrate_gaussian_by_name(
                expression.get(), "x", lower.get(), upper.get(), 3);
            case 2: return lmx_computer_algebra_integrate_adaptive_by_name(
                expression.get(), "x", lower.get(), upper.get(), 1e-10, 12);
            default: return transform_conversion_probe(operation == 3 ? laplace : fourier);
            }
        };
        const char* name = operation == 0
            ? "lmx_computer_algebra_integrate_simpson_by_name"
            : operation == 1 ? "lmx_computer_algebra_integrate_gaussian_by_name"
            : operation == 2 ? "lmx_computer_algebra_integrate_adaptive_by_name"
                             : "transform.conversion";
        std::size_t count = 0;
        for (std::size_t ordinal = 0; ordinal <= count; ++ordinal) {
#ifdef LMMC_DEBUG_LEAKS
            const auto baseline = lmmc_debug_leaks_get_count();
#endif
            {
                test::AllocationProbeScope probe(
                    ordinal == 0 ? std::numeric_limits<std::size_t>::max() : ordinal);
                auto result = adopt_object(invoke());
                const bool correct = ordinal == 0
                    ? result_tuple(result.get(), operation < 3 ? 2 : 3) != nullptr
                    : probe.failed() && has_error(result.get(), "ResourceLimit", name);
                if (ordinal == 0) count = probe.count();
                result.reset();
                if (!correct || count == 0 || probe.outstanding() != 0) return false;
            }
#ifdef LMMC_DEBUG_LEAKS
            if (lmmc_debug_leaks_get_count() != baseline) return false;
#endif
        }
    }
    return true;
}

} // namespace

extern "C" int lmx_test_c_abi_exception_boundaries() noexcept {
    try {
        if (!dense_output_adoption_probe()) return 3;
        if (!dense_output_preconditions_probe()) return 4;
        if (!eigen_order_probe()) return 5;
        if (!decomposition_allocation_probe()) return 6;
        if (!vector_assign_failure_probe()) return 7;
        if (!resized_callback_probe()) return 8;
        if (!copied_inputs_probe()) return 9;
        if (!dense_output_allocation_probe()) return 10;
        if (!quadrature_tuple_probe()) return 11;
        if (!transform_tuple_probe()) return 12;
        if (!tuple_allocation_probe()) return 13;
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

        return real_math_probe() &&
               has_error(allocation.get(), "ResourceLimit",
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
                                 "lmx_cas_groebner_basis_by_symbols") &&
                       has_error(reduced_symbol_first.get(), "InvalidArgument",
                                 "lmx_cas_reduced_groebner_basis_by_symbols") &&
                       has_error(reduced_names_after_symbols.get(),
                                 "InvalidArgument",
                                 "lmx_cas_reduced_groebner_basis_by_symbols") &&
                       has_error(membership_symbol_first.get(), "InvalidArgument",
                                 "lmx_cas_ideal_membership_by_symbols") &&
                       has_error(membership_names_after_symbols.get(),
                                 "InvalidArgument",
                                 "lmx_cas_ideal_membership_by_symbols") &&
                       has_error(elimination_symbol_first.get(), "InvalidArgument",
                                 "lmx_cas_elimination_ideal_by_symbols") &&
                       has_error(elimination_names_after_symbols.get(),
                                 "InvalidArgument",
                                 "lmx_cas_elimination_ideal_by_symbols") &&
                       has_error(polynomial_system_symbol_first.get(),
                                 "InvalidArgument",
                                 "lmx_cas_polynomial_system_by_symbols") &&
                       has_error(polynomial_system_names_after_symbols.get(),
                                 "InvalidArgument",
                                 "lmx_cas_polynomial_system_by_symbols") &&
                       has_error(piecewise_unknown_first.get(), "InvalidArgument",
                                 "lmx_cas_parametric_piecewise_by_symbols") &&
                       has_error(piecewise_parameter_second.get(),
                                 "InvalidArgument",
                                 "lmx_cas_parametric_piecewise_by_symbols") &&
                       has_error(piecewise_names_after_symbols.get(),
                                 "InvalidArgument",
                                 "lmx_cas_parametric_piecewise_by_symbols")
            ? 0
            : 1;
    } catch (...) {
        return 2;
    }
}

} // namespace lmx::bridge
