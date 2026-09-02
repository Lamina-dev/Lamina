#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>

#include <string>
#include <vector>

using namespace lmx::bridge;

extern "C" LM_API ComplexObj* lmx_complex_numbers_from_cartesian(const double real,
                                                  const double imag) noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t value{};
    const auto status = lmmc_complex_create(real, imag, &value);
    if (status != LMMC_STATUS_OK) return nullptr;
    return new ComplexObj(value.real, value.imag);
} catch (...) {
    return nullptr;
}

extern "C" LM_API double lmx_complex_numbers_real_part(ComplexObj* value) noexcept try {
    ensure_lmmc_runtime();
    return value ? value->real() : 0.0;
} catch (...) {
    return 0.0;
}

extern "C" LM_API double lmx_complex_numbers_imaginary_part(ComplexObj* value) noexcept try {
    ensure_lmmc_runtime();
    return value ? value->imag() : 0.0;
} catch (...) {
    return 0.0;
}

extern "C" LM_API AdtObj* lmx_complex_numbers_add(ComplexObj* lhs,
                                             ComplexObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t left{}, right{}, result{};
    if (!checked_complex(lhs, left) || !checked_complex(rhs, right)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "lmx_complex_numbers_add: invalid argument");
    }
    const auto status = lmmc_complex_add(&left, &right, &result);
    return lmmc_complex_result("lmx_complex_numbers_add", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_complex_numbers_subtract(ComplexObj* lhs,
                                             ComplexObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t left{}, right{}, result{};
    if (!checked_complex(lhs, left) || !checked_complex(rhs, right)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "lmx_complex_numbers_subtract: invalid argument");
    }
    const auto status = lmmc_complex_sub(&left, &right, &result);
    return lmmc_complex_result("lmx_complex_numbers_subtract", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_complex_numbers_multiply(ComplexObj* lhs,
                                             ComplexObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t left{}, right{}, result{};
    if (!checked_complex(lhs, left) || !checked_complex(rhs, right)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "lmx_complex_numbers_multiply: invalid argument");
    }
    const auto status = lmmc_complex_mul(&left, &right, &result);
    return lmmc_complex_result("lmx_complex_numbers_multiply", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_complex_numbers_divide(ComplexObj* lhs,
                                             ComplexObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t left{}, right{}, result{};
    if (!checked_complex(lhs, left) || !checked_complex(rhs, right)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "lmx_complex_numbers_divide: invalid argument");
    }
    const auto status = lmmc_complex_div(&left, &right, &result);
    return lmmc_complex_result("lmx_complex_numbers_divide", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_complex_numbers_conjugate(ComplexObj* value) noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t input{}, result{};
    if (!checked_complex(value, input)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "lmx_complex_numbers_conjugate: invalid argument");
    }
    const auto status = lmmc_complex_conj(&input, &result);
    return lmmc_complex_result("lmx_complex_numbers_conjugate", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_complex_numbers_absolute_value(ComplexObj* value) noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t input{};
    if (!checked_complex(value, input)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "lmx_complex_numbers_absolute_value: invalid argument");
    }
    lmmc_real_t result = 0.0;
    const auto status = lmmc_complex_modulus(&input, &result);
    return lmmc_real_result("lmx_complex_numbers_absolute_value", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

namespace {
AdtObj* complex_unary(
    const char* name, ComplexObj* value,
    lmmc_status_t (*operation)(const lmmc_complex_t*, lmmc_complex_t*)) {
    lmmc_complex_t input{}, output{};
    if (!checked_complex(value, input))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string(name) + ": invalid argument");
    return lmmc_complex_result(name, operation(&input, &output), output);
}

AdtObj* fft_transform(ArrayObj* values, const bool inverse) {
    if (!values || values->values().empty())
        return result_error(MathErrorCode::EmptyInput, __func__, "fft: input must not be empty");
    std::vector<double> real;
    std::vector<double> imag;
    real.reserve(values->values().size());
    imag.reserve(values->values().size());
    for (const auto& value : values->values()) {
        if (value.kind != ValueKind::Complex || !value.obj ||
            value.obj->get_kind() != lmx::runtime::ObjectKind::Complex)
            return result_error(MathErrorCode::InvalidArgument, __func__, "fft: input contains a non-complex value");
        const auto* number = static_cast<const ComplexObj*>(value.obj);
        if (!std::isfinite(number->real()) || !std::isfinite(number->imag()))
            return result_error(MathErrorCode::InvalidArgument, __func__, "fft: input contains a non-finite value");
        real.push_back(number->real());
        imag.push_back(number->imag());
    }
    const auto status = inverse
        ? lmmc_fft_inverse(real.data(), imag.data(), real.size())
        : lmmc_fft_forward(real.data(), imag.data(), real.size());
    if (status != LMMC_STATUS_OK) return result_error(status, "fft");
    auto result = make_owned_object<ArrayObj>();
    for (std::size_t index = 0; index < real.size(); ++index) {
        result->append(take_object_value(
            make_owned_object<ComplexObj>(real[index], imag[index]),
            ValueKind::Complex));
    }
    return result_ok(result.release(), ValueKind::Obj);
}
} // namespace

extern "C" LM_API ComplexObj* lmx_complex_numbers_from_polar(
    const double radius, const double angle) noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t output{};
    if (lmmc_complex_from_polar(radius, angle, &output) != LMMC_STATUS_OK)
        return nullptr;
    return new ComplexObj(output.real, output.imag);
} catch (...) {
    return nullptr;
}
extern "C" LM_API AdtObj* lmx_complex_numbers_argument(ComplexObj* value) noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t input{};
    if (!checked_complex(value, input))
        return result_error(MathErrorCode::InvalidArgument, __func__, "complex.arg: invalid argument");
    double output = 0.0;
    const auto status = lmmc_complex_arg(&input, &output);
    return lmmc_real_result("complex.arg", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_complex_numbers_exponential(ComplexObj* value) noexcept try {
    ensure_lmmc_runtime();
    return complex_unary("complex.exp", value, lmmc_complex_exp);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_complex_numbers_natural_logarithm(ComplexObj* value) noexcept try {
    ensure_lmmc_runtime();
    return complex_unary("complex.log", value, lmmc_complex_log);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_complex_numbers_square_root(ComplexObj* value) noexcept try {
    ensure_lmmc_runtime();
    return complex_unary("complex.sqrt", value, lmmc_complex_sqrt);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_complex_numbers_sine(ComplexObj* value) noexcept try {
    ensure_lmmc_runtime();
    return complex_unary("complex.sin", value, lmmc_complex_sin);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_complex_numbers_cosine(ComplexObj* value) noexcept try {
    ensure_lmmc_runtime();
    return complex_unary("complex.cos", value, lmmc_complex_cos);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_complex_numbers_power(
    ComplexObj* base, ComplexObj* exponent) noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t lhs{}, rhs{}, output{};
    if (!checked_complex(base, lhs) || !checked_complex(exponent, rhs))
        return result_error(MathErrorCode::InvalidArgument, __func__, "complex.pow: invalid argument");
    return lmmc_complex_result(
        "complex.pow", lmmc_complex_pow(&lhs, &rhs, &output), output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_fast_fourier_transform_forward(ArrayObj* values) noexcept try {
    ensure_lmmc_runtime();
    return fft_transform(values, false);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_fast_fourier_transform_inverse(ArrayObj* values) noexcept try {
    ensure_lmmc_runtime();
    return fft_transform(values, true);
} catch (...) {
    return c_abi_current_exception(__func__);
}
