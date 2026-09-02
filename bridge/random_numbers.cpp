#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <lmmc/random.h>
#include <lmmc/lsr_stdlib.h>
#include "runtime/object/random.hpp"

using lmx::runtime::RandomObj;

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_random_create(const LmInt seed) noexcept try {
    ensure_lmmc_runtime();
    lmmc_rng_t* rng = nullptr;
    auto status = lmmc_rng_create(&rng);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_rng_seed(rng, static_cast<std::uint64_t>(seed));
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_rng_destroy(rng);
        return result_error(status, "rng");
    }
    return result_ok(new RandomObj(rng), ValueKind::Random);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_random_clone(RandomObj* source) noexcept try {
    ensure_lmmc_runtime();
    if (!source) return result_error(MathErrorCode::InvalidArgument, __func__, "rng_clone: null rng");
    lmmc_rng_t* rng = nullptr;
    const auto status = lmmc_rng_clone(source->handle(), &rng);
    if (status != LMMC_STATUS_OK) return result_error(status, "rng_clone");
    return result_ok(new RandomObj(rng), ValueKind::Random);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_random_jump(RandomObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "rng_jump: null rng");
    const auto status = lmmc_rng_jump(value->handle());
    if (status != LMMC_STATUS_OK) return result_error(status, "rng_jump");
    return result_ok(value->get(), ValueKind::Random);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_random_uniform(RandomObj* value, const double lower,
                                             const double upper) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "random_uniform: null rng");
    lmmc_real_t result = 0.0;
    const auto status = lmmc_rng_uniform(value->handle(), lower, upper, &result);
    return lmmc_real_result("random_uniform", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_random_normal_with_generator(RandomObj* value, const double mean,
                                            const double stddev) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "random_normal: null rng");
    lmmc_real_t result = 0.0;
    const auto status = lmmc_rng_normal(value->handle(), mean, stddev, &result);
    return lmmc_real_result("random_normal", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_random_exponential(RandomObj* value, const double rate) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "random_exponential: null rng");
    lmmc_real_t result = 0.0;
    const auto status = lmmc_rng_exponential(value->handle(), rate, &result);
    return lmmc_real_result("random_exponential", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_random_int(RandomObj* value, const LmInt lower,
                                         const LmInt upper) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "random_int: null rng");
    std::int64_t result = 0;
    const auto status = lmmc_rng_int_uniform(value->handle(), lower, upper, &result);
    if (status != LMMC_STATUS_OK) return result_error(status, "random_int");
    return result_ok(static_cast<LmInt>(result));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_random_vector(RandomObj* value, const LmInt count,
                                            const double lower, const double upper) noexcept try {
    ensure_lmmc_runtime();
    if (!value || count <= 0) return result_error(MathErrorCode::InvalidArgument, __func__, "random_vector: invalid argument");
    std::vector<double> data(static_cast<std::size_t>(count));
    const auto status = lmmc_rng_fill_uniform(value->handle(), lower, upper,
                                               data.data(), data.size());
    if (status != LMMC_STATUS_OK) return result_error(status, "random_vector");
    return result_ok(new VectorObj(std::move(data)), ValueKind::Vector);
} catch (...) {
    return c_abi_current_exception(__func__);
}

namespace {
AdtObj* rng_real_result(
    const char* name, RandomObj* value,
    const std::function<lmmc_status_t(lmmc_rng_t*, double*)>& operation) {
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::string(name) + ": null rng");
    double output = 0.0;
    const auto status = operation(value->handle(), &output);
    return lmmc_real_result(name, status, output);
}
}

extern "C" LM_API AdtObj* lmx_random_long_jump(RandomObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "random.long_jump: null rng");
    const auto status = lmmc_rng_long_jump(value->handle());
    if (status != LMMC_STATUS_OK)
        return result_error(status, "random.long_jump");
    return result_ok(value->get(), ValueKind::Random);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_random_gamma(
    RandomObj* value, const double shape, const double scale) noexcept try {
    ensure_lmmc_runtime();
    return rng_real_result("random.gamma", value, [=](auto* rng, auto* out) {
        return lmmc_rng_gamma(rng, shape, scale, out);
    });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_random_beta(
    RandomObj* value, const double alpha, const double beta) noexcept try {
    ensure_lmmc_runtime();
    return rng_real_result("random.beta", value, [=](auto* rng, auto* out) {
        return lmmc_rng_beta(rng, alpha, beta, out);
    });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_random_chi_squared(
    RandomObj* value, const double degrees_of_freedom) noexcept try {
    ensure_lmmc_runtime();
    return rng_real_result(
        "random.chi_squared", value, [=](auto* rng, auto* out) {
            return lmmc_rng_chi_squared(rng, degrees_of_freedom, out);
        });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_random_fisher_f(
    RandomObj* value, const double first_df, const double second_df) noexcept try {
    ensure_lmmc_runtime();
    return rng_real_result("random.fisher_f", value, [=](auto* rng, auto* out) {
        return lmmc_rng_f(rng, first_df, second_df, out);
    });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_random_student_t(
    RandomObj* value, const double degrees_of_freedom) noexcept try {
    ensure_lmmc_runtime();
    return rng_real_result(
        "random.student_t", value, [=](auto* rng, auto* out) {
            return lmmc_rng_student_t(rng, degrees_of_freedom, out);
        });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_random_poisson(
    RandomObj* value, const double lambda) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "random.poisson: null rng");
    std::size_t output = 0;
    const auto status = lmmc_rng_poisson(value->handle(), lambda, &output);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "random.poisson");
    if (output > static_cast<std::size_t>(
                     std::numeric_limits<LmInt>::max()))
        return result_error(MathErrorCode::NumericalFailure, __func__, "random.poisson: integer overflow");
    return result_ok(static_cast<LmInt>(output));
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_random_binomial(
    RandomObj* value, const LmInt count, const double probability) noexcept try {
    ensure_lmmc_runtime();
    if (!value || count < 0)
        return result_error(MathErrorCode::InvalidArgument, __func__, "random.binomial: invalid argument");
    std::size_t output = 0;
    const auto status = lmmc_rng_binomial(
        value->handle(), static_cast<std::size_t>(count), probability,
        &output);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "random.binomial");
    if (output > static_cast<std::size_t>(
                     std::numeric_limits<LmInt>::max()))
        return result_error(MathErrorCode::NumericalFailure, __func__, "random.binomial: integer overflow");
    return result_ok(static_cast<LmInt>(output));
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_random_shuffle(
    RandomObj* value, VectorObj* input) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !input)
        return result_error(MathErrorCode::InvalidArgument, __func__, "random.shuffle: invalid argument");
    auto data = input->data();
    const auto status = lmmc_rng_shuffle(
        value->handle(), data.data(), data.size(), sizeof(double));
    if (status != LMMC_STATUS_OK)
        return result_error(status, "random.shuffle");
    return result_ok(new VectorObj(std::move(data)), ValueKind::Vector);
} catch (...) {
    return c_abi_current_exception(__func__);
}

namespace {
/**
 * @brief Owns the current thread's default LMMC random engine.
 * @ownership Creates the engine lazily and destroys it before the thread's
 * LMMC runtime lease is released.
 */
struct DefaultRngContext {
    lmmc_rng_t* value = nullptr;
    ~DefaultRngContext() {
        if (value) lmmc_rng_destroy(value);
    }
};

/**
 * @brief Returns the current thread's default engine, creating it lazily.
 * @param error Receives a stable error message on initialization failure.
 * @return Borrowed engine pointer, or null.
 * @ownership The thread-local context owns the returned pointer.
 */
lmmc_rng_t* default_rng(std::string& error) {
    static thread_local DefaultRngContext context;
    if (!context.value) {
        const auto status = lmmc_rng_create(&context.value);
        if (status != LMMC_STATUS_OK) {
            error = std::string("random.default: ") + lmmc_lsr_error_name(status);
            return nullptr;
        }
    }
    return context.value;
}

} // namespace

/** @brief Reseeds the current thread's default random engine. @param seed Deterministic Lamina seed. @return Owning Result bool or error. @ownership Engine remains thread-owned; caller owns return. @threadsafe Each thread has an independent engine. */
extern "C" LM_API AdtObj* lmx_random_seed(const LmInt seed) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    auto* rng = default_rng(error);
    if (!rng) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto status = lmmc_rng_seed(rng, static_cast<std::uint64_t>(seed));
    return status == LMMC_STATUS_OK ? result_ok(true)
        : result_error(status, "random.seed");
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Draws uniform `[0,1)` from the current thread's default engine. @return Owning Result real or error. @ownership Engine remains thread-owned; caller owns return. @threadsafe Each thread has an independent engine. */
extern "C" LM_API AdtObj* lmx_random_random_real() noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    auto* rng = default_rng(error);
    if (!rng) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    lmmc_real_t output = 0.0;
    const auto status = lmmc_rng_uniform(rng, 0.0, 1.0, &output);
    return lmmc_real_result("random.rand", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Draws an integer from the current thread's default engine. @param lower Inclusive lower bound. @param upper Inclusive upper bound. @return Owning Result int or error. @ownership Engine remains thread-owned; caller owns return. @threadsafe Each thread has an independent engine. */
extern "C" LM_API AdtObj* lmx_random_random_integer(LmInt lower, LmInt upper) noexcept try {
    ensure_lmmc_runtime();
    if (lower > upper) return result_error(MathErrorCode::InvalidArgument, __func__, "random.randint: invalid bounds");
    std::string error;
    auto* rng = default_rng(error); if (!rng) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    std::int64_t output = 0;
    const auto status = lmmc_rng_int_uniform(rng, lower, upper, &output);
    return status == LMMC_STATUS_OK ? result_ok(static_cast<LmInt>(output))
        : result_error(status, "random.randint");
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Chooses one element from a dense vector using the current thread's default engine. @param values Borrowed nonempty vector. @return Owning Result real or error. @ownership Engine and input remain borrowed; caller owns return. @threadsafe Each thread has an independent engine. */
extern "C" LM_API AdtObj* lmx_random_choice(VectorObj* values) noexcept try {
    ensure_lmmc_runtime();
    if (!values || values->data().empty())
        return result_error(MathErrorCode::EmptyInput, __func__, "random.choice: empty vector");
    std::string error;
    auto* rng = default_rng(error); if (!rng) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    std::int64_t index = 0;
    const auto status = lmmc_rng_int_uniform(
        rng, 0, static_cast<std::int64_t>(values->data().size() - 1), &index);
    return status == LMMC_STATUS_OK
        ? result_ok(values->data()[static_cast<std::size_t>(index)])
        : result_error(status, "random.choice");
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Draws a normal variate from the current thread's default engine. @param mean Distribution mean. @param stddev Positive standard deviation. @return Owning Result real or error. @ownership Engine remains thread-owned; caller owns return. @threadsafe Each thread has an independent engine. */
extern "C" LM_API AdtObj* lmx_random_normal(
    double mean, double stddev) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    auto* rng = default_rng(error); if (!rng) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    lmmc_real_t output = 0.0;
    const auto status = lmmc_rng_normal(rng, mean, stddev, &output);
    return lmmc_real_result("random.normal", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}
