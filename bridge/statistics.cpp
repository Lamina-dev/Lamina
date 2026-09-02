#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>

#include <cstdint>
#include <string>
#include <lmmc/stats.h>
#include <lmmc/lsr_stdlib.h>
#include <vector>

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_statistics_mean(VectorObj* value) noexcept try {
    ensure_lmmc_runtime();
    return vector_stat_result("stats_mean", value, lmmc_vec_mean);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_statistics_variance(VectorObj* value) noexcept try {
    ensure_lmmc_runtime();
    return vector_stat_result("stats_variance", value, lmmc_vec_variance_sample);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_statistics_variance_population(VectorObj* value) noexcept try {
    ensure_lmmc_runtime();
    return vector_stat_result("stats_variance_population", value,
                              lmmc_vec_variance_population);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_statistics_standard_deviation(VectorObj* value) noexcept try {
    ensure_lmmc_runtime();
    return vector_stat_result("stats_stddev", value, lmmc_vec_stddev_sample);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_statistics_stddev_population(VectorObj* value) noexcept try {
    ensure_lmmc_runtime();
    return vector_stat_result("stats_stddev_population", value,
                              lmmc_vec_stddev_population);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_statistics_median(VectorObj* value) noexcept try {
    ensure_lmmc_runtime();
    return vector_stat_result("stats_median", value, lmmc_vec_median);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_statistics_quantile(VectorObj* value, const double p) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "stats_quantile: null vector");
    auto input = vector_view(value);
    lmmc_real_t result = 0.0;
    const auto status = lmmc_vec_quantile(&input, p, &result);
    return lmmc_real_result("stats_quantile", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_statistics_covariance(VectorObj* lhs, VectorObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    if (!lhs || !rhs) return result_error(MathErrorCode::InvalidArgument, __func__, "stats_covariance: null vector");
    auto left = vector_view(lhs);
    auto right = vector_view(rhs);
    lmmc_real_t result = 0.0;
    const auto status = lmmc_vec_covariance_sample(&left, &right, &result);
    return lmmc_real_result("stats_covariance", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_statistics_correlation(VectorObj* lhs, VectorObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    if (!lhs || !rhs) return result_error(MathErrorCode::InvalidArgument, __func__, "stats_correlation: null vector");
    auto left = vector_view(lhs);
    auto right = vector_view(rhs);
    lmmc_real_t result = 0.0;
    const auto status = lmmc_vec_correlation_sample(&left, &right, &result);
    return lmmc_real_result("stats_correlation", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_statistics_histogram(VectorObj* value, const LmInt bins) noexcept try {
    ensure_lmmc_runtime();
    if (!value || bins <= 0) return result_error(MathErrorCode::InvalidArgument, __func__, "stats_histogram: invalid argument");
    auto input = vector_view(value);
    std::vector<double> edges(static_cast<std::size_t>(bins) + 1);
    std::vector<std::size_t> counts(static_cast<std::size_t>(bins));
    const auto status = lmmc_vec_histogram(&input, static_cast<std::size_t>(bins),
                                           edges.data(), counts.data());
    if (status != LMMC_STATUS_OK) return result_error(status, "stats_histogram");
    std::vector<double> count_values;
    count_values.reserve(counts.size());
    for (const auto count : counts) count_values.push_back(static_cast<double>(count));
    std::vector<TableObj::Entry> entries;
    entries.emplace_back("counts", take_object_value(
        make_owned_object<VectorObj>(std::move(count_values)),
        ValueKind::Vector));
    entries.emplace_back("edges", take_object_value(
        make_owned_object<VectorObj>(std::move(edges)), ValueKind::Vector));
    return result_ok(new TableObj(std::move(entries)), ValueKind::Table);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_statistics_factorial(const LmInt n) noexcept try {
    ensure_lmmc_runtime();
    if (n < 0 || n > std::numeric_limits<std::uint32_t>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "stats.factorial: invalid argument");
    double output = 0.0;
    lmmc_stats_factorial(&output, static_cast<std::uint32_t>(n));
    return std::isfinite(output)
        ? result_ok(output)
        : result_error(MathErrorCode::NumericalFailure, __func__, "stats.factorial: numerical overflow");
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_combinations(
    const LmInt n, const LmInt r) noexcept try {
    ensure_lmmc_runtime();
    if (n < 0 || r < 0 || r > n ||
        n > std::numeric_limits<std::uint32_t>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "stats.ncr: invalid argument");
    double output = 0.0;
    lmmc_stats_nCr(
        &output, static_cast<std::uint32_t>(n),
        static_cast<std::uint32_t>(r));
    return std::isfinite(output)
        ? result_ok(output)
        : result_error(MathErrorCode::NumericalFailure, __func__, "stats.ncr: numerical overflow");
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_permutations(
    const LmInt n, const LmInt r) noexcept try {
    ensure_lmmc_runtime();
    if (n < 0 || r < 0 || r > n ||
        n > std::numeric_limits<std::uint32_t>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "stats.npr: invalid argument");
    double output = 0.0;
    lmmc_stats_nPr(
        &output, static_cast<std::uint32_t>(n),
        static_cast<std::uint32_t>(r));
    return std::isfinite(output)
        ? result_ok(output)
        : result_error(MathErrorCode::NumericalFailure, __func__, "stats.npr: numerical overflow");
} catch (...) {
    return c_abi_current_exception(__func__);
}

namespace {
AdtObj* matrix_stat_output(
    const char* name, MatrixObj* value,
    lmmc_status_t (*operation)(const lmmc_mat_t*, lmmc_mat_t*)) {
    if (!value || !value->valid())
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string(name) + ": invalid matrix");
    auto input = matrix_view(value);
    lmmc_mat_t output{};
    auto status = lmmc_mat_create(value->cols(), value->cols(), &output);
    if (status == LMMC_STATUS_OK) status = operation(&input, &output);
    return lmmc_matrix_output(name, status, output);
}
}

extern "C" LM_API AdtObj* lmx_statistics_column_mean(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !value->valid())
        return result_error(MathErrorCode::InvalidArgument, __func__, "stats.column_mean: invalid matrix");
    auto input = matrix_view(value);
    std::vector<double> data(value->cols());
    lmmc_vec_t output{data.size(), data.data(), 0};
    const auto status = lmmc_mat_column_mean(&input, &output);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "stats.column_mean");
    return result_ok(new VectorObj(std::move(data)), ValueKind::Vector);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_covariance_matrix_sample(
    MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return matrix_stat_output(
        "stats.covariance_matrix_sample", value,
        lmmc_mat_covariance_sample);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_covariance_matrix_population(
    MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return matrix_stat_output(
        "stats.covariance_matrix_population", value,
        lmmc_mat_covariance_population);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_correlation_matrix_sample(
    MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return matrix_stat_output(
        "stats.correlation_matrix_sample", value,
        lmmc_mat_correlation_sample);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_correlation_matrix_population(
    MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return matrix_stat_output(
        "stats.correlation_matrix_population", value,
        lmmc_mat_correlation_population);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_statistics_normal_probability_density(
    const double x, const double mean, const double stddev) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.normal_probability_density", x, mean, stddev, lmmc_lsr_stats_normal_pdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_normal_cumulative_distribution(
    const double x, const double mean, const double stddev) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.normal_cumulative_distribution", x, mean, stddev, lmmc_lsr_stats_normal_cdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_normal_quantile(
    const double p, const double mean, const double stddev) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.normal_quantile", p, mean, stddev,
        lmmc_lsr_stats_normal_quantile);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_student_t_probability_density(const double x,
                                             const double df) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result("stats.student_t_probability_density", x, df,
                                   lmmc_lsr_stats_t_pdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_student_t_cumulative_distribution(const double x,
                                             const double df) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result("stats.student_t_cumulative_distribution", x, df,
                                   lmmc_lsr_stats_t_cdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_t_quantile(const double p,
                                                  const double df) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result("stats.t_quantile", p, df,
                                   lmmc_lsr_stats_t_quantile);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_chi_squared_probability_density(const double x,
                                                const double df) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result("stats.chi_squared_probability_density", x, df,
                                   lmmc_lsr_stats_chi2_pdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_chi_squared_cumulative_distribution(const double x,
                                                const double df) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result("stats.chi_squared_cumulative_distribution", x, df,
                                   lmmc_lsr_stats_chi2_cdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_chi2_quantile(const double p,
                                                     const double df) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result("stats.chi2_quantile", p, df,
                                   lmmc_lsr_stats_chi2_quantile);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_fisher_f_probability_density(
    const double x, const double df1, const double df2) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.fisher_f_probability_density", x, df1, df2, lmmc_lsr_stats_f_pdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_fisher_f_cumulative_distribution(
    const double x, const double df1, const double df2) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.fisher_f_cumulative_distribution", x, df1, df2, lmmc_lsr_stats_f_cdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_f_quantile(
    const double p, const double df1, const double df2) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.f_quantile", p, df1, df2, lmmc_lsr_stats_f_quantile);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_gamma_pdf(
    const double x, const double shape, const double scale) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.gamma_pdf", x, shape, scale, lmmc_lsr_stats_gamma_pdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_gamma_cdf(
    const double x, const double shape, const double scale) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.gamma_cdf", x, shape, scale, lmmc_lsr_stats_gamma_cdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_gamma_quantile(
    const double p, const double shape, const double scale) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.gamma_quantile", p, shape, scale,
        lmmc_lsr_stats_gamma_quantile);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_beta_pdf(
    const double x, const double alpha, const double beta) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.beta_pdf", x, alpha, beta, lmmc_lsr_stats_beta_pdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_beta_cdf(
    const double x, const double alpha, const double beta) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.beta_cdf", x, alpha, beta, lmmc_lsr_stats_beta_cdf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_beta_quantile(
    const double p, const double alpha, const double beta) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "stats.beta_quantile", p, alpha, beta,
        lmmc_lsr_stats_beta_quantile);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_binomial_probability_mass(
    const LmInt k, const LmInt n, const double p) noexcept try {
    ensure_lmmc_runtime();
    if (k < 0 || n < 0) return result_error(MathErrorCode::InvalidArgument, __func__, "stats.binomial_probability_mass: invalid count");
    lmmc_real_t result = 0.0;
    const auto status = lmmc_lsr_stats_binomial_pmf(
        static_cast<std::size_t>(k), static_cast<std::size_t>(n), p, &result);
    return lmmc_real_result("stats.binomial_probability_mass", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_binomial_cumulative_distribution(
    const LmInt k, const LmInt n, const double p) noexcept try {
    ensure_lmmc_runtime();
    if (k < 0 || n < 0) return result_error(MathErrorCode::InvalidArgument, __func__, "stats.binomial_cumulative_distribution: invalid count");
    lmmc_real_t result = 0.0;
    const auto status = lmmc_lsr_stats_binomial_cdf(
        static_cast<std::size_t>(k), static_cast<std::size_t>(n), p, &result);
    return lmmc_real_result("stats.binomial_cumulative_distribution", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_poisson_probability_mass(const LmInt k,
                                                   const double lambda) noexcept try {
    ensure_lmmc_runtime();
    if (k < 0) return result_error(MathErrorCode::InvalidArgument, __func__, "stats.poisson_probability_mass: invalid count");
    lmmc_real_t result = 0.0;
    const auto status = lmmc_lsr_stats_poisson_pmf(
        static_cast<std::size_t>(k), lambda, &result);
    return lmmc_real_result("stats.poisson_probability_mass", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_statistics_poisson_cumulative_distribution(const LmInt k,
                                                   const double lambda) noexcept try {
    ensure_lmmc_runtime();
    if (k < 0) return result_error(MathErrorCode::InvalidArgument, __func__, "stats.poisson_cumulative_distribution: invalid count");
    lmmc_real_t result = 0.0;
    const auto status = lmmc_lsr_stats_poisson_cdf(
        static_cast<std::size_t>(k), lambda, &result);
    return lmmc_real_result("stats.poisson_cumulative_distribution", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}
