#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "lmmc/interp.h"

using namespace lmx::bridge;

namespace {
AdtObj* interpolate_1d(
    const int algorithm, VectorObj* xs, VectorObj* ys, VectorObj* query) {
    if (!xs || !ys || !query || xs->size() != ys->size() ||
        xs->size() < 2 || query->size() == 0)
        return result_error(MathErrorCode::InvalidArgument, __func__, "interpolate: invalid input dimensions");
    std::vector<double> output(query->size());
    lmmc_status_t status = LMMC_STATUS_OK;
    if (algorithm == 0) {
        for (std::size_t i = 0; i < query->size() && status == LMMC_STATUS_OK; ++i)
            status = lmmc_interp_linear(
                xs->data().data(), ys->data().data(), xs->size(),
                query->data()[i], &output[i]);
    } else if (algorithm == 1) {
        lmmc_interp_lagrange_t* interpolant = nullptr;
        status = lmmc_interp_lagrange_create(
            xs->data().data(), ys->data().data(), xs->size(), &interpolant);
        for (std::size_t i = 0; i < query->size() && status == LMMC_STATUS_OK; ++i)
            status = lmmc_interp_lagrange_eval(
                interpolant, query->data()[i], &output[i]);
        lmmc_interp_lagrange_destroy(interpolant);
    } else if (algorithm == 2) {
        lmmc_interp_cspline_t* interpolant = nullptr;
        status = lmmc_interp_cspline_create(
            xs->data().data(), ys->data().data(), xs->size(), &interpolant);
        for (std::size_t i = 0; i < query->size() && status == LMMC_STATUS_OK; ++i)
            status = lmmc_interp_cspline_eval(
                interpolant, query->data()[i], &output[i]);
        lmmc_interp_cspline_destroy(interpolant);
    } else if (algorithm == 3) {
        lmmc_interp_pchip_t* interpolant = nullptr;
        status = lmmc_interp_pchip_create(
            xs->data().data(), ys->data().data(), xs->size(), &interpolant);
        for (std::size_t i = 0; i < query->size() && status == LMMC_STATUS_OK; ++i)
            status = lmmc_interp_pchip_eval(
                interpolant, query->data()[i], &output[i]);
        lmmc_interp_pchip_destroy(interpolant);
    } else {
        lmmc_interp_akima_t* interpolant = nullptr;
        status = lmmc_interp_akima_create(
            xs->data().data(), ys->data().data(), xs->size(), &interpolant);
        for (std::size_t i = 0; i < query->size() && status == LMMC_STATUS_OK; ++i)
            status = lmmc_interp_akima_eval(
                interpolant, query->data()[i], &output[i]);
        lmmc_interp_akima_destroy(interpolant);
    }
    if (status != LMMC_STATUS_OK)
        return result_error(status, "interpolate");
    return result_ok(new VectorObj(std::move(output)), ValueKind::Vector);
}

AdtObj* interpolate_2d(
    const bool bicubic, VectorObj* xs, VectorObj* ys, MatrixObj* values,
    VectorObj* query_x, VectorObj* query_y) {
    if (!xs || !ys || !values || !values->valid() || !query_x || !query_y ||
        values->rows() != xs->size() || values->cols() != ys->size() ||
        query_x->size() != query_y->size())
        return result_error(MathErrorCode::InvalidArgument, __func__, "interpolate: invalid grid dimensions");
    std::vector<double> output(query_x->size());
    lmmc_status_t status = LMMC_STATUS_OK;
    for (std::size_t i = 0; i < output.size() && status == LMMC_STATUS_OK; ++i) {
        status = bicubic
            ? lmmc_interp_bicubic(
                  xs->data().data(), xs->size(), ys->data().data(),
                  ys->size(), values->data().data(), query_x->data()[i],
                  query_y->data()[i], &output[i])
            : lmmc_interp_bilinear(
                  xs->data().data(), xs->size(), ys->data().data(),
                  ys->size(), values->data().data(), query_x->data()[i],
                  query_y->data()[i], &output[i]);
    }
    if (status != LMMC_STATUS_OK)
        return result_error(status, "interpolate");
    return result_ok(new VectorObj(std::move(output)), ValueKind::Vector);
}
} // namespace

extern "C" LM_API AdtObj* lmx_interpolation_linear(
    VectorObj* x, VectorObj* y, VectorObj* query) {
    return interpolate_1d(0, x, y, query);
}
extern "C" LM_API AdtObj* lmx_interpolation_lagrange(
    VectorObj* x, VectorObj* y, VectorObj* query) {
    return interpolate_1d(1, x, y, query);
}
extern "C" LM_API AdtObj* lmx_interpolation_cubic_spline(
    VectorObj* x, VectorObj* y, VectorObj* query) {
    return interpolate_1d(2, x, y, query);
}
extern "C" LM_API AdtObj* lmx_interpolation_piecewise_cubic_hermite(
    VectorObj* x, VectorObj* y, VectorObj* query) {
    return interpolate_1d(3, x, y, query);
}
extern "C" LM_API AdtObj* lmx_interpolation_akima(
    VectorObj* x, VectorObj* y, VectorObj* query) {
    return interpolate_1d(4, x, y, query);
}
extern "C" LM_API AdtObj* lmx_interpolation_bilinear(
    VectorObj* x, VectorObj* y, MatrixObj* values,
    VectorObj* query_x, VectorObj* query_y) {
    return interpolate_2d(false, x, y, values, query_x, query_y);
}
extern "C" LM_API AdtObj* lmx_interpolation_bicubic(
    VectorObj* x, VectorObj* y, MatrixObj* values,
    VectorObj* query_x, VectorObj* query_y) {
    return interpolate_2d(true, x, y, values, query_x, query_y);
}
