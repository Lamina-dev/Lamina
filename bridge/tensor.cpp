#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/lmmc_internal.hpp"
#include "runtime/object/tensor.hpp"
#include "lmmc/tensor.h"
#include <algorithm>

using namespace lmx::bridge;

namespace {
using lmx::runtime::TensorObj;

bool tensor_shape(ArrayObj* shape, std::vector<std::size_t>& dimensions,
                  std::size_t& count, std::string& error) {
    if (!shape || shape->values().empty() ||
        shape->values().size() > LMMC_TENSOR_MAX_NDIM) {
        error = "tensor rank must be between 1 and 8";
        return false;
    }
    count = 1;
    dimensions.reserve(shape->values().size());
    for (const auto& value : shape->values()) {
        if (value.kind != ValueKind::Int || value.int_val <= 0) {
            error = "tensor dimensions must be positive integers";
            return false;
        }
        const auto dimension = static_cast<std::size_t>(value.int_val);
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            error = "tensor element count overflow";
            return false;
        }
        count *= dimension;
        dimensions.push_back(dimension);
    }
    return true;
}

AdtObj* tensor_output(
    const char* name, const lmmc_status_t status, lmmc_tensor_nd_t& output) {
    if (status != LMMC_STATUS_OK) {
        lmmc_tensor_nd_destroy(&output);
        return result_error(status, name);
    }
    return result_ok(new TensorObj(std::move(output)), ValueKind::Tensor);
}

lmmc_tensor_t tensor3_view(TensorObj* value) {
    if (!value || value->tensor().ndim != 3) return {};
    const auto& tensor = value->tensor();
    return {tensor.dims[0], tensor.dims[1], tensor.dims[2],
            tensor.strides[0], tensor.strides[1], tensor.strides[2],
            tensor.data, 0};
}

lmmc_tensor_nd_t tensor3_to_nd(lmmc_tensor_t& value) {
    lmmc_tensor_nd_t result{};
    result.ndim = 3;
    result.dims[0] = value.dim0;
    result.dims[1] = value.dim1;
    result.dims[2] = value.dim2;
    result.strides[0] = value.stride0;
    result.strides[1] = value.stride1;
    result.strides[2] = value.stride2;
    result.data = value.data;
    result.owns_data = value.owns_data;
    value = {};
    return result;
}

AdtObj* tensor3_binary(
    const char* name, TensorObj* lhs, TensorObj* rhs,
    lmmc_status_t (*operation)(
        const lmmc_tensor_t*, const lmmc_tensor_t*, lmmc_tensor_t*)) {
    auto left = tensor3_view(lhs);
    auto right = tensor3_view(rhs);
    if (!left.data || !right.data)
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string(name) +
                                 ": expected two rank-3 tensors");
    lmmc_tensor_t output{};
    auto status =
        lmmc_tensor3_create(left.dim0, left.dim1, left.dim2, &output);
    if (status == LMMC_STATUS_OK)
        status = operation(&left, &right, &output);
    if (status != LMMC_STATUS_OK) {
        lmmc_tensor_destroy(&output);
        return result_error(status, name);
    }
    auto nd = tensor3_to_nd(output);
    return result_ok(new TensorObj(std::move(nd)), ValueKind::Tensor);
}

AdtObj* tensor3_stat(
    const char* name, TensorObj* value,
    lmmc_status_t (*operation)(const lmmc_tensor_t*, double*)) {
    auto input = tensor3_view(value);
    if (!input.data)
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string(name) +
                                 ": expected a rank-3 tensor");
    double output = 0.0;
    const auto status = operation(&input, &output);
    return lmmc_real_result(name, status, output);
}
} // namespace

extern "C" LM_API AdtObj* lmx_tensor_from_flat(
    VectorObj* values, ArrayObj* shape) {
    if (!values) return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.from_flat: null values");
    std::vector<std::size_t> dimensions;
    std::size_t count = 0;
    std::string error;
    if (!tensor_shape(shape, dimensions, count, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.from_flat: " + error);
    if (count != values->size())
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            "tensor.from_flat: shape does not match value count");
    if (!std::ranges::all_of(values->data(), [](const double value) {
            return std::isfinite(value);
        }))
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.from_flat: non-finite value");
    lmmc_tensor_nd_t output{};
    auto status =
        lmmc_tensor_create(dimensions.size(), dimensions.data(), &output);
    if (status == LMMC_STATUS_OK)
        std::copy(values->data().begin(), values->data().end(), output.data);
    return tensor_output("tensor.from_flat", status, output);
}

extern "C" LM_API ArrayObj* lmx_tensor_shape(TensorObj* value) {
    auto* result = new ArrayObj();
    if (!value || !value->valid()) return result;
    for (std::size_t axis = 0; axis < value->tensor().ndim; ++axis) {
        if (value->tensor().dims[axis] >
            static_cast<std::size_t>(std::numeric_limits<LmInt>::max()))
            return result;
        result->append(
            Value(static_cast<LmInt>(value->tensor().dims[axis])));
    }
    return result;
}

extern "C" LM_API AdtObj* lmx_tensor_element_at(
    TensorObj* value, ArrayObj* indices) {
    if (!value || !value->valid())
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.get: invalid tensor");
    std::vector<std::size_t> checked_indices;
    std::string error;
    if (!checked_nonnegative_ints(indices, checked_indices, error) ||
        checked_indices.size() != value->tensor().ndim)
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.get: invalid indices");
    double output = 0.0;
    const auto status = lmmc_tensor_get_nd(
        &value->tensor(), checked_indices.data(), &output);
    return lmmc_real_result("tensor.get", status, output);
}

extern "C" LM_API AdtObj* lmx_tensor_reshape(
    TensorObj* value, ArrayObj* shape) {
    if (!value || !value->valid())
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.reshape: invalid tensor");
    std::vector<std::size_t> dimensions;
    std::size_t count = 0;
    std::string error;
    if (!tensor_shape(shape, dimensions, count, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.reshape: " + error);
    lmmc_tensor_nd_t view{};
    const auto status = lmmc_tensor_nd_reshape_view(
        &value->tensor(), dimensions.size(), dimensions.data(), &view);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "tensor.reshape");
    return result_ok(
        new TensorObj(value->storage(), view), ValueKind::Tensor);
}

extern "C" LM_API AdtObj* lmx_tensor_permute(
    TensorObj* value, ArrayObj* permutation) {
    if (!value || !value->valid())
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.permute: invalid tensor");
    std::vector<std::size_t> axes;
    std::string error;
    if (!checked_nonnegative_ints(permutation, axes, error) ||
        axes.size() != value->tensor().ndim)
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.permute: invalid permutation");
    lmmc_tensor_nd_t output{};
    return tensor_output(
        "tensor.permute",
        lmmc_tensor_permute(&value->tensor(), axes.data(), &output), output);
}

extern "C" LM_API AdtObj* lmx_tensor_contract(
    TensorObj* lhs, TensorObj* rhs, ArrayObj* lhs_axes,
    ArrayObj* rhs_axes) {
    if (!lhs || !rhs || !lhs->valid() || !rhs->valid())
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.contract: invalid tensor");
    std::vector<std::size_t> left_axes;
    std::vector<std::size_t> right_axes;
    std::string error;
    if (!checked_nonnegative_ints(lhs_axes, left_axes, error) ||
        !checked_nonnegative_ints(rhs_axes, right_axes, error) ||
        left_axes.size() != right_axes.size())
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.contract: invalid axes");
    lmmc_tensor_nd_t output{};
    return tensor_output(
        "tensor.contract",
        lmmc_tensor_contract(
            &lhs->tensor(), &rhs->tensor(), left_axes.data(),
            right_axes.data(), left_axes.size(), &output),
        output);
}

extern "C" LM_API AdtObj* lmx_tensor_mode_n_product(
    TensorObj* value, MatrixObj* matrix, const LmInt mode) {
    if (!value || !value->valid() || !matrix || !matrix->valid() || mode < 0)
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.mode_n_product: invalid argument");
    auto input_matrix = matrix_view(matrix);
    lmmc_tensor_nd_t output{};
    return tensor_output(
        "tensor.mode_n_product",
        lmmc_tensor_mode_n_product(
            &value->tensor(), &input_matrix,
            static_cast<std::size_t>(mode), &output),
        output);
}

extern "C" LM_API AdtObj* lmx_tensor_add(
    TensorObj* lhs, TensorObj* rhs) {
    return tensor3_binary("tensor.add", lhs, rhs, lmmc_tensor_add);
}
extern "C" LM_API AdtObj* lmx_tensor_subtract(
    TensorObj* lhs, TensorObj* rhs) {
    return tensor3_binary("tensor.sub", lhs, rhs, lmmc_tensor_sub);
}
extern "C" LM_API AdtObj* lmx_tensor_multiply(
    TensorObj* lhs, TensorObj* rhs) {
    return tensor3_binary("tensor.mul", lhs, rhs, lmmc_tensor_mul);
}
extern "C" LM_API AdtObj* lmx_tensor_divide(
    TensorObj* lhs, TensorObj* rhs) {
    return tensor3_binary("tensor.div", lhs, rhs, lmmc_tensor_div);
}

extern "C" LM_API AdtObj* lmx_tensor_scale(
    TensorObj* value, const double scalar) {
    auto input = tensor3_view(value);
    if (!input.data || !std::isfinite(scalar))
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.scale: invalid argument");
    lmmc_tensor_t output{};
    auto status =
        lmmc_tensor3_create(input.dim0, input.dim1, input.dim2, &output);
    if (status == LMMC_STATUS_OK)
        status = lmmc_tensor_scale(&input, scalar, &output);
    if (status != LMMC_STATUS_OK) {
        lmmc_tensor_destroy(&output);
        return result_error(status, "tensor.scale");
    }
    auto nd = tensor3_to_nd(output);
    return result_ok(new TensorObj(std::move(nd)), ValueKind::Tensor);
}

extern "C" LM_API AdtObj* lmx_tensor_norm(TensorObj* value) {
    return tensor3_stat("tensor.norm", value, lmmc_tensor_norm_fro);
}
extern "C" LM_API AdtObj* lmx_tensor_sum(TensorObj* value) {
    return tensor3_stat("tensor.sum", value, lmmc_tensor_sum);
}
extern "C" LM_API AdtObj* lmx_tensor_minimum(TensorObj* value) {
    return tensor3_stat("tensor.min", value, lmmc_tensor_min);
}
extern "C" LM_API AdtObj* lmx_tensor_maximum(TensorObj* value) {
    return tensor3_stat("tensor.max", value, lmmc_tensor_max);
}

extern "C" LM_API AdtObj* lmx_tensor_sum_axis(
    TensorObj* value, const LmInt axis) {
    auto input = tensor3_view(value);
    if (!input.data || axis < 0 || axis > 2)
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.sum_axis: invalid argument");
    const std::size_t rows = axis == 0 ? input.dim1 : input.dim0;
    const std::size_t cols = axis == 2 ? input.dim1 : input.dim2;
    lmmc_mat_t output{};
    auto status = lmmc_mat_create(rows, cols, &output);
    if (status == LMMC_STATUS_OK)
        status = lmmc_tensor_sum_axis(
            &input, static_cast<std::size_t>(axis), &output);
    return lmmc_matrix_output("tensor.sum_axis", status, output);
}

extern "C" LM_API AdtObj* lmx_tensor_slice(
    TensorObj* value, const LmInt begin0, const LmInt end0,
    const LmInt begin1, const LmInt end1,
    const LmInt begin2, const LmInt end2) {
    auto input = tensor3_view(value);
    if (!input.data || begin0 < 0 || begin1 < 0 || begin2 < 0 ||
        end0 < 0 || end1 < 0 || end2 < 0)
        return result_error(MathErrorCode::InvalidArgument, __func__, "tensor.slice: invalid argument");
    lmmc_tensor_t output{};
    const auto status = lmmc_tensor_slice_view(
        &input, static_cast<std::size_t>(begin0),
        static_cast<std::size_t>(end0), static_cast<std::size_t>(begin1),
        static_cast<std::size_t>(end1), static_cast<std::size_t>(begin2),
        static_cast<std::size_t>(end2), &output);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "tensor.slice");
    auto nd = tensor3_to_nd(output);
    return result_ok(
        new TensorObj(value->storage(), nd), ValueKind::Tensor);
}
