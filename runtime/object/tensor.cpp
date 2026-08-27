#include "tensor.hpp"

#include <functional>
#include <limits>

namespace lmx::runtime {
namespace {

struct TensorDeleter {
    void operator()(lmmc_tensor_nd_t* tensor) const noexcept {
        if (!tensor) return;
        lmmc_tensor_nd_destroy(tensor);
        delete tensor;
    }
};

void hash_combine(std::size_t& seed, const std::size_t value) noexcept {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
}

} // namespace

TensorObj::TensorObj(lmmc_tensor_nd_t&& tensor)
    : Object(ObjectKind::Tensor),
      storage_(new lmmc_tensor_nd_t(tensor), TensorDeleter{}),
      view_(tensor) {
    view_.owns_data = 0;
    tensor = {};
}

TensorObj::TensorObj(std::shared_ptr<lmmc_tensor_nd_t> storage,
                     const lmmc_tensor_nd_t& view) noexcept
    : Object(ObjectKind::Tensor), storage_(std::move(storage)), view_(view) {
    view_.owns_data = 0;
}

std::size_t TensorObj::element_count() const noexcept {
    if (view_.ndim == 0 || view_.ndim > LMMC_TENSOR_MAX_NDIM) return 0;
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < view_.ndim; ++axis) {
        if (view_.dims[axis] == 0 ||
            count > std::numeric_limits<std::size_t>::max() / view_.dims[axis])
            return 0;
        count *= view_.dims[axis];
    }
    return count;
}

bool TensorObj::valid() const noexcept {
    return storage_ && view_.data && element_count() > 0;
}

bool TensorObj::equals(const TensorObj& other) const noexcept {
    if (view_.ndim != other.view_.ndim) return false;
    for (std::size_t axis = 0; axis < view_.ndim; ++axis)
        if (view_.dims[axis] != other.view_.dims[axis]) return false;
    const auto count = element_count();
    if (count != other.element_count()) return false;
    for (std::size_t linear = 0; linear < count; ++linear) {
        std::size_t remainder = linear;
        std::size_t left_offset = 0;
        std::size_t right_offset = 0;
        for (std::size_t axis = view_.ndim; axis-- > 0;) {
            const auto index = remainder % view_.dims[axis];
            remainder /= view_.dims[axis];
            left_offset += index * view_.strides[axis];
            right_offset += index * other.view_.strides[axis];
        }
        if (view_.data[left_offset] != other.view_.data[right_offset]) return false;
    }
    return true;
}

std::size_t TensorObj::hash() const noexcept {
    std::size_t result = std::hash<std::size_t>{}(view_.ndim);
    for (std::size_t axis = 0; axis < view_.ndim; ++axis)
        hash_combine(result, std::hash<std::size_t>{}(view_.dims[axis]));
    const auto count = element_count();
    for (std::size_t linear = 0; linear < count; ++linear) {
        std::size_t remainder = linear;
        std::size_t offset = 0;
        for (std::size_t axis = view_.ndim; axis-- > 0;) {
            const auto index = remainder % view_.dims[axis];
            remainder /= view_.dims[axis];
            offset += index * view_.strides[axis];
        }
        hash_combine(result, std::hash<double>{}(view_.data[offset]));
    }
    return result;
}

std::string TensorObj::to_string() const noexcept {
    std::string result = "tensor(";
    for (std::size_t axis = 0; axis < view_.ndim; ++axis) {
        if (axis) result += 'x';
        result += std::to_string(view_.dims[axis]);
    }
    result += ')';
    return result;
}

} // namespace lmx::runtime
