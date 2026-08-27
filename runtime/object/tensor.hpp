#pragma once

#include "object.hpp"
#include "lmmc/tensor.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace lmx::runtime {

class TensorObj final : public Object {
    std::shared_ptr<lmmc_tensor_nd_t> storage_;
    lmmc_tensor_nd_t view_{};

public:
    explicit TensorObj(lmmc_tensor_nd_t&& tensor);
    TensorObj(std::shared_ptr<lmmc_tensor_nd_t> storage,
              const lmmc_tensor_nd_t& view) noexcept;

    [[nodiscard]] const lmmc_tensor_nd_t& tensor() const noexcept { return view_; }
    [[nodiscard]] lmmc_tensor_nd_t& tensor() noexcept { return view_; }
    [[nodiscard]] const std::shared_ptr<lmmc_tensor_nd_t>& storage() const noexcept {
        return storage_;
    }
    [[nodiscard]] std::size_t element_count() const noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool equals(const TensorObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
};

} // namespace lmx::runtime
