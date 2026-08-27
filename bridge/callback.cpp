#include "bridge/callback.hpp"

#include "runtime/object/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <utility>

namespace lmx::bridge {
namespace {

using lmx::runtime::LaminaVM;
using lmx::runtime::MatrixObj;
using lmx::runtime::Value;
using lmx::runtime::ValueKind;
using lmx::runtime::VectorObj;

std::expected<Value, std::string> invoke(
    CallbackContext& context, const std::span<const Value> arguments) {
    if (!context.function) {
        return std::unexpected("null Lamina callback");
    }
    auto* vm = LaminaVM::current();
    if (!vm) {
        return std::unexpected("Lamina callback invoked outside an active VM");
    }
    return vm->invoke(*context.function, arguments);
}

bool finite_vector(const std::vector<double>& values) noexcept {
    return std::ranges::all_of(values, [](const double value) {
        return std::isfinite(value);
    });
}

const VectorObj* checked_vector(CallbackContext& context, const Value& value,
                                const std::size_t expected_size) {
    if (value.kind != ValueKind::Vector || !value.obj ||
        value.obj->get_kind() != lmx::runtime::ObjectKind::Vector) {
        context.fail("Lamina callback returned a non-vector value");
        return nullptr;
    }
    const auto* vector = static_cast<const VectorObj*>(value.obj);
    if (vector->size() != expected_size) {
        context.fail("Lamina callback returned vector dimension " +
                     std::to_string(vector->size()) + ", expected " +
                     std::to_string(expected_size));
        return nullptr;
    }
    if (!finite_vector(vector->data())) {
        context.fail("Lamina callback returned a non-finite vector");
        return nullptr;
    }
    return vector;
}

void load_argument(VectorObj& argument, const double* data,
                   const std::size_t size) {
    auto& destination = argument.data();
    if (destination.size() != size) destination.resize(size);
    std::copy_n(data, size, destination.begin());
}

} // namespace

void CallbackContext::fail(std::string message) {
    if (error.empty()) error = std::move(message);
}

VectorCallbackContext::VectorCallbackContext(
    const lmx::runtime::FuncObj* function_, const std::size_t argument_size,
    const std::size_t result_size_)
    : CallbackContext(function_),
      argument(std::vector<double>(argument_size)),
      result_size(result_size_) {}

VectorScalarCallbackContext::VectorScalarCallbackContext(
    const lmx::runtime::FuncObj* function_, const std::size_t argument_size)
    : VectorCallbackContext(function_, argument_size, 0) {}

MatrixCallbackContext::MatrixCallbackContext(
    const lmx::runtime::FuncObj* function_, const std::size_t argument_size,
    const std::size_t rows_, const std::size_t cols_)
    : CallbackContext(function_),
      argument(std::vector<double>(argument_size)),
      rows(rows_),
      cols(cols_) {}

OdeCallbackContext::OdeCallbackContext(
    const lmx::runtime::FuncObj* function_, const std::size_t dimension_)
    : CallbackContext(function_),
      argument(std::vector<double>(dimension_)),
      dimension(dimension_) {}

OdeMatrixCallbackContext::OdeMatrixCallbackContext(
    const lmx::runtime::FuncObj* function_, const std::size_t dimension_)
    : CallbackContext(function_),
      argument(std::vector<double>(dimension_)),
      dimension(dimension_) {}

double scalar_callback_trampoline(const double value, void* user_data) noexcept {
    auto* context = static_cast<ScalarCallbackContext*>(user_data);
    if (!context) return std::numeric_limits<double>::quiet_NaN();
    if (context->failed()) return 0.0;
    try {
        const Value argument(value);
        const auto result = invoke(*context, std::span(&argument, 1));
        if (!result) {
            context->fail(result.error());
            return 0.0;
        }
        if (result->kind != ValueKind::Real) {
            context->fail("Lamina callback returned a non-real value");
            return 0.0;
        }
        if (!std::isfinite(result->real_val)) {
            context->fail("Lamina callback returned a non-finite real");
            return 0.0;
        }
        return result->real_val;
    } catch (const std::exception& error) {
        context->fail(error.what());
    } catch (...) {
        context->fail("unknown Lamina callback failure");
    }
    return 0.0;
}

lmmc_status_t vector_callback_trampoline(
    const lmmc_vec_t* input, lmmc_vec_t* output, void* user_data) noexcept {
    auto* context = static_cast<VectorCallbackContext*>(user_data);
    if (!context || !input || !input->data || !output) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    try {
        load_argument(context->argument, input->data, input->size);
        const Value argument(context->argument.get(), ValueKind::Vector);
        const auto result = invoke(*context, std::span(&argument, 1));
        if (!result) {
            context->fail(result.error());
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        const auto* vector =
            checked_vector(*context, *result, context->result_size);
        if (!vector) return LMMC_STATUS_DIMENSION_MISMATCH;
        if (!output->data) {
            const auto status = lmmc_vec_create(context->result_size, output);
            if (status != LMMC_STATUS_OK) return status;
        }
        if (output->size != context->result_size) {
            context->fail("native vector callback output dimension mismatch");
            return LMMC_STATUS_DIMENSION_MISMATCH;
        }
        std::copy(vector->data().begin(), vector->data().end(), output->data);
        return LMMC_STATUS_OK;
    } catch (const std::exception& error) {
        context->fail(error.what());
    } catch (...) {
        context->fail("unknown Lamina callback failure");
    }
    return LMMC_STATUS_NUMERICAL_FAILURE;
}

lmmc_status_t matrix_callback_trampoline(
    const lmmc_vec_t* input, lmmc_mat_t* output, void* user_data) noexcept {
    auto* context = static_cast<MatrixCallbackContext*>(user_data);
    if (!context || !input || !input->data || !output) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    try {
        load_argument(context->argument, input->data, input->size);
        const Value argument(context->argument.get(), ValueKind::Vector);
        const auto result = invoke(*context, std::span(&argument, 1));
        if (!result) {
            context->fail(result.error());
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (result->kind != ValueKind::Matrix || !result->obj ||
            result->obj->get_kind() != lmx::runtime::ObjectKind::Matrix) {
            context->fail("Lamina callback returned a non-matrix value");
            return LMMC_STATUS_DIMENSION_MISMATCH;
        }
        const auto* matrix = static_cast<const MatrixObj*>(result->obj);
        if (matrix->rows() != context->rows ||
            matrix->cols() != context->cols || !matrix->valid()) {
            context->fail("Lamina callback returned an invalid matrix dimension");
            return LMMC_STATUS_DIMENSION_MISMATCH;
        }
        if (!finite_vector(matrix->data())) {
            context->fail("Lamina callback returned a non-finite matrix");
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (!output->data) {
            const auto status =
                lmmc_mat_create(context->rows, context->cols, output);
            if (status != LMMC_STATUS_OK) return status;
        }
        if (output->rows != context->rows || output->cols != context->cols ||
            output->stride < context->cols) {
            context->fail("native matrix callback output dimension mismatch");
            return LMMC_STATUS_DIMENSION_MISMATCH;
        }
        for (std::size_t row = 0; row < context->rows; ++row) {
            std::copy_n(matrix->data().data() + row * context->cols,
                        context->cols, output->data + row * output->stride);
        }
        return LMMC_STATUS_OK;
    } catch (const std::exception& error) {
        context->fail(error.what());
    } catch (...) {
        context->fail("unknown Lamina callback failure");
    }
    return LMMC_STATUS_NUMERICAL_FAILURE;
}
double vector_scalar_callback_trampoline(
    const lmmc_vec_t* input, void* user_data) noexcept {
    auto* context = static_cast<VectorScalarCallbackContext*>(user_data);
    if (!context || !input || !input->data)
        return std::numeric_limits<double>::quiet_NaN();
    if (context->failed()) return 0.0;
    try {
        load_argument(context->argument, input->data, input->size);
        const Value argument(context->argument.get(), ValueKind::Vector);
        const auto result = invoke(*context, std::span(&argument, 1));
        if (!result) {
            context->fail(result.error());
            return 0.0;
        }
        if (result->kind != ValueKind::Real ||
            !std::isfinite(result->real_val)) {
            context->fail("Lamina callback returned an invalid real value");
            return 0.0;
        }
        return result->real_val;
    } catch (const std::exception& error) {
        context->fail(error.what());
    } catch (...) {
        context->fail("unknown Lamina callback failure");
    }
    return 0.0;
}


lmmc_status_t ode_callback_trampoline(
    const double time, const double* state, double* derivative,
    const std::size_t dimension, void* user_data) noexcept {
    auto* context = static_cast<OdeCallbackContext*>(user_data);
    if (!context || !state || !derivative || dimension != context->dimension) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    try {
        load_argument(context->argument, state, dimension);
        const Value arguments[] = {
            Value(time),
            Value(context->argument.get(), ValueKind::Vector),
        };
        const auto result = invoke(*context, arguments);
        if (!result) {
            context->fail(result.error());
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        const auto* vector = checked_vector(*context, *result, dimension);
        if (!vector) return LMMC_STATUS_DIMENSION_MISMATCH;
        std::copy(vector->data().begin(), vector->data().end(), derivative);
        return LMMC_STATUS_OK;
    } catch (const std::exception& error) {
        context->fail(error.what());
    } catch (...) {
        context->fail("unknown Lamina callback failure");
    }
    return LMMC_STATUS_NUMERICAL_FAILURE;
}
lmmc_status_t ode_matrix_callback_trampoline(
    const double time, const double* state, double* jacobian,
    const std::size_t dimension, void* user_data) noexcept {
    auto* context = static_cast<OdeMatrixCallbackContext*>(user_data);
    if (!context || !state || !jacobian ||
        dimension != context->dimension)
        return LMMC_STATUS_INVALID_ARGUMENT;
    try {
        load_argument(context->argument, state, dimension);
        const Value arguments[] = {
            Value(time),
            Value(context->argument.get(), ValueKind::Vector),
        };
        const auto result = invoke(*context, arguments);
        if (!result) {
            context->fail(result.error());
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (result->kind != ValueKind::Matrix || !result->obj ||
            result->obj->get_kind() != lmx::runtime::ObjectKind::Matrix) {
            context->fail("Lamina callback returned a non-matrix Jacobian");
            return LMMC_STATUS_DIMENSION_MISMATCH;
        }
        const auto* matrix = static_cast<const MatrixObj*>(result->obj);
        if (!matrix->valid() || matrix->rows() != dimension ||
            matrix->cols() != dimension || !finite_vector(matrix->data())) {
            context->fail("Lamina callback returned an invalid Jacobian");
            return LMMC_STATUS_DIMENSION_MISMATCH;
        }
        std::copy(matrix->data().begin(), matrix->data().end(), jacobian);
        return LMMC_STATUS_OK;
    } catch (const std::exception& error) {
        context->fail(error.what());
    } catch (...) {
        context->fail("unknown Lamina callback failure");
    }
    return LMMC_STATUS_NUMERICAL_FAILURE;
}


} // namespace lmx::bridge
