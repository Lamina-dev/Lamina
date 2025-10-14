#pragma once
#include "../backend_interface.hpp"

/**
 * @brief CPU compute backend (default)
 *
 * This backend uses CPU for all computations. It serves as the default
 * fallback when specialized backends are not available.
 */
class LAMINA_API CPUBackend : public ComputeBackend {
public:
    CPUBackend();
    ~CPUBackend() override = default;

    std::string name() const override { return "cpu"; }
    bool initialize() override;
    bool is_available() const override { return true; }
    Value call_function(const std::string& func_name, const std::vector<Value>& args) override;
    std::vector<std::string> available_functions() const override;
    bool has_function(const std::string& func_name) const override;

private:
    void register_functions();

    // CPU implementation of operations
    Value matmul(const std::vector<Value>& args);
    Value add(const std::vector<Value>& args);
    Value sub(const std::vector<Value>& args);
    Value mul(const std::vector<Value>& args);
    Value div(const std::vector<Value>& args);
    Value dot(const std::vector<Value>& args);
    Value cross(const std::vector<Value>& args);
    Value transpose(const std::vector<Value>& args);
};
