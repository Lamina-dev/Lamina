#pragma once
#include "../value.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifdef LAMINA_CORE_EXPORTS
#define LAMINA_API __declspec(dllexport)
#else
#define LAMINA_API __declspec(dllimport)
#endif
#else
#define LAMINA_API
#endif

/**
 * @brief Base class for compute backends
 *
 * This class defines the interface for compute backends that can accelerate
 * mathematical operations using different hardware (CPU, GPU, etc.)
 */
class LAMINA_API ComputeBackend {
public:
    using BackendFunction = std::function<Value(const std::vector<Value>&)>;

    virtual ~ComputeBackend() = default;

    /**
     * @brief Get the name of this backend
     */
    virtual std::string name() const = 0;

    /**
     * @brief Initialize the backend
     * @return true if initialization succeeded
     */
    virtual bool initialize() = 0;

    /**
     * @brief Check if the backend is available on this system
     */
    virtual bool is_available() const = 0;

    /**
     * @brief Call a function on this backend
     * @param func_name Name of the function to call
     * @param args Arguments to pass to the function
     * @return Result value
     */
    virtual Value call_function(const std::string& func_name, const std::vector<Value>& args) = 0;

    /**
     * @brief Get a list of all available functions on this backend
     */
    virtual std::vector<std::string> available_functions() const = 0;

    /**
     * @brief Check if a function is available on this backend
     */
    virtual bool has_function(const std::string& func_name) const = 0;

protected:
    std::unordered_map<std::string, BackendFunction> functions_;
};

/**
 * @brief Backend manager
 *
 * Manages all available compute backends and provides a unified interface
 * for backend registration and function calls.
 */
class LAMINA_API BackendManager {
public:
    static BackendManager& instance();

    /**
     * @brief Register a compute backend
     */
    void register_backend(std::shared_ptr<ComputeBackend> backend);

    /**
     * @brief Get a backend by name
     */
    std::shared_ptr<ComputeBackend> get_backend(const std::string& name) const;

    /**
     * @brief Check if a backend is registered
     */
    bool has_backend(const std::string& name) const;

    /**
     * @brief Get a list of all registered backends
     */
    std::vector<std::string> available_backends() const;

    /**
     * @brief Set the default backend for a scope
     */
    void push_default_backend(const std::string& name);

    /**
     * @brief Restore the previous default backend
     */
    void pop_default_backend();

    /**
     * @brief Get the current default backend
     */
    std::string current_default_backend() const;

private:
    BackendManager() = default;
    std::unordered_map<std::string, std::shared_ptr<ComputeBackend>> backends_;
    std::vector<std::string> default_backend_stack_{"cpu"};
};
