#include "backend_interface.hpp"
#include <iostream>

BackendManager& BackendManager::instance() {
    static BackendManager instance;
    return instance;
}

void BackendManager::register_backend(std::shared_ptr<ComputeBackend> backend) {
    if (backend && backend->initialize()) {
        backends_[backend->name()] = backend;
        std::cout << "Registered compute backend: " << backend->name() << std::endl;
    } else {
        std::cerr << "Failed to register backend" << std::endl;
    }
}

std::shared_ptr<ComputeBackend> BackendManager::get_backend(const std::string& name) const {
    auto it = backends_.find(name);
    if (it != backends_.end()) {
        return it->second;
    }
    return nullptr;
}

bool BackendManager::has_backend(const std::string& name) const {
    return backends_.find(name) != backends_.end();
}

std::vector<std::string> BackendManager::available_backends() const {
    std::vector<std::string> result;
    for (const auto& pair : backends_) {
        result.push_back(pair.first);
    }
    return result;
}

void BackendManager::push_default_backend(const std::string& name) {
    if (has_backend(name)) {
        default_backend_stack_.push_back(name);
    } else {
        std::cerr << "Warning: Backend '" << name << "' not found, keeping current default" << std::endl;
    }
}

void BackendManager::pop_default_backend() {
    if (default_backend_stack_.size() > 1) {
        default_backend_stack_.pop_back();
    }
}

std::string BackendManager::current_default_backend() const {
    return default_backend_stack_.back();
}
