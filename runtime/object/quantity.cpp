#include "quantity.hpp"

#include "lmmc/lsr_stdlib.h"

#include <functional>
#include <sstream>

namespace lmx::runtime {

bool QuantityObj::equals(const QuantityObj& other) const noexcept {
    if (si_value_ != other.si_value_) return false;
    double ignored = 0.0;
    return lmmc_lsr_units_convert(1.0, unit_.c_str(), other.unit_.c_str(),
                                  &ignored) == LMMC_STATUS_OK;
}

std::size_t QuantityObj::hash() const noexcept {
    return std::hash<double>{}(si_value_ == 0.0 ? 0.0 : si_value_);
}

std::string QuantityObj::to_string() const noexcept {
    double displayed = si_value_;
    if (lmmc_lsr_units_convert_from_si(si_value_, unit_.c_str(), &displayed) !=
        LMMC_STATUS_OK) {
        displayed = si_value_;
    }
    std::ostringstream out;
    out << displayed;
    if (unit_ != "1") out << ' ' << unit_;
    return out.str();
}

}
