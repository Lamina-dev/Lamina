#include "random.hpp"

#include "lmmc/random.h"

namespace lmx::runtime {

RandomObj::~RandomObj() noexcept {
    lmmc_rng_destroy(rng_);
    rng_ = nullptr;
}

}
