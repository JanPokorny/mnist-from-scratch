#ifndef PV021_PROJECT_NABLA_H
#define PV021_PROJECT_NABLA_H

#include <new>

#include "types.h"

template<typename network_type>
struct alignas(128) Nabla {
    using tail_nabla_type = typename network_type::tail_network_type::nabla_type;
    typename network_type::output_type biases = {};
    typename network_type::weights_type weights = {};
    tail_nabla_type tail_nabla = {};

    constexpr Nabla operator+(Nabla const& other) const {
        Nabla result = {};
        result.biases = this->biases + other.biases;
        result.weights = this->weights + other.weights;
        result.tail_nabla = this->tail_nabla + other.tail_nabla;
        return std::move(result);
    }
};

#endif //PV021_PROJECT_NABLA_H
