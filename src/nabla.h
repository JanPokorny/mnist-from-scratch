#ifndef PV021_PROJECT_NABLA_H
#define PV021_PROJECT_NABLA_H

#include <numeric>
#include <memory>
#include <thread>

#include "types.h"
#include "vec_ops.h"
#include "math_fn.h"
#include "layers.h"

template<typename network_type>
struct Nabla {
    using tail_nabla_type = typename network_type::tail_network_type::nabla_type;
    tail_nabla_type tail_nabla = {};
    typename network_type::output_type biases = {};
    typename network_type::weights_type weights = {};
};



#endif //PV021_PROJECT_NABLA_H
