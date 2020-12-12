#ifndef PV021_PROJECT_MATH_FN_H
#define PV021_PROJECT_MATH_FN_H

#include <cmath>

number sigmoid(number z) {
    return 1.0 / (1.0 + std::exp(-z));
}

number sigmoid_prime(number z) {
    return sigmoid(z) * (1 - sigmoid(z));
}

number relu(number z) {
    return std::max(number(0), z);
}

number relu_prime(number z) {
    return number(z > 0);
}

template<size_t R>
vec<R> softmax(vec<R> z) {
    vec<R> result = {};
    number z_max = std::max(z.begin(), z.end());
    for (size_t i = 0; i < R; i++)
        result[i] = std::exp(z[i] - z_max);
    number sum = std::accumulate(z.begin(), z.end(), vec<R>());
    for (size_t i = 0; i < R; i++)
        result[i] /= sum;
    return std::move(result);
}

#endif //PV021_PROJECT_MATH_FN_H
