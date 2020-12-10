#ifndef PV021_PROJECT_MATH_FN_H
#define PV021_PROJECT_MATH_FN_H

#include <cmath>

number sigmoid(number z) {
    return 1.0 / (1.0 + std::exp(-z));
}

number sigmoid_prime(number z) {
    return sigmoid(z) * (1 - sigmoid(z));
}

#endif //PV021_PROJECT_MATH_FN_H
