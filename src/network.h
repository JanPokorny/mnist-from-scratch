#ifndef PV021_PROJECT_NETWORK_H
#define PV021_PROJECT_NETWORK_H

#include <random>
#include <numeric>
#include <memory>
#include <thread>

#include "types.h"
#include "vec_ops.h"
#include "math_fn.h"
#include "layers.h"
#include "nabla.h"

template<typename... args>
struct Network;

template<typename previous_layer, typename current_layer, typename ...args>
struct Network<previous_layer, current_layer, args...> {
    using tail_network_type = Network<current_layer, args...>;
    using nabla_type = Nabla<Network<previous_layer, current_layer, args...>>;

    using input_type = vec<previous_layer::size>;
    using output_type = vec<current_layer::size>;
    using final_output_type = typename tail_network_type::final_output_type;
    using weights_type = mat<previous_layer::size, current_layer::size>;

    tail_network_type tail_network;

    output_type biases = {};
    weights_type weights = {};

    explicit Network(std::default_random_engine &random_engine) : tail_network(random_engine) {
        std::normal_distribution<number> distribution;

        if constexpr(!current_layer::is_input)
            for (number &b : biases)
                b = distribution(random_engine);

        for (number &w : weights)
            w = distribution(random_engine);
    }

    void print_to(std::ostream &out) const {
        out << "LAYER size " << current_layer::size << std::endl;
        out << "Biases: " << biases << std::endl;
        out << "Weights: " << std::endl << weights << std::endl;
        tail_network.print_to(out);
    }

    auto feedforward(input_type const &a) const {
        return tail_network.feedforward(current_layer::activation_fn(dot(weights, a) + biases));
    }

    size_t predict(input_type const &a) const {
        return argmax(feedforward(a));
    }

    std::vector<size_t> predict(std::vector<input_type> const &as) const {
        std::vector<size_t> predicted_labels(as.size());
        for (size_t i = 0; i < as.size(); i++)
            predicted_labels[i] = predict(as[i]);
        return std::move(predicted_labels);
    }

    output_type backprop(nabla_type &nabla, input_type const& prev_activation, final_output_type const& y) const {
        output_type z = dot(weights, prev_activation) + biases;

        output_type delta;
        if constexpr (current_layer::is_output) {
            delta = (current_layer::activation_fn(z) - y) * current_layer::activation_fn_prime(z);
        } else {
            auto next_delta = tail_network.backprop(nabla.tail_nabla, current_layer::activation_fn(z), y);
            delta = dot_t(tail_network.weights, next_delta) * current_layer::activation_fn_prime(z);
        }

        nabla.biases = nabla.biases + delta;
        nabla.weights = nabla.weights + dot(delta, prev_activation);

        return delta;
    }

    void update_weights(nabla_type nabla, number eta_piece) {
        weights = weights - nabla.weights * eta_piece;
        biases = biases - nabla.biases * eta_piece;
        tail_network.update_weights(nabla.tail_nabla, eta_piece);
    }
};

template<typename output_layer>
struct Network<output_layer> {
    using nabla_type = char;
    using final_output_type = vec<output_layer::size>;

    explicit Network(__attribute__((unused)) std::default_random_engine const &random_engine) {}

    void print_to(std::ostream &out) const {
        out << "OUTPUT LAYER size " << output_layer::size << std::endl;
    };

    vec<output_layer::size> feedforward(vec<output_layer::size> a) const { return a; }

    void update_weights(__attribute__((unused)) nabla_type tail_nabla, __attribute__((unused)) number eta_piece) {}
};

template<typename ...args>
std::ostream &operator<<(std::ostream &out, const Network<args...> &network) {
    network.print_to(out);
    return out;
}


#endif //PV021_PROJECT_NETWORK_H
