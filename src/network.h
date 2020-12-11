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

template<typename... args>
class Network;

template<typename previous_layer, typename current_layer, typename ...args>
class Network<previous_layer, current_layer, args...> {
public:
    using tail_network_type = Network<current_layer, args...>;

    using input_type = vec<previous_layer::size>;
    using output_type = vec<current_layer::size>;
    using final_output_type = typename tail_network_type::final_output_type;
    using weights_type = mat<previous_layer::size, current_layer::size>;

    tail_network_type tail_network;

    output_type biases = {};
    weights_type weights = {};

    output_type nabla_biases = {};
    weights_type nabla_weights = {};

    std::shared_ptr<std::default_random_engine> random_engine_ptr;

    explicit Network(size_t seed) : Network(std::make_shared<std::default_random_engine>(seed)) {}

    explicit Network(std::shared_ptr<std::default_random_engine> const &random_engine_ptr)
            : random_engine_ptr(random_engine_ptr),
              tail_network(random_engine_ptr) {
        std::normal_distribution<number> distribution;

        if constexpr(!current_layer::is_input)
            for (number &b : biases)
                b = distribution(*random_engine_ptr);

        for (number &w : weights)
            w = distribution(*random_engine_ptr);
    }

    void print_to(std::ostream &out) const {
        out << "LAYER size " << current_layer::size << std::endl;
        out << "Biases: " << biases << std::endl;
        out << "Weights: " << std::endl << weights << std::endl;
        tail_network.print_to(out);
    }

    auto feedforward(input_type const &a) const {
        return tail_network.feedforward(vec_map<sigmoid>(dot(weights, a) + biases));
    }

    size_t predict(input_type const &a) const {
        return argmax(feedforward(a));
    }

    std::vector<size_t> predict(std::vector<input_type> const &as) const {
        std::vector<size_t> predicted_labels(as.size());
        for (size_t i = 0; i < as.size(); i++)
            predicted_labels[i] = predict(as[i]);
        return predicted_labels;
    }

    number evaluate_accuracy(std::vector<input_type> const &xs, std::vector<size_t> const &ys) const {
        auto predicted_labels = predict(xs);
        size_t predicted_correctly = 0;
        for (size_t i = 0; i < xs.size(); i++)
            predicted_correctly += predicted_labels[i] == ys[i] ? 1 : 0;
        return (number) predicted_correctly / xs.size();
    }

    output_type backprop(input_type const& activation, final_output_type const& y) {
        output_type z = dot(weights, activation) + biases;
        output_type next_activation = vec_map<sigmoid>(z);

        output_type delta;
        if constexpr (!current_layer::is_output) {
            auto next_delta = tail_network.backprop(next_activation, y);
            delta = dot_t(tail_network.weights, next_delta) * vec_map<sigmoid_prime>(z);
        } else {
            delta = (next_activation - y) * vec_map<sigmoid_prime>(z);
        }

#pragma omp critical
        nabla_biases = nabla_biases + delta;
#pragma omp critical
        nabla_weights = nabla_weights + dot(delta, activation);

        return delta;
    }

    void update_weights(number eta_piece) {
        weights = weights - nabla_weights * eta_piece;
        biases = biases - nabla_biases * eta_piece;
        nabla_weights = {};
        nabla_biases = {};
        tail_network.update_weights(eta_piece);
    }

    template<size_t mini_batch_size>
    void SGD(std::ostream &out, std::vector<input_type> const &xs, std::vector<label_type> const &ys,
             std::vector<input_type> const &test_xs, std::vector<label_type> const &test_ys, size_t epochs,
             number eta) {
        std::vector<size_t> idx(xs.size());
        std::iota(idx.begin(), idx.end(), 0);
        for (size_t epoch = 0; epoch < epochs; epoch++) {
            auto start_clock = std::chrono::high_resolution_clock::now();
            std::shuffle(idx.begin(), idx.end(), *random_engine_ptr);
            for(size_t mini_batch_start = 0; mini_batch_start < xs.size(); mini_batch_start += mini_batch_size) {
                size_t mini_batch_end = std::min(mini_batch_start + mini_batch_size, xs.size());
#pragma omp parallel for num_threads(4) default(none) shared(idx, xs, ys, mini_batch_start, mini_batch_end)
                for (size_t i = mini_batch_start; i < mini_batch_end; i++) {
                    backprop(xs[idx[i]], onehot<final_output_type::rows>(ys[idx[i]]));
                }
                update_weights(eta / mini_batch_size);
            }

            auto end_clock = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end_clock - start_clock;
            out << "Epoch " << epoch << " | train_acc: " << evaluate_accuracy(xs, ys) << ", test_acc: "
                << evaluate_accuracy(test_xs, test_ys) << ", took " << elapsed_seconds.count() << "s" << std::endl;
        }
    }

};

template<typename output_layer>
class Network<output_layer> {
public:
    using final_output_type = vec<output_layer::size>;

    explicit Network(__attribute__((unused)) std::shared_ptr<std::default_random_engine> const &random_engine) {};

    void print_to(std::ostream &out) const {
        out << "OUTPUT LAYER size " << output_layer::size << std::endl;
    };

    vec<output_layer::size> feedforward(vec<output_layer::size> a) const { return a; }

    void update_weights(__attribute__((unused)) number eta_piece) {}
};

template<typename ...args>
std::ostream &operator<<(std::ostream &out, const Network<args...> &network) {
    network.print_to(out);
    return out;
}


#endif //PV021_PROJECT_NETWORK_H
