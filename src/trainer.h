#ifndef PV021_PROJECT_TRAINER_H
#define PV021_PROJECT_TRAINER_H

template<typename network_type>
struct Trainer {
    using input_type = typename network_type::input_type;
    using nabla_type = typename network_type::nabla_type;
    using final_output_type = typename network_type::final_output_type;

    network_type &network;
    std::vector<input_type> const &train_xs;
    std::vector<label_type> const &train_ys;

    template<size_t mini_batch_size>
    void SGD_full(std::default_random_engine &random_engine, size_t epochs, number eta, double eta_decrease_rate) {
        std::vector<size_t> idx(train_xs.size());
        std::iota(idx.begin(), idx.end(), 0);
        double total_elapsed_seconds = 0.0;
        for (size_t epoch = 0; epoch < epochs; epoch++) {
            std::cerr << "Epoch " << epoch;
            auto start_clock = std::chrono::high_resolution_clock::now();
            std::shuffle(idx.begin(), idx.end(), random_engine);
            for (size_t mini_batch_start = 0; mini_batch_start < train_xs.size(); mini_batch_start += mini_batch_size) {
                nabla_type nabla = {};
                size_t mini_batch_end = std::min(mini_batch_start + mini_batch_size, train_xs.size());
                for (size_t i = mini_batch_start; i < mini_batch_end; i++) {
                    network.backprop(nabla, train_xs[idx[i]], onehot<final_output_type::rows>(train_ys[idx[i]]));
                }
                network.update_weights(nabla, eta / mini_batch_size);
                eta = eta_decrease_rate * eta;
            }
            auto end_clock = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end_clock - start_clock;
            total_elapsed_seconds += elapsed_seconds.count();
            std::cerr << " @ " << total_elapsed_seconds << "s" << std::endl;
        }
    }
};

#endif //PV021_PROJECT_TRAINER_H
