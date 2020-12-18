#ifndef PV021_PROJECT_TRAINER_H
#define PV021_PROJECT_TRAINER_H

#include <atomic>
#include <condition_variable>

using namespace std::chrono_literals;

template<typename network_type>
struct Trainer {
    using input_type = typename network_type::input_type;
    using nabla_type = typename network_type::nabla_type;
    using final_output_type = typename network_type::final_output_type;

    network_type &network;
    std::vector<input_type> const &train_xs;
    std::vector<label_type> const &train_ys;
    std::vector<input_type> const &test_xs;
    std::vector<label_type> const &test_ys;

    number evaluate_accuracy(std::vector<input_type> const &xs, std::vector<size_t> const &ys) const {
        size_t predicted_correctly = 0;
        for (size_t i = 0; i < xs.size(); i++)
            predicted_correctly += network.predict(xs[i]) == ys[i] ? 1 : 0;
        return (number) predicted_correctly / xs.size();
    }

//lambda added
    template<size_t mini_batch_size>
    void SGD_full(std::default_random_engine &random_engine, size_t epochs, number eta, number lambda) {
        std::vector<size_t> idx(train_xs.size());
        std::iota(idx.begin(), idx.end(), 0);
        double total_elapsed_seconds = 0.0;
	double eta_decrease_rate = 0.99999;
	std::cerr << "eta decrease rate " << eta_decrease_rate << std::endl;
        for (size_t epoch = 0; epoch < epochs; epoch++) {
            std::cerr << "Epoch " << epoch;
            auto start_clock = std::chrono::high_resolution_clock::now();
            std::shuffle(idx.begin(), idx.end(), random_engine);
            for(size_t mini_batch_start = 0; mini_batch_start < train_xs.size(); mini_batch_start += mini_batch_size) {
                nabla_type nabla = {};
                size_t mini_batch_end = std::min(mini_batch_start + mini_batch_size, train_xs.size());
                for (size_t i = mini_batch_start; i < mini_batch_end; i++) {
                    network.backprop(nabla, train_xs[idx[i]], onehot<final_output_type::rows>(train_ys[idx[i]]));
                }
                network.update_weights(nabla, eta / mini_batch_size, lambda);
                eta = eta_decrease_rate * eta;
            }
            auto end_clock = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end_clock - start_clock;
            total_elapsed_seconds += elapsed_seconds.count();
            double test_accuracy = evaluate_accuracy(test_xs, test_ys);
            std::cerr << " | test_acc=" << test_accuracy << " @ " << total_elapsed_seconds << "s" << "| eta " << eta << " lambda " << lambda << std::endl;
        }
    }

    template<size_t mini_batch_size>
    void SGD_dumb(std::default_random_engine &random_engine, size_t epochs, number eta, number lambda) {
        std::uniform_int_distribution<size_t> train_distribution(0, train_xs.size() - 1);
        double total_elapsed_seconds = 0.0;
        for (size_t epoch = 0; epoch < epochs; epoch++) {
            std::cerr << "Epoch " << epoch;
            auto start_clock = std::chrono::high_resolution_clock::now();
            for(size_t mini_batch_start = 0; mini_batch_start < train_xs.size(); mini_batch_start += mini_batch_size) {
                nabla_type nabla = {};
                for (size_t i = 0; i < mini_batch_size; i++) {
                    size_t idx = train_distribution(random_engine);
                    network.backprop(nabla, train_xs[idx], onehot<final_output_type::rows>(train_ys[idx]));
                }
                network.update_weights(nabla, eta / mini_batch_size, lambda);
            }

            auto end_clock = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end_clock - start_clock;
            total_elapsed_seconds += elapsed_seconds.count();
            double test_accuracy = evaluate_accuracy(test_xs, test_ys);
            std::cerr << " | test_acc=" << test_accuracy << " @ " << total_elapsed_seconds << "s" << std::endl;
        }
    }

    // TODO: fix, learns slower compared to full / dumb, investigate
    template<size_t mini_batch_size>
    void SGD_parallel(std::default_random_engine &random_engine, size_t epochs, number eta, number lambda) {
        constexpr size_t num_threads = 2;
        constexpr size_t thread_batch_size = mini_batch_size / num_threads;

        if(mini_batch_size % num_threads > 0) throw;

        size_t mini_batch_items_remaining = 0;
        size_t mini_batch_items_done = 0;
        bool done = false;
        std::array<nabla_type, num_threads> nablas;
        std::mutex m;
        std::condition_variable cv_worker;
        std::condition_variable cv_boss;

        auto thread_work = [&, this] (size_t id, size_t seed) {
            std::default_random_engine random_engine(seed);
            std::uniform_int_distribution<size_t> train_distribution(0, train_xs.size() - 1);

            while(!done) {
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv_worker.wait(lk, [&] { return mini_batch_items_remaining > 0 || done; });
                    if (done) return;
                    mini_batch_items_remaining -= thread_batch_size;
                }
                for(size_t i = 0; i < thread_batch_size; i++) {
                    size_t idx = train_distribution(random_engine);
                    network.backprop(nablas[id], train_xs[idx], onehot<final_output_type::rows>(train_ys[idx]));
                }
                {
                    std::unique_lock<std::mutex> lk(m);
                    mini_batch_items_done += thread_batch_size;
                }
                cv_boss.notify_all();
            }
        };

        std::vector<std::thread> threads(num_threads);
        for(size_t i = 0; i < num_threads; i++) {
            threads[i] = std::thread(thread_work, i, random_engine());
        }

        double total_elapsed_seconds = 0.0;
        for (size_t epoch = 0; epoch < epochs; epoch++) {
            std::cerr << "Epoch " << epoch;
            auto start_clock = std::chrono::high_resolution_clock::now();
            size_t mini_batches_per_epoch = train_xs.size() / mini_batch_size;
            for(size_t i = 0; i < mini_batches_per_epoch; i++) {
                std::unique_lock<std::mutex> lk(m);
                mini_batch_items_done = 0;
                mini_batch_items_remaining = mini_batch_size;
                cv_worker.notify_all();
                cv_boss.wait(lk, [&]{return mini_batch_items_done == mini_batch_size;});
                network.update_weights(std::accumulate(nablas.begin(), nablas.end(), nabla_type()), eta / mini_batch_size, lambda);
                for(auto &nabla : nablas)
                    nabla = {};
            }
            auto end_clock = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = std::chrono::duration<double>(end_clock - start_clock).count();
            total_elapsed_seconds += elapsed_seconds;
            double test_accuracy = evaluate_accuracy(test_xs, test_ys);
            std::cerr << " | test_acc=" << test_accuracy << " @ " << total_elapsed_seconds << "s" << std::endl;
        }

        done = true;
        cv_worker.notify_all();
        for(size_t i = 0; i < num_threads; i++) {
            threads[i].join();
        }
    }
};

#endif //PV021_PROJECT_TRAINER_H
