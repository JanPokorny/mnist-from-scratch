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

    template<size_t mini_batch_size>
    void SGD_full(std::default_random_engine &random_engine, size_t epochs, number eta) {
        std::vector<size_t> idx(train_xs.size());
        std::iota(idx.begin(), idx.end(), 0);
        for (size_t epoch = 0; epoch < epochs; epoch++) {
            auto start_clock = std::chrono::high_resolution_clock::now();
            std::shuffle(idx.begin(), idx.end(), random_engine);
            for(size_t mini_batch_start = 0; mini_batch_start < train_xs.size(); mini_batch_start += mini_batch_size) {
                nabla_type nabla = {};
                size_t mini_batch_end = std::min(mini_batch_start + mini_batch_size, train_xs.size());
                for (size_t i = mini_batch_start; i < mini_batch_end; i++) {
                    network.backprop(nabla, train_xs[idx[i]], onehot<final_output_type::rows>(train_ys[idx[i]]));
                }
                network.update_weights(nabla, eta / mini_batch_size);
            }

            auto end_clock = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end_clock - start_clock;
            std::cerr << "Epoch " << epoch << " | train_acc: " << evaluate_accuracy(train_xs, train_ys) << ", test_acc: "
                      << evaluate_accuracy(test_xs, test_ys) << ", took " << elapsed_seconds.count() << "s" << std::endl;
        }
    }

    template<size_t mini_batch_size>
    void SGD_dumb(std::default_random_engine &random_engine, size_t epochs, number eta) {
        std::uniform_int_distribution<size_t> train_distribution(0, train_xs.size() - 1);
        for (size_t epoch = 0; epoch < epochs; epoch++) {
            auto start_clock = std::chrono::high_resolution_clock::now();
            for(size_t mini_batch_start = 0; mini_batch_start < train_xs.size(); mini_batch_start += mini_batch_size) {
                nabla_type nabla = {};
                for (size_t i = 0; i < mini_batch_size; i++) {
                    size_t idx = train_distribution(random_engine);
                    network.backprop(nabla, train_xs[idx], onehot<final_output_type::rows>(train_ys[idx]));
                }
                network.update_weights(nabla, eta / mini_batch_size);
            }

            auto end_clock = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end_clock - start_clock;
            std::cerr << "Epoch " << epoch << " | train_acc: " << evaluate_accuracy(train_xs, train_ys) << ", test_acc: "
                      << evaluate_accuracy(test_xs, test_ys) << ", took " << elapsed_seconds.count() << "s" << std::endl;
        }
    }

    template<size_t mini_batch_size>
    void SGD_parallel(std::default_random_engine &random_engine, size_t epochs, number eta) {
        size_t num_threads = std::thread::hardware_concurrency();
        if(num_threads == 0)
            num_threads = 8;

        size_t mini_batch_items_remaining = 0;
        size_t mini_batch_items_done = 0;
        bool done = false;
        std::vector<nabla_type> nablas(num_threads);
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
                    mini_batch_items_remaining--;
                }
                size_t idx = train_distribution(random_engine);
                network.backprop(nablas[id], train_xs[idx], onehot<final_output_type::rows>(train_ys[idx]));
                {
                    std::unique_lock<std::mutex> lk(m);
                    mini_batch_items_done++;
                }
                cv_boss.notify_all();
            }
        };

        std::vector<std::thread> threads(num_threads);
        for(size_t i = 0; i < num_threads; i++) {
            threads[i] = std::thread(thread_work, i, random_engine());
        }

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
                network.update_weights(std::accumulate(nablas.begin(), nablas.end(), nabla_type()), eta / mini_batch_size);
                for(auto &nabla : nablas)
                    nabla = {};
            }
            auto end_clock = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end_clock - start_clock;
            std::cerr << " done in " << elapsed_seconds.count() << "s | train_acc: " << evaluate_accuracy(train_xs, train_ys) << ", test_acc: " << evaluate_accuracy(test_xs, test_ys) << std::endl;
        }

        done =true;
        cv_worker.notify_all();
        for(size_t i = 0; i < num_threads; i++) {
            threads[i].join();
        }
    }
};

#endif //PV021_PROJECT_TRAINER_H
