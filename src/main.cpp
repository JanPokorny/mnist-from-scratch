#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

#include "network.h"
#include "io.h"
#include "trainer.h"

constexpr size_t input_size = 28 * 28;
constexpr size_t output_size = 10;

constexpr char train_images_path[] = "../data/fashion_mnist_train_vectors.csv";
constexpr char train_labels_path[] = "../data/fashion_mnist_train_labels.csv";

constexpr char test_images_path[] = "../data/fashion_mnist_test_vectors.csv";
constexpr char test_labels_path[] = "../data/fashion_mnist_test_labels.csv";

constexpr char predicted_train_labels_path[] = "../out/trainPredictions";
constexpr char predicted_test_labels_path[] = "../out/actualTestPredictions";

int main() {
    omp_set_num_threads(4);

    auto start_clock = std::chrono::high_resolution_clock::now();
    auto start_time = std::chrono::high_resolution_clock::to_time_t(start_clock);
    std::cerr << "Program started at " << std::ctime(&start_time);

    auto random_engine = std::default_random_engine(42); // NOLINT(cert-msc51-cpp)

    std::cerr << "Loading train images..." << std::endl;
    std::ifstream train_images_infile(train_images_path);
    std::vector<vec<input_size>> train_images = load_images<input_size>(train_images_infile);
    train_images_infile.close();

    std::cerr << "Loading train labels..." << std::endl;
    std::ifstream train_labels_infile(train_labels_path);
    std::vector<label_type> train_labels = load_labels(train_labels_infile);
    train_labels_infile.close();

    std::cerr << "Training the network..." << std::endl;
    Network<InputLayer<input_size>, HiddenLayer<64>, OutputLayer<output_size>> network(random_engine);
    Trainer<decltype(network)> trainer {
            network,
            train_images,
            train_labels
    };

    size_t epochs = 28;
    constexpr size_t batch_size = 64;
    number eta_orig = 0.15;
	double eta_decrease_rate = 0.99999;

    trainer.SGD_full<batch_size>(random_engine, epochs, eta_orig, eta_decrease_rate);

    std::cerr << "Inferring train predictions..." << std::endl;
    std::vector<label_type> predicted_train_labels = network.predict(train_images);

    std::cerr << "Writing train predictions..." << std::endl;
    std::ofstream predicted_train_labels_outfile(predicted_train_labels_path);
    save_labels(predicted_train_labels_outfile, predicted_train_labels);
    predicted_train_labels_outfile.close();

    std::cerr << "Loading test images..." << std::endl;
    std::ifstream test_images_infile(test_images_path);
    std::vector<vec<input_size>> test_images = load_images<input_size>(test_images_infile);
    test_images_infile.close();

    std::cerr << "Loading test labels..." << std::endl;
    std::ifstream test_labels_infile(test_labels_path);
    std::vector<label_type> test_labels = load_labels(test_labels_infile);
    test_labels_infile.close();

    std::cerr << "Inferring test predictions..." << std::endl;
    std::vector<label_type> predicted_test_labels = network.predict(test_images);

    std::cerr << "Writing test predictions..." << std::endl;
    std::ofstream predicted_test_labels_outfile(predicted_test_labels_path);
    save_labels(predicted_test_labels_outfile, predicted_test_labels);
    predicted_test_labels_outfile.close();

    auto end_clock = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::to_time_t(start_clock);
    std::chrono::duration<double> elapsed_seconds = end_clock - start_clock;
    std::cerr << "Program ended at " << std::ctime(&end_time);
    std::cerr << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    return 0;
}
