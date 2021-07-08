# MNIST from scratch

This is a feed-forward neural network, written in pure C++17 with no libraries. It is designed to work on the [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, which is a drop-in replacement for the classic MNIST dataset that instead of handwritten digits uses product pictures of clothing from the Zalando catalogue. Our network achieves accuracy **88%** in just over 3 minutes on a laptop CPU.

There is just one hidden layer of 64 neurons. The network is trained using batched stochastic gradient descent with learning rate decay.

The code makes use of C++ templates and recursive types, allowing most loops to have a compile-time-known iteration count. This in turn allows the compiler to perform aggressive optimizations, like unrolling, vectorization and inlining.

This was developed for the class PV021 Neural Networks @ FI MU, by Jan Pokorný (@JanPokorny) and Tran Anh Minh (@TAnhMinh).
