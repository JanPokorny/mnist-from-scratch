#ifndef PV021_PROJECT_LAYERS_H
#define PV021_PROJECT_LAYERS_H

struct Layer {
};

template<size_t S>
struct HiddenLayer : Layer {
    static constexpr size_t size = S;
    static constexpr bool is_input = false;
    static constexpr bool is_output = false;

    template<size_t R>
    static constexpr vec<R> activation_fn(vec<R> const& z) {
        return vec_map<relu>(z);
    }

    template<size_t R>
    static constexpr vec<R> activation_fn_prime(vec<R> const& z) {
        return vec_map<relu_prime>(z);
    }
};

template<size_t S>
struct InputLayer : HiddenLayer<S> {
    static constexpr bool is_input = true;
};

template<size_t S>
struct OutputLayer : HiddenLayer<S> {
    static constexpr bool is_output = true;
};

#endif //PV021_PROJECT_LAYERS_H
