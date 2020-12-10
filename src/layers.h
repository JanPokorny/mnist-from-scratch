#ifndef PV021_PROJECT_LAYERS_H
#define PV021_PROJECT_LAYERS_H

struct Layer {
};

template<size_t S>
struct HiddenLayer : Layer {
    static constexpr size_t size = S;
    static constexpr bool is_input = false;
    static constexpr bool is_output = false;
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
