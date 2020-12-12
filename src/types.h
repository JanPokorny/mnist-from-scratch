#ifndef PV021_PROJECT_TYPES_H
#define PV021_PROJECT_TYPES_H

#include <array>

typedef float number;

template<size_t R>
struct vec : std::array<number, R> {
    static constexpr size_t rows = R;
    static constexpr size_t columns = 1;
};

template<size_t R>
std::ostream &operator<<(std::ostream &out, const vec<R> &v) {
    out << "[";
    for (size_t i = 0; i < R - 1; ++i)
        out << v[i] << " ";
    out << v[R - 1] << "]";
    return out;
}


template<size_t R, size_t C>
struct mat : std::array<number, R * C> {
    static constexpr size_t rows = R;
    static constexpr size_t columns = C;

    constexpr number &at(size_t r, size_t c) {
        return std::array<number, R * C>::operator[](c * R + r);
    }

    [[nodiscard]] constexpr number const &at(size_t r, size_t c) const {
        return std::array<number, R * C>::operator[](c * R + r);
    }
};

template<size_t R, size_t C>
std::ostream &operator<<(std::ostream &out, const mat<R, C> &m) {
    for (size_t r = 0; r < R; r++) {
        out << "[";
        for (size_t c = 0; c < C - 1; ++c)
            out << m.at(r, c) << " ";
        out << m.at(r, C - 1) << "]";
        out << std::endl;
    }
    return out;
}

using label_type = size_t;


#endif //PV021_PROJECT_TYPES_H
