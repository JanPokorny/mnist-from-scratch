#ifndef PV021_PROJECT_VEC_OPS_H
#define PV021_PROJECT_VEC_OPS_H

#include <algorithm>

#include "types.h"


template<size_t R>
vec<R> operator+(vec<R> const &a, vec<R> const &b) {
    vec<R> result = {};
    for (size_t i = 0; i < R; i++)
        result[i] = a[i] + b[i];
    return std::move(result);
}


template<size_t R>
vec<R> operator-(vec<R> const &a, vec<R> const &b) {
    vec<R> result = {};
    for (size_t i = 0; i < R; i++)
        result[i] = a[i] - b[i];
    return std::move(result);
}


template<size_t R>
vec<R> operator*(vec<R> const &a, number b) {
    vec<R> result = {};
    for (size_t i = 0; i < R; i++)
        result[i] = a[i] * b;
    return std::move(result);
}


template<size_t R>
vec<R> operator*(vec<R> const &a, vec<R> const &b) {
    vec<R> result = {};
    for (size_t i = 0; i < R; i++)
        result[i] = a[i] * b[i];
    return std::move(result);
}


template<size_t R, size_t C>
mat<R, C> operator+(mat<R, C> const &a, mat<R, C> const &b) {
    mat<R, C> result = {};
    for (size_t r = 0; r < R; r++)
        for (size_t c = 0; c < C; c++)
            result.at(r, c) = a.at(r, c) + b.at(r, c);
    return std::move(result);
}


template<size_t R, size_t C>
mat<R, C> operator-(mat<R, C> const &a, mat<R, C> const &b) {
    mat<R, C> result = {};
    for (size_t r = 0; r < R; r++)
        for (size_t c = 0; c < C; c++)
            result.at(r, c) = a.at(r, c) - b.at(r, c);
    return std::move(result);
}


template<size_t R, size_t C>
mat<R, C> operator*(mat<R, C> const &a, number b) {
    mat<R, C> result = {};
    for (size_t c = 0; c < C; c++)
        for (size_t r = 0; r < R; r++)
            result.at(r, c) = a.at(r, c) * b;
    return std::move(result);
}


template<size_t R, size_t C>mat<
R, C> dot(vec<C> const &a, vec<R> const &b) {
    mat<R, C> result = {};
    for (size_t r = 0; r < R; r++)
        for (size_t c = 0; c < C; c++)
            result.at(r, c) = a[c] * b[r];
    return std::move(result);
}


template<size_t R, size_t C>
vec<C> dot(mat<R, C> const &a, vec<R> const &b) {
    vec<C> result = {};
    for (size_t c = 0; c < C; c++)
        for (size_t r = 0; r < R; r++)
            result[c] += a.at(r, c) * b[r];
    return std::move(result);
}


template<size_t R, size_t C>
vec<C> dot_t(mat<C, R> const &a, vec<R> const &b) {
    vec<C> result = {};
    for (size_t r = 0; r < R; r++)
        for (size_t c = 0; c < C; c++)
            result[c] += a.at(c, r) * b[r];
    return std::move(result);
}


template<number F(number), size_t R>
vec<R> vec_map(vec<R> const &z) {
    vec<R> result = {};
    for (size_t i = 0; i < R; i++)
        result[i] = F(z[i]);
    return std::move(result);
}

template<size_t R>
size_t argmax(vec<R> const &a) {
    return std::distance(a.begin(), std::max_element(a.begin(), a.end()));
}

template<size_t R>
vec<R> onehot(size_t i) {
    vec<R> result = {};
    result[i] = 1.0;
    return std::move(result);
}

#endif //PV021_PROJECT_VEC_OPS_H
