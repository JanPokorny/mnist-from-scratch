#ifndef PV021_PROJECT_IO_H
#define PV021_PROJECT_IO_H

template<size_t N>
std::vector<vec<N>> load_images(std::istream &in) {
    std::vector<vec<N>> output;
    while (!in.eof()) {
        vec<N> image;
        for (size_t i = 0; i < N; i++) {
            size_t pixelValue = 0;
            in >> pixelValue;
            image[i] = pixelValue / 256.0;
            in.ignore(); // skip comma
        }
        output.push_back(image);
        in >> std::ws;
    }
    return output;
}

std::vector<label_type> load_labels(std::istream &in) {
    std::vector<label_type> output;
    while (!in.eof()) {
        label_type label = 0;
        in >> label;
        output.push_back(label);
        in >> std::ws;
    }
    return output;
}

void save_labels(std::ostream &out, std::vector<label_type> const &labels) {
    for (label_type label : labels) {
        out << label << std::endl;
    }
}

#endif //PV021_PROJECT_IO_H
