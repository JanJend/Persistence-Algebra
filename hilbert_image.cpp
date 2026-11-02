// Binary PNG on stdout is directly consumable by IPython.display.Image.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <grlina/invariants/hilbert_function.hpp>
#include <iostream>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: hilbert_image input.scc [width height] [--bounds xmin ymin xmax ymax] [--scale min max] [--sqrt-x] [--maximum-only]\n"
                     "--sqrt-x plots squared-radius input in radius; bounds are in displayed coordinates.\n"
                     "--maximum-only reports the exact whole-module maximum, independent of display bounds.\n";
        return 1;
    }
    try {
        std::size_t width = 800, height = 600;
        int position = 2;
        if (position < argc && std::string(argv[position]).rfind("--", 0) != 0) {
            if (position + 1 == argc) throw std::invalid_argument("Specify both width and height");
            width = std::stoull(argv[position++]);
            height = std::stoull(argv[position++]);
        }
        std::vector<double> xs, ys;
        double positive_min = 0, positive_max = 0;
        bool sqrt_x = false;
        bool maximum_only = false;
        auto number = [&] {
            if (++position == argc) throw std::invalid_argument("Missing image option value");
            std::size_t used = 0;
            const std::string text(argv[position]);
            const double value = std::stod(text, &used);
            if (used != text.size() || !std::isfinite(value))
                throw std::invalid_argument("Image option values must be finite numbers");
            return value;
        };
        for (; position < argc; ++position) {
            const std::string option(argv[position]);
            if (option == "--bounds") {
                const double xmin = number(), ymin = number(), xmax = number(), ymax = number();
                if (xmin >= xmax || ymin >= ymax) throw std::invalid_argument("Image bounds must increase");
                xs = {xmin, xmax}; ys = {ymin, ymax};
            } else if (option == "--scale") {
                positive_min = number(); positive_max = number();
                if (positive_min <= 0 || positive_max < positive_min)
                    throw std::invalid_argument("Image scale needs 0 < min <= max");
            } else if (option == "--sqrt-x") sqrt_x = true;
            else if (option == "--maximum-only") maximum_only = true;
            else throw std::invalid_argument("Unknown image option: " + option);
        }
        // Keep parser/algebra diagnostics out of the binary image stream.
        if (sqrt_x && !xs.empty() && xs.front() < 0)
            throw std::invalid_argument("Radius plot bounds must be nonnegative");
        auto* output = std::cout.rdbuf(std::cerr.rdbuf());
        graded_linalg::R2Module<int> module(argv[1]);
        if (sqrt_x) graded_linalg::image_detail::square_root_x(module);
        module.compute_projective_resolution();
        if (maximum_only) {
            const auto maximum = module.hilbert_function().maximum;
            std::cout.rdbuf(output);
            std::cout << maximum << '\n';
            return std::cout ? 0 : 1;
        }
        const auto png = xs.empty() && positive_min == 0
            ? module.hilbert_function_image(width, height)
            : xs.empty()
                ? graded_linalg::hilbert_function_image(module.hilbert_function(), width, height,
                                                       positive_min, positive_max)
                : graded_linalg::hilbert_function_image(module.hilbert_function(), xs, ys, width, height,
                                                       positive_min, positive_max);
        std::cout.rdbuf(output);
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        std::cout.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
