// Binary PNG on stdout is directly consumable by IPython.display.Image.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <grlina/invariants/hilbert_function.hpp>
#include <iomanip>
#include <iostream>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: hilbert_image input.scc [width height] [--bounds xmin ymin xmax ymax] [--scale min max] [--x-label caption] [--y-label caption] [--sqrt-x] [--maximum-only] [--values-only]\n"
                     "--sqrt-x plots squared-radius input in radius; bounds are in displayed coordinates.\n"
                     "--maximum-only reports the exact whole-module maximum, independent of display bounds.\n"
                     "--values-only returns JSON with x_grid, y_grid, values[x][y], and the sampled maximum, without rendering.\n";
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
        std::vector<long double> xs, ys;
        double positive_min = 0, positive_max = 0;
        bool sqrt_x = false;
        bool maximum_only = false;
        bool values_only = false;
        std::string x_label, y_label;
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
            else if (option == "--x-label" || option == "--y-label") {
                if (++position == argc) throw std::invalid_argument("Missing axis caption");
                (option == "--x-label" ? x_label : y_label) = argv[position];
            }
            else if (option == "--maximum-only") maximum_only = true;
            else if (option == "--values-only") values_only = true;
            else throw std::invalid_argument("Unknown image option: " + option);
        }
        if (maximum_only && values_only)
            throw std::invalid_argument("Choose maximum-only or values-only output");
        if (!maximum_only) graded_linalg::image_detail::validate_size(width, height);
        // Keep parser/algebra diagnostics out of the binary image stream.
        if (sqrt_x && !xs.empty() && xs.front() < 0)
            throw std::invalid_argument("Radius plot bounds must be nonnegative");
        auto* output = std::cout.rdbuf(std::cerr.rdbuf());
        graded_linalg::R2Module<int> module(argv[1]);
        if (sqrt_x) graded_linalg::image_detail::square_root_x(module);
        module.compute_projective_resolution();
        const auto function = module.hilbert_function();
        if (maximum_only) {
            std::cout.rdbuf(output);
            std::cout << function.maximum << '\n';
            return std::cout ? 0 : 1;
        }
        if (xs.empty()) {
            // Match the existing PNG bounds: 5% padding unless a scale is supplied.
            auto bounds = [padding = positive_min == 0 ? 0.05 : 0.0](const auto& axis) {
                const long double low = axis.empty() ? 0 : axis.front();
                const long double high = axis.empty() ? 0 : axis.back();
                const auto margin = (high > low ? high - low : 2) * padding;
                return std::vector<long double>{low - margin, high + margin};
            };
            xs = bounds(function.x_grid);
            ys = bounds(function.y_grid);
        }
        if (values_only) {
            const auto grid = function.on_grid(
                graded_linalg::image_detail::uniform_axis(xs.front(), xs.back(), width),
                graded_linalg::image_detail::uniform_axis(ys.front(), ys.back(), height));
            std::cout.rdbuf(output);
            std::cout << std::setprecision(std::numeric_limits<long double>::max_digits10);
            auto write_array = [](const auto& values) {
                std::cout << '[';
                for (std::size_t i = 0; i < values.size(); ++i) {
                    if (i) std::cout << ',';
                    std::cout << values[i];
                }
                std::cout << ']';
            };
            std::cout << "{\"maximum\":" << grid.maximum << ",\"x_grid\":";
            write_array(grid.x_grid);
            std::cout << ",\"y_grid\":";
            write_array(grid.y_grid);
            std::cout << ",\"values\":[";
            for (std::size_t x = 0; x < grid.values.size(); ++x) {
                if (x) std::cout << ',';
                write_array(grid.values[x]);
            }
            std::cout << "]}\n";
            return std::cout ? 0 : 1;
        }
        const auto png = graded_linalg::hilbert_function_image(function, xs, ys, width, height,
                                                              positive_min, positive_max, "", x_label, y_label);
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
