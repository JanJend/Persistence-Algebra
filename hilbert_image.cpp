// Binary PNG on stdout is directly consumable by IPython.display.Image.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <grlina/invariants/hilbert_function.hpp>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <sstream>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace graded_linalg {

R2Module<int> read_hilbert_module(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open SCC input file: " + path);
    auto header_line = [&] {
        std::string line;
        while (std::getline(input,line)) {
            line = line.substr(0,line.find('#'));
            const auto first = line.find_first_not_of(" \t\r");
            if (first != std::string::npos)
                return line.substr(first,line.find_last_not_of(" \t\r")-first+1);
        }
        return std::string{};
    };
    const auto header = header_line(), parameters = header_line();
    if (header == "scc2020" && parameters == "3") {
        // Older R2 resolution files wrote the number of chain groups here.
        // The normal R2 parser still requires exactly two coordinates per grade;
        // genuine three-parameter input is rejected rather than projected.
        std::stringstream normalized;
        normalized << "scc2020\n2\n" << input.rdbuf();
        return R2Module<int>(normalized);
    }
    input.clear();
    input.seekg(0);
    return R2Module<int>(input);
}

} // namespace graded_linalg

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " input.scc [width height] [--bounds xmin ymin xmax ymax] [--scale min max] [--x-label caption] [--y-label caption] [--sqrt-x] [--betti] [--betti-output file.csv] [--image-only] [--maximum-only] [--values-only]\n"
                     "--sqrt-x plots squared-radius input in radius; bounds are in displayed coordinates.\n"
                     "--maximum-only reports the exact whole-module maximum, independent of display bounds.\n"
                     "--values-only returns JSON with x_grid, y_grid, values[x][y], and the sampled maximum, without rendering.\n"
                     "--betti minimizes the input and marks generators, relations and syzygies (default for betti_image).\n"
                     "--betti-output also saves x,y,beta0,beta1,beta2 in the original input coordinates.\n"
                     "--image-only keeps the image and dimension colour bar, without captions, axis ticks or a Betti legend.\n";
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
        bool image_only = false;
#ifdef GRLINA_BETTI_IMAGE
        bool show_betti = true;
#else
        bool show_betti = false;
#endif
        std::string betti_output;
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
            else if (option == "--image-only") image_only = true;
            else if (option == "--betti") show_betti = true;
            else if (option == "--betti-output") {
                if (++position == argc) throw std::invalid_argument("Missing Betti CSV path");
                betti_output = argv[position];
                show_betti = true;
            }
            else throw std::invalid_argument("Unknown image option: " + option);
        }
        if (maximum_only && values_only)
            throw std::invalid_argument("Choose maximum-only or values-only output");
        if (!maximum_only) graded_linalg::image_detail::validate_size(width, height);
        // Keep parser/algebra diagnostics out of the binary image stream.
        if (sqrt_x && !xs.empty() && xs.front() < 0)
            throw std::invalid_argument("Radius plot bounds must be nonnegative");
        if (!betti_output.empty() && std::filesystem::exists(betti_output) &&
            std::filesystem::equivalent(argv[1],betti_output))
            throw std::invalid_argument("Betti CSV must not overwrite the input module");
        auto* output = std::cout.rdbuf(std::cerr.rdbuf());
        auto module = graded_linalg::read_hilbert_module(argv[1]);
        auto betti = show_betti ? graded_linalg::graded_betti_numbers(module) :
                                 std::vector<graded_linalg::BettiDegree>{};
        if (sqrt_x) graded_linalg::image_detail::square_root_x(module);
        if (!betti_output.empty()) {
            std::ofstream csv;
            csv.exceptions(std::ios::badbit | std::ios::failbit);
            csv.open(betti_output);
            csv << std::setprecision(std::numeric_limits<long double>::max_digits10)
                << "x,y,beta0,beta1,beta2\n";
            for (const auto& point : betti)
                csv << point.x << ',' << point.y << ',' << point.multiplicities[0] << ','
                    << point.multiplicities[1] << ',' << point.multiplicities[2] << '\n';
            csv.close();
        }
        if (sqrt_x) for (auto& point : betti) point.x = std::sqrt(static_cast<double>(point.x));
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
                                                              positive_min, positive_max, "", x_label, y_label, betti, image_only);
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
