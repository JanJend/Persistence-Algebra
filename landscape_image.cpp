// Binary PNG on stdout is directly consumable by IPython.display.Image.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <grlina/invariants/landscape.hpp>
#include <iomanip>
#include <iostream>
#include <optional>
#include <boost/property_tree/json_parser.hpp>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// JSON is an optional transport for Python; the default output remains raw PNG.
void write_grid(const graded_linalg::Landscape::Grid& grid, bool values,
                const std::vector<unsigned char>& png = {}) {
    std::cout << std::setprecision(17) << "{\"maximum\":" << grid.maximum << ",\"layer\":" << grid.layer
              << ",\"bounds\":[" << grid.bounds[0] << ',' << grid.bounds[1] << ',' << grid.bounds[2] << ',' << grid.bounds[3] << ']';
    if (!png.empty()) {
        constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::cout << ",\"png\":\"";
        for (std::size_t i=0; i<png.size(); i+=3) {
            const unsigned word = (unsigned(png[i]) << 16) |
                (i+1 < png.size() ? unsigned(png[i+1]) << 8 : 0) | (i+2 < png.size() ? png[i+2] : 0);
            std::cout << alphabet[(word >> 18) & 63] << alphabet[(word >> 12) & 63]
                      << (i+1 < png.size() ? alphabet[(word >> 6) & 63] : '=')
                      << (i+2 < png.size() ? alphabet[word & 63] : '=');
        }
        std::cout << '"';
    }
    if (values) {
        std::cout << ",\"values\":[";
        for (std::size_t x=0; x<grid.values.size(); ++x) {
            if (x) std::cout << ',';
            std::cout << '[';
            for (std::size_t y=0; y<grid.values[x].size(); ++y) {
                if (y) std::cout << ',';
                std::cout << grid.values[x][y];
            }
            std::cout << ']';
        }
        std::cout << ']';
    }
    std::cout << "}\n";
}

graded_linalg::Landscape::Grid read_grid() {
    boost::property_tree::ptree json;
    boost::property_tree::read_json(std::cin, json);
    graded_linalg::Landscape::Grid grid{};
    grid.layer = json.get<std::size_t>("layer");
    if (!grid.layer) throw std::invalid_argument("Landscape layer must be positive");
    const auto& bounds = json.get_child("bounds");
    if (bounds.size() != 4) throw std::invalid_argument("Landscape bounds need four coordinates");
    std::size_t i = 0;
    for (const auto& entry : bounds) grid.bounds[i++] = entry.second.get_value<double>();
    for (i=0; i<2; ++i)
        if (!std::isfinite(grid.bounds[i]) || !std::isfinite(grid.bounds[i+2]) || grid.bounds[i] >= grid.bounds[i+2])
            throw std::invalid_argument("Landscape bounds must be finite and increasing");
    const auto& values = json.get_child("values");
    const auto height = values.empty() ? 0 : values.front().second.size();
    graded_linalg::image_detail::validate_size(values.size(), height);
    grid.values.reserve(values.size());
    for (const auto& column : values) {
        if (column.second.size() != height) throw std::invalid_argument("Landscape values must form a rectangular grid");
        grid.values.emplace_back();
        grid.values.back().reserve(height);
        for (const auto& entry : column.second) {
            // Avoid property_tree's per-pixel stringstream conversion.
            std::size_t used = 0;
            const auto value = std::stod(entry.second.data(), &used);
            if (used != entry.second.data().size() || !std::isfinite(value) || value < 0)
                throw std::invalid_argument("Landscape values must be finite and nonnegative");
            grid.values.back().push_back(value);
            grid.maximum = std::max(grid.maximum, value);
        }
    }
    return grid;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: landscape_image input.scc [width height] [--layer K] "
                     "[--bounds xmin ymin xmax ymax] [--direction vx vy] [--resolution relative_spacing] "
                     "[--colormap blue|inferno] [--scale MAX] [--maximum-only] [--json] [--include-values] "
                     "[--values-only] [--no-vineyard]\n"
                     "       landscape_image --render-values [--colormap blue|inferno] [--scale MAX] [--json]\n"
                     "JSON output includes the sampled maximum; --render-values reads a grid JSON from stdin.\n";
        return 1;
    }
    try {
        std::size_t width = 800, height = 600, layer = 1;
        double resolution = 0.01;
        double colour_maximum = 0;
        bool maximum_only = false;
        bool json_output = false, include_values = false, values_only = false, vineyard = true;
        const bool render_values = std::string(argv[1]) == "--render-values";
        graded_linalg::ColourMap colormap = graded_linalg::ColourMap::blue;
        graded_linalg::r2degree direction{1, 1};
        std::optional<graded_linalg::Landscape::Box> bounds;
        auto integer = [](const char* argument) {
            const std::string text(argument);
            if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("Image sizes and layer must be positive integers");
            const auto value = std::stoull(text);
            if (!value || value > std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument("Image sizes and layer must be positive integers");
            return static_cast<std::size_t>(value);
        };
        int position = 2;
        if (position < argc && std::string(argv[position]).rfind("--", 0) != 0) {
            if (position + 1 == argc) throw std::invalid_argument("Specify both width and height");
            width = integer(argv[position++]);
            height = integer(argv[position++]);
        }
        auto next = [&]() -> const char* {
            if (++position == argc) throw std::invalid_argument("Missing image option value");
            return argv[position];
        };
        auto number = [&] {
            const std::string text(next());
            std::size_t used = 0;
            const double value = std::stod(text, &used);
            if (used != text.size() || !std::isfinite(value))
                throw std::invalid_argument("Image option values must be finite numbers");
            return value;
        };
        for (; position < argc; ++position) {
            const std::string option(argv[position]);
            if (option == "--layer") layer = integer(next());
            else if (option == "--bounds") {
                bounds.emplace();
                for (auto& value : *bounds) value = number();
            } else if (option == "--direction") {
                for (auto& value : direction) value = number();
            } else if (option == "--resolution") resolution = number();
            else if (option == "--colormap") colormap = graded_linalg::parse_colourmap(next());
            else if (option == "--scale") {
                colour_maximum = number();
                if (colour_maximum <= 0)
                    throw std::invalid_argument("Landscape colour maximum must be positive");
            } else if (option == "--maximum-only") maximum_only = true;
            else if (option == "--json") json_output = true;
            else if (option == "--include-values") { json_output = true; include_values = true; }
            else if (option == "--values-only") { json_output = true; include_values = true; values_only = true; }
            else if (option == "--no-vineyard") vineyard = false;
            else throw std::invalid_argument("Unknown image option: " + option);
        }
        graded_linalg::image_detail::validate_size(width, height);
        if (maximum_only && json_output) throw std::invalid_argument("Choose maximum-only or JSON output");
        // Keep parser/algebra diagnostics out of the binary image stream.
        auto* output = std::cout.rdbuf(std::cerr.rdbuf());
        const auto grid = [&] {
            if (render_values) return read_grid();
            const graded_linalg::R2Module<int> module(argv[1]);
            const graded_linalg::Landscape landscape(module,
                bounds.value_or(graded_linalg::Landscape::default_bounds(module)), resolution, direction, vineyard);
            return landscape.sample_grid(layer, width, height);
        }();
        if (maximum_only) {
            std::cout.rdbuf(output);
            std::cout << std::setprecision(17) << grid.maximum << '\n';
            return std::cout ? 0 : 1;
        }
        const auto png = values_only ? std::vector<unsigned char>{} : grid.image(colormap, colour_maximum);
        std::cout.rdbuf(output);
        if (json_output) {
            write_grid(grid, include_values, png);
            return std::cout ? 0 : 1;
        }
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
