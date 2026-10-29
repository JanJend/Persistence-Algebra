// Binary PNG on stdout is directly consumable by IPython.display.Image.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <grlina/invariants/landscape.hpp>
#include <iostream>
#include <optional>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: landscape_image input.scc [width height] [--layer K] "
                     "[--bounds xmin ymin xmax ymax] [--direction vx vy] [--resolution relative_spacing]\n";
        return 1;
    }
    try {
        std::size_t width = 800, height = 600, layer = 1;
        double resolution = 0.01;
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
            else throw std::invalid_argument("Unknown image option: " + option);
        }
        graded_linalg::image_detail::validate_size(width, height);
        // Keep parser/algebra diagnostics out of the binary image stream.
        auto* output = std::cout.rdbuf(std::cerr.rdbuf());
        const graded_linalg::R2Module<int> module(argv[1]);
        const graded_linalg::Landscape landscape(module,
            bounds.value_or(graded_linalg::Landscape::default_bounds(module)), resolution, direction);
        const auto png = landscape.image(layer, width, height);
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
