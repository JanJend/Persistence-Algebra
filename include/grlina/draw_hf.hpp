#pragma once
#include <stb_image_write.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace graded_linalg {
namespace detail {

// Small bitmap labels keep PNG rendering independent of GUI/font libraries.
inline const unsigned char* hilbert_glyph(char c) {
    static constexpr unsigned char glyphs[][5] = {
        {0x3e,0x51,0x49,0x45,0x3e},{0,0x42,0x7f,0x40,0},{0x42,0x61,0x51,0x49,0x46},
        {0x21,0x41,0x45,0x4b,0x31},{0x18,0x14,0x12,0x7f,0x10},{0x27,0x45,0x45,0x45,0x39},
        {0x3c,0x4a,0x49,0x49,0x30},{1,0x71,9,5,3},{0x36,0x49,0x49,0x49,0x36},
        {6,0x49,0x49,0x29,0x1e},
        {0x7e,0x11,0x11,0x11,0x7e},{0x7f,0x49,0x49,0x49,0x36},{0x3e,0x41,0x41,0x41,0x22},
        {0x7f,0x41,0x41,0x22,0x1c},{0x7f,0x49,0x49,0x49,0x41},{0x7f,9,9,9,1},
        {0x3e,0x41,0x49,0x49,0x7a},{0x7f,8,8,8,0x7f},{0,0x41,0x7f,0x41,0},
        {0x20,0x40,0x41,0x3f,1},{0x7f,8,0x14,0x22,0x41},{0x7f,0x40,0x40,0x40,0x40},
        {0x7f,2,0x0c,2,0x7f},{0x7f,4,8,0x10,0x7f},{0x3e,0x41,0x41,0x41,0x3e},
        {0x7f,9,9,9,6},{0x3e,0x41,0x51,0x21,0x5e},{0x7f,9,0x19,0x29,0x46},
        {0x46,0x49,0x49,0x49,0x31},{1,1,0x7f,1,1},{0x3f,0x40,0x40,0x40,0x3f},
        {0x1f,0x20,0x40,0x20,0x1f},{0x3f,0x40,0x38,0x40,0x3f},{0x63,0x14,8,0x14,0x63},
        {7,8,0x70,8,7},{0x61,0x51,0x49,0x45,0x43},
        {0,0x60,0x60,0,0},{8,8,8,8,8},{8,8,0x3e,8,8},{0,0x36,0x36,0,0},{0,0,0,0,0}
    };
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    int i = c >= '0' && c <= '9' ? c-'0' : c >= 'A' && c <= 'Z' ? c-'A'+10 :
            c == '.' ? 36 : c == '-' ? 37 : c == '+' ? 38 : c == ':' ? 39 : 40;
    return glyphs[i];
}

inline std::array<unsigned char,3> hilbert_colour(double t) {
    t = std::clamp(t, 0.0, 1.0);
    return t < 0.5 ? std::array<unsigned char,3>{static_cast<unsigned char>(212*(1-2*t)),
        static_cast<unsigned char>(234-256*t),255} : std::array<unsigned char,3>{0,
        static_cast<unsigned char>(106*(2-2*t)),static_cast<unsigned char>(255*(2-2*t))};
}

} // namespace detail

/** Draw a Module::R2HilbertGrid as a labelled PNG. Positive dimensions use the
 * Python visualise_reso.py light-blue/blue/black logarithmic scale; zero is white.
 * Pass the same positive_min/positive_max to compare multiple grids fairly.
 * Define STB_IMAGE_WRITE_IMPLEMENTATION in exactly one consuming .cpp file.
 */
template <typename Grid>
void save_hilbert_png(const Grid& grid, const std::string& filename,
                      int positive_min, int positive_max, const std::string& title = "HILBERT FUNCTION") {
    const int w = static_cast<int>(grid.x_grid.size()), h = static_cast<int>(grid.y_grid.size());
    if (w < 2 || h < 2) throw std::invalid_argument("Hilbert image needs at least two grid points on each axis");
    const int left = 85, top = 65, width = w+205, height = h+140;
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width)*height*3, 255);
    auto pixel = [&](int x, int y, std::array<unsigned char,3> colour) {
        if (x < 0 || y < 0 || x >= width || y >= height) return;
        const auto offset = (static_cast<std::size_t>(y)*width+x)*3;
        std::copy(colour.begin(), colour.end(), pixels.begin()+offset);
    };
    auto text = [&](int x, int y, const std::string& label, int scale = 1) {
        for (char c : label) {
            const auto* glyph = detail::hilbert_glyph(c);
            for (int col=0; col<5; ++col) for (int row=0; row<7; ++row)
                if (glyph[col] & (1<<row)) for (int dx=0; dx<scale; ++dx) for (int dy=0; dy<scale; ++dy)
                    pixel(x+col*scale+dx,y+row*scale+dy,{30,30,30});
            x += 6*scale;
        }
    };
    auto number = [](double value) { std::ostringstream out; out << std::setprecision(3) << value; return out.str(); };
    positive_min = std::max(1, positive_min);
    const double upper = positive_max > positive_min ? positive_max : positive_min + 1.0;
    const double log_min = std::log(positive_min), log_range = std::log(upper)-log_min;
    int maximum = 0;
    for (int x=0; x<w; ++x) for (int y=0; y<h; ++y) {
        const int value = grid.values[x][y];
        maximum = std::max(maximum, value);
        if (value > 0) pixel(left+x,top+h-1-y,detail::hilbert_colour((std::log(value)-log_min)/log_range));
    }
    for (int x=left-1; x<=left+w; ++x) { pixel(x,top-1,{50,50,50}); pixel(x,top+h,{50,50,50}); }
    for (int y=top-1; y<=top+h; ++y) { pixel(left-1,y,{50,50,50}); pixel(left+w,y,{50,50,50}); }
    text(left,15,title,w >= 320 ? 2 : 1);
    text(left,40,"MAX DIMENSION: "+std::to_string(maximum));
    const int intervals = w < 240 ? 2 : 4;
    for (int i=0; i<=intervals; ++i) {
        const int x=i*(w-1)/intervals, y=i*(h-1)/intervals;
        text(left+x-15,top+h+12,number(grid.x_grid[x]));
        text(4,top+h-1-y,number(grid.y_grid[y]));
    }
    text(left+w/2,top+h+35,"X",2);
    text(15,top-25,"Y",2);
    text(left+w+18,top-20,"DIMENSION");
    if (positive_max > 0) {
        for (int y=0; y<h; ++y) for (int x=0; x<18; ++x)
            pixel(left+w+20+x,top+y,detail::hilbert_colour(1.0-static_cast<double>(y)/(h-1)));
        text(left+w+45,top,number(upper));
        text(left+w+45,top+h-7,number(positive_min));
        if (positive_max > positive_min) text(left+w+45,top+h/2,number(std::sqrt(upper*positive_min)));
    } else text(left+w+18,top+10,"ALL ZERO");
    text(left,top+h+60,"WHITE: ZERO    LOG COLOUR SCALE");
    if (!stbi_write_png(filename.c_str(),width,height,3,pixels.data(),width*3))
        throw std::runtime_error("Could not write Hilbert image: "+filename);
}

} // namespace graded_linalg
