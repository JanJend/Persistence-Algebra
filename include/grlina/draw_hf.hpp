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
#ifdef GRLINA_HILBERT_CORETEXT
#include <CoreText/CoreText.h>
#include <memory>
#endif

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

/** Draw a uniformly sampled Module::R2HilbertGrid as a labelled PNG.
 * Positive dimensions use the Python light-blue/blue/black logarithmic scale;
 * zero is white. Pass shared positive_min/positive_max for comparable images.
 * Define STB_IMAGE_WRITE_IMPLEMENTATION in one consuming .cpp file. On macOS,
 * GRLINA_HILBERT_CORETEXT enables system fonts (link CoreText, CoreGraphics and CoreFoundation);
 * otherwise the built-in bitmap font keeps the renderer dependency-free.
 */
template <typename Grid>
void save_hilbert_png(const Grid& grid, const std::string& filename,
                      int positive_min, int positive_max, const std::string& title = "Hilbert function") {
    const int nx = static_cast<int>(grid.x_grid.size()), ny = static_cast<int>(grid.y_grid.size());
    if (nx < 2 || ny < 2) throw std::invalid_argument("Hilbert image needs at least two grid points on each axis");
    // Leave room for labels even when the sampling grid is small.
    const double enlargement = std::max({1.0, 360.0/nx, 300.0/ny});
    const int w = std::lround(nx*enlargement), h = std::lround(ny*enlargement);
    const int left = 100, top = 85, width = left+w+165, height = top+h+100;
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width)*height*4, 255);
    auto pixel = [&](int x, int y, std::array<unsigned char,3> colour) {
        if (x < 0 || y < 0 || x >= width || y >= height) return;
        const auto offset = (static_cast<std::size_t>(y)*width+x)*4;
        std::copy(colour.begin(), colour.end(), pixels.begin()+offset);
    };
#ifdef GRLINA_HILBERT_CORETEXT
    auto space = CGColorSpaceCreateDeviceRGB();
    std::unique_ptr<CGContext, decltype(&CGContextRelease)> context(CGBitmapContextCreate(pixels.data(), width, height,
        8, width*4, space, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big), CGContextRelease);
    CGColorSpaceRelease(space);
    if (!context) throw std::runtime_error("Could not create Hilbert image context");
    CGContextSetRGBFillColor(context.get(), 0.15, 0.17, 0.20, 1);
#endif
    // align = 0: left, 0.5: centred, 1: right; y is the top of the text.
    auto text = [&](double x, double y, const std::string& label, int size = 14, double align = 0) {
#ifdef GRLINA_HILBERT_CORETEXT
        auto font = CTFontCreateWithName(CFSTR("Helvetica"), size, nullptr);
        const void* keys[] = {kCTFontAttributeName, kCTForegroundColorFromContextAttributeName};
        const void* values[] = {font, kCFBooleanTrue};
        auto attributes = CFDictionaryCreate(nullptr, keys, values, 2,
            &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        auto string = CFStringCreateWithCString(nullptr, label.c_str(), kCFStringEncodingUTF8);
        auto styled = CFAttributedStringCreate(nullptr, string, attributes);
        auto line = CTLineCreateWithAttributedString(styled);
        CGFloat ascent = 0;
        const double advance = CTLineGetTypographicBounds(line, &ascent, nullptr, nullptr);
        CGContextSetTextPosition(context.get(), x-align*advance, height-y-ascent);
        CTLineDraw(line, context.get());
        CFRelease(line); CFRelease(styled); CFRelease(string); CFRelease(attributes); CFRelease(font);
#else
        const int scale = std::max(1, size/7);
        x -= align * (label.size()*6-1)*scale;
        for (char c : label) {
            const auto* glyph = detail::hilbert_glyph(c);
            for (int col=0; col<5; ++col) for (int row=0; row<7; ++row)
                if (glyph[col] & (1<<row)) for (int dx=0; dx<scale; ++dx) for (int dy=0; dy<scale; ++dy)
                    pixel(std::lround(x)+col*scale+dx,std::lround(y)+row*scale+dy,{38,43,51});
            x += 6*scale;
        }
#endif
    };
    auto number = [](double value) {
        std::ostringstream out;
        out << std::setprecision(4) << (value == 0 ? 0 : value);
        return out.str();
    };
    positive_min = std::max(1, positive_min);
    const double log_min = std::log(positive_min);
    const double log_range = positive_max > positive_min ? std::log(positive_max)-log_min : 0;
    int maximum = 0;
    for (const auto& column : grid.values) for (int value : column) maximum = std::max(maximum, value);
    for (int x=0; x<w; ++x) for (int y=0; y<h; ++y) {
        const int value = grid.values[x*nx/w][y*ny/h];
        if (value > 0) pixel(left+x,top+h-1-y,
            detail::hilbert_colour(log_range > 0 ? (std::log(value)-log_min)/log_range : 0));
    }
    const std::array<unsigned char,3> ink{90,96,105};
    auto frame = [&](int x, int y, int dx, int dy) {
        for (int i=x; i<=x+dx; ++i) { pixel(i,y,ink); pixel(i,y+dy,ink); }
        for (int j=y; j<=y+dy; ++j) { pixel(x,j,ink); pixel(x+dx,j,ink); }
    };
    frame(left-1,top-1,w+1,h+1);
    text(left+w/2.0,16,title,22,0.5);
    text(left+w/2.0,49,"Max dimension: "+std::to_string(maximum),14,0.5);
    // Rounded ticks, positioned in coordinate space rather than at grid indices.
    auto ticks = [&](double low, double high, bool horizontal) {
        const double raw = (high-low)/4;
        if (!(raw > 0)) return;
        const double magnitude = std::pow(10.0,std::floor(std::log10(raw)));
        const double unit = raw/magnitude;
        const double step = (unit < 1.5 ? 1 : unit < 2.25 ? 2 : unit < 3.75 ? 2.5 : unit < 7.5 ? 5 : 10)*magnitude;
        const double first = std::ceil(low/step)*step;
        for (int i=0; i<10; ++i) {
            double value = first+i*step;
            if (value > high+step*1e-8) break;
            if (std::abs(value) < step*1e-8) value = 0;
            const double t = (value-low)/(high-low);
            const int x = left+std::lround(t*(w-1)), y = top+h-1-std::lround(t*(h-1));
            for (int d=1; d<=5; ++d) pixel(horizontal ? x : left-d,horizontal ? top+h+d : y,ink);
            if (horizontal) text(x,top+h+13,number(value),14,0.5);
            else text(left-12,y-8,number(value),14,1);
        }
    };
    ticks(grid.x_grid.front(),grid.x_grid.back(),true);
    ticks(grid.y_grid.front(),grid.y_grid.back(),false);
    text(left+w/2.0,top+h+43,"X",18,0.5);
    text(16,top+h/2.0-10,"Y",18);
    const int bar = left+w+35, bar_width = 20;
    text(bar-4,top-29,"Dimension",14);
    for (int y=0; y<h; ++y) for (int x=0; x<bar_width; ++x)
        if (positive_max > 0) pixel(bar+x,top+y,
            detail::hilbert_colour(log_range > 0 ? 1.0-static_cast<double>(y)/(h-1) : 0));
    frame(bar-1,top-1,bar_width+1,h+1);
    if (log_range > 0) {
        std::vector<int> values{positive_min,positive_max};
        for (double power=1; power<=positive_max; power*=10)
            for (int factor : {1,2,3,5}) {
                const double v = factor*power;
                if (v > positive_min && v < positive_max) values.push_back(static_cast<int>(v));
            }
        std::sort(values.begin(),values.end());
        int previous = h+30;
        for (int value : values) {
            const int y = std::lround((1-(std::log(value)-log_min)/log_range)*(h-1));
            if (value != positive_max && (previous-y < 25 || y < 25)) continue;
            for (int d=0; d<=5; ++d) pixel(bar+bar_width+d,top+y,ink);
            text(bar+bar_width+11,top+y-8,std::to_string(value));
            previous = y;
        }
    } else text(bar+bar_width+11,top+h/2.0-8,std::to_string(std::max(0,positive_max)));
    // A separate white swatch makes zero explicit without including it in log space.
    frame(left,top+h+77,12,12);
    text(left+21,top+h+75,"White: 0",12);
    text(left+w,top+h+75,"Log colour scale",12,1);
#ifdef GRLINA_HILBERT_CORETEXT
    CGContextFlush(context.get());
#endif
    if (!stbi_write_png(filename.c_str(),width,height,4,pixels.data(),width*4))
        throw std::runtime_error("Could not write Hilbert image: "+filename);
}

} // namespace graded_linalg
