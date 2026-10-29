#pragma once
#include <grlina/general_checks.hpp>
#include <grlina/third_party/stb_image_write.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef GRLINA_HILBERT_CORETEXT
#include <CoreText/CoreText.h>
#include <memory>
#endif

namespace graded_linalg {
namespace hilbert_detail {

/** Axes may be empty; nonempty axes must be finite and nondecreasing. */
template <typename Scalar>
void validate_axis(const std::vector<Scalar>& axis) {
    for (std::size_t i = 0; i < axis.size(); ++i)
        if (!std::isfinite(static_cast<long double>(axis[i])) ||
            (i && axis[i] < axis[i - 1]))
            throw std::invalid_argument("Hilbert axes must be finite and sorted");
}

} // namespace hilbert_detail


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


namespace image_detail {

inline void validate_size(std::size_t width, std::size_t height) {
    if (width < 2 || height < 2 ||
        width > static_cast<std::size_t>(std::numeric_limits<int>::max() / 4 - 265) ||
        height > static_cast<std::size_t>(std::numeric_limits<int>::max() - 185))
        throw std::invalid_argument("Image needs width and height >= 2, within PNG limits");
    if ((width + 265) * 4 + 1 > static_cast<std::size_t>(std::numeric_limits<int>::max()) / (height + 185))
        throw std::invalid_argument("Image exceeds the PNG encoder's size limit");
}

template <typename Scalar>
std::vector<Scalar> uniform_axis(Scalar low, Scalar high, std::size_t count) {
    std::vector<Scalar> axis(count);
    for (std::size_t i = 0; i < count; ++i)
        axis[i] = i == count - 1 ? high : low + (high-low) * (static_cast<Scalar>(i)/(count-1));
    return axis;
}

/** Draw uniformly sampled values, with coordinate labels, a colour bar and title.
 * Width/height count heatmap pixels; labels add 265 x 185 pixels.
 * Hilbert dimensions use logarithmic colours; landscape heights use linear colours.
 */
template <typename Scalar, typename Value>
std::vector<unsigned char> heatmap_image(const std::vector<Scalar>& xs, const std::vector<Scalar>& ys,
    const std::vector<std::vector<Value>>& values, const std::string& title,
    const std::string& value_label, bool logarithmic,
    long double positive_min = 0, long double positive_max = 0) {
    validate_size(xs.size(), ys.size());
    const int w = static_cast<int>(xs.size()), h = static_cast<int>(ys.size());
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
    if (!context) throw std::runtime_error("Could not create heatmap image context");
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
            const auto* glyph = hilbert_glyph(c);
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
    long double maximum = 0;
    for (const auto& column : values) for (auto value : column) {
        if (!std::isfinite(static_cast<long double>(value)))
            throw std::invalid_argument("Heatmap values must be finite");
        maximum = std::max(maximum, static_cast<long double>(value));
    }
    if (positive_min <= 0) {
        positive_min = std::numeric_limits<long double>::infinity();
        for (const auto& column : values) for (auto value : column)
            if (value > 0) positive_min = std::min(positive_min, static_cast<long double>(value));
        if (!std::isfinite(positive_min)) positive_min = 1;
    }
    if (positive_max <= 0) positive_max = maximum;
    if (!std::isfinite(positive_min) || !std::isfinite(positive_max))
        throw std::invalid_argument("Heatmap colour limits must be finite");
    const double log_min = std::log(positive_min);
    const double log_range = positive_max > positive_min ? std::log(positive_max)-log_min : 0;
    auto normalise = [&](long double value) {
        if (!logarithmic) return positive_max > 0 ? value / positive_max : 0.0L;
        return log_range > 0 ? (std::log(value)-log_min)/log_range : 0.0L;
    };
    auto colour = [&](double t) {
        if (logarithmic) return hilbert_colour(t);
        const auto shade = static_cast<unsigned char>(std::lround(255 * (1 - std::clamp(t, 0.0, 1.0))));
        return std::array<unsigned char,3>{shade,shade,255};
    };
    for (int x=0; x<w; ++x) for (int y=0; y<h; ++y) {
        const auto value = values[x][y];
        if (value > 0) pixel(left+x,top+h-1-y,colour(normalise(value)));
    }
    const std::array<unsigned char,3> ink{90,96,105};
    auto frame = [&](int x, int y, int dx, int dy) {
        for (int i=x; i<=x+dx; ++i) { pixel(i,y,ink); pixel(i,y+dy,ink); }
        for (int j=y; j<=y+dy; ++j) { pixel(x,j,ink); pixel(x+dx,j,ink); }
    };
    frame(left-1,top-1,w+1,h+1);
    text(left+w/2.0,16,title,22,0.5);
    text(left+w/2.0,49,"Max "+value_label+": "+number(maximum),14,0.5);
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
    ticks(xs.front(),xs.back(),true);
    ticks(ys.front(),ys.back(),false);
    text(left+w/2.0,top+h+43,"X",18,0.5);
    text(16,top+h/2.0-10,"Y",18);
    const int bar = left+w+35, bar_width = 20;
    text(bar-4,top-29,value_label,14);
    for (int y=0; y<h; ++y) for (int x=0; x<bar_width; ++x)
        if (positive_max > 0) pixel(bar+x,top+y,
            colour(!logarithmic || log_range > 0 ? 1.0-static_cast<double>(y)/(h-1) : 0));
    frame(bar-1,top-1,bar_width+1,h+1);
    std::vector<long double> colour_ticks;
    if (!logarithmic && positive_max > 0) {
        for (int i=0; i<=4; ++i) colour_ticks.push_back(positive_max*i/4);
    } else if (log_range > 0) {
        colour_ticks = {positive_min,positive_max};
        for (double power=1; power<=positive_max; power*=10)
            for (int factor : {1,2,3,5}) {
                const double value = factor*power;
                if (value > positive_min && value < positive_max) colour_ticks.push_back(value);
            }
        std::sort(colour_ticks.begin(),colour_ticks.end());
    }
    int previous = h+30;
    for (auto value : colour_ticks) {
        const int y = std::lround((1-normalise(value))*(h-1));
        if (value != positive_max && (previous-y < 25 || y < 25)) continue;
        for (int d=0; d<=5; ++d) pixel(bar+bar_width+d,top+y,ink);
        text(bar+bar_width+11,top+y-8,number(value));
        previous = y;
    }
    if (colour_ticks.empty()) text(bar+bar_width+11,top+h/2.0-8,number(std::max(0.0L,positive_max)));
    // A separate white swatch makes zero explicit without including it in log space.
    frame(left,top+h+77,12,12);
    text(left+21,top+h+75,"White: 0",12);
    text(left+w,top+h+75,logarithmic ? "Log colour scale" : "Linear colour scale",12,1);
#ifdef GRLINA_HILBERT_CORETEXT
    CGContextFlush(context.get());
#endif
    std::vector<unsigned char> png;
    auto append = [](void* context, void* data, int size) {
        auto& bytes = *static_cast<std::vector<unsigned char>*>(context);
        const auto* begin = static_cast<unsigned char*>(data);
        bytes.insert(bytes.end(), begin, begin + size);
    };
    if (!stbi_write_png_to_func(append, &png, width, height, 4, pixels.data(), width * 4))
        throw std::runtime_error("Could not encode heatmap image");
    return png;
}

} // namespace image_detail

/** Draw a Hilbert function as PNG bytes, without opening a file.
 * xs/ys select the visible range; the full numerical table is sampled at
 * equidistant pixels by coordinate predecessors. Labels add 265 x 185 pixels.
 * Zero colour limits select the logarithmic scale automatically; shared
 * positive_min/max allow comparison across images.
 * Define STB_IMAGE_WRITE_IMPLEMENTATION before any grlina include in one .cpp.
 * GRLINA_HILBERT_CORETEXT optionally enables macOS system fonts.
 */
template <typename Function, typename Scalar>
std::vector<unsigned char> hilbert_function_image(const Function& function,
    const std::vector<Scalar>& xs, const std::vector<Scalar>& ys,
    std::size_t heatmap_width = 800, std::size_t heatmap_height = 600,
    long double positive_min = 0, long double positive_max = 0,
    const std::string& title = "Hilbert function") {
    image_detail::validate_size(heatmap_width, heatmap_height);
    if (xs.empty() || ys.empty()) throw std::invalid_argument("Hilbert image axes must be nonempty");
    hilbert_detail::validate_axis(xs);
    hilbert_detail::validate_axis(ys);
    const auto grid = function.on_grid(
        image_detail::uniform_axis<long double>(xs.front(), xs.back(), heatmap_width),
        image_detail::uniform_axis<long double>(ys.front(), ys.back(), heatmap_height));
    return image_detail::heatmap_image(grid.x_grid, grid.y_grid, grid.values,
                                       title, "Dimension", true, positive_min, positive_max);
}

/** Render within the numerical table's coordinate range, without padding. */
template <typename Function>
std::vector<unsigned char> hilbert_function_image(const Function& function,
    std::size_t width = 800, std::size_t height = 600,
    long double positive_min = 0, long double positive_max = 0,
    const std::string& title = "Hilbert function") {
    if (function.x_grid.empty() || function.y_grid.empty()) {
        auto xs = function.x_grid, ys = function.y_grid;
        if (xs.empty()) xs.push_back(0);
        if (ys.empty()) ys.push_back(0);
        return hilbert_function_image(function, xs, ys, width, height, positive_min, positive_max, title);
    }
    return hilbert_function_image(function, function.x_grid, function.y_grid,
                                  width, height, positive_min, positive_max, title);
}

/** Write existing PNG bytes only when explicitly requested. */
inline void save_png(const std::vector<unsigned char>& png, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    file.close();
}

} // namespace graded_linalg
