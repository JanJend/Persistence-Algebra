

#include "grlina/r2graded_matrix.hpp"
#include "stb_image_write.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

//TO-DO: this is ai translated from the python file, check if it does what it should

template<typename index>
void save_hilbert_png(const R2Resolution<index>& resolution, 
                      const std::string& filename,
                      int width = 500, int height = 500) {
    
    const auto& generators = resolution.d1.row_degrees;
    const auto& relations = resolution.d1.col_degrees;
    const auto& syzygies = resolution.d2.col_degrees;
    
    // Find bounds
    double x_min = std::numeric_limits<double>::max();
    double x_max = std::numeric_limits<double>::lowest();
    double y_min = std::numeric_limits<double>::max();
    double y_max = std::numeric_limits<double>::lowest();
    
    auto update_bounds = [&](const std::vector<std::pair<double,double>>& pts) {
        for (const auto& [x, y] : pts) {
            x_min = std::min(x_min, x); x_max = std::max(x_max, x);
            y_min = std::min(y_min, y); y_max = std::max(y_max, y);
        }
    };
    
    update_bounds(generators);
    update_bounds(relations);
    update_bounds(syzygies);
    
    double padding = 0.1;
    x_max += padding; y_max += padding;
    
    // Compute Hilbert function on grid
    std::vector<int> hilbert(width * height, 0);
    int max_val = 0;
    
    for (int py = 0; py < height; ++py) {
        for (int px = 0; px < width; ++px) {
            double x = x_min + (x_max - x_min) * px / (width - 1);
            double y = y_min + (y_max - y_min) * py / (height - 1);
            int val = 0;
            
            // Add generators and syzygies
            for (const auto& [gx, gy] : generators) 
                if (x >= gx && y >= gy) val++;
            for (const auto& [sx, sy] : syzygies) 
                if (x >= sx && y >= sy) val++;
            
            // Subtract relations
            for (const auto& [rx, ry] : relations) 
                if (x >= rx && y >= ry) val--;
            
            val = std::max(val, 0);
            hilbert[py * width + px] = val;
            max_val = std::max(max_val, val);
        }
    }
    
    // Find min positive value for log normalization
    int min_pos = max_val;
    for (int val : hilbert) 
        if (val > 0) min_pos = std::min(min_pos, val);
    if (min_pos == max_val && min_pos > 0) max_val = min_pos + 1;
    
    // Generate RGB image with log color mapping
    std::vector<uint8_t> pixels(width * height * 3);
    
    auto log_normalize = [&](int val) -> double {
        if (val <= 0) return 0.0;
        return (std::log(val) - std::log(min_pos)) / 
               (std::log(max_val) - std::log(min_pos));
    };
    
    auto color_map = [](double t) -> std::tuple<uint8_t,uint8_t,uint8_t> {
        // white (#d4eaff) -> blue (#006aff) -> black
        if (t < 0.5) {
            double s = t * 2.0;
            uint8_t r = 212 + (0 - 212) * s;
            uint8_t g = 234 + (106 - 234) * s;
            uint8_t b = 255;
            return {r, g, b};
        } else {
            double s = (t - 0.5) * 2.0;
            uint8_t r = 0;
            uint8_t g = 106 * (1 - s);
            uint8_t b = 255 * (1 - s);
            return {r, g, b};
        }
    };
    
    for (int py = 0; py < height; ++py) {
        for (int px = 0; px < width; ++px) {
            int idx = (height - 1 - py) * width + px; // flip vertically
            int val = hilbert[idx];
            
            auto [r, g, b] = (val > 0) ? color_map(log_normalize(val)) 
                                       : std::tuple<uint8_t,uint8_t,uint8_t>{255, 255, 255};
            
            int pidx = (py * width + px) * 3;
            pixels[pidx] = r;
            pixels[pidx + 1] = g;
            pixels[pidx + 2] = b;
        }
    }
    
    stbi_write_png(filename.c_str(), width, height, 3, pixels.data(), width * 3);
}