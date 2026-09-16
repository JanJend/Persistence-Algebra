#pragma once
#ifndef GENERAL_HPP
#define GENERAL_HPP

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>
#include <filesystem>
#include <grlina/progress.hpp>

inline std::string insert_suffix_before_extension(const std::string& filepath, const std::string& suffix, const std::string& new_extension = "") {
    std::filesystem::path path(filepath);
    std::string stem = path.stem().string();  
    std::string extension;      // filename without extension
    if (!new_extension.empty()) {
      extension = new_extension;
    } else {
        extension =  path.extension().string(); 
    }
    std::filesystem::path new_path = path.parent_path() / (stem + suffix + extension);
    return new_path.string();
}


// Preserve the global name without creating a competing overload under ADL.
using graded_linalg::timed_with_progress;

#endif // GENERAL_HPP
