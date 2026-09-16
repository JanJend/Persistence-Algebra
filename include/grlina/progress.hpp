#pragma once
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace graded_linalg {

/** Console timing utility; joins its worker even if the computation throws.
 * A condition variable avoids an unconditional one-second wait on fast calls.
 */
template <typename Func, typename... Args>
auto timed_with_progress(const std::string& label, Func&& func, Args&&... args)
    -> std::invoke_result_t<Func, Args...> {
    const auto start = std::chrono::steady_clock::now();
    std::mutex mutex;
    std::condition_variable changed;
    bool done = false;
    std::thread worker([&] {
        std::unique_lock<std::mutex> lock(mutex);
        while (!changed.wait_for(lock, std::chrono::seconds(1), [&] { return done; })) {
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << '\r' << label << ": " << seconds << "s" << std::flush;
        }
    });
    auto finish = [&] {
        { std::lock_guard<std::mutex> lock(mutex); done = true; }
        changed.notify_one();
        worker.join();
    };
    auto report = [&] {
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << '\r' << label << " completed in: " << seconds << "s" << std::endl;
    };
    try {
        if constexpr (std::is_void_v<std::invoke_result_t<Func, Args...>>) {
            std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
            finish();
            report();
        } else {
            decltype(auto) result = std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
            finish();
            report();
            return result;
        }
    } catch (...) {
        if (worker.joinable()) finish();
        throw;
    }
}
} // namespace graded_linalg
