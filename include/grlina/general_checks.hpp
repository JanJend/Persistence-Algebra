/** Build policy for defensive checks; explicit validators always remain active. */
#pragma once

// GCC/Clang define __OPTIMIZE__ for -O1/-O2/-O3/-Os/-Og. NDEBUG also
// supports conventional Release builds on compilers without that macro.
// Define GRLINA_ENABLE_CHECKS=0 or 1 consistently across translation units
// to override the default (e.g. optimized diagnostic builds).
#ifndef GRLINA_ENABLE_CHECKS
#if defined(__OPTIMIZE__) || defined(NDEBUG)
#define GRLINA_ENABLE_CHECKS 0
#else
#define GRLINA_ENABLE_CHECKS 1
#endif
#endif

#if GRLINA_ENABLE_CHECKS != 0 && GRLINA_ENABLE_CHECKS != 1
#error "GRLINA_ENABLE_CHECKS must be 0 or 1"
#endif

#if GRLINA_ENABLE_CHECKS
#include <cstdio>
#include <cstdlib>
#define GRLINA_DEBUG_CHECK(...) do { __VA_ARGS__; } while (false)
#define GRLINA_ASSERT(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Assertion failed: %s (%s:%d)\n", #condition, __FILE__, __LINE__); \
    std::abort(); \
} } while (false)
#else
#define GRLINA_DEBUG_CHECK(...) do {} while (false)
#define GRLINA_ASSERT(condition) do {} while (false)
#endif
