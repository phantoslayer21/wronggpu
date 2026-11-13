// SPDX-License-Identifier: MIT
// Minimal check macros shared by the proxy tests.
#pragma once
#include <stdio.h>

static int g_checks = 0, g_failures = 0;
#define CHECK(cond)                                                                  \
  do {                                                                               \
    ++g_checks;                                                                      \
    if (!(cond)) {                                                                   \
      ++g_failures;                                                                  \
      if (g_failures < 60) fprintf(stderr, "%s:%d: FAILED %s\n", __FILE__, __LINE__, #cond); \
    }                                                                                \
  } while (0)
#define CHECK_EQ(a, b)                                                                              \
  do {                                                                                              \
    ++g_checks;                                                                                     \
    auto va_ = (a);                                                                                 \
    auto vb_ = (b);                                                                                 \
    if (!(va_ == vb_)) {                                                                            \
      ++g_failures;                                                                                 \
      if (g_failures < 60)                                                                          \
        fprintf(stderr, "%s:%d: FAILED %s == %s (%lld vs %lld)\n", __FILE__, __LINE__, #a, #b,      \
                (long long)va_, (long long)vb_);                                                    \
    }                                                                                               \
  } while (0)
#define TEST_SUMMARY(name)                                                              \
  do {                                                                                  \
    printf("%s: %d checks, %d failures\n", name, g_checks, g_failures);                 \
    return g_failures ? 1 : 0;                                                          \
  } while (0)
