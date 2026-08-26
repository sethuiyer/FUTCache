#ifndef FUTCACHE_TEST_H
#define FUTCACHE_TEST_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "futcache/futcache.h"

#define TEST_ASSERT(condition)                                                     \
    do {                                                                           \
        if (!(condition)) {                                                         \
            fprintf(stderr, "%s:%d: assertion failed: %s\n",                     \
                __FILE__, __LINE__, #condition);                                    \
            return false;                                                          \
        }                                                                          \
    } while (0)

#define TEST_STATUS(expression, expected)                                          \
    do {                                                                           \
        futcache_status_t test_actual_status = (expression);                        \
        futcache_status_t test_expected_status = (expected);                        \
        if (test_actual_status != test_expected_status) {                           \
            fprintf(stderr, "%s:%d: status %s, expected %s: %s\n",               \
                __FILE__, __LINE__,                                                  \
                futcache_status_string(test_actual_status),                         \
                futcache_status_string(test_expected_status), #expression);         \
            return false;                                                          \
        }                                                                          \
    } while (0)

#define TEST_NEAR(actual, expected, tolerance)                                     \
    do {                                                                           \
        double test_actual_value = (actual);                                        \
        double test_expected_value = (expected);                                    \
        double test_tolerance_value = (tolerance);                                  \
        if (fabs(test_actual_value - test_expected_value) >                         \
            test_tolerance_value) {                                                 \
            fprintf(stderr, "%s:%d: %.17g != %.17g (tol %.3g)\n",                \
                __FILE__, __LINE__, test_actual_value, test_expected_value,          \
                test_tolerance_value);                                              \
            return false;                                                          \
        }                                                                          \
    } while (0)

typedef bool (*test_function_t)(void);

typedef struct test_case {
    const char *name;
    test_function_t function;
} test_case_t;

int run_test_cases(const char *suite_name, const test_case_t *tests, size_t count);
int futcache_test_suite(void);
int tower_test_suite(void);
int pack_test_suite(void);
int pack_vptree_test_suite(void);
int pack_stress_test_suite(void);
int box_test_suite(void);
int crdt_test_suite(void);
int embed_test_suite(void);
int select_test_suite(void);
int persist_test_suite(void);
int persist_nd_test_suite(void);
int mdl_test_suite(void);

#endif
