#include "test.h"

int run_test_cases(const char *suite_name, const test_case_t *tests, size_t count)
{
    size_t index;
    int failures = 0;

    for (index = 0U; index < count; ++index) {
        bool passed = tests[index].function();
        printf("[%s] %s: %s\n", suite_name, tests[index].name,
            passed ? "PASS" : "FAIL");
        if (!passed) {
            failures++;
        }
    }
    return failures;
}

int main(void)
{
    int failures = 0;
    failures += futcache_test_suite();
    failures += tower_test_suite();
    failures += pack_test_suite();
    failures += box_test_suite();
    failures += crdt_test_suite();
    if (failures != 0) {
        fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }
    puts("All FUTCache tests passed.");
    return 0;
}
