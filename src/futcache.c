#define _POSIX_C_SOURCE 200809L

#include "futcache/futcache.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

enum {
    FUTCACHE_SERIAL_HEADER_SIZE = 72,
    FUTCACHE_SERIAL_INTERVAL_SIZE = 16,
    FUTCACHE_SERIAL_CRC_SIZE = 4,
    FUTCACHE_SERIAL_VERSION = 1
};

static const uint8_t futcache_serial_magic[8] = {
    (uint8_t)'F', (uint8_t)'U', (uint8_t)'T', (uint8_t)'C',
    (uint8_t)'A', (uint8_t)'C', (uint8_t)'H', (uint8_t)'E'
};

typedef struct futcache_node {
    double lower;
    double upper;
    int height;
    struct futcache_node *left;
    struct futcache_node *right;
} futcache_node_t;

struct futcache {
    double domain_min;
    double domain_max;
    double epsilon;
    futcache_allocator_t allocator;
    pthread_rwlock_t lock;
    futcache_node_t *root;
    size_t interval_count;
    long double covered_measure;
    uint64_t observations;
    uint64_t novel_observations;
    uint64_t generation;
};

typedef struct validation_summary {
    bool valid;
    bool has_node;
    double minimum_lower;
    double maximum_upper;
    int height;
    size_t count;
    long double measure;
} validation_summary_t;

static void *default_allocate(void *context, size_t size)
{
    (void)context;
    return malloc(size);
}

static void default_deallocate(void *context, void *pointer)
{
    (void)context;
    free(pointer);
}

static futcache_status_t normalize_allocator(
    const futcache_allocator_t *requested,
    futcache_allocator_t *normalized
)
{
    if (normalized == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    if (requested == NULL ||
        (requested->allocate == NULL && requested->deallocate == NULL)) {
        normalized->allocate = default_allocate;
        normalized->deallocate = default_deallocate;
        normalized->context = NULL;
        return FUTCACHE_OK;
    }

    if (requested->allocate == NULL || requested->deallocate == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    *normalized = *requested;
    return FUTCACHE_OK;
}

static bool config_is_valid(const futcache_config_t *config)
{
    double width;

    if (config == NULL || !isfinite(config->domain_min) ||
        !isfinite(config->domain_max) || !isfinite(config->epsilon) ||
        config->domain_min > config->domain_max || config->epsilon < 0.0) {
        return false;
    }

    width = config->domain_max - config->domain_min;
    return isfinite(width);
}

static bool floating_environment_is_supported(void)
{
    return FLT_EVAL_METHOD == 0 && fegetround() == FE_TONEAREST;
}

static bool point_is_valid(const futcache_t *cache, double x)
{
    return isfinite(x) && x >= cache->domain_min && x <= cache->domain_max;
}

static uint64_t increment_saturating(uint64_t value)
{
    return value == UINT64_MAX ? UINT64_MAX : value + UINT64_C(1);
}

static int node_height(const futcache_node_t *node)
{
    return node == NULL ? 0 : node->height;
}

static int maximum_int(int left, int right)
{
    return left > right ? left : right;
}

static void update_height(futcache_node_t *node)
{
    node->height = 1 + maximum_int(node_height(node->left), node_height(node->right));
}

static futcache_node_t *rotate_right(futcache_node_t *root)
{
    futcache_node_t *pivot;
    futcache_node_t *transfer;

    if (root == NULL || root->left == NULL) {
        return root;
    }
    pivot = root->left;
    transfer = pivot->right;

    pivot->right = root;
    root->left = transfer;
    update_height(root);
    update_height(pivot);
    return pivot;
}

static futcache_node_t *rotate_left(futcache_node_t *root)
{
    futcache_node_t *pivot;
    futcache_node_t *transfer;

    if (root == NULL || root->right == NULL) {
        return root;
    }
    pivot = root->right;
    transfer = pivot->left;

    pivot->left = root;
    root->right = transfer;
    update_height(root);
    update_height(pivot);
    return pivot;
}

static futcache_node_t *rebalance(futcache_node_t *root)
{
    int balance;

    if (root == NULL) {
        return NULL;
    }

    update_height(root);
    balance = node_height(root->left) - node_height(root->right);

    if (balance > 1) {
        if (node_height(root->left->left) < node_height(root->left->right)) {
            root->left = rotate_left(root->left);
        }
        return rotate_right(root);
    }

    if (balance < -1) {
        if (node_height(root->right->right) < node_height(root->right->left)) {
            root->right = rotate_right(root->right);
        }
        return rotate_left(root);
    }

    return root;
}

static futcache_node_t *node_create(
    futcache_t *cache,
    double lower,
    double upper
)
{
    futcache_node_t *node = cache->allocator.allocate(
        cache->allocator.context,
        sizeof(*node)
    );

    if (node == NULL) {
        return NULL;
    }

    node->lower = lower;
    node->upper = upper;
    node->height = 1;
    node->left = NULL;
    node->right = NULL;
    return node;
}

static futcache_node_t *tree_insert(
    futcache_node_t *root,
    futcache_node_t *node
)
{
    if (root == NULL) {
        return node;
    }

    if (node->lower < root->lower) {
        root->left = tree_insert(root->left, node);
    } else {
        root->right = tree_insert(root->right, node);
    }
    return rebalance(root);
}

static futcache_node_t *minimum_node(futcache_node_t *root)
{
    while (root->left != NULL) {
        root = root->left;
    }
    return root;
}

static futcache_node_t *tree_delete(
    futcache_t *cache,
    futcache_node_t *root,
    double lower
)
{
    if (root == NULL) {
        return NULL;
    }

    if (lower < root->lower) {
        root->left = tree_delete(cache, root->left, lower);
    } else if (lower > root->lower) {
        root->right = tree_delete(cache, root->right, lower);
    } else if (root->left == NULL || root->right == NULL) {
        futcache_node_t *replacement = root->left != NULL ? root->left : root->right;
        cache->allocator.deallocate(cache->allocator.context, root);
        return replacement;
    } else {
        futcache_node_t *successor = minimum_node(root->right);
        root->lower = successor->lower;
        root->upper = successor->upper;
        root->right = tree_delete(cache, root->right, successor->lower);
    }

    return rebalance(root);
}

static void tree_destroy(futcache_t *cache, futcache_node_t *root)
{
    if (root == NULL) {
        return;
    }
    tree_destroy(cache, root->left);
    tree_destroy(cache, root->right);
    cache->allocator.deallocate(cache->allocator.context, root);
}

static futcache_node_t *find_overlap(
    futcache_node_t *root,
    double lower,
    double upper
)
{
    while (root != NULL) {
        if (root->upper < lower) {
            root = root->right;
        } else if (root->lower > upper) {
            root = root->left;
        } else {
            return root;
        }
    }
    return NULL;
}

static bool tree_covers_point(const futcache_node_t *root, double x)
{
    while (root != NULL) {
        if (x < root->lower) {
            root = root->left;
        } else if (x > root->upper) {
            root = root->right;
        } else {
            return true;
        }
    }
    return false;
}

static void make_epsilon_ball(
    const futcache_t *cache,
    double x,
    double *out_lower,
    double *out_upper
)
{
    double lower = x - cache->epsilon;
    double upper = x + cache->epsilon;

    /*
     * Convert the exact real-valued ball to representable doubles using
     * inward-directed endpoints. Knuth TwoSum recovers the rounding residual
     * without changing the process-wide floating-point rounding mode.
     */
    if (isfinite(lower)) {
        double virtual_epsilon = lower - x;
        double residual = (x - (lower - virtual_epsilon)) +
            (-cache->epsilon - virtual_epsilon);
        if (residual > 0.0) {
            lower = nextafter(lower, INFINITY);
        }
    }
    if (isfinite(upper)) {
        double virtual_epsilon = upper - x;
        double residual = (x - (upper - virtual_epsilon)) +
            (cache->epsilon - virtual_epsilon);
        if (residual < 0.0) {
            upper = nextafter(upper, -INFINITY);
        }
    }

    if (lower == 0.0) {
        lower = 0.0;
    }
    if (upper == 0.0) {
        upper = 0.0;
    }
    *out_lower = !isfinite(lower) || lower <= cache->domain_min
        ? cache->domain_min
        : lower;
    *out_upper = !isfinite(upper) || upper >= cache->domain_max
        ? cache->domain_max
        : upper;
}

/* Allocation happens before mutation, so an allocation failure is atomic. */
static futcache_status_t merge_ball_locked(
    futcache_t *cache,
    double lower,
    double upper
)
{
    futcache_node_t *overlap = find_overlap(cache->root, lower, upper);
    futcache_node_t *replacement;

    if (overlap != NULL && overlap->lower <= lower && overlap->upper >= upper) {
        return FUTCACHE_OK;
    }

    replacement = node_create(cache, lower, upper);
    if (replacement == NULL) {
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }

    while ((overlap = find_overlap(cache->root, lower, upper)) != NULL) {
        double old_lower = overlap->lower;
        double old_upper = overlap->upper;

        if (old_lower < lower) {
            lower = old_lower;
        }
        if (old_upper > upper) {
            upper = old_upper;
        }

        cache->covered_measure -= (long double)old_upper - (long double)old_lower;
        cache->root = tree_delete(cache, cache->root, old_lower);
        cache->interval_count--;
    }

    replacement->lower = lower;
    replacement->upper = upper;
    cache->root = tree_insert(cache->root, replacement);
    cache->interval_count++;
    cache->covered_measure += (long double)upper - (long double)lower;
    return FUTCACHE_OK;
}

static futcache_status_t read_lock(const futcache_t *cache)
{
    return pthread_rwlock_rdlock((pthread_rwlock_t *)&cache->lock) == 0
        ? FUTCACHE_OK
        : FUTCACHE_ERROR_SYSTEM;
}

static futcache_status_t write_lock(futcache_t *cache)
{
    return pthread_rwlock_wrlock(&cache->lock) == 0
        ? FUTCACHE_OK
        : FUTCACHE_ERROR_SYSTEM;
}

static futcache_status_t unlock_cache(const futcache_t *cache)
{
    return pthread_rwlock_unlock((pthread_rwlock_t *)&cache->lock) == 0
        ? FUTCACHE_OK
        : FUTCACHE_ERROR_SYSTEM;
}

static size_t memory_usage_locked(const futcache_t *cache)
{
    if (cache->interval_count > (SIZE_MAX - sizeof(*cache)) / sizeof(futcache_node_t)) {
        return SIZE_MAX;
    }
    return sizeof(*cache) + cache->interval_count * sizeof(futcache_node_t);
}

static bool fully_covered_locked(const futcache_t *cache)
{
    return cache->interval_count == 1U && cache->root != NULL &&
        cache->root->lower == cache->domain_min &&
        cache->root->upper == cache->domain_max;
}

static void copy_intervals_in_order(
    const futcache_node_t *root,
    futcache_interval_t *intervals,
    size_t *index
)
{
    if (root == NULL) {
        return;
    }
    copy_intervals_in_order(root->left, intervals, index);
    intervals[*index].lower = root->lower;
    intervals[*index].upper = root->upper;
    (*index)++;
    copy_intervals_in_order(root->right, intervals, index);
}

static validation_summary_t validate_node(
    const futcache_t *cache,
    const futcache_node_t *node
)
{
    validation_summary_t result = {
        true, false, 0.0, 0.0, 0, 0U, 0.0L
    };
    validation_summary_t left;
    validation_summary_t right;
    int balance;

    if (node == NULL) {
        return result;
    }

    left = validate_node(cache, node->left);
    right = validate_node(cache, node->right);
    result.valid = left.valid && right.valid && isfinite(node->lower) &&
        isfinite(node->upper) && node->lower <= node->upper &&
        node->lower >= cache->domain_min && node->upper <= cache->domain_max;

    if (left.has_node && left.maximum_upper >= node->lower) {
        result.valid = false;
    }
    if (right.has_node && node->upper >= right.minimum_lower) {
        result.valid = false;
    }

    balance = left.height - right.height;
    result.height = 1 + maximum_int(left.height, right.height);
    if (node->height != result.height || balance < -1 || balance > 1) {
        result.valid = false;
    }

    if (right.count == SIZE_MAX || left.count > SIZE_MAX - right.count - 1U) {
        result.valid = false;
        result.count = SIZE_MAX;
    } else {
        result.count = left.count + right.count + 1U;
    }
    result.measure = left.measure + right.measure +
        ((long double)node->upper - (long double)node->lower);
    result.has_node = true;
    result.minimum_lower = left.has_node ? left.minimum_lower : node->lower;
    result.maximum_upper = right.has_node ? right.maximum_upper : node->upper;
    return result;
}

static bool double_serialization_supported(void)
{
    return CHAR_BIT == 8 && sizeof(double) == sizeof(uint64_t) &&
        FLT_RADIX == 2 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024;
}

static void write_u16_le(uint8_t *destination, uint16_t value)
{
    destination[0] = (uint8_t)(value & UINT16_C(0xff));
    destination[1] = (uint8_t)((value >> 8U) & UINT16_C(0xff));
}

static void write_u32_le(uint8_t *destination, uint32_t value)
{
    size_t index;
    for (index = 0U; index < 4U; ++index) {
        destination[index] = (uint8_t)((value >> (index * 8U)) & UINT32_C(0xff));
    }
}

static void write_u64_le(uint8_t *destination, uint64_t value)
{
    size_t index;
    for (index = 0U; index < 8U; ++index) {
        destination[index] = (uint8_t)((value >> (index * 8U)) & UINT64_C(0xff));
    }
}

static uint16_t read_u16_le(const uint8_t *source)
{
    return (uint16_t)((uint16_t)source[0] | ((uint16_t)source[1] << 8U));
}

static uint32_t read_u32_le(const uint8_t *source)
{
    uint32_t value = 0U;
    size_t index;
    for (index = 0U; index < 4U; ++index) {
        value |= (uint32_t)source[index] << (index * 8U);
    }
    return value;
}

static uint64_t read_u64_le(const uint8_t *source)
{
    uint64_t value = UINT64_C(0);
    size_t index;
    for (index = 0U; index < 8U; ++index) {
        value |= (uint64_t)source[index] << (index * 8U);
    }
    return value;
}

static void write_double_le(uint8_t *destination, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    write_u64_le(destination, bits);
}

static double read_double_le(const uint8_t *source)
{
    uint64_t bits = read_u64_le(source);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t crc32_bytes(const uint8_t *data, size_t size)
{
    uint32_t crc = UINT32_C(0xffffffff);
    size_t index;

    for (index = 0U; index < size; ++index) {
        unsigned int bit;
        crc ^= (uint32_t)data[index];
        for (bit = 0U; bit < 8U; ++bit) {
            uint32_t mask = (uint32_t)(-(int32_t)(crc & UINT32_C(1)));
            crc = (crc >> 1U) ^ (UINT32_C(0xedb88320) & mask);
        }
    }
    return ~crc;
}

static void serialize_intervals_in_order(
    const futcache_node_t *root,
    uint8_t *destination,
    size_t *offset
)
{
    if (root == NULL) {
        return;
    }
    serialize_intervals_in_order(root->left, destination, offset);
    write_double_le(destination + *offset, root->lower);
    write_double_le(destination + *offset + 8U, root->upper);
    *offset += FUTCACHE_SERIAL_INTERVAL_SIZE;
    serialize_intervals_in_order(root->right, destination, offset);
}

void futcache_config_init(futcache_config_t *config)
{
    if (config == NULL) {
        return;
    }
    config->domain_min = 0.0;
    config->domain_max = 1.0;
    config->epsilon = 0.1;
    config->allocator.allocate = NULL;
    config->allocator.deallocate = NULL;
    config->allocator.context = NULL;
}

futcache_status_t futcache_create(
    const futcache_config_t *config,
    futcache_t **out_cache
)
{
    futcache_allocator_t allocator;
    futcache_t *cache;
    futcache_status_t status;

    if (out_cache == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    *out_cache = NULL;
    if (!floating_environment_is_supported()) {
        return FUTCACHE_ERROR_UNSUPPORTED_PLATFORM;
    }
    if (!config_is_valid(config)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    status = normalize_allocator(&config->allocator, &allocator);
    if (status != FUTCACHE_OK) {
        return status;
    }

    cache = allocator.allocate(allocator.context, sizeof(*cache));
    if (cache == NULL) {
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    memset(cache, 0, sizeof(*cache));
    cache->domain_min = config->domain_min == 0.0 ? 0.0 : config->domain_min;
    cache->domain_max = config->domain_max == 0.0 ? 0.0 : config->domain_max;
    cache->epsilon = config->epsilon == 0.0 ? 0.0 : config->epsilon;
    cache->allocator = allocator;

    if (pthread_rwlock_init(&cache->lock, NULL) != 0) {
        allocator.deallocate(allocator.context, cache);
        return FUTCACHE_ERROR_SYSTEM;
    }

    *out_cache = cache;
    return FUTCACHE_OK;
}

void futcache_destroy(futcache_t *cache)
{
    futcache_allocator_t allocator;

    if (cache == NULL) {
        return;
    }
    allocator = cache->allocator;
    tree_destroy(cache, cache->root);
    (void)pthread_rwlock_destroy(&cache->lock);
    allocator.deallocate(allocator.context, cache);
}

futcache_status_t futcache_is_novel(
    const futcache_t *cache,
    double x,
    bool *out_is_novel
)
{
    futcache_status_t status;
    bool is_novel;

    if (cache == NULL || out_is_novel == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!point_is_valid(cache, x)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    status = read_lock(cache);
    if (status != FUTCACHE_OK) {
        return status;
    }
    is_novel = !tree_covers_point(cache->root, x);
    status = unlock_cache(cache);
    if (status == FUTCACHE_OK) {
        *out_is_novel = is_novel;
    }
    return status;
}

futcache_status_t futcache_observe(
    futcache_t *cache,
    double x,
    bool *out_was_novel
)
{
    futcache_status_t status;
    futcache_status_t unlock_status;
    bool was_novel;
    double lower;
    double upper;

    if (cache == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!floating_environment_is_supported()) {
        return FUTCACHE_ERROR_UNSUPPORTED_PLATFORM;
    }
    if (!point_is_valid(cache, x)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    status = write_lock(cache);
    if (status != FUTCACHE_OK) {
        return status;
    }

    was_novel = !tree_covers_point(cache->root, x);
    make_epsilon_ball(cache, x, &lower, &upper);
    status = merge_ball_locked(cache, lower, upper);
    if (status == FUTCACHE_OK) {
        cache->observations = increment_saturating(cache->observations);
        cache->generation = increment_saturating(cache->generation);
        if (was_novel) {
            cache->novel_observations = increment_saturating(cache->novel_observations);
        }
    }

    unlock_status = unlock_cache(cache);
    if (status == FUTCACHE_OK && unlock_status != FUTCACHE_OK) {
        status = unlock_status;
    }
    if (status == FUTCACHE_OK && out_was_novel != NULL) {
        *out_was_novel = was_novel;
    }
    return status;
}

futcache_status_t futcache_get_stats(
    const futcache_t *cache,
    futcache_stats_t *out_stats
)
{
    futcache_status_t status;
    futcache_stats_t stats;

    if (cache == NULL || out_stats == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    status = read_lock(cache);
    if (status != FUTCACHE_OK) {
        return status;
    }

    stats.observations = cache->observations;
    stats.novel_observations = cache->novel_observations;
    stats.generation = cache->generation;
    stats.interval_count = cache->interval_count;
    stats.tree_height = (size_t)node_height(cache->root);
    stats.covered_measure = (double)cache->covered_measure;
    stats.memory_bytes = memory_usage_locked(cache);
    stats.fully_covered = fully_covered_locked(cache);

    status = unlock_cache(cache);
    if (status == FUTCACHE_OK) {
        *out_stats = stats;
    }
    return status;
}

futcache_status_t futcache_get_parameters(
    const futcache_t *cache,
    futcache_parameters_t *out_parameters
)
{
    futcache_status_t status;
    futcache_parameters_t parameters;

    if (cache == NULL || out_parameters == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    status = read_lock(cache);
    if (status != FUTCACHE_OK) {
        return status;
    }
    parameters.domain_min = cache->domain_min;
    parameters.domain_max = cache->domain_max;
    parameters.epsilon = cache->epsilon;
    status = unlock_cache(cache);
    if (status == FUTCACHE_OK) {
        *out_parameters = parameters;
    }
    return status;
}

futcache_status_t futcache_copy_intervals(
    const futcache_t *cache,
    futcache_interval_t *intervals,
    size_t *inout_count
)
{
    futcache_status_t status;
    size_t capacity;
    size_t required;
    size_t index = 0U;

    if (cache == NULL || inout_count == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    capacity = *inout_count;
    if (intervals == NULL && capacity != 0U) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }

    status = read_lock(cache);
    if (status != FUTCACHE_OK) {
        return status;
    }
    required = cache->interval_count;
    if (intervals != NULL && capacity >= required) {
        copy_intervals_in_order(cache->root, intervals, &index);
    }
    status = unlock_cache(cache);
    *inout_count = required;
    if (status != FUTCACHE_OK) {
        return status;
    }
    if (intervals == NULL) {
        return capacity == 0U ? FUTCACHE_OK : FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    return capacity >= required ? FUTCACHE_OK : FUTCACHE_ERROR_BUFFER_TOO_SMALL;
}

futcache_status_t futcache_clear(futcache_t *cache)
{
    futcache_status_t status;
    futcache_status_t unlock_status;

    if (cache == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    status = write_lock(cache);
    if (status != FUTCACHE_OK) {
        return status;
    }

    tree_destroy(cache, cache->root);
    cache->root = NULL;
    cache->interval_count = 0U;
    cache->covered_measure = 0.0L;
    cache->observations = UINT64_C(0);
    cache->novel_observations = UINT64_C(0);
    cache->generation = increment_saturating(cache->generation);

    unlock_status = unlock_cache(cache);
    return unlock_status;
}

futcache_status_t futcache_validate(const futcache_t *cache)
{
    futcache_status_t status;
    validation_summary_t summary;
    long double difference;
    long double tolerance;

    if (cache == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    status = read_lock(cache);
    if (status != FUTCACHE_OK) {
        return status;
    }

    summary = validate_node(cache, cache->root);
    difference = fabsl(summary.measure - cache->covered_measure);
    tolerance = 64.0L * LDBL_EPSILON * (1.0L + fabsl(summary.measure));
    if (!summary.valid || summary.count != cache->interval_count ||
        difference > tolerance ||
        cache->interval_count > (size_t)UINT64_MAX ||
        (uint64_t)cache->interval_count > cache->novel_observations ||
        cache->novel_observations > cache->observations ||
        (cache->interval_count == 0U && cache->observations != UINT64_C(0)) ||
        cache->generation < cache->observations) {
        status = FUTCACHE_ERROR_CORRUPT_DATA;
    }

    {
        futcache_status_t unlock_status = unlock_cache(cache);
        if (status == FUTCACHE_OK && unlock_status != FUTCACHE_OK) {
            status = unlock_status;
        }
    }
    return status;
}

futcache_status_t futcache_serialize(
    const futcache_t *cache,
    void *buffer,
    size_t buffer_size,
    size_t *out_size
)
{
    futcache_status_t status;
    futcache_status_t unlock_status;
    size_t required;
    size_t offset;
    uint8_t *bytes = buffer;

    if (cache == NULL || out_size == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!double_serialization_supported()) {
        return FUTCACHE_ERROR_UNSUPPORTED_PLATFORM;
    }

    status = read_lock(cache);
    if (status != FUTCACHE_OK) {
        return status;
    }
    if (cache->interval_count >
        (SIZE_MAX - FUTCACHE_SERIAL_HEADER_SIZE - FUTCACHE_SERIAL_CRC_SIZE) /
            FUTCACHE_SERIAL_INTERVAL_SIZE ||
        cache->interval_count > (size_t)UINT64_MAX) {
        (void)unlock_cache(cache);
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    required = FUTCACHE_SERIAL_HEADER_SIZE +
        cache->interval_count * FUTCACHE_SERIAL_INTERVAL_SIZE +
        FUTCACHE_SERIAL_CRC_SIZE;
    *out_size = required;

    if (buffer == NULL) {
        return unlock_cache(cache);
    }
    if (buffer_size < required) {
        (void)unlock_cache(cache);
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }

    memcpy(bytes, futcache_serial_magic, sizeof(futcache_serial_magic));
    write_u16_le(bytes + 8U, (uint16_t)FUTCACHE_SERIAL_VERSION);
    write_u16_le(bytes + 10U, (uint16_t)FUTCACHE_SERIAL_HEADER_SIZE);
    write_u32_le(bytes + 12U, UINT32_C(0));
    write_double_le(bytes + 16U, cache->epsilon);
    write_double_le(bytes + 24U, cache->domain_min);
    write_double_le(bytes + 32U, cache->domain_max);
    write_u64_le(bytes + 40U, cache->observations);
    write_u64_le(bytes + 48U, cache->novel_observations);
    write_u64_le(bytes + 56U, cache->generation);
    write_u64_le(bytes + 64U, (uint64_t)cache->interval_count);
    offset = FUTCACHE_SERIAL_HEADER_SIZE;
    serialize_intervals_in_order(cache->root, bytes, &offset);
    write_u32_le(bytes + offset, crc32_bytes(bytes, offset));

    unlock_status = unlock_cache(cache);
    return unlock_status;
}

futcache_status_t futcache_deserialize(
    const void *data,
    size_t data_size,
    const futcache_allocator_t *allocator,
    futcache_t **out_cache
)
{
    const uint8_t *bytes = data;
    uint64_t serialized_count;
    size_t interval_count;
    size_t expected_size;
    size_t index;
    uint32_t expected_crc;
    uint32_t actual_crc;
    futcache_config_t config;
    futcache_t *cache = NULL;
    futcache_status_t status;
    double previous_upper = 0.0;
    bool has_previous = false;

    if (out_cache == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    *out_cache = NULL;
    if (data == NULL || data_size <
        FUTCACHE_SERIAL_HEADER_SIZE + FUTCACHE_SERIAL_CRC_SIZE) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!double_serialization_supported()) {
        return FUTCACHE_ERROR_UNSUPPORTED_PLATFORM;
    }
    if (memcmp(bytes, futcache_serial_magic, sizeof(futcache_serial_magic)) != 0 ||
        read_u16_le(bytes + 8U) != (uint16_t)FUTCACHE_SERIAL_VERSION ||
        read_u16_le(bytes + 10U) != (uint16_t)FUTCACHE_SERIAL_HEADER_SIZE ||
        read_u32_le(bytes + 12U) != UINT32_C(0)) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    serialized_count = read_u64_le(bytes + 64U);
    if (serialized_count > (uint64_t)SIZE_MAX) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }
    interval_count = (size_t)serialized_count;
    if (interval_count >
        (SIZE_MAX - FUTCACHE_SERIAL_HEADER_SIZE - FUTCACHE_SERIAL_CRC_SIZE) /
            FUTCACHE_SERIAL_INTERVAL_SIZE) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }
    expected_size = FUTCACHE_SERIAL_HEADER_SIZE +
        interval_count * FUTCACHE_SERIAL_INTERVAL_SIZE +
        FUTCACHE_SERIAL_CRC_SIZE;
    if (data_size != expected_size) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    expected_crc = read_u32_le(bytes + data_size - FUTCACHE_SERIAL_CRC_SIZE);
    actual_crc = crc32_bytes(bytes, data_size - FUTCACHE_SERIAL_CRC_SIZE);
    if (expected_crc != actual_crc) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    futcache_config_init(&config);
    config.epsilon = read_double_le(bytes + 16U);
    config.domain_min = read_double_le(bytes + 24U);
    config.domain_max = read_double_le(bytes + 32U);
    if (allocator != NULL) {
        config.allocator = *allocator;
    }
    if (!config_is_valid(&config) ||
        read_u64_le(bytes + 48U) > read_u64_le(bytes + 40U)) {
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    status = futcache_create(&config, &cache);
    if (status != FUTCACHE_OK) {
        return status;
    }

    for (index = 0U; index < interval_count; ++index) {
        size_t offset = FUTCACHE_SERIAL_HEADER_SIZE +
            index * FUTCACHE_SERIAL_INTERVAL_SIZE;
        double lower = read_double_le(bytes + offset);
        double upper = read_double_le(bytes + offset + 8U);
        futcache_node_t *node;

        if (lower == 0.0) {
            lower = 0.0;
        }
        if (upper == 0.0) {
            upper = 0.0;
        }

        if (!isfinite(lower) || !isfinite(upper) || lower > upper ||
            lower < cache->domain_min || upper > cache->domain_max ||
            (has_previous && previous_upper >= lower)) {
            futcache_destroy(cache);
            return FUTCACHE_ERROR_CORRUPT_DATA;
        }
        node = node_create(cache, lower, upper);
        if (node == NULL) {
            futcache_destroy(cache);
            return FUTCACHE_ERROR_OUT_OF_MEMORY;
        }
        cache->root = tree_insert(cache->root, node);
        cache->interval_count++;
        cache->covered_measure += (long double)upper - (long double)lower;
        previous_upper = upper;
        has_previous = true;
    }

    cache->observations = read_u64_le(bytes + 40U);
    cache->novel_observations = read_u64_le(bytes + 48U);
    cache->generation = read_u64_le(bytes + 56U);
    status = futcache_validate(cache);
    if (status != FUTCACHE_OK) {
        futcache_destroy(cache);
        return status;
    }

    *out_cache = cache;
    return FUTCACHE_OK;
}

const char *futcache_status_string(futcache_status_t status)
{
    switch (status) {
    case FUTCACHE_OK:
        return "success";
    case FUTCACHE_ERROR_INVALID_ARGUMENT:
        return "invalid argument";
    case FUTCACHE_ERROR_OUT_OF_MEMORY:
        return "out of memory";
    case FUTCACHE_ERROR_OUT_OF_RANGE:
        return "value out of range";
    case FUTCACHE_ERROR_BUFFER_TOO_SMALL:
        return "buffer too small";
    case FUTCACHE_ERROR_CORRUPT_DATA:
        return "corrupt data or invariant violation";
    case FUTCACHE_ERROR_UNSUPPORTED_PLATFORM:
        return "unsupported platform";
    case FUTCACHE_ERROR_SYSTEM:
        return "system synchronization error";
    default:
        return "unknown FUTCache status";
    }
}
