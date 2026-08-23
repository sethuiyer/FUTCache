#define _POSIX_C_SOURCE 200809L

#include "futcache/tower.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct tower_level {
    size_t cell_count;
    size_t word_count;
    size_t discovered_count;
    uint64_t *occupancy;
    size_t *fenwick;
    size_t *discovery_log;
} tower_level_t;

struct futcache_tower {
    double domain_min;
    double domain_max;
    size_t level_count;
    size_t root_cells;
    size_t total_cells;
    size_t memory_bytes;
    uint64_t observations;
    uint64_t generation;
    futcache_allocator_t allocator;
    pthread_rwlock_t lock;
    tower_level_t *levels;
};

static void *tower_default_allocate(void *context, size_t size)
{
    (void)context;
    return malloc(size);
}

static void tower_default_deallocate(void *context, void *pointer)
{
    (void)context;
    free(pointer);
}

static futcache_status_t tower_normalize_allocator(
    const futcache_allocator_t *requested,
    futcache_allocator_t *normalized
)
{
    if (normalized == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (requested == NULL ||
        (requested->allocate == NULL && requested->deallocate == NULL)) {
        normalized->allocate = tower_default_allocate;
        normalized->deallocate = tower_default_deallocate;
        normalized->context = NULL;
        return FUTCACHE_OK;
    }
    if (requested->allocate == NULL || requested->deallocate == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    *normalized = *requested;
    return FUTCACHE_OK;
}

static uint64_t tower_increment_saturating(uint64_t value)
{
    return value == UINT64_MAX ? UINT64_MAX : value + UINT64_C(1);
}

static bool tower_config_is_valid(const futcache_tower_config_t *config)
{
    double width;

    if (config == NULL || !isfinite(config->domain_min) ||
        !isfinite(config->domain_max) ||
        config->domain_min >= config->domain_max ||
        config->level_count == 0U || config->root_cells == 0U) {
        return false;
    }
    width = config->domain_max - config->domain_min;
    return isfinite(width);
}

static bool tower_floating_environment_is_supported(void)
{
    return FLT_EVAL_METHOD == 0 && fegetround() == FE_TONEAREST;
}

static bool checked_add_size(size_t left, size_t right, size_t *out)
{
    if (left > SIZE_MAX - right) {
        return false;
    }
    *out = left + right;
    return true;
}

static bool checked_multiply_size(size_t left, size_t right, size_t *out)
{
    if (left != 0U && right > SIZE_MAX / left) {
        return false;
    }
    *out = left * right;
    return true;
}

static void *tower_allocate_zeroed(
    const futcache_allocator_t *allocator,
    size_t size
)
{
    void *memory = allocator->allocate(allocator->context, size);
    if (memory != NULL) {
        memset(memory, 0, size);
    }
    return memory;
}

static bool level_layout(
    size_t cells,
    size_t *out_words,
    size_t *out_occupancy_bytes,
    size_t *out_fenwick_bytes,
    size_t *out_log_bytes,
    size_t *out_total_bytes
)
{
    size_t words = cells / 64U + (cells % 64U == 0U ? 0U : 1U);
    size_t occupancy_bytes;
    size_t fenwick_entries;
    size_t fenwick_bytes;
    size_t log_bytes;
    size_t total;

    if (!checked_multiply_size(words, sizeof(uint64_t), &occupancy_bytes) ||
        !checked_add_size(cells, 1U, &fenwick_entries) ||
        !checked_multiply_size(fenwick_entries, sizeof(size_t), &fenwick_bytes) ||
        !checked_multiply_size(cells, sizeof(size_t), &log_bytes) ||
        !checked_add_size(occupancy_bytes, fenwick_bytes, &total) ||
        !checked_add_size(total, log_bytes, &total)) {
        return false;
    }

    *out_words = words;
    *out_occupancy_bytes = occupancy_bytes;
    *out_fenwick_bytes = fenwick_bytes;
    *out_log_bytes = log_bytes;
    *out_total_bytes = total;
    return true;
}

static futcache_status_t tower_read_lock(const futcache_tower_t *tower)
{
    return pthread_rwlock_rdlock((pthread_rwlock_t *)&tower->lock) == 0
        ? FUTCACHE_OK
        : FUTCACHE_ERROR_SYSTEM;
}

static futcache_status_t tower_write_lock(futcache_tower_t *tower)
{
    return pthread_rwlock_wrlock(&tower->lock) == 0
        ? FUTCACHE_OK
        : FUTCACHE_ERROR_SYSTEM;
}

static futcache_status_t tower_unlock(const futcache_tower_t *tower)
{
    return pthread_rwlock_unlock((pthread_rwlock_t *)&tower->lock) == 0
        ? FUTCACHE_OK
        : FUTCACHE_ERROR_SYSTEM;
}

static bool tower_point_is_valid(const futcache_tower_t *tower, double x)
{
    return isfinite(x) && x >= tower->domain_min && x <= tower->domain_max;
}

static size_t tower_map_cell(
    const futcache_tower_t *tower,
    size_t cell_count,
    double x
)
{
    long double numerator;
    long double denominator;
    long double scaled;
    size_t cell;

    if (x == tower->domain_max) {
        return cell_count - 1U;
    }
    numerator = (long double)x - (long double)tower->domain_min;
    denominator = (long double)tower->domain_max - (long double)tower->domain_min;
    scaled = (numerator / denominator) * (long double)cell_count;
    if (scaled <= 0.0L) {
        return 0U;
    }
    if (scaled >= (long double)cell_count) {
        return cell_count - 1U;
    }
    cell = (size_t)floorl(scaled);
    return cell < cell_count ? cell : cell_count - 1U;
}

static size_t tower_map_level_cell(
    const futcache_tower_t *tower,
    size_t level,
    double x
)
{
    size_t finest_level = tower->level_count - 1U;
    size_t finest_cell = tower_map_cell(
        tower,
        tower->levels[finest_level].cell_count,
        x
    );
    return finest_cell >> (finest_level - level);
}

static bool level_is_seen(const tower_level_t *level, size_t cell)
{
    size_t word = cell / 64U;
    unsigned int bit = (unsigned int)(cell % 64U);
    return (level->occupancy[word] & (UINT64_C(1) << bit)) != 0U;
}

static void level_mark_seen(tower_level_t *level, size_t cell)
{
    size_t word = cell / 64U;
    unsigned int bit = (unsigned int)(cell % 64U);
    size_t index;

    level->occupancy[word] |= UINT64_C(1) << bit;
    level->discovery_log[level->discovered_count] = cell;
    level->discovered_count++;

    index = cell + 1U;
    while (index <= level->cell_count) {
        size_t low_bit;
        level->fenwick[index]++;
        low_bit = index & (~index + 1U);
        if (low_bit > level->cell_count - index) {
            break;
        }
        index += low_bit;
    }
}

static size_t fenwick_prefix(const tower_level_t *level, size_t cell_inclusive)
{
    size_t index = cell_inclusive + 1U;
    size_t count = 0U;

    while (index != 0U) {
        count += level->fenwick[index];
        index -= index & (~index + 1U);
    }
    return count;
}

static size_t highest_power_of_two_not_greater_than(size_t value)
{
    size_t power = 1U;
    while (power <= value / 2U) {
        power <<= 1U;
    }
    return power;
}

static void tower_release_levels(futcache_tower_t *tower)
{
    size_t level_index;

    if (tower->levels == NULL) {
        return;
    }
    for (level_index = 0U; level_index < tower->level_count; ++level_index) {
        tower_level_t *level = &tower->levels[level_index];
        if (level->occupancy != NULL) {
            tower->allocator.deallocate(tower->allocator.context, level->occupancy);
        }
        if (level->fenwick != NULL) {
            tower->allocator.deallocate(tower->allocator.context, level->fenwick);
        }
        if (level->discovery_log != NULL) {
            tower->allocator.deallocate(tower->allocator.context, level->discovery_log);
        }
    }
    tower->allocator.deallocate(tower->allocator.context, tower->levels);
    tower->levels = NULL;
}

void futcache_tower_config_init(futcache_tower_config_t *config)
{
    if (config == NULL) {
        return;
    }
    config->domain_min = 0.0;
    config->domain_max = 1.0;
    config->level_count = 2U;
    config->root_cells = 2U;
    config->allocator.allocate = NULL;
    config->allocator.deallocate = NULL;
    config->allocator.context = NULL;
}

futcache_status_t futcache_tower_create(
    const futcache_tower_config_t *config,
    futcache_tower_t **out_tower
)
{
    futcache_allocator_t allocator;
    futcache_tower_t *tower;
    futcache_status_t status;
    size_t levels_bytes;
    size_t cells;
    size_t total_cells = 0U;
    size_t total_memory;
    size_t level_index;

    if (out_tower == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    *out_tower = NULL;
    if (!tower_floating_environment_is_supported()) {
        return FUTCACHE_ERROR_UNSUPPORTED_PLATFORM;
    }
    if (!tower_config_is_valid(config)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    status = tower_normalize_allocator(&config->allocator, &allocator);
    if (status != FUTCACHE_OK) {
        return status;
    }
    if (!checked_multiply_size(config->level_count, sizeof(tower_level_t), &levels_bytes)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    total_memory = sizeof(*tower);
    if (!checked_add_size(total_memory, levels_bytes, &total_memory)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    cells = config->root_cells;
    for (level_index = 0U; level_index < config->level_count; ++level_index) {
        size_t words;
        size_t occupancy_bytes;
        size_t fenwick_bytes;
        size_t log_bytes;
        size_t level_bytes;

        if (!level_layout(cells, &words, &occupancy_bytes, &fenwick_bytes,
                &log_bytes, &level_bytes) ||
            !checked_add_size(total_cells, cells, &total_cells) ||
            !checked_add_size(total_memory, level_bytes, &total_memory)) {
            return FUTCACHE_ERROR_OUT_OF_RANGE;
        }
        (void)words;
        (void)occupancy_bytes;
        (void)fenwick_bytes;
        (void)log_bytes;
        if (level_index + 1U < config->level_count) {
            if (cells > SIZE_MAX / 2U) {
                return FUTCACHE_ERROR_OUT_OF_RANGE;
            }
            cells *= 2U;
        }
    }

    tower = tower_allocate_zeroed(&allocator, sizeof(*tower));
    if (tower == NULL) {
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    tower->domain_min = config->domain_min == 0.0 ? 0.0 : config->domain_min;
    tower->domain_max = config->domain_max == 0.0 ? 0.0 : config->domain_max;
    tower->level_count = config->level_count;
    tower->root_cells = config->root_cells;
    tower->total_cells = total_cells;
    tower->memory_bytes = total_memory;
    tower->allocator = allocator;
    if (pthread_rwlock_init(&tower->lock, NULL) != 0) {
        allocator.deallocate(allocator.context, tower);
        return FUTCACHE_ERROR_SYSTEM;
    }

    tower->levels = tower_allocate_zeroed(&allocator, levels_bytes);
    if (tower->levels == NULL) {
        (void)pthread_rwlock_destroy(&tower->lock);
        allocator.deallocate(allocator.context, tower);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }

    cells = config->root_cells;
    for (level_index = 0U; level_index < config->level_count; ++level_index) {
        tower_level_t *level = &tower->levels[level_index];
        size_t occupancy_bytes;
        size_t fenwick_bytes;
        size_t log_bytes;
        size_t ignored_total;

        if (!level_layout(cells, &level->word_count, &occupancy_bytes,
                &fenwick_bytes, &log_bytes, &ignored_total)) {
            tower_release_levels(tower);
            (void)pthread_rwlock_destroy(&tower->lock);
            allocator.deallocate(allocator.context, tower);
            return FUTCACHE_ERROR_OUT_OF_RANGE;
        }
        level->cell_count = cells;
        level->occupancy = tower_allocate_zeroed(&allocator, occupancy_bytes);
        level->fenwick = tower_allocate_zeroed(&allocator, fenwick_bytes);
        level->discovery_log = tower_allocate_zeroed(&allocator, log_bytes);
        if (level->occupancy == NULL || level->fenwick == NULL ||
            level->discovery_log == NULL) {
            tower_release_levels(tower);
            (void)pthread_rwlock_destroy(&tower->lock);
            allocator.deallocate(allocator.context, tower);
            return FUTCACHE_ERROR_OUT_OF_MEMORY;
        }
        if (level_index + 1U < config->level_count) {
            cells *= 2U;
        }
    }

    *out_tower = tower;
    return FUTCACHE_OK;
}

void futcache_tower_destroy(futcache_tower_t *tower)
{
    futcache_allocator_t allocator;

    if (tower == NULL) {
        return;
    }
    allocator = tower->allocator;
    tower_release_levels(tower);
    (void)pthread_rwlock_destroy(&tower->lock);
    allocator.deallocate(allocator.context, tower);
}

futcache_status_t futcache_tower_query(
    const futcache_tower_t *tower,
    double x,
    uint8_t *out_novel,
    size_t output_capacity
)
{
    futcache_status_t status;
    size_t level_index;
    size_t finest_cell;

    if (tower == NULL || out_novel == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (output_capacity < tower->level_count) {
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }
    if (!tower_floating_environment_is_supported()) {
        return FUTCACHE_ERROR_UNSUPPORTED_PLATFORM;
    }
    if (!tower_point_is_valid(tower, x)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    status = tower_read_lock(tower);
    if (status != FUTCACHE_OK) {
        return status;
    }
    finest_cell = tower_map_cell(tower,
        tower->levels[tower->level_count - 1U].cell_count, x);
    for (level_index = 0U; level_index < tower->level_count; ++level_index) {
        const tower_level_t *level = &tower->levels[level_index];
        size_t cell = finest_cell >> (tower->level_count - 1U - level_index);
        out_novel[level_index] = (uint8_t)(level_is_seen(level, cell) ? 0U : 1U);
    }
    return tower_unlock(tower);
}

futcache_status_t futcache_tower_observe(
    futcache_tower_t *tower,
    double x,
    uint8_t *out_was_novel,
    size_t output_capacity
)
{
    futcache_status_t status;
    futcache_status_t unlock_status;
    size_t level_index;
    size_t finest_cell;

    if (tower == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (out_was_novel != NULL && output_capacity < tower->level_count) {
        return FUTCACHE_ERROR_BUFFER_TOO_SMALL;
    }
    if (!tower_floating_environment_is_supported()) {
        return FUTCACHE_ERROR_UNSUPPORTED_PLATFORM;
    }
    if (!tower_point_is_valid(tower, x)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    status = tower_write_lock(tower);
    if (status != FUTCACHE_OK) {
        return status;
    }

    finest_cell = tower_map_cell(tower,
        tower->levels[tower->level_count - 1U].cell_count, x);

    for (level_index = 0U; level_index < tower->level_count; ++level_index) {
        tower_level_t *level = &tower->levels[level_index];
        size_t cell = finest_cell >> (tower->level_count - 1U - level_index);
        bool novel = !level_is_seen(level, cell);
        if (out_was_novel != NULL) {
            out_was_novel[level_index] = novel ? 1U : 0U;
        }
        if (novel) {
            level_mark_seen(level, cell);
        }
    }
    tower->observations = tower_increment_saturating(tower->observations);
    tower->generation = tower_increment_saturating(tower->generation);

    unlock_status = tower_unlock(tower);
    return unlock_status;
}

futcache_status_t futcache_tower_cell_index(
    const futcache_tower_t *tower,
    size_t level,
    double x,
    size_t *out_cell
)
{
    if (tower == NULL || out_cell == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!tower_floating_environment_is_supported()) {
        return FUTCACHE_ERROR_UNSUPPORTED_PLATFORM;
    }
    if (level >= tower->level_count || !tower_point_is_valid(tower, x)) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    *out_cell = tower_map_level_cell(tower, level, x);
    return FUTCACHE_OK;
}

futcache_status_t futcache_tower_level_info(
    const futcache_tower_t *tower,
    size_t level,
    futcache_tower_level_info_t *out_info
)
{
    futcache_status_t status;
    futcache_tower_level_info_t info;

    if (tower == NULL || out_info == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (level >= tower->level_count) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    status = tower_read_lock(tower);
    if (status != FUTCACHE_OK) {
        return status;
    }
    info.cell_count = tower->levels[level].cell_count;
    info.discovered_count = tower->levels[level].discovered_count;
    status = tower_unlock(tower);
    if (status == FUTCACHE_OK) {
        *out_info = info;
    }
    return status;
}

futcache_status_t futcache_tower_prefix_count(
    const futcache_tower_t *tower,
    size_t level,
    size_t cell_inclusive,
    size_t *out_count
)
{
    futcache_status_t status;
    size_t count;

    if (tower == NULL || out_count == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (level >= tower->level_count ||
        cell_inclusive >= tower->levels[level].cell_count) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    status = tower_read_lock(tower);
    if (status != FUTCACHE_OK) {
        return status;
    }
    count = fenwick_prefix(&tower->levels[level], cell_inclusive);
    status = tower_unlock(tower);
    if (status == FUTCACHE_OK) {
        *out_count = count;
    }
    return status;
}

futcache_status_t futcache_tower_select_occupied(
    const futcache_tower_t *tower,
    size_t level,
    size_t ordinal,
    size_t *out_cell
)
{
    futcache_status_t status;
    const tower_level_t *tower_level;
    size_t target;
    size_t index = 0U;
    size_t step;

    if (tower == NULL || out_cell == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (level >= tower->level_count) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    status = tower_read_lock(tower);
    if (status != FUTCACHE_OK) {
        return status;
    }
    tower_level = &tower->levels[level];
    if (ordinal >= tower_level->discovered_count) {
        (void)tower_unlock(tower);
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }

    target = ordinal + 1U;
    step = highest_power_of_two_not_greater_than(tower_level->cell_count);
    while (step != 0U) {
        if (step <= tower_level->cell_count - index) {
            size_t next = index + step;
            if (tower_level->fenwick[next] < target) {
                index = next;
                target -= tower_level->fenwick[next];
            }
        }
        step >>= 1U;
    }
    *out_cell = index;
    return tower_unlock(tower);
}

futcache_status_t futcache_tower_discovery_at(
    const futcache_tower_t *tower,
    size_t level,
    size_t discovery_index,
    size_t *out_cell
)
{
    futcache_status_t status;
    size_t cell;

    if (tower == NULL || out_cell == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (level >= tower->level_count) {
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    status = tower_read_lock(tower);
    if (status != FUTCACHE_OK) {
        return status;
    }
    if (discovery_index >= tower->levels[level].discovered_count) {
        (void)tower_unlock(tower);
        return FUTCACHE_ERROR_OUT_OF_RANGE;
    }
    cell = tower->levels[level].discovery_log[discovery_index];
    status = tower_unlock(tower);
    if (status == FUTCACHE_OK) {
        *out_cell = cell;
    }
    return status;
}

futcache_status_t futcache_tower_get_stats(
    const futcache_tower_t *tower,
    futcache_tower_stats_t *out_stats
)
{
    futcache_status_t status;
    futcache_tower_stats_t stats;
    size_t level_index;

    if (tower == NULL || out_stats == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    status = tower_read_lock(tower);
    if (status != FUTCACHE_OK) {
        return status;
    }
    stats.observations = tower->observations;
    stats.generation = tower->generation;
    stats.level_count = tower->level_count;
    stats.total_cells = tower->total_cells;
    stats.total_discoveries = 0U;
    stats.memory_bytes = tower->memory_bytes;
    for (level_index = 0U; level_index < tower->level_count; ++level_index) {
        stats.total_discoveries += tower->levels[level_index].discovered_count;
    }
    status = tower_unlock(tower);
    if (status == FUTCACHE_OK) {
        *out_stats = stats;
    }
    return status;
}

futcache_status_t futcache_tower_clear(futcache_tower_t *tower)
{
    futcache_status_t status;
    size_t level_index;

    if (tower == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    status = tower_write_lock(tower);
    if (status != FUTCACHE_OK) {
        return status;
    }
    for (level_index = 0U; level_index < tower->level_count; ++level_index) {
        tower_level_t *level = &tower->levels[level_index];
        memset(level->occupancy, 0, level->word_count * sizeof(uint64_t));
        memset(level->fenwick, 0, (level->cell_count + 1U) * sizeof(size_t));
        memset(level->discovery_log, 0, level->cell_count * sizeof(size_t));
        level->discovered_count = 0U;
    }
    tower->observations = UINT64_C(0);
    tower->generation = tower_increment_saturating(tower->generation);
    return tower_unlock(tower);
}

futcache_status_t futcache_tower_validate(const futcache_tower_t *tower)
{
    futcache_status_t status;
    size_t expected_cells;
    size_t level_index;
    size_t validated_total_cells = 0U;

    if (tower == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    status = tower_read_lock(tower);
    if (status != FUTCACHE_OK) {
        return status;
    }

    if (tower->generation < tower->observations) {
        status = FUTCACHE_ERROR_CORRUPT_DATA;
    }

    expected_cells = tower->root_cells;
    for (level_index = 0U; level_index < tower->level_count && status == FUTCACHE_OK;
         ++level_index) {
        const tower_level_t *level = &tower->levels[level_index];
        uint64_t *discovery_seen;
        size_t bit_count = 0U;
        size_t cell;

        if (level->cell_count != expected_cells ||
            level->discovered_count > level->cell_count ||
            level->discovered_count > (size_t)UINT64_MAX ||
            (uint64_t)level->discovered_count > tower->observations ||
            ((level->discovered_count == 0U) !=
                (tower->observations == UINT64_C(0))) ||
            !checked_add_size(validated_total_cells, level->cell_count,
                &validated_total_cells)) {
            status = FUTCACHE_ERROR_CORRUPT_DATA;
            break;
        }
        discovery_seen = tower_allocate_zeroed(
            &tower->allocator,
            level->word_count * sizeof(uint64_t)
        );
        if (discovery_seen == NULL) {
            status = FUTCACHE_ERROR_OUT_OF_MEMORY;
            break;
        }

        for (cell = 0U; cell < level->cell_count && status == FUTCACHE_OK; ++cell) {
            size_t fenwick_index = cell + 1U;
            size_t low_bit = fenwick_index & (~fenwick_index + 1U);
            size_t range_start = fenwick_index - low_bit;
            size_t scan;
            size_t expected_fenwick = 0U;

            if (level_is_seen(level, cell)) {
                bit_count++;
            }
            for (scan = range_start; scan < fenwick_index; ++scan) {
                if (level_is_seen(level, scan)) {
                    expected_fenwick++;
                }
            }
            if (level->fenwick[fenwick_index] != expected_fenwick) {
                status = FUTCACHE_ERROR_CORRUPT_DATA;
            }
        }

        if (bit_count != level->discovered_count) {
            status = FUTCACHE_ERROR_CORRUPT_DATA;
        }
        for (cell = 0U; cell < level->discovered_count && status == FUTCACHE_OK;
             ++cell) {
            size_t discovered_cell = level->discovery_log[cell];
            size_t word;
            unsigned int bit;
            uint64_t mask;

            if (discovered_cell >= level->cell_count ||
                !level_is_seen(level, discovered_cell)) {
                status = FUTCACHE_ERROR_CORRUPT_DATA;
                break;
            }
            word = discovered_cell / 64U;
            bit = (unsigned int)(discovered_cell % 64U);
            mask = UINT64_C(1) << bit;
            if ((discovery_seen[word] & mask) != 0U) {
                status = FUTCACHE_ERROR_CORRUPT_DATA;
                break;
            }
            discovery_seen[word] |= mask;
        }
        tower->allocator.deallocate(tower->allocator.context, discovery_seen);

        if (level_index + 1U < tower->level_count) {
            expected_cells *= 2U;
        }
    }

    if (status == FUTCACHE_OK && validated_total_cells != tower->total_cells) {
        status = FUTCACHE_ERROR_CORRUPT_DATA;
    }

    {
        futcache_status_t unlock_status = tower_unlock(tower);
        if (status == FUTCACHE_OK && unlock_status != FUTCACHE_OK) {
            status = unlock_status;
        }
    }
    return status;
}
