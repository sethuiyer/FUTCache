/*
 * box.c — exact bounded-dim L_inf box-union novelty cache.
 *
 * The state is a list of closed d-dimensional axis-aligned boxes
 * (lo, hi); a point is redundant iff it lies inside any stored box.
 * A box is appended on every novel observation.
 *
 * Why the representation is exact but non-canonical (not minimal).
 *
 *   Each admitted box is the clipped epsilon-ball of a novel center x,
 *   i.e. every prior center c satisfies d_inf(x, c) > epsilon. A newly
 *   admitted box can therefore never strictly contain a previously
 *   admitted box: if it did, it would contain that box's center c, so
 *   d_inf(x, c) <= epsilon, contradicting novelty. The symmetric case
 *   (new box contained in a prior box) is likewise impossible, and a
 *   fortiori no two admitted boxes are exact duplicates. Boxes may
 *   still partially overlap, which is what makes the union generally
 *   non-canonical; the representation is never merged or split.
 *
 * Consequently box_count is a storage diagnostic (equal to the number
 * of novel observations), bounded above by the packing number P(K,
 * epsilon) but not a canonical minimal cell count. For a canonical
 * 1-D union use the interval cache in futcache.c; a future disjoint
 * cell backend could replace this representation without changing the
 * public API.
 */

#include "futcache/box.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double *lo;
    double *hi;
} box_rect_t;

static void *box_default_allocate(void *context, size_t size)
{
    (void)context;
    return malloc(size);
}

static void box_default_deallocate(void *context, void *pointer)
{
    (void)context;
    free(pointer);
}

static futcache_status_t box_normalize_allocator(
    const futcache_allocator_t *requested, futcache_allocator_t *normalized)
{
    if (requested == NULL ||
        (requested->allocate == NULL && requested->deallocate == NULL)) {
        normalized->allocate = box_default_allocate;
        normalized->deallocate = box_default_deallocate;
        normalized->context = NULL;
        return FUTCACHE_OK;
    }
    if (requested->allocate == NULL || requested->deallocate == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    *normalized = *requested;
    return FUTCACHE_OK;
}

struct futcache_box {
    futcache_box_config_t c;
    futcache_allocator_t allocator;
    double *min;
    double *max;
    box_rect_t *rects;
    size_t count;
    size_t capacity;
    size_t peak_count;
    uint64_t observations;
    uint64_t novel;
    uint64_t generation;
    /* Approximate bytes owned by the cache; recomputed on clear. */
    size_t memory;
    pthread_rwlock_t lock;
};

static void *a_malloc(const futcache_allocator_t *a, size_t n)
{
    return a->allocate(a->context, n);
}

static void a_free(const futcache_allocator_t *a, void *p)
{
    if (p != NULL) a->deallocate(a->context, p);
}

/* Saturating increment for telemetry. Mirrors the helper in futcache.c. */
static uint64_t box_saturating(uint64_t value)
{
    return value < UINT64_MAX ? value + 1U : value;
}

static bool box_valid_point(const futcache_box_t *x, const double *p)
{
    if (p == NULL) return false;
    for (size_t i = 0; i < x->c.dimension; ++i) {
        if (!isfinite(p[i])) return false;
    }
    return true;
}

static bool box_in_domain(const futcache_box_t *x, const double *p)
{
    for (size_t i = 0; i < x->c.dimension; ++i) {
        if (p[i] < x->min[i] || p[i] > x->max[i]) return false;
    }
    return true;
}

static bool box_covered(const futcache_box_t *x, const double *p)
{
    for (size_t j = 0; j < x->count; ++j) {
        bool inside = true;
        for (size_t i = 0; i < x->c.dimension; ++i) {
            if (p[i] < x->rects[j].lo[i] || p[i] > x->rects[j].hi[i]) {
                inside = false;
                break;
            }
        }
        if (inside) return true;
    }
    return false;
}

void futcache_box_config_init(futcache_box_config_t *c)
{
    if (c == NULL) return;
    memset(c, 0, sizeof(*c));
    c->dimension = 1U;
    c->epsilon = 0.0;
    c->domain_min = NULL;
    c->domain_max = NULL;
}

futcache_status_t futcache_box_create(const futcache_box_config_t *c,
                                      futcache_box_t **out)
{
    if (c == NULL || out == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    if (c->dimension == 0 || c->dimension > 8) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!isfinite(c->epsilon) || c->epsilon < 0.0) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (c->domain_min == NULL || c->domain_max == NULL) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < c->dimension; ++i) {
        if (!isfinite(c->domain_min[i]) || !isfinite(c->domain_max[i]) ||
            c->domain_min[i] >= c->domain_max[i]) {
            return FUTCACHE_ERROR_INVALID_ARGUMENT;
        }
    }

    futcache_allocator_t allocator;
    futcache_status_t allocator_status =
        box_normalize_allocator(&c->allocator, &allocator);
    if (allocator_status != FUTCACHE_OK) return allocator_status;

    futcache_box_t *x = a_malloc(&allocator, sizeof(*x));
    if (x == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
    memset(x, 0, sizeof(*x));
    x->c = *c;
    x->allocator = allocator;

    if (pthread_rwlock_init(&x->lock, NULL) != 0) {
        a_free(&allocator, x);
        return FUTCACHE_ERROR_SYSTEM;
    }

    size_t bytes = c->dimension * sizeof(double);
    x->min = a_malloc(&allocator, bytes);
    x->max = a_malloc(&allocator, bytes);
    if (x->min == NULL || x->max == NULL) {
        a_free(&allocator, x->min);
        a_free(&allocator, x->max);
        pthread_rwlock_destroy(&x->lock);
        a_free(&allocator, x);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    memcpy(x->min, c->domain_min, bytes);
    memcpy(x->max, c->domain_max, bytes);

    /* baseline: struct + bounds + rects array (zero slots). */
    x->memory = sizeof(*x) + 2U * bytes;
    *out = x;
    return FUTCACHE_OK;
}

void futcache_box_destroy(futcache_box_t *x)
{
    if (x == NULL) return;
    for (size_t j = 0; j < x->count; ++j) {
        a_free(&x->allocator, x->rects[j].lo);
        a_free(&x->allocator, x->rects[j].hi);
    }
    a_free(&x->allocator, x->rects);
    a_free(&x->allocator, x->min);
    a_free(&x->allocator, x->max);
    pthread_rwlock_destroy(&x->lock);
    a_free(&x->allocator, x);
}

static futcache_status_t box_grow(futcache_box_t *x)
{
    if (x->count < x->capacity) return FUTCACHE_OK;
    size_t new_cap = x->capacity == 0 ? 8U : x->capacity * 2U;
    box_rect_t *nr = a_malloc(&x->allocator, new_cap * sizeof(*nr));
    if (nr == NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY;
    if (x->rects != NULL && x->count > 0) {
        memcpy(nr, x->rects, x->count * sizeof(*nr));
    }
    a_free(&x->allocator, x->rects);
    x->rects = nr;
    x->capacity = new_cap;
    x->memory += (new_cap - x->count) * sizeof(*nr);
    return FUTCACHE_OK;
}

futcache_status_t futcache_box_is_novel(const futcache_box_t *x,
                                        const double *p, bool *out)
{
    if (x == NULL || out == NULL || !box_valid_point(x, p)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!box_in_domain(x, p)) return FUTCACHE_ERROR_OUT_OF_RANGE;

    pthread_rwlock_t *lock = (pthread_rwlock_t *)&x->lock;
    pthread_rwlock_rdlock(lock);
    *out = !box_covered(x, p);
    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_box_observe(futcache_box_t *x, const double *p,
                                       bool *out)
{
    if (x == NULL || !box_valid_point(x, p)) {
        return FUTCACHE_ERROR_INVALID_ARGUMENT;
    }
    if (!box_in_domain(x, p)) return FUTCACHE_ERROR_OUT_OF_RANGE;

    pthread_rwlock_wrlock(&x->lock);

    /* A covered centre can still contribute previously uncovered area.  For
     * example, in one dimension with epsilon=1, the ball around 0 covers the
     * centre 0.9, but the latter's ball extends coverage from 1 to 1.9.
     * Retaining every observed box is what makes this engine an exact
     * full-history union rather than a representative packing. */
    bool was_novel = !box_covered(x, p);

    futcache_status_t st = box_grow(x);
    if (st != FUTCACHE_OK) {
        pthread_rwlock_unlock(&x->lock);
        return st;
    }

    size_t d = x->c.dimension;
    size_t bytes = d * sizeof(double);
    double *lo = a_malloc(&x->allocator, bytes);
    double *hi = a_malloc(&x->allocator, bytes);
    if (lo == NULL || hi == NULL) {
        a_free(&x->allocator, lo);
        a_free(&x->allocator, hi);
        pthread_rwlock_unlock(&x->lock);
        return FUTCACHE_ERROR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < d; ++i) {
        lo[i] = (p[i] - x->c.epsilon) < x->min[i] ? x->min[i]
                                                   : (p[i] - x->c.epsilon);
        hi[i] = (p[i] + x->c.epsilon) > x->max[i] ? x->max[i]
                                                   : (p[i] + x->c.epsilon);
    }
    x->rects[x->count].lo = lo;
    x->rects[x->count].hi = hi;
    x->count++;
    if (x->count > x->peak_count) x->peak_count = x->count;
    x->memory += 2U * bytes;

    x->observations = box_saturating(x->observations);
    if (was_novel) x->novel = box_saturating(x->novel);
    x->generation = box_saturating(x->generation);

    if (out != NULL) *out = was_novel;
    pthread_rwlock_unlock(&x->lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_box_get_stats(const futcache_box_t *x,
                                         futcache_box_stats_t *out)
{
    if (x == NULL || out == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    pthread_rwlock_t *lock = (pthread_rwlock_t *)&x->lock;
    pthread_rwlock_rdlock(lock);
    out->observations = x->observations;
    out->novel_observations = x->novel;
    out->generation = x->generation;
    out->box_count = x->count;
    out->peak_box_count = x->peak_count;
    out->memory_bytes = x->memory;
    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_box_clear(futcache_box_t *x)
{
    if (x == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    pthread_rwlock_wrlock(&x->lock);
    for (size_t j = 0; j < x->count; ++j) {
        a_free(&x->allocator, x->rects[j].lo);
        a_free(&x->allocator, x->rects[j].hi);
    }
    x->count = 0;
    x->peak_count = 0;
    size_t bytes = x->c.dimension * sizeof(double);
    x->memory = sizeof(*x) + 2U * bytes;
    x->observations = 0;
    x->novel = 0;
    x->generation = box_saturating(x->generation);
    pthread_rwlock_unlock(&x->lock);
    return FUTCACHE_OK;
}

futcache_status_t futcache_box_validate(const futcache_box_t *x)
{
    if (x == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    pthread_rwlock_t *lock = (pthread_rwlock_t *)&x->lock;
    pthread_rwlock_rdlock(lock);

    /* Telemetry lifecycle invariants (mirror the interval engine). */
    if (x->generation < x->observations ||
        x->novel > x->observations ||
        x->count != (size_t)x->observations ||
        x->count > x->peak_count) {
        pthread_rwlock_unlock(lock);
        return FUTCACHE_ERROR_CORRUPT_DATA;
    }

    size_t d = x->c.dimension;
    for (size_t j = 0; j < x->count; ++j) {
        for (size_t i = 0; i < d; ++i) {
            if (x->rects[j].lo[i] < x->min[i] ||
                x->rects[j].hi[i] > x->max[i] ||
                x->rects[j].lo[i] > x->rects[j].hi[i]) {
                pthread_rwlock_unlock(lock);
                return FUTCACHE_ERROR_CORRUPT_DATA;
            }
        }
    }

    pthread_rwlock_unlock(lock);
    return FUTCACHE_OK;
}
