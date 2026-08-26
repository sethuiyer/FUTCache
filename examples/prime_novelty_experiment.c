/* Arithmetic prime novelty experiment for FUTCache.
 *
 * The online verdict is epsilon-dependent FUTCache state. The ranking score
 * is the distance to the nearest earlier prime over the full history. Final
 * representative persistence is nearest-representative distance - epsilon;
 * it is a reduced summary, not a full persistent-homology diagram.
 */

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "futcache/pack.h"

enum { DEFAULT_COUNT = 512, DEFAULT_TOP = 10, MAX_LOCAL_PRIMES = 32 };

typedef enum metric_kind {
    METRIC_ARCHIMEDEAN,
    METRIC_RESIDUE,
    METRIC_PADIC,
    METRIC_COMBINED
} metric_kind_t;

typedef struct metric_context {
    metric_kind_t kind;
    const uint64_t *local_primes;
    const double *local_weights;
    size_t local_count;
    size_t padic_index;
    double arch_weight;
    char name[32];
} metric_context_t;

typedef struct observation_result {
    uint64_t prime;
    uint64_t nearest_prior_prime;
    double nearest_prior_distance;
    double final_persistence;
    size_t novelty_rank;
    bool has_nearest_prior;
    bool was_novel;
    bool is_representative;
} observation_result_t;

typedef struct rank_entry {
    size_t observation_index;
    uint64_t prime;
    double score;
} rank_entry_t;

typedef struct options {
    size_t count;
    size_t top_count;
    size_t burn_in;
    double epsilon;
    double arch_weight;
    uint64_t local_primes[MAX_LOCAL_PRIMES];
    size_t local_count;
    const char *csv_path;
    bool burn_in_explicit;
} options_t;

static void usage(FILE *stream, const char *program)
{
    fprintf(stream,
        "Usage: %s [options]\n"
        "  --count N             number of primes (default %d)\n"
        "  --epsilon E           FUTCache radius in [0,1] (default 0.65)\n"
        "  --local-primes LIST   r-adic lenses (default 2,3,5,7,11,13)\n"
        "  --arch-weight W       combined Archimedean weight (default 0.35)\n"
        "  --burn-in N           exclude first N primes from ranking\n"
        "                        (default 32, or count/4 for small runs)\n"
        "  --top K               summary rows per metric (default %d)\n"
        "  --csv PATH            detailed long-form CSV; '-' is stdout\n"
        "  --help                show this help\n",
        program, DEFAULT_COUNT, DEFAULT_TOP);
}

static bool parse_size(const char *text, size_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > SIZE_MAX) {
        return false;
    }
    *out = (size_t)value;
    return true;
}

static bool parse_double_option(const char *text, double *out)
{
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
        return false;
    }
    *out = value;
    return true;
}

static bool is_u64_prime(uint64_t value)
{
    if (value < 2U) return false;
    if (value % 2U == 0U) return value == 2U;
    for (uint64_t divisor = 3U; divisor <= value / divisor; divisor += 2U) {
        if (value % divisor == 0U) return false;
    }
    return true;
}

static bool parse_local_primes(const char *text, options_t *options)
{
    size_t length = strlen(text);
    char *copy = (char *)malloc(length + 1U);
    if (copy == NULL) return false;
    memcpy(copy, text, length + 1U);

    options->local_count = 0U;
    char *cursor = copy;
    while (*cursor != '\0') {
        char *comma = strchr(cursor, ',');
        if (comma != NULL) *comma = '\0';
        size_t parsed = 0U;
        if (!parse_size(cursor, &parsed) || parsed > UINT32_MAX ||
            !is_u64_prime((uint64_t)parsed) ||
            options->local_count == MAX_LOCAL_PRIMES) {
            free(copy);
            return false;
        }
        for (size_t i = 0U; i < options->local_count; ++i) {
            if (options->local_primes[i] == (uint64_t)parsed) {
                free(copy);
                return false;
            }
        }
        options->local_primes[options->local_count++] = (uint64_t)parsed;
        if (comma == NULL) break;
        cursor = comma + 1;
        if (*cursor == '\0') {
            free(copy);
            return false;
        }
    }
    free(copy);
    return options->local_count != 0U;
}

static bool parse_options(int argc, char **argv, options_t *options)
{
    memset(options, 0, sizeof(*options));
    options->count = DEFAULT_COUNT;
    options->top_count = DEFAULT_TOP;
    options->epsilon = 0.65;
    options->arch_weight = 0.35;
    if (!parse_local_primes("2,3,5,7,11,13", options)) return false;

    for (int i = 1; i < argc; ++i) {
        const char *argument = argv[i];
        if (strcmp(argument, "--help") == 0) {
            usage(stdout, argv[0]);
            exit(EXIT_SUCCESS);
        }
        if (i + 1 >= argc) return false;
        const char *value = argv[++i];
        if (strcmp(argument, "--count") == 0) {
            if (!parse_size(value, &options->count)) return false;
        } else if (strcmp(argument, "--epsilon") == 0) {
            if (!parse_double_option(value, &options->epsilon)) return false;
        } else if (strcmp(argument, "--local-primes") == 0) {
            if (!parse_local_primes(value, options)) return false;
        } else if (strcmp(argument, "--arch-weight") == 0) {
            if (!parse_double_option(value, &options->arch_weight)) return false;
        } else if (strcmp(argument, "--top") == 0) {
            if (!parse_size(value, &options->top_count)) return false;
        } else if (strcmp(argument, "--burn-in") == 0) {
            if (!parse_size(value, &options->burn_in)) return false;
            options->burn_in_explicit = true;
        } else if (strcmp(argument, "--csv") == 0) {
            options->csv_path = value;
        } else {
            return false;
        }
    }
    if (!options->burn_in_explicit) {
        options->burn_in = options->count < 64U ? options->count / 4U : 32U;
    }
    return options->count >= 2U && options->top_count != 0U &&
        options->burn_in < options->count &&
        options->epsilon >= 0.0 && options->epsilon <= 1.0 &&
        options->arch_weight >= 0.0 && options->arch_weight <= 1.0;
}

static bool generate_primes(size_t count, uint64_t **out_primes)
{
    *out_primes = NULL;
    double n = (double)count;
    double estimate = count < 6U ? 15.0
        : n * (log(n) + log(log(n))) + 16.0;
    if (!isfinite(estimate) || estimate > (double)(SIZE_MAX - 1U) ||
        count > SIZE_MAX / sizeof(uint64_t)) return false;
    size_t bound = (size_t)ceil(estimate);
    uint64_t *primes = (uint64_t *)malloc(count * sizeof(uint64_t));
    if (primes == NULL) return false;

    for (;;) {
        bool *composite = (bool *)calloc(bound + 1U, sizeof(bool));
        if (composite == NULL) {
            free(primes);
            return false;
        }
        for (size_t candidate = 2U; candidate <= bound / candidate;
             ++candidate) {
            if (composite[candidate]) continue;
            for (size_t multiple = candidate * candidate; multiple <= bound;) {
                composite[multiple] = true;
                if (multiple > bound - candidate) break;
                multiple += candidate;
            }
        }
        size_t found = 0U;
        for (size_t candidate = 2U; candidate <= bound && found < count;
             ++candidate) {
            if (!composite[candidate]) primes[found++] = (uint64_t)candidate;
        }
        free(composite);
        if (found == count) {
            *out_primes = primes;
            return true;
        }
        if (bound > (SIZE_MAX - 1U) / 2U) {
            free(primes);
            return false;
        }
        bound = bound * 2U + 1U;
    }
}

static double archimedean_distance(uint64_t p, uint64_t q)
{
    uint64_t difference = p > q ? p - q : q - p;
    if (difference == 0U) return 0.0;
    double delta = (double)difference;
    return delta / (1.0 + delta);
}

static double padic_distance(uint64_t p, uint64_t q, uint64_t local_prime)
{
    uint64_t difference = p > q ? p - q : q - p;
    if (difference == 0U) return 0.0;
    double distance = 1.0;
    while (difference % local_prime == 0U) {
        difference /= local_prime;
        distance /= (double)local_prime;
    }
    return distance;
}

static double residue_distance(uint64_t p, uint64_t q,
                               const metric_context_t *context)
{
    double sum = 0.0;
    for (size_t i = 0U; i < context->local_count; ++i) {
        uint64_t local_prime = context->local_primes[i];
        double left = (double)(p % local_prime) / (double)local_prime;
        double right = (double)(q % local_prime) / (double)local_prime;
        double delta = left - right;
        sum += context->local_weights[i] * delta * delta;
    }
    return sqrt(sum);
}

static double arithmetic_distance(const double *a, const double *b,
                                  size_t dimension, void *opaque)
{
    (void)dimension;
    const metric_context_t *context = (const metric_context_t *)opaque;
    uint64_t p = (uint64_t)a[0];
    uint64_t q = (uint64_t)b[0];
    if (p == q) return 0.0;
    if (context->kind == METRIC_ARCHIMEDEAN) {
        return archimedean_distance(p, q);
    }
    if (context->kind == METRIC_RESIDUE) {
        return residue_distance(p, q, context);
    }
    if (context->kind == METRIC_PADIC) {
        return padic_distance(p, q,
            context->local_primes[context->padic_index]);
    }
    double local = 0.0;
    for (size_t i = 0U; i < context->local_count; ++i) {
        local += context->local_weights[i] *
            padic_distance(p, q, context->local_primes[i]);
    }
    return context->arch_weight * archimedean_distance(p, q) +
        (1.0 - context->arch_weight) * local;
}

static bool initialize_metrics(const options_t *options,
                               double *local_weights,
                               metric_context_t *metrics,
                               size_t metric_count)
{
    double sum = 0.0;
    for (size_t i = 0U; i < options->local_count; ++i) {
        local_weights[i] = 1.0 / log1p((double)options->local_primes[i]);
        sum += local_weights[i];
    }
    if (!(sum > 0.0) || metric_count != options->local_count + 3U) return false;
    for (size_t i = 0U; i < options->local_count; ++i) local_weights[i] /= sum;

    memset(metrics, 0, metric_count * sizeof(*metrics));
    for (size_t i = 0U; i < metric_count; ++i) {
        metrics[i].local_primes = options->local_primes;
        metrics[i].local_weights = local_weights;
        metrics[i].local_count = options->local_count;
        metrics[i].arch_weight = options->arch_weight;
    }
    metrics[0].kind = METRIC_ARCHIMEDEAN;
    (void)snprintf(metrics[0].name, sizeof(metrics[0].name), "archimedean");
    metrics[1].kind = METRIC_RESIDUE;
    (void)snprintf(metrics[1].name, sizeof(metrics[1].name), "residue-l2");
    for (size_t i = 0U; i < options->local_count; ++i) {
        metrics[2U + i].kind = METRIC_PADIC;
        metrics[2U + i].padic_index = i;
        (void)snprintf(metrics[2U + i].name,
            sizeof(metrics[2U + i].name), "%" PRIu64 "-adic",
            options->local_primes[i]);
    }
    metrics[metric_count - 1U].kind = METRIC_COMBINED;
    (void)snprintf(metrics[metric_count - 1U].name,
        sizeof(metrics[metric_count - 1U].name), "combined");
    return true;
}

static bool validate_metrics(const uint64_t *primes, size_t prime_count,
                             const metric_context_t *metrics,
                             size_t metric_count)
{
    if (padic_distance(3U, 11U, 2U) != 0.125 ||
        padic_distance(7U, 7U, 3U) != 0.0) return false;
    size_t samples = prime_count < 24U ? prime_count : 24U;
    for (size_t metric = 0U; metric < metric_count; ++metric) {
        for (size_t i = 0U; i < samples; ++i) {
            double a[1] = {(double)primes[i]};
            if (arithmetic_distance(a, a, 1U, (void *)&metrics[metric]) != 0.0) {
                return false;
            }
            for (size_t j = 0U; j < samples; ++j) {
                double b[1] = {(double)primes[j]};
                double ab = arithmetic_distance(a, b, 1U,
                    (void *)&metrics[metric]);
                double ba = arithmetic_distance(b, a, 1U,
                    (void *)&metrics[metric]);
                if (!isfinite(ab) || ab < 0.0 || ab > 1.0 || ab != ba) {
                    return false;
                }
                for (size_t k = 0U; k < samples; ++k) {
                    double c[1] = {(double)primes[k]};
                    double ac = arithmetic_distance(a, c, 1U,
                        (void *)&metrics[metric]);
                    double bc = arithmetic_distance(b, c, 1U,
                        (void *)&metrics[metric]);
                    if (ac > ab + bc + 32.0 * DBL_EPSILON) {
                        fprintf(stderr, "triangle inequality failed for %s\n",
                            metrics[metric].name);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static int rank_compare(const void *left, const void *right)
{
    const rank_entry_t *a = (const rank_entry_t *)left;
    const rank_entry_t *b = (const rank_entry_t *)right;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    if (a->prime < b->prime) return -1;
    if (a->prime > b->prime) return 1;
    return 0;
}

static void assign_ranks(observation_result_t *results, size_t count,
                         size_t burn_in, rank_entry_t *ranking)
{
    size_t rankable = count - burn_in;
    for (size_t i = burn_in; i < count; ++i) {
        size_t offset = i - burn_in;
        ranking[offset].observation_index = i;
        ranking[offset].prime = results[i].prime;
        ranking[offset].score = results[i].nearest_prior_distance;
    }
    qsort(ranking, rankable, sizeof(*ranking), rank_compare);
    for (size_t rank = 0U; rank < rankable; ++rank) {
        results[ranking[rank].observation_index].novelty_rank = rank + 1U;
    }
}

static bool attach_persistence(futcache_pack_t *cache,
                               const metric_context_t *metric,
                               double epsilon,
                               observation_result_t *results,
                               size_t count)
{
    size_t representative_count = 0U;
    futcache_status_t status = futcache_pack_copy_representatives(
        cache, NULL, &representative_count);
    if (status != FUTCACHE_OK || representative_count == 0U) return false;
    double *representatives = (double *)malloc(
        representative_count * sizeof(double));
    if (representatives == NULL) return false;
    size_t copied = representative_count;
    status = futcache_pack_copy_representatives(cache, representatives, &copied);
    if (status != FUTCACHE_OK || copied != representative_count) {
        free(representatives);
        return false;
    }

    for (size_t i = 0U; i < representative_count; ++i) {
        double nearest = HUGE_VAL;
        for (size_t j = 0U; j < representative_count; ++j) {
            if (i == j) continue;
            double distance = arithmetic_distance(&representatives[i],
                &representatives[j], 1U, (void *)metric);
            if (distance < nearest) nearest = distance;
        }
        uint64_t prime = (uint64_t)representatives[i];
        size_t lo = 0U;
        size_t hi = count;
        while (lo < hi) {
            size_t middle = lo + (hi - lo) / 2U;
            if (results[middle].prime < prime) lo = middle + 1U;
            else hi = middle;
        }
        if (lo == count || results[lo].prime != prime ||
            !results[lo].was_novel) {
            free(representatives);
            return false;
        }
        results[lo].is_representative = true;
        results[lo].final_persistence = isinf(nearest)
            ? HUGE_VAL : nearest - epsilon;
    }
    free(representatives);
    return true;
}

static bool run_metric(const uint64_t *primes, size_t count, double epsilon,
                       size_t burn_in,
                       const metric_context_t *metric,
                       observation_result_t *results,
                       rank_entry_t *ranking,
                       size_t *out_novel_count,
                       size_t *out_representative_count)
{
    double domain_min[1] = {(double)primes[0]};
    double domain_max[1] = {(double)primes[count - 1U]};
    futcache_pack_config_t config;
    futcache_pack_config_init(&config);
    config.dimension = 1U;
    config.epsilon = epsilon;
    config.distance = arithmetic_distance;
    config.distance_context = (void *)metric;
    config.domain_min = domain_min;
    config.domain_max = domain_max;
    futcache_pack_t *cache = NULL;
    futcache_status_t status = futcache_pack_create(&config, &cache);
    if (status != FUTCACHE_OK) return false;

    size_t novel_count = 0U;
    for (size_t i = 0U; i < count; ++i) {
        observation_result_t *result = &results[i];
        memset(result, 0, sizeof(*result));
        result->prime = primes[i];
        result->nearest_prior_distance = HUGE_VAL;
        result->final_persistence = NAN;
        double point[1] = {(double)primes[i]};
        for (size_t j = 0U; j < i; ++j) {
            double prior[1] = {(double)primes[j]};
            double distance = arithmetic_distance(point, prior, 1U,
                (void *)metric);
            if (distance < result->nearest_prior_distance) {
                result->nearest_prior_distance = distance;
                result->nearest_prior_prime = primes[j];
                result->has_nearest_prior = true;
            }
        }
        status = futcache_pack_observe(cache, point, &result->was_novel);
        if (status != FUTCACHE_OK) {
            futcache_pack_destroy(cache);
            return false;
        }
        if (result->was_novel) novel_count++;
    }

    futcache_pack_stats_t stats;
    status = futcache_pack_get_stats(cache, &stats);
    if (status != FUTCACHE_OK || stats.novel_observations != novel_count ||
        !attach_persistence(cache, metric, epsilon, results, count)) {
        futcache_pack_destroy(cache);
        return false;
    }
    assign_ranks(results, count, burn_in, ranking);
    *out_novel_count = novel_count;
    *out_representative_count = stats.representative_count;
    futcache_pack_destroy(cache);
    return true;
}

static bool write_csv(FILE *stream, size_t count,
                      const metric_context_t *metrics, size_t metric_count,
                      const observation_result_t *all_results)
{
    if (fprintf(stream,
        "metric,sequence_index,prime,nearest_prior_prime,"
        "nearest_prior_distance,was_novel,is_representative,"
        "final_representative_persistence,novelty_rank\n") < 0) return false;
    for (size_t metric = 0U; metric < metric_count; ++metric) {
        const observation_result_t *results = all_results + metric * count;
        for (size_t i = 0U; i < count; ++i) {
            const observation_result_t *result = &results[i];
            if (fprintf(stream, "%s,%zu,%" PRIu64 ",",
                    metrics[metric].name, i, result->prime) < 0) return false;
            if (result->has_nearest_prior &&
                fprintf(stream, "%" PRIu64,
                    result->nearest_prior_prime) < 0) return false;
            if (fputc(',', stream) == EOF) return false;
            if (result->has_nearest_prior &&
                fprintf(stream, "%.17g",
                    result->nearest_prior_distance) < 0) return false;
            if (fprintf(stream, ",%d,%d,", result->was_novel ? 1 : 0,
                    result->is_representative ? 1 : 0) < 0) return false;
            if (result->is_representative &&
                fprintf(stream, "%.17g", result->final_persistence) < 0) {
                return false;
            }
            if (fprintf(stream, ",%zu\n", result->novelty_rank) < 0) {
                return false;
            }
        }
    }
    return fflush(stream) == 0;
}

static void print_summary(FILE *stream, const options_t *options,
                          const metric_context_t *metrics,
                          size_t metric_count,
                          const observation_result_t *all_results,
                          const size_t *novel_counts,
                          const size_t *representative_counts)
{
    fprintf(stream, "FUTCache arithmetic prime novelty experiment\n");
    fprintf(stream, "primes=%zu epsilon=%.6g arch_weight=%.6g local_primes=",
        options->count, options->epsilon, options->arch_weight);
    for (size_t i = 0U; i < options->local_count; ++i) {
        fprintf(stream, "%s%" PRIu64, i == 0U ? "" : ",",
            options->local_primes[i]);
    }
    fprintf(stream,
        "\nranking score = distance to nearest earlier prime; burn_in=%zu\n",
        options->burn_in);
    size_t top = options->top_count;
    if (top > options->count - options->burn_in) {
        top = options->count - options->burn_in;
    }
    for (size_t metric = 0U; metric < metric_count; ++metric) {
        const observation_result_t *results =
            all_results + metric * options->count;
        fprintf(stream, "\n[%s] novel=%zu representatives=%zu\n",
            metrics[metric].name, novel_counts[metric],
            representative_counts[metric]);
        for (size_t rank = 1U; rank <= top; ++rank) {
            for (size_t i = options->burn_in; i < options->count; ++i) {
                if (results[i].novelty_rank == rank) {
                    fprintf(stream, "  #%zu prime=%" PRIu64
                            " score=%.9g nearest=%" PRIu64 " verdict=%s\n",
                        rank, results[i].prime,
                        results[i].nearest_prior_distance,
                        results[i].nearest_prior_prime,
                        results[i].was_novel ? "novel" : "redundant");
                    break;
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    options_t options;
    if (!parse_options(argc, argv, &options)) {
        usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }
    uint64_t *primes = NULL;
    if (!generate_primes(options.count, &primes) ||
        primes[options.count - 1U] > UINT64_C(9007199254740992)) {
        fputs("failed to generate an exactly representable prime sequence\n",
            stderr);
        free(primes);
        return EXIT_FAILURE;
    }

    size_t metric_count = options.local_count + 3U;
    if (options.count > SIZE_MAX / metric_count ||
        options.count * metric_count >
            SIZE_MAX / sizeof(observation_result_t)) {
        free(primes);
        return EXIT_FAILURE;
    }
    double *weights = (double *)malloc(options.local_count * sizeof(double));
    metric_context_t *metrics = (metric_context_t *)malloc(
        metric_count * sizeof(metric_context_t));
    observation_result_t *all_results = (observation_result_t *)calloc(
        options.count * metric_count, sizeof(observation_result_t));
    rank_entry_t *ranking = (rank_entry_t *)malloc(
        (options.count - 1U) * sizeof(rank_entry_t));
    size_t *novel_counts = (size_t *)calloc(metric_count, sizeof(size_t));
    size_t *representative_counts = (size_t *)calloc(
        metric_count, sizeof(size_t));
    bool success = weights != NULL && metrics != NULL && all_results != NULL &&
        ranking != NULL && novel_counts != NULL &&
        representative_counts != NULL &&
        initialize_metrics(&options, weights, metrics, metric_count) &&
        validate_metrics(primes, options.count, metrics, metric_count);

    for (size_t metric = 0U; success && metric < metric_count; ++metric) {
        success = run_metric(primes, options.count, options.epsilon,
            options.burn_in, &metrics[metric],
            all_results + metric * options.count, ranking,
            &novel_counts[metric], &representative_counts[metric]);
    }
    if (success) {
        FILE *summary = options.csv_path != NULL &&
            strcmp(options.csv_path, "-") == 0 ? stderr : stdout;
        print_summary(summary, &options, metrics, metric_count, all_results,
            novel_counts, representative_counts);
    }
    if (success && options.csv_path != NULL) {
        FILE *csv = stdout;
        if (strcmp(options.csv_path, "-") != 0) {
            csv = fopen(options.csv_path, "w");
            if (csv == NULL) {
                fprintf(stderr, "cannot open CSV '%s': %s\n",
                    options.csv_path, strerror(errno));
                success = false;
            }
        }
        if (success) {
            success = write_csv(csv, options.count, metrics, metric_count,
                all_results);
        }
        if (csv != NULL && csv != stdout && fclose(csv) != 0) success = false;
    }

    free(primes);
    free(weights);
    free(metrics);
    free(all_results);
    free(ranking);
    free(novel_counts);
    free(representative_counts);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
