#define _POSIX_C_SOURCE 200809L

#include "../navokoj_checkpoint.h"

/*
 * NitroSAT V3 - bounded-memory, disk-streaming CNF solver.
 *
 * V2 intentionally remains unchanged.  V3 stores clauses as an appendable
 * binary stream and never constructs an in-memory clause table or a global
 * variable-to-clause index.  The resident set is O(variables + batch literals
 * + active-cache literals), independent of the total number of clauses.
 */

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include "cdcl_internal.h"

#define STORE_VERSION 2u
#define STORE_HEADER_SIZE 40u
#define STORE_FLAG_WEIGHTED 1u
#define CLAUSE_FLAG_HARD 1u
#define CLAUSE_FLAG_WEIGHTED 2u
#define DEFAULT_BATCH_CLAUSES 65536u
#define DEFAULT_ACTIVE_CLAUSES 65536u
#define DEFAULT_ACTIVE_LITERALS 1048576u
#define DEFAULT_EPOCHS 20u
#define DEFAULT_FINISHER_PASSES 256u
#define DEFAULT_FINISHER_MAX_CLAUSES UINT64_C(100000)
#define DEFAULT_FINISHER_BATCH_FLIPS 64u
#define DEFAULT_LR 0.015
#define DEFAULT_EXACT_MAX_CLAUSES UINT64_C(1000000)
#define DEFAULT_EXACT_MEMORY_MB UINT64_C(1024)
#define DEFAULT_EXACT_MAX_CONFLICTS UINT64_C(5000000)
#define DEFAULT_INDEXED_FLIPS UINT64_C(200000)
#define DEFAULT_INDEXED_CHECKPOINT 1000u
#define MAX_CLAUSE_LITERALS UINT32_MAX

static const unsigned char STORE_MAGIC[8] = {'N','S','A','T','V','3','\r','\n'};

typedef struct {
    uint32_t num_vars;
    uint32_t flags;
    uint64_t num_clauses;
    uint64_t num_literals;
} StoreMeta;

typedef struct {
    FILE *fp;
    uint8_t *map;
    size_t map_size;
    const unsigned char *cursor;
    const unsigned char *end;
    int mapped;
    StoreMeta meta;
    uint64_t clauses_read;
    int32_t *lits;
    size_t lit_cap;
} ClauseReader;

typedef struct {
    uint32_t *offs;
    int32_t *lits;
    uint64_t *weights;
    uint8_t *hard;
    uint32_t clauses;
    uint32_t clause_cap;
    uint32_t literals;
    uint32_t literal_cap;
} ActiveCache;

typedef struct {
    double *x;
    double *m;
    double *v;
    double *grad;
    double *degrees;
    double mean_degree;
    uint32_t *marks;
    uint32_t *touched;
    uint32_t mark_generation;
    uint32_t touched_count;
    uint32_t num_vars;
    uint64_t rng;
    uint64_t batches;
} SolverState;

typedef struct {
    uint64_t satisfied;
    uint64_t hard_unsatisfied;
    uint64_t soft_unsatisfied;
    uint64_t soft_cost;
    uint64_t total_soft_weight;
    int cost_overflow;
} Verification;

typedef struct {
    int fd;
    uint64_t *starts;
    uint32_t num_vars;
    uint64_t entries;
} OccurrenceIndex;

typedef struct {
    const char *base_cnf;
    const char **add_files;
    int add_count;
    const char *store_path;
    const char *solution_path;
    int load_store;
    int keep_store;
    int store_only;
    uint32_t epochs;
    uint32_t finisher_passes;
    uint32_t finisher_batch_flips;
    uint64_t finisher_max_clauses;
    uint32_t batch_clauses;
    uint32_t active_clauses;
    uint32_t active_literals;
    double learning_rate;
    uint64_t seed;
    int json;
    int exact;
    const char *proof_path;
    const char *weights_path;
    uint64_t exact_max_clauses;
    uint64_t exact_memory_mb;
    uint64_t exact_max_conflicts;
    int indexed_finisher;
    uint64_t indexed_flips;
    uint32_t indexed_checkpoint;
    NavokojCheckpoint checkpoint;
} Options;

static void put_u32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24);
}

static void put_u64(unsigned char *p, uint64_t v)
{
    for (unsigned i = 0; i < 8; ++i) p[i] = (unsigned char)(v >> (i * 8));
}

static uint32_t get_u32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t get_u64(const unsigned char *p)
{
    uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (i * 8);
    return v;
}

static int write_header(FILE *fp, const StoreMeta *m)
{
    unsigned char h[STORE_HEADER_SIZE] = {0};
    memcpy(h, STORE_MAGIC, sizeof(STORE_MAGIC));
    put_u32(h + 8, STORE_VERSION);
    put_u32(h + 12, STORE_HEADER_SIZE);
    put_u32(h + 16, m->num_vars);
    put_u32(h + 20, m->flags);
    put_u64(h + 24, m->num_clauses);
    put_u64(h + 32, m->num_literals);
    if (fseeko(fp, 0, SEEK_SET) != 0) return 0;
    return fwrite(h, 1, sizeof(h), fp) == sizeof(h);
}

static int read_header(FILE *fp, StoreMeta *m)
{
    unsigned char h[STORE_HEADER_SIZE];
    if (fseeko(fp, 0, SEEK_SET) != 0 || fread(h, 1, sizeof(h), fp) != sizeof(h)) return 0;
    if (memcmp(h, STORE_MAGIC, sizeof(STORE_MAGIC)) != 0 ||
        get_u32(h + 8) != STORE_VERSION || get_u32(h + 12) != STORE_HEADER_SIZE) return 0;
    m->num_vars = get_u32(h + 16);
    m->flags = get_u32(h + 20);
    m->num_clauses = get_u64(h + 24);
    m->num_literals = get_u64(h + 32);
    return 1;
}

static int write_clause(FILE *fp, const int32_t *lits, uint32_t count,
                        uint64_t weight, uint32_t flags)
{
    unsigned char b[8];
    put_u32(b, count);
    put_u32(b + 4, flags);
    if (fwrite(b, 1, sizeof(b), fp) != sizeof(b)) return 0;
    put_u64(b, weight);
    if (fwrite(b, 1, sizeof(b), fp) != sizeof(b)) return 0;
    for (uint32_t i = 0; i < count; ++i) {
        put_u32(b, (uint32_t)lits[i]);
        if (fwrite(b, 1, 4, fp) != 4) return 0;
    }
    return 1;
}

static int grow_literals(int32_t **lits, size_t *cap, size_t need)
{
    if (need <= *cap) return 1;
    size_t n = *cap ? *cap : 16;
    while (n < need) {
        if (n > SIZE_MAX / 2) { errno = ENOMEM; return 0; }
        n *= 2;
    }
    if (n > SIZE_MAX / sizeof(**lits)) { errno = ENOMEM; return 0; }
    int32_t *p = realloc(*lits, n * sizeof(*p));
    if (!p) return 0;
    *lits = p; *cap = n;
    return 1;
}

/* Token scanner handles arbitrary line length, comments, and clauses spanning lines. */
static int next_dimacs_token(FILE *fp, int64_t *value, int *is_problem,
                             uint32_t *problem_vars, uint64_t *problem_clauses,
                             int *problem_weighted, uint64_t *problem_top,
                             int *problem_has_top, int *is_hard_marker)
{
    int ch;
    *is_problem = 0;
    *is_hard_marker = 0;
    for (;;) {
        ch = fgetc(fp);
        if (ch == EOF) return feof(fp) ? 0 : -1;
        if (isspace((unsigned char)ch)) continue;
        if (ch == 'c') {
            while ((ch = fgetc(fp)) != EOF && ch != '\n') {}
            continue;
        }
        if (ch == 'p') {
            char line[512], kind[16], extra;
            unsigned long long clauses, vars, top = 0;
            if (!fgets(line, sizeof(line), fp)) return -1;
            int fields = sscanf(line, "%15s %llu %llu %llu %c",
                                kind, &vars, &clauses, &top, &extra);
            int weighted = !strcmp(kind, "wcnf");
            if (vars > INT32_MAX ||
                (!weighted && (strcmp(kind, "cnf") != 0 || fields != 3)) ||
                (weighted && fields != 3 && fields != 4) ||
                (weighted && fields == 4 && top == 0))
                return -1;
            *problem_vars = (uint32_t)vars;
            *problem_clauses = (uint64_t)clauses;
            *problem_weighted = weighted;
            *problem_has_top = weighted && fields == 4;
            *problem_top = (uint64_t)top;
            *is_problem = 1;
            return 1;
        }
        ungetc(ch, fp);
        char token[64];
        size_t n = 0;
        while ((ch = fgetc(fp)) != EOF && !isspace((unsigned char)ch)) {
            if (n + 1 >= sizeof(token)) return -1;
            token[n++] = (char)ch;
        }
        token[n] = '\0';
        if (!strcmp(token, "h")) {
            *is_hard_marker = 1;
            *value = 0;
            return 1;
        }
        char *end = NULL;
        errno = 0;
        long long parsed = strtoll(token, &end, 10);
        if (errno || end == token || *end != '\0') return -1;
        *value = (int64_t)parsed;
        return 1;
    }
}

static int append_dimacs(FILE *store, const char *path, StoreMeta *total)
{
    FILE *in = fopen(path, "rb");
    if (!in) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); return 0; }
    uint32_t declared_vars = 0;
    uint64_t declared_clauses = 0, parsed_clauses = 0;
    uint64_t top = 0, clause_weight_value = 1;
    int have_header = 0, weighted = 0, has_top = 0, have_clause_weight = 0;
    int clause_explicit_hard = 0;
    int32_t *clause = NULL;
    size_t clause_cap = 0, clause_len = 0;
    int ok = 1;

    for (;;) {
        int64_t tok = 0;
        int is_problem = 0;
        uint32_t pv = 0;
        uint64_t pc = 0;
        int pw = 0, pht = 0, hard_marker = 0;
        uint64_t pt = 0;
        int status = next_dimacs_token(in, &tok, &is_problem, &pv, &pc,
                                       &pw, &pt, &pht, &hard_marker);
        if (status == 0) break;
        if (status < 0) { fprintf(stderr, "%s: invalid DIMACS token\n", path); ok = 0; break; }
        if (is_problem) {
            if (have_header || parsed_clauses || clause_len || have_clause_weight) {
                fprintf(stderr, "%s: misplaced or duplicate problem line\n", path); ok = 0; break;
            }
            declared_vars = pv; declared_clauses = pc; weighted = pw;
            top = pt; has_top = pht; have_header = 1;
            if (weighted) total->flags |= STORE_FLAG_WEIGHTED;
            continue;
        }
        if (!have_header) { fprintf(stderr, "%s: clause before problem line\n", path); ok = 0; break; }
        if (weighted && !have_clause_weight) {
            if (hard_marker) {
                clause_weight_value = 1;
                clause_explicit_hard = 1;
                have_clause_weight = 1;
                continue;
            }
            if (tok <= 0) {
                fprintf(stderr, "%s: WCNF clause weight must be positive\n", path); ok = 0; break;
            }
            clause_weight_value = (uint64_t)tok;
            have_clause_weight = 1;
            continue;
        }
        if (hard_marker) {
            fprintf(stderr, "%s: hard marker must start a WCNF clause\n", path); ok = 0; break;
        }
        if (tok == 0) {
            uint32_t clause_flags = weighted ? CLAUSE_FLAG_WEIGHTED : CLAUSE_FLAG_HARD;
            if (weighted && (clause_explicit_hard ||
                             (has_top && clause_weight_value >= top)))
                clause_flags |= CLAUSE_FLAG_HARD;
            if (clause_len > MAX_CLAUSE_LITERALS ||
                !write_clause(store, clause, (uint32_t)clause_len,
                              clause_weight_value, clause_flags)) {
                fprintf(stderr, "%s: failed writing clause: %s\n", path, strerror(errno)); ok = 0; break;
            }
            parsed_clauses++;
            if (UINT64_MAX - total->num_literals < clause_len) { ok = 0; errno = EOVERFLOW; break; }
            total->num_literals += (uint64_t)clause_len;
            clause_len = 0;
            clause_weight_value = 1;
            have_clause_weight = 0;
            clause_explicit_hard = 0;
        } else {
            if (tok < INT32_MIN || tok > INT32_MAX || tok == INT32_MIN ||
                (uint64_t)llabs((long long)tok) > declared_vars) {
                fprintf(stderr, "%s: literal %" PRId64 " exceeds declared variable range\n", path, tok);
                ok = 0; break;
            }
            if (!grow_literals(&clause, &clause_cap, clause_len + 1)) { ok = 0; break; }
            clause[clause_len++] = (int32_t)tok;
        }
    }
    if (ok && (clause_len || have_clause_weight)) {
        fprintf(stderr, "%s: unterminated final clause\n", path); ok = 0;
    }
    if (ok && parsed_clauses != declared_clauses) {
        fprintf(stderr, "%s: clause count mismatch (header=%" PRIu64 ", parsed=%" PRIu64 ")\n",
                path, declared_clauses, parsed_clauses); ok = 0;
    }
    if (ok) {
        if (UINT64_MAX - total->num_clauses < parsed_clauses) { ok = 0; errno = EOVERFLOW; }
        else total->num_clauses += parsed_clauses;
        if (declared_vars > total->num_vars) total->num_vars = declared_vars;
    }
    if (!ok && errno) fprintf(stderr, "%s: %s\n", path, strerror(errno));
    free(clause);
    fclose(in);
    return ok;
}

static int create_store(const Options *opt, StoreMeta *meta)
{
    FILE *fp = fopen(opt->store_path, "w+b");
    if (!fp) { fprintf(stderr, "cannot create %s: %s\n", opt->store_path, strerror(errno)); return 0; }
    memset(meta, 0, sizeof(*meta));
    int ok = write_header(fp, meta) && fseeko(fp, STORE_HEADER_SIZE, SEEK_SET) == 0;
    if (ok) ok = append_dimacs(fp, opt->base_cnf, meta);
    for (int i = 0; ok && i < opt->add_count; ++i) ok = append_dimacs(fp, opt->add_files[i], meta);
    if (ok) ok = write_header(fp, meta) && fflush(fp) == 0;
    if (fclose(fp) != 0) ok = 0;
    if (!ok) unlink(opt->store_path);
    return ok;
}

static int load_or_extend_store(const Options *opt, StoreMeta *meta)
{
    FILE *fp = fopen(opt->store_path, opt->add_count ? "r+b" : "rb");
    if (!fp) { fprintf(stderr, "cannot open store %s: %s\n", opt->store_path, strerror(errno)); return 0; }
    StoreMeta original = {0};
    int ok = read_header(fp, &original);
    off_t original_size = 0;
    if (ok && fseeko(fp, 0, SEEK_END) == 0) original_size = ftello(fp);
    else ok = 0;
    *meta = original;
    for (int i = 0; ok && i < opt->add_count; ++i) ok = append_dimacs(fp, opt->add_files[i], meta);
    if (ok && opt->add_count) ok = write_header(fp, meta) && fflush(fp) == 0;
    if (!ok && opt->add_count) {
        /* Roll back both bytes and metadata if any increment is invalid. */
        clearerr(fp);
        /* Flush pending append bytes before truncating, otherwise fclose could
           replay the stdio buffer beyond the restored end of file. */
        int rollback_ok = fflush(fp) == 0 && ftruncate(fileno(fp), original_size) == 0;
        if (!rollback_ok || !write_header(fp, &original) || fflush(fp) != 0)
            fprintf(stderr, "warning: failed to roll back store %s\n", opt->store_path);
        *meta = original;
    }
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int reader_open(ClauseReader *r, const char *path)
{
    memset(r, 0, sizeof(*r));
    r->fp = fopen(path, "rb");
    if (!r->fp) return 0;
    if (!read_header(r->fp, &r->meta)) {
        fclose(r->fp); r->fp = NULL; return 0;
    }
    struct stat st;
    if (fstat(fileno(r->fp), &st) == 0 && S_ISREG(st.st_mode) &&
        st.st_size >= (off_t)STORE_HEADER_SIZE &&
        (uintmax_t)st.st_size <= (uintmax_t)SIZE_MAX) {
        void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fileno(r->fp), 0);
        if (m != MAP_FAILED) {
            r->map = (uint8_t *)m;
            r->map_size = (size_t)st.st_size;
            r->mapped = 1;
            r->cursor = r->map + STORE_HEADER_SIZE;
            r->end = r->map + r->map_size;
            return 1;
        }
    }
    if (fseeko(r->fp, STORE_HEADER_SIZE, SEEK_SET) != 0) {
        fclose(r->fp); r->fp = NULL; return 0;
    }
    return 1;
}

static int reader_rewind(ClauseReader *r)
{
    r->clauses_read = 0;
    if (r->mapped) {
        r->cursor = r->map + STORE_HEADER_SIZE;
        return 1;
    }
    clearerr(r->fp);
    return fseeko(r->fp, STORE_HEADER_SIZE, SEEK_SET) == 0;
}

static uint64_t reader_tell(const ClauseReader *r)
{
    if (r->mapped) return (uint64_t)(r->cursor - r->map);
    return (uint64_t)ftello(r->fp);
}

static int reader_next(ClauseReader *r, const int32_t **lits, uint32_t *count,
                       uint64_t *weight, uint32_t *flags)
{
    if (r->clauses_read >= r->meta.num_clauses) return 0;
    if (r->mapped) {
        if ((size_t)(r->end - r->cursor) < 16) return -1;
        uint32_t n = get_u32(r->cursor);
        uint32_t clause_flags = get_u32(r->cursor + 4);
        if (clause_flags & ~(CLAUSE_FLAG_HARD | CLAUSE_FLAG_WEIGHTED)) return -1;
        uint64_t clause_weight_value = get_u64(r->cursor + 8);
        if (!clause_weight_value) return -1;
        r->cursor += 16;
        if ((size_t)(r->end - r->cursor) < (size_t)n * 4u) return -1;
        if (!grow_literals(&r->lits, &r->lit_cap, n)) return -1;
        memcpy(r->lits, r->cursor, (size_t)n * 4u);
        r->cursor += (size_t)n * 4u;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t v = (uint32_t)(r->lits[i] < 0 ? -(int64_t)r->lits[i] : r->lits[i]);
            if (!v || v > r->meta.num_vars) return -1;
        }
        r->clauses_read++;
        *lits = r->lits; *count = n;
        *weight = clause_weight_value; *flags = clause_flags;
        return 1;
    }
    unsigned char b[8];
    if (fread(b, 1, 8, r->fp) != 8) return -1;
    uint32_t n = get_u32(b);
    uint32_t clause_flags = get_u32(b + 4);
    if (clause_flags & ~(CLAUSE_FLAG_HARD | CLAUSE_FLAG_WEIGHTED)) return -1;
    if (fread(b, 1, 8, r->fp) != 8) return -1;
    uint64_t clause_weight_value = get_u64(b);
    if (!clause_weight_value) return -1;
    if (!grow_literals(&r->lits, &r->lit_cap, n)) return -1;
    for (uint32_t i = 0; i < n; ++i) {
        if (fread(b, 1, 4, r->fp) != 4) return -1;
        r->lits[i] = (int32_t)get_u32(b);
        uint32_t v = (uint32_t)(r->lits[i] < 0 ? -(int64_t)r->lits[i] : r->lits[i]);
        if (!v || v > r->meta.num_vars) return -1;
    }
    r->clauses_read++;
    *lits = r->lits; *count = n;
    *weight = clause_weight_value; *flags = clause_flags;
    return 1;
}

static void reader_close(ClauseReader *r)
{
    if (r->mapped && r->map) munmap(r->map, r->map_size);
    if (r->fp) fclose(r->fp);
    free(r->lits);
    memset(r, 0, sizeof(*r));
}

static void occurrence_index_close(OccurrenceIndex *index)
{
    if (index->fd >= 0) close(index->fd);
    free(index->starts);
    memset(index, 0, sizeof(*index));
    index->fd = -1;
}

static int occurrence_index_build(ClauseReader *reader, OccurrenceIndex *index, int hard_only)
{
    memset(index, 0, sizeof(*index));
    index->fd = -1;
    index->num_vars = reader->meta.num_vars;
    size_t vars = (size_t)index->num_vars + 2u;
    if (vars > SIZE_MAX / sizeof(uint64_t)) { errno = EOVERFLOW; return 0; }
    uint64_t *counts = calloc(vars, sizeof(*counts));
    index->starts = calloc(vars, sizeof(*index->starts));
    uint64_t *cursor = calloc(vars, sizeof(*cursor));
    if (!counts || !index->starts || !cursor) {
        free(counts); free(cursor); occurrence_index_close(index); return 0;
    }

    if (!reader_rewind(reader)) goto fail;
    const int32_t *lits;
    uint32_t n, flags;
    uint64_t weight;
    int status;
    while ((status = reader_next(reader, &lits, &n, &weight, &flags)) > 0) {
        (void)weight; (void)flags;
        if (hard_only && !(flags & CLAUSE_FLAG_HARD)) continue;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t v = (uint32_t)(lits[i] < 0 ? -(int64_t)lits[i] : lits[i]);
            if (counts[v] == UINT64_MAX) { errno = EOVERFLOW; goto fail; }
            counts[v]++;
        }
    }
    if (status < 0) goto fail;
    for (uint32_t v = 1; v <= index->num_vars; ++v) {
        index->starts[v] = index->entries;
        if (UINT64_MAX - index->entries < counts[v]) { errno = EOVERFLOW; goto fail; }
        index->entries += counts[v];
    }
    index->starts[index->num_vars + 1u] = index->entries;
    memcpy(cursor, index->starts, vars * sizeof(*cursor));
    if (index->entries > SIZE_MAX / sizeof(uint64_t) ||
        index->entries > (uint64_t)INT64_MAX / sizeof(uint64_t)) {
        errno = EOVERFLOW; goto fail;
    }

    char path[] = "/tmp/nitrosat-occ-XXXXXX";
    index->fd = mkstemp(path);
    if (index->fd < 0) goto fail;
    unlink(path);
    size_t bytes = (size_t)index->entries * sizeof(uint64_t);
    if (ftruncate(index->fd, (off_t)bytes) != 0) goto fail;
    uint64_t *mapped = NULL;
    if (bytes) {
        mapped = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, index->fd, 0);
        if (mapped == MAP_FAILED) { mapped = NULL; goto fail; }
    }
    if (!reader_rewind(reader)) { if (mapped) munmap(mapped, bytes); goto fail; }
    while (1) {
        uint64_t clause_offset = reader_tell(reader);
        status = reader_next(reader, &lits, &n, &weight, &flags);
        if (status <= 0) break;
        if (hard_only && !(flags & CLAUSE_FLAG_HARD)) continue;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t v = (uint32_t)(lits[i] < 0 ? -(int64_t)lits[i] : lits[i]);
            mapped[cursor[v]++] = (uint64_t)clause_offset;
        }
    }
    if (status < 0 || (mapped && msync(mapped, bytes, MS_SYNC) != 0)) {
        if (mapped) munmap(mapped, bytes);
        goto fail;
    }
    if (mapped) {
        (void)posix_madvise(mapped, bytes, POSIX_MADV_DONTNEED);
        munmap(mapped, bytes);
    }
    free(counts); free(cursor);
    return 1;

fail:
    free(counts); free(cursor); occurrence_index_close(index);
    return 0;
}

static int read_clause_at(ClauseReader *r, uint64_t offset, int32_t **lits, size_t *capacity,
                          uint32_t *count, uint64_t *weight, uint32_t *flags)
{
    if (r->mapped) {
        if (offset > (uint64_t)r->map_size || r->map_size - (size_t)offset < 16u) return 0;
        const uint8_t *p = r->map + (size_t)offset;
        uint32_t n = get_u32(p);
        uint32_t clause_flags = get_u32(p + 4);
        if (clause_flags & ~(CLAUSE_FLAG_HARD | CLAUSE_FLAG_WEIGHTED)) return 0;
        uint64_t clause_weight_value = get_u64(p + 8);
        if (!clause_weight_value) return 0;
        p += 16;
        size_t remaining = r->map_size - (size_t)(p - r->map);
        if (remaining < (size_t)n * 4u) return 0;
        if (!grow_literals(lits, capacity, n)) return 0;
        memcpy(*lits, p, (size_t)n * 4u);
        *count = n; *weight = clause_weight_value; *flags = clause_flags;
        return 1;
    }
    if (offset > (uint64_t)INT64_MAX) return 0;
    off_t saved = ftello(r->fp);
    if (saved < 0 || fseeko(r->fp, (off_t)offset, SEEK_SET) != 0) return 0;
    unsigned char b[8];
    if (fread(b, 1, 8, r->fp) != 8) { (void)fseeko(r->fp, saved, SEEK_SET); return 0; }
    uint32_t n = get_u32(b);
    *flags = get_u32(b + 4);
    if (fread(b, 1, 8, r->fp) != 8) { (void)fseeko(r->fp, saved, SEEK_SET); return 0; }
    *weight = get_u64(b);
    if (!grow_literals(lits, capacity, n)) { (void)fseeko(r->fp, saved, SEEK_SET); return 0; }
    for (uint32_t i = 0; i < n; ++i) {
        if (fread(b, 1, 4, r->fp) != 4) { (void)fseeko(r->fp, saved, SEEK_SET); return 0; }
        (*lits)[i] = (int32_t)get_u32(b);
    }
    (void)fseeko(r->fp, saved, SEEK_SET);
    *count = n;
    return 1;
}

static int validate_store_file(const char *path, const StoreMeta *expected)
{
    ClauseReader r = {0};
    if (!reader_open(&r, path)) return 0;
    uint64_t literals = 0;
    const int32_t *lits;
    uint32_t count;
    uint64_t weight;
    uint32_t flags;
    int status;
    while ((status = reader_next(&r, &lits, &count, &weight, &flags)) > 0) {
        (void)lits;
        (void)weight;
        (void)flags;
        if (UINT64_MAX - literals < count) { status = -1; break; }
        literals += count;
    }
    int trailing_ok;
    if (r.mapped) {
        trailing_ok = (r.cursor == r.end);
    } else {
        int trailing = fgetc(r.fp);
        trailing_ok = (trailing == EOF && feof(r.fp));
    }
    int ok = status == 0 && r.clauses_read == expected->num_clauses &&
             literals == expected->num_literals && trailing_ok;
    reader_close(&r);
    return ok;
}

static uint64_t rng_next(SolverState *s)
{
    uint64_t x = s->rng;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    s->rng = x;
    return x * UINT64_C(2685821657736338717);
}

static double rng_unit(SolverState *s)
{
    return (double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

static int solver_init(SolverState *s, uint32_t vars, uint64_t seed)
{
    memset(s, 0, sizeof(*s));
    s->num_vars = vars;
    s->rng = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);
    size_t n = (size_t)vars + 1;
    if (n > SIZE_MAX / sizeof(double)) return 0;
    s->x = malloc(n * sizeof(double));
    s->m = calloc(n, sizeof(double));
    s->v = calloc(n, sizeof(double));
    s->grad = calloc(n, sizeof(double));
    s->degrees = calloc(n, sizeof(double));
    s->marks = calloc(n, sizeof(uint32_t));
    s->touched = malloc(n * sizeof(uint32_t));
    if (!s->x || !s->m || !s->v || !s->grad || !s->degrees ||
        !s->marks || !s->touched) return 0;
    for (uint32_t i = 1; i <= vars; ++i) s->x[i] = 0.45 + 0.1 * rng_unit(s);
    s->mark_generation = 1;
    return 1;
}

static void solver_free(SolverState *s)
{
    free(s->x); free(s->m); free(s->v); free(s->grad); free(s->degrees);
    free(s->marks); free(s->touched);
    memset(s, 0, sizeof(*s));
}

static int compute_stream_degrees(ClauseReader *reader, SolverState *solver)
{
    if (!reader_rewind(reader)) return 0;
    const int32_t *lits;
    uint32_t n, flags;
    uint64_t weight;
    int status;
    while ((status = reader_next(reader, &lits, &n, &weight, &flags)) > 0) {
        (void)weight; (void)flags;
        double contribution = n ? (double)(n - 1) : 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t v = (uint32_t)(lits[i] < 0 ? -(int64_t)lits[i] : lits[i]);
            solver->degrees[v] += contribution;
        }
    }
    if (status < 0 || reader->clauses_read != reader->meta.num_clauses) return 0;
    double sum = 0.0;
    for (uint32_t v = 1; v <= solver->num_vars; ++v) sum += solver->degrees[v];
    solver->mean_degree = solver->num_vars ? sum / solver->num_vars : 0.0;
    return 1;
}

static int cache_init(ActiveCache *c, uint32_t clauses, uint32_t literals)
{
    memset(c, 0, sizeof(*c));
    c->clause_cap = clauses; c->literal_cap = literals;
    c->offs = malloc(((size_t)clauses + 1) * sizeof(uint32_t));
    c->lits = malloc((size_t)(literals ? literals : 1) * sizeof(int32_t));
    c->weights = malloc((size_t)clauses * sizeof(uint64_t));
    c->hard = malloc((size_t)clauses * sizeof(uint8_t));
    if (!c->offs || !c->lits || !c->weights || !c->hard) return 0;
    c->offs[0] = 0;
    return 1;
}

static void cache_reset(ActiveCache *c)
{
    c->clauses = 0; c->literals = 0; c->offs[0] = 0;
}

static void cache_add(ActiveCache *c, const int32_t *lits, uint32_t n,
                      uint64_t weight, int hard)
{
    if (c->clauses >= c->clause_cap || n > c->literal_cap - c->literals) return;
    memcpy(c->lits + c->literals, lits, (size_t)n * sizeof(*lits));
    c->weights[c->clauses] = weight;
    c->hard[c->clauses] = (uint8_t)hard;
    c->literals += n;
    c->offs[++c->clauses] = c->literals;
}

static void cache_free(ActiveCache *c)
{
    free(c->offs); free(c->lits); free(c->weights); free(c->hard);
    memset(c, 0, sizeof(*c));
}

static int literal_is_satisfied(int32_t lit, double x)
{
    return lit > 0 ? x > 0.5 : x <= 0.5;
}

static double clause_weight(uint64_t clause_id)
{
    /* Procedural approximation to V2's nth-prime weighting; no 1B-entry sieve. */
    double n = (double)clause_id + 2.0;
    double p = n < 6.0 ? (2.0 + n) : n * (log(n) + log(log(n)) - 1.0);
    return 50.0 / (1.0 + log(fmax(p, 2.0)));
}

static void touch_gradient(SolverState *s, uint32_t var, double contribution)
{
    if (s->marks[var] != s->mark_generation) {
        s->marks[var] = s->mark_generation;
        s->grad[var] = 0.0;
        s->touched[s->touched_count++] = var;
    }
    s->grad[var] += contribution;
}

static void apply_batch(SolverState *s, double lr, double grad_clip)
{
    if (!s->touched_count) return;
    s->batches++;
    double b1 = 0.9, b2 = 0.999;
    double b1c = 1.0 - pow(b1, (double)s->batches);
    double b2c = 1.0 - pow(b2, (double)s->batches);
    for (uint32_t i = 0; i < s->touched_count; ++i) {
        uint32_t var = s->touched[i];
        double heat = exp(-0.5 * s->degrees[var] / (s->mean_degree + 1.0));
        double g = s->grad[var] * (1.0 + 0.1 * heat);
        if (g > grad_clip) g = grad_clip;
        if (g < -grad_clip) g = -grad_clip;
        s->m[var] = b1 * s->m[var] + (1.0 - b1) * g;
        s->v[var] = b2 * s->v[var] + (1.0 - b2) * g * g;
        s->x[var] -= lr * (s->m[var] / b1c) / (sqrt(s->v[var] / b2c) + 1e-8);
        if (s->x[var] < 1e-6) s->x[var] = 1e-6;
        if (s->x[var] > 1.0 - 1e-6) s->x[var] = 1.0 - 1e-6;
    }
    s->touched_count = 0;
    if (++s->mark_generation == 0) {
        memset(s->marks, 0, ((size_t)s->num_vars + 1) * sizeof(uint32_t));
        s->mark_generation = 1;
    }
}

static int process_epoch(ClauseReader *r, SolverState *s, ActiveCache *cache,
                         uint32_t batch_clauses, double lr, int hard_focus,
                         double grad_clip, uint64_t *unsat_out)
{
    if (!reader_rewind(r)) return 0;
    cache_reset(cache);
    uint64_t unsat = 0, clause_id = 0;
    uint32_t in_batch = 0;
    const int32_t *lits;
    uint32_t n;
    uint64_t stored_weight;
    uint32_t clause_flags;
    int status;
    while ((status = reader_next(r, &lits, &n, &stored_weight, &clause_flags)) > 0) {
        int satisfied = 0;
        double violation = 1.0;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t var = (uint32_t)(lits[i] < 0 ? -(int64_t)lits[i] : lits[i]);
            double x = s->x[var];
            if (literal_is_satisfied(lits[i], x)) satisfied = 1;
            double lv = lits[i] > 0 ? 1.0 - x : x;
            violation *= fmax(lv, 1e-12);
        }
        int weighted = (clause_flags & CLAUSE_FLAG_WEIGHTED) != 0;
        int hard = (clause_flags & CLAUSE_FLAG_HARD) != 0;
        int optimize_clause = !hard_focus || !weighted || hard;
        if (!satisfied) {
            unsat++;
            if (optimize_clause) cache_add(cache, lits, n, stored_weight, hard);
        }
        if (!optimize_clause) {
            clause_id++;
            if (++in_batch == batch_clauses) { apply_batch(s, lr, grad_clip); in_batch = 0; }
            continue;
        }
        double base_weight = (clause_flags & CLAUSE_FLAG_WEIGHTED)
                           ? (hard ? fmin((double)stored_weight, 1e8) : fmin((double)stored_weight, 1e5))
                           : clause_weight(clause_id);
        double coef = base_weight * violation;
        if (weighted && hard) {
            double slack = fmax(1.0 - violation, 1e-12);
            coef = base_weight * violation / slack;
        }
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t var = (uint32_t)(lits[i] < 0 ? -(int64_t)lits[i] : lits[i]);
            double lv = lits[i] > 0 ? 1.0 - s->x[var] : s->x[var];
            touch_gradient(s, var, coef * (lits[i] > 0 ? -1.0 : 1.0) / fmax(lv, 1e-12));
        }
        clause_id++;
        if (++in_batch == batch_clauses) { apply_batch(s, lr, grad_clip); in_batch = 0; }
    }
    apply_batch(s, lr, grad_clip);
    if (status < 0 || r->clauses_read != r->meta.num_clauses) return 0;
    *unsat_out = unsat;
    return 1;
}

static void repair_active_cache(ActiveCache *c, SolverState *s)
{
    /* Bounded stochastic local repair. Recheck because earlier flips overlap. */
    if (!c->clauses) return;
    uint64_t rounds = (uint64_t)c->clauses * 2;
    for (uint64_t r = 0; r < rounds; ++r) {
        uint32_t ci = (uint32_t)(rng_next(s) % c->clauses);
        uint32_t alternative = (uint32_t)(rng_next(s) % c->clauses);
        if ((c->hard[alternative] && !c->hard[ci]) ||
            (c->hard[alternative] == c->hard[ci] && c->weights[alternative] > c->weights[ci]))
            ci = alternative;
        uint32_t begin = c->offs[ci], end = c->offs[ci + 1];
        int sat = 0;
        for (uint32_t j = begin; j < end; ++j) {
            int32_t lit = c->lits[j];
            uint32_t v = (uint32_t)(lit < 0 ? -(int64_t)lit : lit);
            if (literal_is_satisfied(lit, s->x[v])) { sat = 1; break; }
        }
        if (!sat && end > begin) {
            int32_t lit = c->lits[begin + (uint32_t)(rng_next(s) % (end - begin))];
            uint32_t v = (uint32_t)(lit < 0 ? -(int64_t)lit : lit);
            s->x[v] = lit > 0 ? 0.999 : 0.001;
        }
    }
}

static void add_cost(uint64_t *total, uint64_t value, int *overflow)
{
    if (UINT64_MAX - *total < value) { *total = UINT64_MAX; *overflow = 1; }
    else *total += value;
}

static int verify_assignment(ClauseReader *r, SolverState *s, Verification *result)
{
    if (!reader_rewind(r)) return 0;
    memset(result, 0, sizeof(*result));
    const int32_t *lits;
    uint32_t n;
    uint64_t weight;
    uint32_t flags;
    int status;
    while ((status = reader_next(r, &lits, &n, &weight, &flags)) > 0) {
        int clause_sat = 0;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t v = (uint32_t)(lits[i] < 0 ? -(int64_t)lits[i] : lits[i]);
            if (literal_is_satisfied(lits[i], s->x[v])) { clause_sat = 1; break; }
        }
        result->satisfied += (uint64_t)clause_sat;
        if (!(flags & CLAUSE_FLAG_HARD)) {
            add_cost(&result->total_soft_weight, weight, &result->cost_overflow);
            if (!clause_sat) {
                result->soft_unsatisfied++;
                add_cost(&result->soft_cost, weight, &result->cost_overflow);
            }
        } else if (!clause_sat) result->hard_unsatisfied++;
    }
    if (status < 0 || r->clauses_read != r->meta.num_clauses) return 0;
    return 1;
}

static int verification_better(const Verification *candidate, const Verification *best)
{
    if (candidate->hard_unsatisfied != best->hard_unsatisfied)
        return candidate->hard_unsatisfied < best->hard_unsatisfied;
    if (candidate->soft_cost != best->soft_cost)
        return candidate->soft_cost < best->soft_cost;
    return candidate->satisfied > best->satisfied;
}

static void retain_best(const Verification *candidate, Verification *best,
                        int *have_best, double *best_x, const SolverState *solver)
{
    if (!*have_best || verification_better(candidate, best)) {
        *best = *candidate;
        memcpy(best_x, solver->x, ((size_t)solver->num_vars + 1) * sizeof(double));
        *have_best = 1;
    }
}

static double optimization_weight(uint64_t clause_id, uint64_t stored_weight,
                                  uint32_t clause_flags)
{
    if (!(clause_flags & CLAUSE_FLAG_WEIGHTED)) return clause_weight(clause_id);
    return (clause_flags & CLAUSE_FLAG_HARD)
         ? 1e6 : fmin((double)stored_weight, 1e5);
}

static int streaming_make_break_finisher(ClauseReader *reader, SolverState *solver,
                                         uint32_t max_passes, uint32_t batch_flips,
                                         Verification *best,
                                         int *have_best, double *best_x,
                                         uint32_t *passes_run)
{
    size_t vars_size = (size_t)solver->num_vars + 1;
    double *make = calloc(vars_size, sizeof(double));
    double *brk = calloc(vars_size, sizeof(double));
    double *heat_force = calloc(vars_size, sizeof(double));
    uint32_t *batch_marks = calloc(vars_size, sizeof(uint32_t));
    uint32_t *candidates = malloc((size_t)batch_flips * sizeof(uint32_t));
    double *candidate_scores = malloc((size_t)batch_flips * sizeof(double));
    int32_t *escape_clause = NULL;
    size_t escape_cap = 0;
    uint32_t previous_var = 0;
    if (!make || !brk || !heat_force || !batch_marks || !candidates || !candidate_scores) {
        free(make); free(brk); free(heat_force); free(batch_marks);
        free(candidates); free(candidate_scores);
        return 0;
    }

    for (uint32_t pass = 0; pass < max_passes; ++pass) {
        memset(make, 0, vars_size * sizeof(double));
        memset(brk, 0, vars_size * sizeof(double));
        memset(heat_force, 0, vars_size * sizeof(double));
        if (!reader_rewind(reader)) {
            free(make); free(brk); free(heat_force); free(batch_marks); free(candidates);
            free(candidate_scores); free(escape_clause); return 0;
        }
        uint64_t clause_id = 0;
        uint64_t escape_seen = 0;
        uint32_t escape_n = 0;
        int escape_is_hard = 0;
        const int32_t *lits;
        uint32_t n, flags;
        uint64_t stored_weight;
        int status;
        while ((status = reader_next(reader, &lits, &n, &stored_weight, &flags)) > 0) {
            uint32_t sat_count = 0, sole_var = 0;
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t var = (uint32_t)(lits[i] < 0 ? -(int64_t)lits[i] : lits[i]);
                if (literal_is_satisfied(lits[i], solver->x[var])) {
                    sat_count++;
                    sole_var = var;
                }
            }
            double weight = optimization_weight(clause_id, stored_weight, flags);
            if (sat_count == 0) {
                for (uint32_t i = 0; i < n; ++i) {
                    uint32_t var = (uint32_t)(lits[i] < 0 ? -(int64_t)lits[i] : lits[i]);
                    make[var] += weight;
                    double hk = exp(-0.5 * solver->degrees[var] / (solver->mean_degree + 1.0));
                    heat_force[var] += weight * hk;
                }
                int hard = (flags & CLAUSE_FLAG_HARD) != 0;
                if (n && hard && !escape_is_hard) {
                    escape_seen = 0;
                    escape_n = 0;
                    escape_is_hard = 1;
                }
                if (n && hard == escape_is_hard) {
                    escape_seen++;
                    int replace = !escape_n || rng_next(solver) % escape_seen == 0;
                    if (!replace) { clause_id++; continue; }
                    if (!grow_literals(&escape_clause, &escape_cap, n)) {
                        free(make); free(brk); free(heat_force); free(batch_marks); free(candidates);
                        free(candidate_scores); free(escape_clause); return 0;
                    }
                    memcpy(escape_clause, lits, (size_t)n * sizeof(*lits));
                    escape_n = n;
                    escape_is_hard = hard;
                }
            } else if (sat_count == 1) brk[sole_var] += weight;
            clause_id++;
        }
        if (status < 0 || reader->clauses_read != reader->meta.num_clauses) {
            free(make); free(brk); free(heat_force); free(batch_marks); free(candidates);
            free(candidate_scores); free(escape_clause); return 0;
        }
        if (!escape_n) break;

        uint32_t selected = 0, candidate_count = 0;
        double best_gain = -INFINITY;
        for (uint32_t v = 1; v <= solver->num_vars; ++v) {
            if (v == previous_var) continue;
            double gain = make[v] - brk[v];
            if (gain > best_gain) { best_gain = gain; selected = v; }
            if (gain > 1e-12) {
                double score = gain + 0.1 * heat_force[v];
                if (candidate_count < batch_flips) {
                    candidates[candidate_count] = v;
                    candidate_scores[candidate_count++] = score;
                } else {
                    uint32_t min_i = 0;
                    for (uint32_t i = 1; i < candidate_count; ++i)
                        if (candidate_scores[i] < candidate_scores[min_i]) min_i = i;
                    if (score > candidate_scores[min_i]) {
                        candidates[min_i] = v; candidate_scores[min_i] = score;
                    }
                }
            }
        }
        /* At a local minimum, use the least-damaging variable from a currently
           unsatisfied clause. This is a bounded WalkSAT escape informed by the
           global streamed break scores. */
        if (!(best_gain > 1e-12)) {
            selected = 0;
            best_gain = -INFINITY;
            if (rng_unit(solver) < 0.15) {
                for (uint32_t tries = 0; tries < escape_n * 2; ++tries) {
                    int32_t lit = escape_clause[rng_next(solver) % escape_n];
                    uint32_t v = (uint32_t)(lit < 0 ? -(int64_t)lit : lit);
                    if (v != previous_var || escape_n == 1) { selected = v; break; }
                }
            } else {
                for (uint32_t i = 0; i < escape_n; ++i) {
                    uint32_t v = (uint32_t)(escape_clause[i] < 0
                                          ? -(int64_t)escape_clause[i] : escape_clause[i]);
                    if (v == previous_var && escape_n > 1) continue;
                    double gain = make[v] - brk[v];
                    if (gain > best_gain ||
                        (gain == best_gain && (rng_next(solver) & 1))) {
                        best_gain = gain; selected = v;
                    }
                }
            }
        }
        if (!selected) break;
        if (best_gain > 1e-12 && candidate_count) {
            memset(batch_marks, 0, vars_size * sizeof(uint32_t));
            for (uint32_t i = 0; i < candidate_count; ++i) batch_marks[candidates[i]] = i + 1;
            /* Conflict screen: a clause may contain at most one simultaneous
               flip. Keep its highest heat/gain candidate and suppress peers. */
            if (candidate_count > 1) {
                if (!reader_rewind(reader)) {
                    free(make); free(brk); free(heat_force); free(batch_marks);
                    free(candidates); free(candidate_scores); free(escape_clause); return 0;
                }
                while ((status = reader_next(reader, &lits, &n, &stored_weight, &flags)) > 0) {
                    uint32_t keep = 0;
                    for (uint32_t j = 0; j < n; ++j) {
                        uint32_t v = (uint32_t)(lits[j] < 0 ? -(int64_t)lits[j] : lits[j]);
                        uint32_t mark = batch_marks[v];
                        if (mark && (!keep || candidate_scores[mark-1] > candidate_scores[keep-1]))
                            keep = mark;
                    }
                    if (keep) for (uint32_t j = 0; j < n; ++j) {
                        uint32_t v = (uint32_t)(lits[j] < 0 ? -(int64_t)lits[j] : lits[j]);
                        if (batch_marks[v] && batch_marks[v] != keep) batch_marks[v] = 0;
                    }
                }
                if (status < 0) {
                    free(make); free(brk); free(heat_force); free(batch_marks);
                    free(candidates); free(candidate_scores); free(escape_clause); return 0;
                }
            }
            for (uint32_t i = 0; i < candidate_count; ++i) {
                uint32_t v = candidates[i];
                if (batch_marks[v]) solver->x[v] = solver->x[v] > 0.5 ? 0.001 : 0.999;
            }
            previous_var = 0;
        } else {
            previous_var = selected;
            solver->x[selected] = solver->x[selected] > 0.5 ? 0.001 : 0.999;
        }
        Verification current;
        if (!verify_assignment(reader, solver, &current)) {
            free(make); free(brk); free(heat_force); free(batch_marks); free(candidates);
            free(candidate_scores); free(escape_clause); return 0;
        }
        retain_best(&current, best, have_best, best_x, solver);
        *passes_run = pass + 1;
        if (current.satisfied == reader->meta.num_clauses) break;
    }
    free(make); free(brk); free(heat_force); free(batch_marks); free(candidates);
    free(candidate_scores); free(escape_clause);
    return 1;
}

static int grow_offsets(uint64_t **offsets, size_t *capacity, size_t need)
{
    if (need <= *capacity) return 1;
    size_t next = *capacity ? *capacity : 256u;
    while (next < need) {
        if (next > SIZE_MAX / 2u) { errno = EOVERFLOW; return 0; }
        next *= 2u;
    }
    uint64_t *grown = realloc(*offsets, next * sizeof(*grown));
    if (!grown) return 0;
    *offsets = grown; *capacity = next;
    return 1;
}

static uint32_t clause_sat_count_after_flip(const int32_t *lits, uint32_t n,
                                             const SolverState *solver, uint32_t flip_var)
{
    uint32_t count = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t v = (uint32_t)(lits[i] < 0 ? -(int64_t)lits[i] : lits[i]);
        double x = solver->x[v];
        if (v == flip_var) x = x > 0.5 ? 0.001 : 0.999;
        if (literal_is_satisfied(lits[i], x)) count++;
    }
    return count;
}

static int occurrence_offsets(const OccurrenceIndex *index, uint32_t var,
                              uint64_t **buffer, size_t *capacity, size_t *count)
{
    uint64_t begin = index->starts[var], end = index->starts[var + 1u];
    if (end < begin || end - begin > SIZE_MAX) { errno = EOVERFLOW; return 0; }
    *count = (size_t)(end - begin);
    if (!grow_offsets(buffer, capacity, *count)) return 0;
    size_t bytes = *count * sizeof(**buffer);
    off_t position = (off_t)(begin * sizeof(**buffer));
    size_t done = 0;
    while (done < bytes) {
        ssize_t got = pread(index->fd, (unsigned char *)*buffer + done, bytes - done,
                            position + (off_t)done);
        if (got <= 0) return 0;
        done += (size_t)got;
    }
    return 1;
}

static int collect_unsatisfied_pool(ClauseReader *reader, SolverState *solver,
                                    uint64_t *pool, uint32_t pool_capacity,
                                    uint32_t *pool_size, Verification *result,
                                    int hard_only)
{
    if (!reader_rewind(reader)) return 0;
    memset(result, 0, sizeof(*result));
    *pool_size = 0;
    uint64_t unsat_seen = 0;
    const int32_t *lits;
    uint32_t n, flags;
    uint64_t weight;
    int status;
    while (1) {
        uint64_t offset = reader_tell(reader);
        status = reader_next(reader, &lits, &n, &weight, &flags);
        if (status <= 0) break;
        int sat = clause_sat_count_after_flip(lits, n, solver, 0) != 0;
        result->satisfied += (uint64_t)sat;
        if (!(flags & CLAUSE_FLAG_HARD)) {
            add_cost(&result->total_soft_weight, weight, &result->cost_overflow);
            if (!sat) {
                result->soft_unsatisfied++;
                add_cost(&result->soft_cost, weight, &result->cost_overflow);
            }
        } else if (!sat) result->hard_unsatisfied++;
        if (!sat && (!hard_only || (flags & CLAUSE_FLAG_HARD))) {
            unsat_seen++;
            if (*pool_size < pool_capacity) pool[(*pool_size)++] = (uint64_t)offset;
            else {
                uint64_t pick = rng_next(solver) % unsat_seen;
                if (pick < pool_capacity) pool[pick] = (uint64_t)offset;
            }
        }
    }
    return status == 0 && reader->clauses_read == reader->meta.num_clauses;
}

static int indexed_walksat_finisher(ClauseReader *reader,
                                    SolverState *solver, uint64_t max_flips,
                                    uint32_t checkpoint_interval, uint32_t pool_capacity,
                                    Verification *best, int *have_best, double *best_x,
                                    uint64_t *flips_run, int hard_only,
                                    uint64_t *index_entries_out)
{
    OccurrenceIndex index;
    if (!occurrence_index_build(reader, &index, hard_only)) return 0;
    if (index_entries_out) *index_entries_out = index.entries;
    uint64_t *pool = malloc((size_t)pool_capacity * sizeof(*pool));
    uint64_t *occurrences = NULL;
    size_t occurrence_capacity = 0;
    int32_t *clause = NULL;
    size_t clause_capacity = 0;
    int32_t *target_clause = NULL;
    size_t target_capacity = 0;
    if (!pool) {
        free(pool); occurrence_index_close(&index); return 0;
    }

    uint32_t pool_size = 0;
    Verification current;
    if (!collect_unsatisfied_pool(reader, solver, pool, pool_capacity, &pool_size, &current, hard_only))
        goto fail;
    retain_best(&current, best, have_best, best_x, solver);
    uint32_t n, flags;
    uint64_t weight;

    uint32_t stale_tries = 0;
    uint64_t attempts = 0;
    uint64_t max_attempts = max_flips > UINT64_MAX / 10u ? UINT64_MAX : max_flips * 10u;
    while (*flips_run < max_flips && pool_size && attempts++ < max_attempts) {
        uint32_t target_position = (uint32_t)(rng_next(solver) % pool_size);
        uint64_t target_offset = pool[target_position];
        if (!read_clause_at(reader, target_offset, &target_clause, &target_capacity,
                            &n, &weight, &flags)) goto fail;
        uint32_t target_n = n;
        if (clause_sat_count_after_flip(target_clause, target_n, solver, 0)) {
            pool[target_position] = pool[--pool_size];
            if (++stale_tries > pool_size * 2u) {
                if (!collect_unsatisfied_pool(reader, solver, pool, pool_capacity,
                                              &pool_size, &current, hard_only)) goto fail;
                retain_best(&current, best, have_best, best_x, solver);
                stale_tries = 0;
            }
            continue;
        }
        stale_tries = 0;

        uint32_t selected = 0;
        int64_t best_gain = INT64_MIN;
        if (target_n && rng_unit(solver) < 0.15) {
            int32_t lit = target_clause[rng_next(solver) % target_n];
            selected = (uint32_t)(lit < 0 ? -(int64_t)lit : lit);
        } else {
            for (uint32_t candidate = 0; candidate < target_n; ++candidate) {
                uint32_t var = (uint32_t)(target_clause[candidate] < 0
                                       ? -(int64_t)target_clause[candidate] : target_clause[candidate]);
                size_t occurrence_count;
                if (!occurrence_offsets(&index, var, &occurrences, &occurrence_capacity,
                                        &occurrence_count)) goto fail;
                int64_t gain = 0;
                uint64_t previous = UINT64_MAX;
                for (size_t i = 0; i < occurrence_count; ++i) {
                    uint64_t offset = occurrences[i];
                    if (offset == previous) continue;
                    previous = offset;
                    if (!read_clause_at(reader, offset, &clause, &clause_capacity,
                                        &n, &weight, &flags)) goto fail;
                    uint32_t before = clause_sat_count_after_flip(clause, n, solver, 0);
                    uint32_t after = clause_sat_count_after_flip(clause, n, solver, var);
                    if (!before && after) gain++;
                    else if (before && !after) gain--;
                }
                if (gain > best_gain || (gain == best_gain && (rng_next(solver) & 1u))) {
                    best_gain = gain; selected = var;
                }
            }
        }
        if (!selected) break;
        solver->x[selected] = solver->x[selected] > 0.5 ? 0.001 : 0.999;
        (*flips_run)++;

        size_t occurrence_count;
        if (!occurrence_offsets(&index, selected, &occurrences, &occurrence_capacity,
                                &occurrence_count)) goto fail;
        uint64_t previous = UINT64_MAX;
        int replaced_target = 0;
        for (size_t i = 0; i < occurrence_count; ++i) {
            uint64_t offset = occurrences[i];
            if (offset == previous) continue;
            previous = offset;
            if (!read_clause_at(reader, offset, &clause, &clause_capacity,
                                &n, &weight, &flags)) goto fail;
            if (!clause_sat_count_after_flip(clause, n, solver, 0)) {
                if (!replaced_target) {
                    pool[target_position] = offset;
                    replaced_target = 1;
                } else if (pool_size < pool_capacity) pool[pool_size++] = offset;
                else pool[rng_next(solver) % pool_capacity] = offset;
            }
        }
        if (!replaced_target) pool[target_position] = pool[--pool_size];

        if (*flips_run % checkpoint_interval == 0) {
            if (!collect_unsatisfied_pool(reader, solver, pool, pool_capacity,
                                          &pool_size, &current, hard_only)) goto fail;
            retain_best(&current, best, have_best, best_x, solver);
            if (hard_only ? (current.hard_unsatisfied == 0) : (current.satisfied == reader->meta.num_clauses)) break;
        }
    }
    {
        if (!collect_unsatisfied_pool(reader, solver, pool, pool_capacity,
                                      &pool_size, &current, hard_only)) goto fail;
        retain_best(&current, best, have_best, best_x, solver);
    }
    free(pool); free(occurrences); free(clause); free(target_clause);
    occurrence_index_close(&index);
    return 1;

fail:
    free(pool); free(occurrences); free(clause); free(target_clause);
    occurrence_index_close(&index);
    return 0;
}

static double elapsed_ms(const struct timespec *a, const struct timespec *b)
{
    return (double)(b->tv_sec - a->tv_sec) * 1000.0 +
           (double)(b->tv_nsec - a->tv_nsec) / 1e6;
}

static int parse_u32(const char *s, uint32_t *out)
{
    char *end = NULL; errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s || *end || v > UINT32_MAX) return 0;
    *out = (uint32_t)v; return 1;
}

static int parse_u64(const char *s, uint64_t *out)
{
    char *end = NULL; errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s || *end) return 0;
    *out = (uint64_t)v; return 1;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
        "Usage: %s BASE.cnf [options]\n"
        "       %s --load-store FILE [options]\n"
        "  --add FILE              append an incremental DIMACS constraint set (repeatable)\n"
        "  --store FILE            binary clause-store path (default: temporary)\n"
        "  --load-store FILE       solve/extend an existing V3 store\n"
        "  --solution FILE         write the final Boolean assignment\n"
        "  --keep-store            retain the binary store after solving\n"
        "  --store-only            build/validate the store without solving\n"
        "  --epochs N              streaming passes (default: %u)\n"
        "  --finisher-passes N     global streamed make/break flips (default: %u)\n"
        "  --finisher-batch-flips N  positive-gain flips per pass (default: %u)\n"
        "  --finisher-max-clauses N  clause ceiling for finisher (default: %" PRIu64 ")\n"
        "  --indexed-finisher      disk-indexed WalkSAT finisher (hard-only for WCNF)\n"
        "  --indexed-flips N       indexed-finisher flip budget (default: %" PRIu64 ")\n"
        "  --indexed-checkpoint N  full verification interval (default: %u)\n"
        "  --batch-clauses N       optimizer batch size (default: %u)\n"
        "  --active-clauses N      local-repair clause bound (default: %u)\n"
        "  --active-literals N     local-repair literal bound (default: %u)\n"
        "  --learning-rate X       optimizer learning rate (default: %.3f)\n"
        "  --seed N                deterministic random seed\n"
        "  --cinematic             text output instead of JSON\n"
        "  --exact                 run formal CDCL solver if heuristic fails\n"
        "  --proof FILE            emit DRAT proof of unsatisfiability to FILE\n"
        "  --weights FILE          emit heuristic variable importance weights to FILE\n"
        "  --exact-max-clauses N   exact-mode clause ceiling (default: %" PRIu64 ")\n"
        "  --exact-memory-mb N     exact CDCL memory ceiling (default: %" PRIu64 ")\n"
        "  --exact-max-conflicts N exact CDCL conflict ceiling (default: %" PRIu64 ")\n",
        argv0, argv0, DEFAULT_EPOCHS, DEFAULT_FINISHER_PASSES,
        DEFAULT_FINISHER_BATCH_FLIPS,
        DEFAULT_FINISHER_MAX_CLAUSES, DEFAULT_INDEXED_FLIPS, DEFAULT_INDEXED_CHECKPOINT,
        DEFAULT_BATCH_CLAUSES,
        DEFAULT_ACTIVE_CLAUSES, DEFAULT_ACTIVE_LITERALS, DEFAULT_LR,
        DEFAULT_EXACT_MAX_CLAUSES, DEFAULT_EXACT_MEMORY_MB,
        DEFAULT_EXACT_MAX_CONFLICTS);
}

static int parse_options(int argc, char **argv, Options *o)
{
    memset(o, 0, sizeof(*o));
    o->epochs = DEFAULT_EPOCHS; o->finisher_passes = DEFAULT_FINISHER_PASSES;
    o->finisher_batch_flips = DEFAULT_FINISHER_BATCH_FLIPS;
    o->finisher_max_clauses = DEFAULT_FINISHER_MAX_CLAUSES;
    o->batch_clauses = DEFAULT_BATCH_CLAUSES;
    o->active_clauses = DEFAULT_ACTIVE_CLAUSES; o->active_literals = DEFAULT_ACTIVE_LITERALS;
    o->learning_rate = DEFAULT_LR; o->seed = 42; o->json = 1;
    o->exact_max_clauses = DEFAULT_EXACT_MAX_CLAUSES;
    o->exact_memory_mb = DEFAULT_EXACT_MEMORY_MB;
    o->exact_max_conflicts = DEFAULT_EXACT_MAX_CONFLICTS;
    o->indexed_flips = DEFAULT_INDEXED_FLIPS;
    o->indexed_checkpoint = DEFAULT_INDEXED_CHECKPOINT;
    if (argc < 2) return 0;
    o->add_files = calloc((size_t)argc, sizeof(char *));
    if (!o->add_files) return 0;
    int first_option;
    if (!strcmp(argv[1], "--load-store")) {
        if (argc < 3) return 0;
        o->load_store = 1; o->store_path = argv[2]; o->keep_store = 1; first_option = 3;
    } else if (argv[1][0] != '-') {
        o->base_cnf = argv[1]; first_option = 2;
    } else return 0;
    for (int i = first_option; i < argc; ++i) {
        if (!strcmp(argv[i], "--add") && i + 1 < argc) o->add_files[o->add_count++] = argv[++i];
        else if (!strcmp(argv[i], "--store") && i + 1 < argc) o->store_path = argv[++i];
        else if (!strcmp(argv[i], "--solution") && i + 1 < argc) o->solution_path = argv[++i];
        else if (!strcmp(argv[i], "--checkpoint") && i + 1 < argc) o->checkpoint.path = argv[++i];
        else if (!strcmp(argv[i], "--checkpoint-interval-ms") && i + 1 < argc) {
            double ms = atof(argv[++i]);
            o->checkpoint.interval_seconds = ms < 0.0 ? 0.0 : ms / 1000.0;
        }
        else if (!strcmp(argv[i], "--keep-store")) o->keep_store = 1;
        else if (!strcmp(argv[i], "--store-only")) o->store_only = 1;
        else if (!strcmp(argv[i], "--cinematic")) o->json = 0;
        else if (!strcmp(argv[i], "--indexed-finisher")) o->indexed_finisher = 1;
        else if (!strcmp(argv[i], "--indexed-flips") && i + 1 < argc && parse_u64(argv[i+1], &o->indexed_flips)) i++;
        else if (!strcmp(argv[i], "--indexed-checkpoint") && i + 1 < argc && parse_u32(argv[i+1], &o->indexed_checkpoint)) i++;
        else if (!strcmp(argv[i], "--exact")) o->exact = 1;
        else if (!strcmp(argv[i], "--proof") && i + 1 < argc) o->proof_path = argv[++i];
        else if (!strcmp(argv[i], "--weights") && i + 1 < argc) o->weights_path = argv[++i];
        else if (!strcmp(argv[i], "--exact-max-clauses") && i + 1 < argc && parse_u64(argv[i+1], &o->exact_max_clauses)) i++;
        else if (!strcmp(argv[i], "--exact-memory-mb") && i + 1 < argc && parse_u64(argv[i+1], &o->exact_memory_mb)) i++;
        else if (!strcmp(argv[i], "--exact-max-conflicts") && i + 1 < argc && parse_u64(argv[i+1], &o->exact_max_conflicts)) i++;
        else if (!strcmp(argv[i], "--epochs") && i + 1 < argc && parse_u32(argv[i+1], &o->epochs)) i++;
        else if (!strcmp(argv[i], "--finisher-passes") && i + 1 < argc && parse_u32(argv[i+1], &o->finisher_passes)) i++;
        else if (!strcmp(argv[i], "--finisher-batch-flips") && i + 1 < argc && parse_u32(argv[i+1], &o->finisher_batch_flips)) i++;
        else if (!strcmp(argv[i], "--finisher-max-clauses") && i + 1 < argc && parse_u64(argv[i+1], &o->finisher_max_clauses)) i++;
        else if (!strcmp(argv[i], "--batch-clauses") && i + 1 < argc && parse_u32(argv[i+1], &o->batch_clauses)) i++;
        else if (!strcmp(argv[i], "--active-clauses") && i + 1 < argc && parse_u32(argv[i+1], &o->active_clauses)) i++;
        else if (!strcmp(argv[i], "--active-literals") && i + 1 < argc && parse_u32(argv[i+1], &o->active_literals)) i++;
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc && parse_u64(argv[i+1], &o->seed)) i++;
        else if (!strcmp(argv[i], "--learning-rate") && i + 1 < argc) {
            char *end = NULL; errno = 0; o->learning_rate = strtod(argv[++i], &end);
            if (errno || end == argv[i] || *end || !(o->learning_rate > 0.0)) return 0;
        } else return 0;
    }
    if (o->proof_path) o->exact = 1;
    return o->batch_clauses > 0 && o->active_clauses > 0 &&
           o->active_literals > 0 && o->finisher_batch_flips > 0 &&
           o->exact_max_clauses > 0 && o->exact_memory_mb > 0 &&
           o->exact_max_conflicts > 0 && o->indexed_flips > 0 &&
           o->indexed_checkpoint > 0;
}

static int write_solution(const char *path, const SolverState *solver)
{
    if (!path) return 1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    for (uint32_t i = 1; i <= solver->num_vars; ++i) {
        int64_t lit = solver->x[i] > 0.5 ? (int64_t)i : -(int64_t)i;
        if (fprintf(fp, "%" PRId64 " ", lit) < 0) { fclose(fp); return 0; }
    }
    int ok = fprintf(fp, "0\n") >= 0;
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int write_importance(const char *path, const SolverState *solver, const double *x)
{
    if (!path) return 1;
    FILE *fp = fopen(path, "w");
    if (!fp) return 0;
    for (uint32_t i = 1; i <= solver->num_vars; ++i) {
        double frustration = 1.0 - 2.0 * fabs(x[i] - 0.5);
        double degree = solver->degrees[i];
        double importance = frustration * 10.0 + degree;
        if (fprintf(fp, "%u %.6f %.6f %.6f\n", i, frustration, degree, importance) < 0) {
            fclose(fp);
            return 0;
        }
    }
    int ok = 1;
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int streaming_cdcl_exact(ClauseReader *reader, SolverState *solver,
                                Verification *best, int *have_best, double *best_x,
                                int *exact_result, int *proof_generated,
                                const Options *opt)
{
    if (!reader_rewind(reader)) return 0;
    if (opt->exact_memory_mb > SIZE_MAX / (1024u * 1024u)) { errno = EOVERFLOW; return 0; }
    size_t memory_limit = (size_t)opt->exact_memory_mb * 1024u * 1024u;
    cdcl_t *cdcl = cdcl_new(solver->num_vars, memory_limit, opt->exact_max_conflicts);
    if (!cdcl) {
        if (errno == E2BIG) { *exact_result = CDCL_RESULT_LIMIT; return 1; }
        return 0;
    }
    char *proof_temp = NULL;
    if (opt->proof_path) {
        size_t path_len = strlen(opt->proof_path);
        if (path_len > SIZE_MAX - 16u) { cdcl_free(cdcl); errno = EOVERFLOW; return 0; }
        proof_temp = malloc(path_len + 16u);
        if (!proof_temp) { cdcl_free(cdcl); errno = ENOMEM; return 0; }
        snprintf(proof_temp, path_len + 16u, "%s.tmp.XXXXXX", opt->proof_path);
        int fd = mkstemp(proof_temp);
        if (fd < 0) { free(proof_temp); cdcl_free(cdcl); return 0; }
        cdcl->drat_fp = fdopen(fd, "w");
        if (!cdcl->drat_fp) { close(fd); unlink(proof_temp); free(proof_temp); cdcl_free(cdcl); return 0; }
    }

    for (uint32_t i = 1; i <= solver->num_vars; ++i) {
        double frustration = 1.0 - 2.0 * fabs(solver->x[i] - 0.5);
        cdcl->activity[i] = frustration * 10.0 + solver->degrees[i];
        cdcl->phase[i] = solver->x[i] > 0.5 ? 1 : -1;
    }

    const int32_t *lits;
    uint32_t n, flags;
    uint64_t weight;
    int status;
    int32_t *temp_lits = NULL;
    size_t temp_capacity = 0;

    while ((status = reader_next(reader, &lits, &n, &weight, &flags)) > 0) {
        if ((flags & CLAUSE_FLAG_HARD) || !(flags & CLAUSE_FLAG_WEIGHTED)) {
            if (n > temp_capacity) {
                int32_t *new_lits = realloc(temp_lits, (size_t)n * sizeof(*new_lits));
                if (!new_lits) { errno = ENOMEM; status = -1; break; }
                temp_lits = new_lits;
                temp_capacity = n;
            }
            for (uint32_t i = 0; i < n; ++i) temp_lits[i] = lits[i];
            if (!cdcl_add_clause(cdcl, temp_lits, n)) {
                if (cdcl->oom || cdcl->proof_error) status = -1;
                break;
            }
        }
    }
    if (status < 0) {
        if (cdcl->drat_fp) fclose(cdcl->drat_fp);
        if (proof_temp) unlink(proof_temp);
        free(proof_temp); free(temp_lits); cdcl_free(cdcl); return 0;
    }

    int result = cdcl_solve(cdcl);
    int solver_errno = cdcl->oom ? ENOMEM : EIO;
    *exact_result = result;
    if (result == 1) { // SAT
        for (uint32_t i = 1; i <= solver->num_vars; ++i) {
            solver->x[i] = cdcl->assigns[i] == 1 ? 0.999 : 0.001;
        }
        Verification current;
        if (!verify_assignment(reader, solver, &current) ||
            (!(reader->meta.flags & STORE_FLAG_WEIGHTED) &&
             current.satisfied != reader->meta.num_clauses) ||
            ((reader->meta.flags & STORE_FLAG_WEIGHTED) && current.hard_unsatisfied != 0)) {
            errno = EPROTO;
            if (cdcl->drat_fp) fclose(cdcl->drat_fp);
            cdcl->drat_fp = NULL;
            if (proof_temp) unlink(proof_temp);
            free(proof_temp); free(temp_lits); cdcl_free(cdcl); return 0;
        }
        retain_best(&current, best, have_best, best_x, solver);
    }
    if (cdcl->drat_fp) {
        int proof_ok = fflush(cdcl->drat_fp) == 0 && !ferror(cdcl->drat_fp);
        if (fclose(cdcl->drat_fp) != 0) proof_ok = 0;
        cdcl->drat_fp = NULL;
        if (!proof_ok) {
            unlink(proof_temp); free(proof_temp); free(temp_lits); cdcl_free(cdcl); return 0;
        }
        if (result == 0) {
            if (rename(proof_temp, opt->proof_path) != 0) {
                unlink(proof_temp); free(proof_temp); free(temp_lits); cdcl_free(cdcl); return 0;
            }
            *proof_generated = 1;
        } else {
            unlink(proof_temp);
        }
    }
    free(proof_temp);
    free(temp_lits);
    cdcl_free(cdcl);
    if (result == CDCL_RESULT_ERROR) { errno = solver_errno; return 0; }
    return 1;
}

int main(int argc, char **argv)
{
    Options opt;
    if (!parse_options(argc, argv, &opt)) { usage(argv[0]); free(opt.add_files); return 2; }
    char temp_path[] = "/tmp/nitrosatv3-XXXXXX";
    if (!opt.store_path) {
        int fd = mkstemp(temp_path);
        if (fd < 0) { perror("mkstemp"); free(opt.add_files); return 2; }
        close(fd); opt.store_path = temp_path;
    } else if (!opt.load_store) {
        /* An explicitly named store is a useful artifact unless removal is requested by omission. */
        opt.keep_store = 1;
    }

    struct timespec start, built, finished;
    clock_gettime(CLOCK_MONOTONIC, &start);
    StoreMeta meta;
    int store_ok = opt.load_store ? load_or_extend_store(&opt, &meta) : create_store(&opt, &meta);
    if (!store_ok) { free(opt.add_files); return 2; }
    clock_gettime(CLOCK_MONOTONIC, &built);

    if ((meta.flags & STORE_FLAG_WEIGHTED) && opt.proof_path) {
        fprintf(stderr, "--proof is supported only for CNF; WCNF hard-clause proofs require an exported projection\n");
        if (!opt.keep_store) unlink(opt.store_path);
        free(opt.add_files);
        return 2;
    }

    if (opt.store_only) {
        if (opt.load_store && !validate_store_file(opt.store_path, &meta)) {
            fprintf(stderr, "store validation failed: %s\n", opt.store_path);
            free(opt.add_files);
            return 2;
        }
        printf("{\"store\":\"%s\",\"format\":\"%s\",\"variables\":%u,"
               "\"clauses\":%" PRIu64 ",\"literals\":%" PRIu64
               ",\"increment_files\":%d}\n",
               opt.store_path, (meta.flags & STORE_FLAG_WEIGHTED) ? "wcnf" : "cnf",
               meta.num_vars, meta.num_clauses, meta.num_literals, opt.add_count);
        free(opt.add_files);
        return 0;
    }

    ClauseReader reader = {0};
    SolverState solver = {0};
    ActiveCache cache = {0};
    if (!reader_open(&reader, opt.store_path) || !solver_init(&solver, meta.num_vars, opt.seed) ||
        !compute_stream_degrees(&reader, &solver) ||
        !cache_init(&cache, opt.active_clauses, opt.active_literals)) {
        fprintf(stderr, "V3 initialization failed: %s\n", strerror(errno));
        reader_close(&reader); solver_free(&solver); cache_free(&cache);
        if (!opt.keep_store) unlink(opt.store_path);
        free(opt.add_files); return 2;
    }

    uint64_t unsat = meta.num_clauses;
    uint32_t epochs_run = 0;
    int io_ok = 1;
    int weighted_store = (meta.flags & STORE_FLAG_WEIGHTED) != 0;
    Verification best_verification = {0};
    int have_best = 0;
    double *best_x = malloc(((size_t)meta.num_vars + 1) * sizeof(double));

    /* Anytime checkpoint: open the periodic-best file so a Python watchdog
     * kill (SIGKILL) salvages whatever V3 had found. */
    if (!navokoj_checkpoint_open(&opt.checkpoint,
                                 opt.checkpoint.path,
                                 opt.checkpoint.interval_seconds)) {
        io_ok = 0;
    }
    /* Seed the checkpoint with the initial solver state so a kill before
     * the first retain_best() still leaves a parseable file. */
    if (io_ok) {
        uint8_t *seed_assignment = (uint8_t *)calloc(((size_t)meta.num_vars + 1u), sizeof(uint8_t));
        if (seed_assignment) {
            for (uint32_t v = 1; v <= meta.num_vars; ++v)
                seed_assignment[v] = solver.x[v] > 0.5 ? 1u : 0u;
            Verification initial_verification = {0};
            if (verify_assignment(&reader, &solver, &initial_verification)) {
                navokoj_checkpoint_update(&opt.checkpoint, meta.num_vars,
                                          seed_assignment,
                                          initial_verification.hard_unsatisfied);
                navokoj_checkpoint_flush(&opt.checkpoint);
            }
            free(seed_assignment);
        }
    }
    if (!best_x) io_ok = 0;
    for (uint32_t e = 0; io_ok && e < opt.epochs; ++e) {
        int hard_focus = weighted_store &&
                         (!have_best || best_verification.hard_unsatisfied > 0);
        double grad_clip = hard_focus ? 1e8 : 1000.0;
        if (!process_epoch(&reader, &solver, &cache, opt.batch_clauses,
                           opt.learning_rate / sqrt((double)e + 1.0),
                           hard_focus, grad_clip, &unsat)) {
            io_ok = 0; break;
        }
        epochs_run = e + 1;
        if (!opt.json) fprintf(stderr, "epoch %u: unsatisfied=%" PRIu64 "/%" PRIu64 " cached=%u\n",
                               epochs_run, unsat, meta.num_clauses, cache.clauses);
        if (weighted_store) {
            Verification before_repair;
            if (!verify_assignment(&reader, &solver, &before_repair)) { io_ok = 0; break; }
            retain_best(&before_repair, &best_verification, &have_best, best_x, &solver);
            if (before_repair.satisfied == meta.num_clauses) break;
        }
        if (unsat) repair_active_cache(&cache, &solver);
        Verification current;
        if (!verify_assignment(&reader, &solver, &current)) { io_ok = 0; break; }
        retain_best(&current, &best_verification, &have_best, best_x, &solver);
        /* Anytime checkpoint: capture the latest best_x after this epoch. */
        if (have_best) {
            /* Round best_x (double in [0,1]) to a uint8_t assignment that
             * matches write_solution()'s format. */
            uint8_t *assignment_u8 = (uint8_t *)calloc((size_t)meta.num_vars + 1u, sizeof(uint8_t));
            if (assignment_u8) {
                for (uint32_t v = 1; v <= meta.num_vars; ++v)
                    assignment_u8[v] = best_x[v] > 0.5 ? 1u : 0u;
                navokoj_checkpoint_update(&opt.checkpoint, meta.num_vars,
                                          assignment_u8,
                                          best_verification.hard_unsatisfied);
                navokoj_checkpoint_maybe(&opt.checkpoint);
                free(assignment_u8);
            }
        }
        if (current.satisfied == meta.num_clauses) break;
    }

    if (io_ok && !have_best) {
        Verification initial;
        io_ok = verify_assignment(&reader, &solver, &initial);
        if (io_ok) retain_best(&initial, &best_verification, &have_best, best_x, &solver);
    }

    Verification verification = {0};
    uint32_t finisher_passes_run = 0;
    uint64_t indexed_flips_run = 0;
    uint64_t index_entries = 0;
    if (io_ok && have_best && best_verification.satisfied < meta.num_clauses &&
        (!weighted_store || best_verification.hard_unsatisfied > 0) &&
        (opt.indexed_finisher ||
         (!weighted_store && meta.num_clauses > opt.finisher_max_clauses))) {
        if (!opt.indexed_finisher && meta.num_clauses > opt.finisher_max_clauses)
            fprintf(stderr, "indexed finisher auto-enabled (%" PRIu64 " clauses > streamed finisher max %" PRIu64 ")\n",
                    meta.num_clauses, opt.finisher_max_clauses);
        /* Scale flip/checkpoint budgets so checkpoint full-store rescans stay
           bounded regardless of the number of flips. */
        uint64_t effective_flips = opt.indexed_flips;
        if (meta.num_clauses < effective_flips) effective_flips = meta.num_clauses;
        if (effective_flips < 1) effective_flips = 1;
        uint64_t effective_checkpoint64 = meta.num_clauses / 64u;
        if (effective_checkpoint64 > (uint64_t)UINT32_MAX)
            effective_checkpoint64 = (uint64_t)UINT32_MAX;
        if (effective_checkpoint64 < (uint64_t)opt.indexed_checkpoint)
            effective_checkpoint64 = (uint64_t)opt.indexed_checkpoint;
        memcpy(solver.x, best_x, ((size_t)meta.num_vars + 1) * sizeof(double));
        io_ok = indexed_walksat_finisher(&reader, &solver,
                                         effective_flips,
                                         (uint32_t)effective_checkpoint64,
                                         opt.active_clauses, &best_verification,
                                         &have_best, best_x, &indexed_flips_run,
                                         weighted_store, &index_entries);
        if (!io_ok) fprintf(stderr, "indexed finisher failed: %s\n", strerror(errno));
    }
    
    if (io_ok && have_best && best_verification.satisfied < meta.num_clauses &&
        opt.finisher_passes && meta.num_clauses <= opt.finisher_max_clauses) {
        memcpy(solver.x, best_x, ((size_t)meta.num_vars + 1) * sizeof(double));
        io_ok = streaming_make_break_finisher(&reader, &solver, opt.finisher_passes,
                                              opt.finisher_batch_flips,
                                              &best_verification, &have_best, best_x,
                                              &finisher_passes_run);
    }

    double *importance_x = NULL;
    if (io_ok && opt.weights_path) {
        importance_x = malloc(((size_t)meta.num_vars + 1u) * sizeof(*importance_x));
        if (!importance_x) io_ok = 0;
        else if (have_best) memcpy(importance_x, best_x, ((size_t)meta.num_vars + 1u) * sizeof(*importance_x));
        else memcpy(importance_x, solver.x, ((size_t)meta.num_vars + 1u) * sizeof(*importance_x));
    }

    int exact_attempted = 0;
    int exact_result = CDCL_RESULT_NOT_RUN;
    int proof_generated = 0;
    if (io_ok && opt.exact && have_best && best_verification.satisfied < meta.num_clauses &&
        (!weighted_store || best_verification.hard_unsatisfied > 0)) {
        exact_attempted = 1;
        if (meta.num_clauses > opt.exact_max_clauses) {
            exact_result = CDCL_RESULT_LIMIT;
        } else {
            io_ok = streaming_cdcl_exact(&reader, &solver, &best_verification, &have_best, best_x,
                                         &exact_result, &proof_generated, &opt);
        }
        if (!io_ok) fprintf(stderr, "exact solver failed: %s\n", strerror(errno));
    }

    if (io_ok && have_best) {
        memcpy(solver.x, best_x, ((size_t)meta.num_vars + 1) * sizeof(double));
        verification = best_verification;
    } else if (io_ok) io_ok = verify_assignment(&reader, &solver, &verification);
    navokoj_checkpoint_close(&opt.checkpoint);
    if (io_ok && exact_result != 0 && !write_solution(opt.solution_path, &solver)) {
        fprintf(stderr, "cannot write solution %s: %s\n", opt.solution_path, strerror(errno));
        io_ok = 0;
    }
    if (io_ok && !write_importance(opt.weights_path, &solver,
                                    importance_x ? importance_x : solver.x)) {
        fprintf(stderr, "cannot write weights %s: %s\n", opt.weights_path, strerror(errno));
        io_ok = 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &finished);
    int solved = io_ok && verification.satisfied == meta.num_clauses;
    int feasible = io_ok && verification.hard_unsatisfied == 0;
    double resident_mb = (((double)meta.num_vars + 1.0) *
                          (5.0 * sizeof(double) + 3.0 * sizeof(uint32_t)) +
                          ((double)opt.active_clauses + 1.0) * sizeof(uint32_t) +
                          (double)opt.active_clauses * (sizeof(uint64_t) + sizeof(uint8_t)) +
                          ((double)meta.num_vars + 1.0) * sizeof(double) +
                          3.0 * ((double)meta.num_vars + 1.0) * sizeof(double) +
                          (double)opt.finisher_batch_flips *
                          (sizeof(uint32_t) + sizeof(double)) +
                          (double)opt.active_literals * sizeof(int32_t)) / (1024.0 * 1024.0);
    double indexed_index_mb = 0.0;
    if (opt.indexed_finisher) {
        uint64_t lits_for_index = index_entries ? index_entries : meta.num_literals;
        indexed_index_mb = ((double)lits_for_index * sizeof(uint64_t) +
                            ((double)meta.num_vars + 2.0) * sizeof(uint64_t)) / (1024.0 * 1024.0);
    }
    resident_mb += indexed_index_mb;
    if (opt.json) {
        const char *status_str = solved ? "SATISFIED" :
                                 (exact_result == 0 ? (weighted_store ? "INFEASIBLE" : "UNSATISFIABLE") :
                                 (exact_result == CDCL_RESULT_LIMIT ? "UNKNOWN" : "PARTIAL"));
        printf("{\"solver\":\"NitroSAT V3\",\"format\":\"%s\",\"status\":\"%s\",\"solved\":%s,"
               "\"feasible\":%s,\"variables\":%u,"
               "\"clauses\":%" PRIu64 ",\"literals\":%" PRIu64 ","
               "\"satisfied\":%" PRIu64 ",\"unsatisfied\":%" PRIu64 ","
               "\"hard_unsatisfied\":%" PRIu64 ",\"soft_unsatisfied\":%" PRIu64 ","
               "\"soft_cost\":%" PRIu64 ",\"total_soft_weight\":%" PRIu64 ","
               "\"cost_overflow\":%s,"
               "\"exact_attempted\":%s,\"exact_result\":\"%s\","
               "\"proof_requested\":%s,\"proof_generated\":%s,"
               "\"exact_memory_bounded\":%s,"
               "\"epochs\":%u,\"finisher_passes\":%u,\"indexed_flips\":%" PRIu64 ",\"increment_files\":%d,"
               "\"bounded_memory_mb\":%.2f,\"indexed_index_mb\":%.3f,"
               "\"store_build_ms\":%.3f,\"solve_ms\":%.3f}\n",
               (meta.flags & STORE_FLAG_WEIGHTED) ? "wcnf" : "cnf",
               status_str,
               solved ? "true" : "false", feasible ? "true" : "false",
               meta.num_vars, meta.num_clauses, meta.num_literals,
               verification.satisfied, meta.num_clauses - verification.satisfied,
               verification.hard_unsatisfied, verification.soft_unsatisfied,
               verification.soft_cost, verification.total_soft_weight,
               verification.cost_overflow ? "true" : "false",
               exact_attempted ? "true" : "false",
               exact_result == 1 ? "SAT" : (exact_result == 0 ? "UNSAT" :
               (exact_result == CDCL_RESULT_LIMIT ? "LIMIT" : "NOT_RUN")),
               opt.proof_path ? "true" : "false", proof_generated ? "true" : "false",
               exact_attempted ? "false" : "true",
               epochs_run, finisher_passes_run, indexed_flips_run, opt.add_count,
               resident_mb, indexed_index_mb,
               elapsed_ms(&start, &built), elapsed_ms(&built, &finished));
    } else {
        printf("NitroSAT V3 (%s): %s\nVariables: %u  Clauses: %" PRIu64 "  Literals: %" PRIu64
               "\nSatisfied: %" PRIu64 "  Unsatisfied: %" PRIu64
               "\nHard unsatisfied: %" PRIu64 "  Soft cost: %" PRIu64
               "\nBounded solver allocation: %.2f MiB\n",
               (meta.flags & STORE_FLAG_WEIGHTED) ? "WCNF" : "CNF",
               solved ? "all clauses satisfied" :
               (exact_result == 0 ? (weighted_store ? "hard clauses proven infeasible" :
                                                      "proven UNSAT") :
               (weighted_store && feasible ? "weighted feasible assignment" :
                                               "best assignment (not proven UNSAT)")),
               meta.num_vars, meta.num_clauses, meta.num_literals, verification.satisfied,
               meta.num_clauses - verification.satisfied, verification.hard_unsatisfied,
               verification.soft_cost, resident_mb);
    }

    free(importance_x);
    free(best_x);
    reader_close(&reader); solver_free(&solver); cache_free(&cache);
    if (!opt.keep_store) unlink(opt.store_path);
    free(opt.add_files);
    if (!io_ok) { fprintf(stderr, "clause store read failed\n"); return 2; }
    return (weighted_store ? feasible : solved) ? 0 : 1;
}
