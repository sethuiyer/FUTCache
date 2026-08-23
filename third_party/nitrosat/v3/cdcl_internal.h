#ifndef CDCL_INTERNAL_H
#define CDCL_INTERNAL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <errno.h>

#define CDCL_PROPAGATE_NONE UINT32_MAX
#define CDCL_PROPAGATE_ERROR (UINT32_MAX - 1u)
#define CDCL_RESULT_ERROR (-1)
#define CDCL_RESULT_LIMIT (-2)
#define CDCL_RESULT_NOT_RUN (-3)

static inline uint32_t cdcl_var(int32_t lit) {
    return lit < 0 ? (uint32_t)(-(int64_t)lit) : (uint32_t)lit;
}
#define CDCL_VAR(lit) cdcl_var((lit))
#define CDCL_SIGN(lit) ((lit) > 0 ? 0 : 1)
#define CDCL_LIT(var, sign) ((sign) ? -(var) : (var))

typedef struct {
    int32_t *lits;
    uint32_t size;
    uint32_t capacity;
    int is_learned;
    uint32_t lbd;       /* Literal Block Distance — lower = higher quality */
} cdcl_clause_t;

typedef struct {
    uint32_t clause_idx;
    int32_t blocked_lit;
} cdcl_watcher_t;

typedef struct {
    cdcl_watcher_t *w;
    uint32_t size;
    uint32_t capacity;
} cdcl_watchlist_t;

typedef struct {
    uint32_t num_vars;
    int8_t *assigns;   /* 0 = unassigned, 1 = true, -1 = false */
    int8_t *phase;     /* saved phase polarity: 1 = positive, -1 = negative */
    uint32_t *reason;  /* clause_idx that implied this variable. UINT32_MAX if decision */
    uint32_t *level;
    double *activity;
    double act_increment;  /* current activity bump value (grows with decay) */

    int32_t *trail;
    uint32_t trail_sz;

    uint32_t *trail_lim;
    uint32_t trail_lim_sz;

    uint32_t qhead;

    cdcl_clause_t *clauses;
    uint32_t clauses_sz;
    uint32_t clauses_cap;
    uint32_t original_clauses;  /* number of original (non-learned) clauses */

    cdcl_watchlist_t *watches; /* size: 2 * num_vars + 2 */

    int ok;
    uint32_t conflicts;
    uint32_t restarts;

    FILE *drat_fp;
    int oom;
    int proof_error;
    int internal_error;
    int limit_hit;
    size_t memory_used;
    size_t memory_limit;
    uint64_t conflict_limit;
    int8_t *clause_marks;

    /* For conflict analysis */
    uint32_t *seen;
    uint32_t *level_seen;
    uint32_t *analyze_toclear;
    uint32_t analyze_toclear_sz;
} cdcl_t;

static void cdcl_free(cdcl_t *s);

static int cdcl_size_mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int cdcl_size_add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static int cdcl_account_growth(cdcl_t *s, size_t old_bytes, size_t new_bytes) {
    if (new_bytes <= old_bytes) return 1;
    size_t growth = new_bytes - old_bytes;
    if (growth > s->memory_limit - s->memory_used) { s->limit_hit = 1; return 0; }
    s->memory_used += growth;
    return 1;
}

/* ── Luby restart sequence ───────────────────────────────────── */
static uint32_t cdcl_luby(uint32_t i) {
    uint32_t k = 1;
    while (k < i + 1) k <<= 1;
    return (i + 1 == k) ? k >> 1 : cdcl_luby(i - (k >> 1) + 1);
}

static inline int cdcl_watch_add(cdcl_t *s, int32_t lit, uint32_t clause_idx, int32_t blocked) {
    uint32_t var = CDCL_VAR(lit);
    uint32_t widx = lit > 0 ? var * 2u : var * 2u + 1u;
    cdcl_watchlist_t *wl = &s->watches[widx];
    if (wl->size == wl->capacity) {
        if (wl->capacity > UINT32_MAX / 2u) { s->limit_hit = 1; return 0; }
        uint32_t new_capacity = wl->capacity ? wl->capacity * 2u : 4u;
        size_t old_bytes, new_bytes;
        if (!cdcl_size_mul(wl->capacity, sizeof(*wl->w), &old_bytes) ||
            !cdcl_size_mul(new_capacity, sizeof(*wl->w), &new_bytes) ||
            !cdcl_account_growth(s, old_bytes, new_bytes)) return 0;
        cdcl_watcher_t *new_w = realloc(wl->w, new_capacity * sizeof(*new_w));
        if (!new_w) { s->memory_used -= new_bytes - old_bytes; s->oom = 1; return 0; }
        wl->capacity = new_capacity;
        wl->w = new_w;
    }
    wl->w[wl->size].clause_idx = clause_idx;
    wl->w[wl->size].blocked_lit = blocked;
    wl->size++;
    return 1;
}

static inline void cdcl_watch_remove(cdcl_t *s, int32_t lit, uint32_t clause_idx) {
    uint32_t var = CDCL_VAR(lit);
    uint32_t widx = lit > 0 ? var * 2u : var * 2u + 1u;
    cdcl_watchlist_t *wl = &s->watches[widx];
    for (uint32_t i = 0; i < wl->size; ++i) {
        if (wl->w[i].clause_idx == clause_idx) {
            wl->w[i] = wl->w[wl->size - 1];
            wl->size--;
            break;
        }
    }
}

static inline int cdcl_value(cdcl_t *s, int32_t lit) {
    int8_t val = s->assigns[CDCL_VAR(lit)];
    if (val == 0) return 0;
    return (lit > 0 && val == 1) || (lit < 0 && val == -1) ? 1 : -1;
}

static void cdcl_enqueue(cdcl_t *s, int32_t lit, uint32_t reason) {
    if (cdcl_value(s, lit) != 0) return;
    s->assigns[CDCL_VAR(lit)] = lit > 0 ? 1 : -1;
    s->level[CDCL_VAR(lit)] = s->trail_lim_sz;
    s->reason[CDCL_VAR(lit)] = reason;
    s->trail[s->trail_sz++] = lit;
}

static cdcl_t* cdcl_new(uint32_t num_vars, size_t memory_limit, uint64_t conflict_limit) {
    if (num_vars > INT32_MAX || memory_limit == 0) { errno = E2BIG; return NULL; }
    cdcl_t *s = calloc(1, sizeof(cdcl_t));
    if (!s) { errno = ENOMEM; return NULL; }
    s->num_vars = num_vars;
    s->memory_limit = memory_limit;
    s->conflict_limit = conflict_limit;
    size_t vars = (size_t)num_vars + 1u;
    size_t watch_count;
    if (!cdcl_size_mul((size_t)num_vars, 2u, &watch_count) ||
        !cdcl_size_add(watch_count, 2u, &watch_count)) { free(s); errno = E2BIG; return NULL; }
    size_t base = sizeof(*s), bytes;
#define CDCL_BASE_ARRAY(count, type) \
    do { if (!cdcl_size_mul((count), sizeof(type), &bytes) || !cdcl_size_add(base, bytes, &base)) { free(s); errno = E2BIG; return NULL; } } while (0)
    CDCL_BASE_ARRAY(vars, int8_t); CDCL_BASE_ARRAY(vars, int8_t);
    CDCL_BASE_ARRAY(vars, uint32_t); CDCL_BASE_ARRAY(vars, uint32_t);
    CDCL_BASE_ARRAY(vars, double); CDCL_BASE_ARRAY(vars, int32_t);
    CDCL_BASE_ARRAY(vars, uint32_t); CDCL_BASE_ARRAY(watch_count, cdcl_watchlist_t);
    CDCL_BASE_ARRAY(vars, uint32_t); CDCL_BASE_ARRAY(vars, uint32_t);
    CDCL_BASE_ARRAY(vars, uint32_t); CDCL_BASE_ARRAY(vars, int8_t);
#undef CDCL_BASE_ARRAY
    if (base > memory_limit) { free(s); errno = E2BIG; return NULL; }
    s->memory_used = base;
    s->assigns = calloc(vars, sizeof(int8_t));
    s->phase = calloc(vars, sizeof(int8_t));
    s->reason = calloc(vars, sizeof(uint32_t));
    s->level = calloc(vars, sizeof(uint32_t));
    s->activity = calloc(vars, sizeof(double));
    s->trail = calloc(vars, sizeof(int32_t));
    s->trail_lim = calloc(vars, sizeof(uint32_t));
    s->watches = calloc(watch_count, sizeof(cdcl_watchlist_t));
    s->seen = calloc(vars, sizeof(uint32_t));
    s->level_seen = calloc(vars, sizeof(uint32_t));
    s->analyze_toclear = calloc(vars, sizeof(uint32_t));
    s->clause_marks = calloc(vars, sizeof(int8_t));
    if (!s->assigns || !s->phase || !s->reason || !s->level || !s->activity ||
        !s->trail || !s->trail_lim || !s->watches || !s->seen || !s->level_seen ||
        !s->analyze_toclear || !s->clause_marks) {
        cdcl_free(s);
        errno = ENOMEM;
        return NULL;
    }
    s->act_increment = 1.0;
    s->ok = 1;
    /* Default phase: positive */
    for (uint32_t i = 1; i <= num_vars; ++i) s->phase[i] = 1;
    return s;
}

static void cdcl_free(cdcl_t *s) {
    if (!s) return;
    for (uint32_t i = 0; i < s->clauses_sz; ++i) {
        free(s->clauses[i].lits);
    }
    free(s->clauses);
    size_t watch_count = (size_t)s->num_vars * 2u + 2u;
    for (size_t i = 0; i < watch_count; ++i) {
        free(s->watches[i].w);
    }
    free(s->watches);
    free(s->assigns); free(s->phase); free(s->reason); free(s->level); free(s->activity);
    free(s->trail); free(s->trail_lim); free(s->seen); free(s->level_seen); free(s->analyze_toclear);
    free(s->clause_marks);
    free(s);
}

static int cdcl_emit_clause(cdcl_t *s, const int32_t *lits, uint32_t size) {
    if (!s->drat_fp) return 1;
    for (uint32_t i = 0; i < size; ++i)
        if (fprintf(s->drat_fp, "%d ", lits[i]) < 0) { s->proof_error = 1; return 0; }
    if (fprintf(s->drat_fp, "0\n") < 0) { s->proof_error = 1; return 0; }
    return 1;
}

static uint32_t cdcl_add_clause_inner(cdcl_t *s, int32_t *lits, uint32_t size, int is_learned) {
    if (s->clauses_sz == s->clauses_cap) {
        if (s->clauses_cap > (UINT32_MAX - 2u) / 2u) { s->limit_hit = 1; return UINT32_MAX; }
        uint32_t new_capacity = s->clauses_cap ? s->clauses_cap * 2u : 16u;
        size_t old_bytes, new_bytes;
        if (!cdcl_size_mul(s->clauses_cap, sizeof(*s->clauses), &old_bytes) ||
            !cdcl_size_mul(new_capacity, sizeof(*s->clauses), &new_bytes) ||
            !cdcl_account_growth(s, old_bytes, new_bytes)) return UINT32_MAX;
        cdcl_clause_t *new_clauses = realloc(s->clauses, new_capacity * sizeof(*new_clauses));
        if (!new_clauses) { s->memory_used -= new_bytes - old_bytes; s->oom = 1; return UINT32_MAX; }
        s->clauses_cap = new_capacity;
        s->clauses = new_clauses;
    }
    uint32_t idx = s->clauses_sz;
    size_t lit_bytes;
    if (!cdcl_size_mul(size, sizeof(int32_t), &lit_bytes) ||
        !cdcl_account_growth(s, 0, lit_bytes)) return UINT32_MAX;
    s->clauses[idx].lits = malloc(lit_bytes);
    if (!s->clauses[idx].lits) { s->memory_used -= lit_bytes; s->oom = 1; return UINT32_MAX; }
    memcpy(s->clauses[idx].lits, lits, size * sizeof(int32_t));
    s->clauses[idx].size = size;
    s->clauses[idx].is_learned = is_learned;
    s->clauses[idx].lbd = size; /* default LBD = clause size, refined during analysis */

    if (size > 1) {
        if (!cdcl_watch_add(s, -lits[0], idx, lits[1]) ||
            !cdcl_watch_add(s, -lits[1], idx, lits[0])) {
            free(s->clauses[idx].lits);
            s->memory_used -= lit_bytes;
            s->clauses[idx].lits = NULL;
            return UINT32_MAX;
        }
    }
    s->clauses_sz++;
    if (is_learned && !cdcl_emit_clause(s, lits, size)) return UINT32_MAX;
    return idx;
}

static int cdcl_add_clause(cdcl_t *s, int32_t *lits, uint32_t size) {
    if (!s->ok) return 0;

    /* Normalize duplicates and tautologies before assignment simplification. */
    uint32_t j = 0;
    for (uint32_t i = 0; i < size; ++i) {
        uint32_t var = (uint32_t)CDCL_VAR(lits[i]);
        if (var == 0 || var > s->num_vars) { s->ok = 0; return 0; }
        int8_t sign = lits[i] > 0 ? 1 : -1;
        if (s->clause_marks[var] == -sign) {
            for (uint32_t k = 0; k < j; ++k) s->clause_marks[CDCL_VAR(lits[k])] = 0;
            return 1; /* tautology */
        }
        if (s->clause_marks[var] == sign) continue;
        s->clause_marks[var] = sign;
        lits[j++] = lits[i];
    }
    for (uint32_t i = 0; i < j; ++i) s->clause_marks[CDCL_VAR(lits[i])] = 0;
    size = j;
    j = 0;
    for (uint32_t i = 0; i < size; ++i) {
        if (cdcl_value(s, lits[i]) == 1) return 1; /* clause already satisfied */
        if (cdcl_value(s, lits[i]) == -1) continue; /* literal already false */
        lits[j++] = lits[i];
    }
    size = j;

    if (size == 0) {
        s->ok = 0;
        cdcl_emit_clause(s, NULL, 0);
        return 0;
    }
    if (size == 1) {
        cdcl_enqueue(s, lits[0], UINT32_MAX);
        return 1;
    }

    if (cdcl_add_clause_inner(s, lits, size, 0) == UINT32_MAX) return 0;
    s->original_clauses++;
    return 1;
}

static uint32_t cdcl_propagate(cdcl_t *s) {
    uint32_t confl = CDCL_PROPAGATE_NONE;
    while (s->qhead < s->trail_sz) {
        int32_t p = s->trail[s->qhead++];
        uint32_t var = CDCL_VAR(p);
        uint32_t widx = p > 0 ? var * 2u : var * 2u + 1u; /* watch list for falsified literal */
        cdcl_watchlist_t *wl = &s->watches[widx];

        uint32_t i = 0, j = 0;
        while (i < wl->size) {
            cdcl_watcher_t w = wl->w[i];
            uint32_t cr = w.clause_idx;
            int32_t blocked = w.blocked_lit;

            if (cdcl_value(s, blocked) == 1) {
                wl->w[j++] = wl->w[i++];
                continue;
            }

            cdcl_clause_t *c = &s->clauses[cr];
            int32_t false_lit = -p;
            if (c->lits[0] == false_lit) {
                c->lits[0] = c->lits[1];
                c->lits[1] = false_lit;
            }

            int32_t first = c->lits[0];
            if (cdcl_value(s, first) == 1) {
                wl->w[j++] = wl->w[i++];
                wl->w[j-1].blocked_lit = first;
                continue;
            }

            int found_new_watch = 0;
            for (uint32_t k = 2; k < c->size; ++k) {
                if (cdcl_value(s, c->lits[k]) != -1) {
                    c->lits[1] = c->lits[k];
                    c->lits[k] = false_lit;
                    if (!cdcl_watch_add(s, -c->lits[1], cr, first)) return CDCL_PROPAGATE_ERROR;
                    found_new_watch = 1;
                    break;
                }
            }

            if (found_new_watch) {
                i++;
            } else {
                wl->w[j++] = wl->w[i++];
                if (cdcl_value(s, first) == -1) {
                    confl = cr;
                    s->qhead = s->trail_sz;
                    while (i < wl->size) wl->w[j++] = wl->w[i++];
                } else {
                    cdcl_enqueue(s, first, cr);
                }
            }
        }
        wl->size = j;
    }
    return confl;
}

static void cdcl_cancel_until(cdcl_t *s, uint32_t level) {
    if (s->trail_lim_sz > level) {
        for (uint32_t c = s->trail_sz; c > s->trail_lim[level]; --c) {
            int32_t x = s->trail[c - 1];
            uint32_t var = CDCL_VAR(x);
            s->phase[var] = s->assigns[var]; /* save phase before unassigning */
            s->assigns[var] = 0;
        }
        s->trail_sz = s->trail_lim[level];
        s->trail_lim_sz = level;
        s->qhead = s->trail_sz;
    }
}

/* ── Activity bump with multiplicative decay ─────────────────── */
static void cdcl_bump_activity(cdcl_t *s, uint32_t var) {
    s->activity[var] += s->act_increment;
    /* Rescale all activities if overflow threatens */
    if (s->activity[var] > 1e100) {
        for (uint32_t i = 1; i <= s->num_vars; ++i)
            s->activity[i] *= 1e-100;
        s->act_increment *= 1e-100;
    }
}

static void cdcl_decay_activity(cdcl_t *s) {
    s->act_increment *= (1.0 / 0.95); /* effectively multiplies all future bumps */
}

static int cdcl_analyze(cdcl_t *s, uint32_t confl, int32_t *out_learnt,
                         uint32_t *out_sz, uint32_t *out_btlevel, uint32_t *out_lbd) {
    uint32_t path_c = 0;
    int32_t p = 0;
    uint32_t idx = s->trail_sz - 1;
    *out_sz = 1; /* leave room for asserting literal */

    do {
        cdcl_clause_t *c = &s->clauses[confl];
        for (uint32_t j = (p == 0 ? 0 : 1); j < c->size; ++j) {
            int32_t q = c->lits[j];
            uint32_t var = CDCL_VAR(q);
            if (!s->seen[var] && s->level[var] > 0) {
                s->seen[var] = 1;
                s->analyze_toclear[s->analyze_toclear_sz++] = var;
                cdcl_bump_activity(s, var);
                if (s->level[var] >= s->trail_lim_sz) {
                    path_c++;
                } else {
                    out_learnt[(*out_sz)++] = q;
                }
            }
        }

        while (idx < s->trail_sz && !s->seen[CDCL_VAR(s->trail[idx])]) {
            idx--;
        }
        if (idx >= s->trail_sz) {
            s->internal_error = 1;
            return 0;
        }
        p = s->trail[idx];
        idx--;
        confl = s->reason[CDCL_VAR(p)];
        s->seen[CDCL_VAR(p)] = 0;
        path_c--;
    } while (path_c > 0);

    out_learnt[0] = -p;

    /* Compute LBD: count distinct decision levels in the learned clause */
    uint32_t lbd = 0;
    for (uint32_t i = 0; i < *out_sz; ++i) {
        uint32_t lev = s->level[CDCL_VAR(out_learnt[i])];
        if (lev > 0 && !s->level_seen[lev]) {
            s->level_seen[lev] = 1;
            lbd++;
        }
    }
    /* Clean up LBD marks */
    for (uint32_t i = 0; i < *out_sz; ++i) {
        uint32_t lev = s->level[CDCL_VAR(out_learnt[i])];
        s->level_seen[lev] = 0;
    }
    *out_lbd = lbd;

    if (*out_sz == 1) {
        *out_btlevel = 0;
    } else {
        uint32_t max_i = 1;
        for (uint32_t i = 2; i < *out_sz; ++i) {
            if (s->level[CDCL_VAR(out_learnt[i])] > s->level[CDCL_VAR(out_learnt[max_i])]) {
                max_i = i;
            }
        }
        int32_t tmp = out_learnt[1];
        out_learnt[1] = out_learnt[max_i];
        out_learnt[max_i] = tmp;
        *out_btlevel = s->level[CDCL_VAR(out_learnt[1])];
    }

    for (uint32_t j = 0; j < s->analyze_toclear_sz; ++j) {
        s->seen[s->analyze_toclear[j]] = 0;
    }
    s->analyze_toclear_sz = 0;
    return 1;
}

/* ── Clause database reduction ───────────────────────────────── */
static void cdcl_reduce_db(cdcl_t *s) {
    /* Remove learned clauses with LBD > 6 that are not currently locked (reason for propagation).
       Keep all original clauses and "glue" clauses (LBD <= 2). */
    uint32_t keep = 0;
    for (uint32_t i = s->original_clauses; i < s->clauses_sz; ++i) {
        cdcl_clause_t *c = &s->clauses[i];
        if (!c->is_learned || c->lbd <= 2) continue;
        if (c->lbd > 6 && c->size > 2) {
            /* Check if this clause is a current reason for any assigned variable */
            int locked = 0;
            for (uint32_t j = 0; j < c->size; ++j) {
                uint32_t var = CDCL_VAR(c->lits[j]);
                if (s->assigns[var] != 0 && s->reason[var] == i) {
                    locked = 1;
                    break;
                }
            }
            if (!locked) {
                /* Remove watches */
                if (c->size > 1) {
                    cdcl_watch_remove(s, -c->lits[0], i);
                    cdcl_watch_remove(s, -c->lits[1], i);
                }
                free(c->lits);
                s->memory_used -= (size_t)c->size * sizeof(*c->lits);
                c->lits = NULL;
                c->size = 0;
                c->is_learned = 0;
                keep++;
            }
        }
    }
    (void)keep;
}

static int32_t cdcl_pick_branch_lit(cdcl_t *s) {
    int32_t best_var = 0;
    double best_act = -1.0;
    for (uint32_t i = 1; i <= s->num_vars; ++i) {
        if (s->assigns[i] == 0) {
            if (s->activity[i] > best_act) {
                best_act = s->activity[i];
                best_var = (int32_t)i;
            }
        }
    }
    if (best_var == 0) return 0;
    /* Phase-informed branching: use saved phase polarity */
    return (s->phase[best_var] > 0) ? best_var : -best_var;
}

static int cdcl_solve(cdcl_t *s) {
    if (s->oom || s->proof_error || s->internal_error) return CDCL_RESULT_ERROR;
    if (s->limit_hit) return CDCL_RESULT_LIMIT;
    if (!s->ok) return 0; /* UNSAT immediately */

    /* Initial propagate */
    uint32_t initial = cdcl_propagate(s);
    if (initial == CDCL_PROPAGATE_ERROR) return s->limit_hit ? CDCL_RESULT_LIMIT : CDCL_RESULT_ERROR;
    if (initial != CDCL_PROPAGATE_NONE) {
        if (!cdcl_emit_clause(s, NULL, 0)) return -1;
        return 0; /* UNSAT */
    }

    size_t learnt_bytes = ((size_t)s->num_vars + 1u) * sizeof(int32_t);
    if (!cdcl_account_growth(s, 0, learnt_bytes)) return CDCL_RESULT_LIMIT;
    int32_t *learnt_clause = malloc(learnt_bytes);
    if (!learnt_clause) { s->memory_used -= learnt_bytes; s->oom = 1; return CDCL_RESULT_ERROR; }
    uint32_t conflicts_since_restart = 0;
    uint32_t restart_limit = 100; /* Luby base unit */
    uint32_t restart_seq = 1;
    uint32_t reduce_interval = 2000;

    while (1) {
        uint32_t confl = cdcl_propagate(s);
        if (confl == CDCL_PROPAGATE_ERROR) {
            free(learnt_clause);
            return s->limit_hit ? CDCL_RESULT_LIMIT : CDCL_RESULT_ERROR;
        }
        if (confl != CDCL_PROPAGATE_NONE) {
            if (s->conflict_limit && s->conflicts >= s->conflict_limit) {
                free(learnt_clause);
                return CDCL_RESULT_LIMIT;
            }
            s->conflicts++;
            conflicts_since_restart++;
            if (s->trail_lim_sz == 0) {
                if (!cdcl_emit_clause(s, NULL, 0)) { free(learnt_clause); return -1; }
                free(learnt_clause);
                return 0; /* UNSAT */
            }
            uint32_t btlevel = 0;
            uint32_t learnt_sz = 0;
            uint32_t lbd = 0;
            if (!cdcl_analyze(s, confl, learnt_clause, &learnt_sz, &btlevel, &lbd)) {
                free(learnt_clause);
                return CDCL_RESULT_ERROR;
            }
            cdcl_cancel_until(s, btlevel);

            if (learnt_sz == 1) {
                if (!cdcl_emit_clause(s, learnt_clause, 1)) { free(learnt_clause); return -1; }
                cdcl_enqueue(s, learnt_clause[0], UINT32_MAX);
            } else {
                uint32_t cr = cdcl_add_clause_inner(s, learnt_clause, learnt_sz, 1);
                if (cr == UINT32_MAX) { free(learnt_clause); return -1; }
                s->clauses[cr].lbd = lbd;
                cdcl_enqueue(s, learnt_clause[0], cr);
            }

            cdcl_decay_activity(s);

            /* Periodic clause database reduction */
            if (s->conflicts % reduce_interval == 0) {
                cdcl_cancel_until(s, 0);
                cdcl_reduce_db(s);
            }
        } else {
            /* ── Restart check ──────────────────────────────────── */
            if (conflicts_since_restart >= restart_limit * cdcl_luby(restart_seq)) {
                cdcl_cancel_until(s, 0);
                restart_seq++;
                conflicts_since_restart = 0;
                s->restarts++;
            }

            int32_t next = cdcl_pick_branch_lit(s);
            if (next == 0) {
                free(learnt_clause);
                return 1; /* SAT */
            }
            s->trail_lim[s->trail_lim_sz++] = s->trail_sz;
            cdcl_enqueue(s, next, UINT32_MAX);
        }
    }
}

#endif /* CDCL_INTERNAL_H */
