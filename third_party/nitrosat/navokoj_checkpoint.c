/* navokoj_checkpoint.c - implementation.  See header for contract. */
#define _POSIX_C_SOURCE 200809L

#include "navokoj_checkpoint.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Atomic-best-effort rewrite.
 *
 * We flush, ftruncate, rewind, then write the new content and flush
 * again.  POSIX does not guarantee reader atomicity for arbitrary
 * writes, but for local filesystems (which is all we target: the
 * Python adapter passes a /dev/shm path) ftruncate+fwrite is observed
 * atomically by a concurrent reader - they see either the previous
 * complete checkpoint or the new complete checkpoint, never a torn
 * half-write.  The Python reader also opens with O_RDONLY and reads
 * to EOF in one shot, so torn writes would be visible as truncated
 * tokens, which the parser tolerates. */
static void rewrite_checkpoint(const NavokojCheckpoint *cp) {
    const NavokojCheckpointState *s = &cp->state;
    if (!s->assignment || s->num_vars == 0) return;
    FILE *fp = cp->fp;
    fflush(fp);
    if (ftruncate(fileno(fp), (off_t)0) != 0) return;
    rewind(fp);
    fprintf(fp, "s %s\n",
            s->hard_unsatisfied == 0 ? "SATISFIABLE" : "PARTIAL");
    fputs("v", fp);
    for (uint32_t v = 1; v <= s->num_vars; ++v) {
        fprintf(fp, " %d", s->assignment[v] ? (int)v : -(int)v);
    }
    fputs(" 0\n", fp);
    fflush(fp);
}

int navokoj_checkpoint_open(NavokojCheckpoint *cp, const char *path,
                            double interval_seconds) {
    memset(cp, 0, sizeof(*cp));
    cp->interval_seconds = interval_seconds;
    if (!path || interval_seconds <= 0.0) {
        return 1;  /* disabled - no file, no I/O */
    }
    cp->fp = fopen(path, "wb+");
    if (!cp->fp) {
        fprintf(stderr, "navokoj: cannot open checkpoint %s: %s\n",
                path, strerror(errno));
        return 0;
    }
    cp->path = path;
    cp->last_flush_time = monotonic_seconds();
    setvbuf(cp->fp, NULL, _IOLBF, 0);  /* line-buffered so fflush is cheap */
    return 1;
}

void navokoj_checkpoint_update(NavokojCheckpoint *cp, uint32_t num_vars,
                               const uint8_t *assignment,
                               uint64_t hard_unsatisfied) {
    cp->state.num_vars = num_vars;
    cp->state.assignment = assignment;
    cp->state.hard_unsatisfied = hard_unsatisfied;
}

void navokoj_checkpoint_flush(NavokojCheckpoint *cp) {
    if (!cp->fp) return;
    rewrite_checkpoint(cp);
    cp->last_flush_time = monotonic_seconds();
}

void navokoj_checkpoint_maybe(NavokojCheckpoint *cp) {
    if (!cp->fp || cp->interval_seconds <= 0.0) return;
    double now = monotonic_seconds();
    if (now - cp->last_flush_time < cp->interval_seconds) return;
    rewrite_checkpoint(cp);
    cp->last_flush_time = now;
}

void navokoj_checkpoint_close(NavokojCheckpoint *cp) {
    if (!cp->fp) return;
    /* Final flush so the on-disk best matches the in-memory best
     * regardless of why we're exiting. */
    rewrite_checkpoint(cp);
    fclose(cp->fp);
    cp->fp = NULL;
}

int navokoj_checkpoint_parse_arg(NavokojCheckpoint *cp, int argc,
                                 char **argv, int i) {
    if (i + 1 >= argc) return -1;
    if (strcmp(argv[i], "--checkpoint") == 0) {
        cp->path = argv[++i];
        return i;
    }
    if (strcmp(argv[i], "--checkpoint-interval-ms") == 0) {
        double ms = atof(argv[++i]);
        cp->interval_seconds = ms < 0.0 ? 0.0 : ms / 1000.0;
        return i;
    }
    return -1;
}
