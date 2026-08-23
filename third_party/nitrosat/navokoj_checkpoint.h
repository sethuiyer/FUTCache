/* navokoj_checkpoint.h - anytime checkpoint support shared across
 * all Nitro-family C solvers (SUTRA, SUTRA v2, NitroSAT V2, V3).
 *
 * The C solvers own a long-running optimization loop.  The Python
 * adapter uses subprocess.run(..., timeout=...) which sends SIGKILL
 * when the watchdog fires, so the in-memory best_assignment is lost.
 * This module writes the current best to a file every
 * ``interval_seconds``, so the Python adapter can read whatever the
 * solver had found before the kill.
 *
 * Wire-up per binary:
 *   1. Include this header.
 *   2. Add ``NavokojCheckpoint checkpoint;`` to your Options struct.
 *   3. In argv parsing: try navokoj_checkpoint_parse_arg().
 *   4. After parsing: navokoj_checkpoint_open().
 *   5. After each best update: navokoj_checkpoint_update() + _maybe().
 *   6. At exit: navokoj_checkpoint_close() (final flush inside).
 */
#ifndef NAVOKOJ_CHECKPOINT_H
#define NAVOKOJ_CHECKPOINT_H

#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque snapshot of the solver's current best.  The pointers are
 * non-owning; the caller must keep the underlying arrays alive. */
typedef struct {
    uint32_t num_vars;
    const uint8_t *assignment;   /* 1-indexed, length num_vars+1 */
    uint64_t hard_unsatisfied;   /* drives "s STATUS" line */
} NavokojCheckpointState;

typedef struct {
    FILE *fp;
    const char *path;
    double interval_seconds;     /* 0 = disabled (no-op) */
    double last_flush_time;
    NavokojCheckpointState state;
} NavokojCheckpoint;

/* Open the checkpoint file.  ``path`` may be NULL or ``interval_seconds``
 * may be 0 - both are no-ops that return success.  Returns 1 on
 * success, 0 on failure (errno set; message already on stderr). */
int navokoj_checkpoint_open(NavokojCheckpoint *cp,
                            const char *path,
                            double interval_seconds);

/* Update the in-memory snapshot.  Cheap (pointer copies).  Call this
 * AFTER you've updated the best_assignment in your solver state. */
void navokoj_checkpoint_update(NavokojCheckpoint *cp,
                               uint32_t num_vars,
                               const uint8_t *assignment,
                               uint64_t hard_unsatisfied);

/* Flush if ``interval_seconds`` has elapsed since the last flush.
 * No-op if disabled or no snapshot has been set yet. */
void navokoj_checkpoint_maybe(NavokojCheckpoint *cp);

/* Force an immediate flush.  Use at exit so on-disk best matches
 * in-memory best regardless of interval. */
void navokoj_checkpoint_flush(NavokojCheckpoint *cp);

/* Close the file (calls _flush first).  Safe on a zero-initialized
 * or never-opened NavokojCheckpoint. */
void navokoj_checkpoint_close(NavokojCheckpoint *cp);

/* Try to consume one or two argv slots for --checkpoint FILE or
 * --checkpoint-interval-ms N.  Returns the new argv index on match,
 * -1 if argv[i] is neither flag.  Mutates cp->path or
 * cp->interval_seconds accordingly. */
int navokoj_checkpoint_parse_arg(NavokojCheckpoint *cp,
                                 int argc, char **argv, int i);

#ifdef __cplusplus
}
#endif

#endif /* NAVOKOJ_CHECKPOINT_H */
