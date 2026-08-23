# Packing-cache snapshot format, version 2

`futcache_pack_serialize` emits an atomic snapshot of the packing engine and
`futcache_pack_deserialize` restores it. All multi-byte fields are
little-endian; floating-point values are IEEE-754 binary64 bit patterns.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | magic: `FUTPACK\0` |
| 8 | 2 | format version (`2`) |
| 10 | 2 | header size (`104`) |
| 12 | 4 | flags; bit 0 means per-representative radii are present |
| 16 | 8 | default epsilon |
| 24 | 8 | dimension `d` |
| 32 | 8 | successful observations |
| 40 | 8 | novel observations |
| 48 | 8 | generation |
| 56 | 8 | live representative count `n` |
| 64 | 8 | peak representative count |
| 72 | 8 | FIFO pressure evictions |
| 80 | 8 | hard memory limit; zero means unlimited |
| 88 | 4 | metric id: 1=L_inf, 2=L1, 3=L2, 4=cosine, 5=Poincare |
| 92 | 4 | backend id: 0=linear, 1=built-in VP-tree |
| 96 | 8 | peak live allocation bytes |
| 104 | `8*d` | domain minima |
| `104+8*d` | `8*d` | domain maxima |
| `104+16*d` | `8*(d+1)*n` | each representative's radius followed by its `d` coordinates, in FIFO/slot order |
| final 4 bytes | 4 | CRC32 of every preceding byte |

The version 2 writer always sets bit 0 and emits every acceptance radius.
Readers remain backward-compatible with version 1 (`flags=0`), whose
representatives contain coordinates only; every restored v1 representative
receives the snapshot's fixed epsilon.

Readers reject unknown versions, flags, metrics, or backends; size arithmetic
overflow; trailing or truncated bytes; checksum mismatches; invalid bounds;
non-finite or out-of-domain representatives; impossible counters; memory
telemetry above the configured ceiling; and representatives that violate the
insertion-order packing invariant (each later centre must lie strictly outside
every earlier centre's stored radius).

Only built-in metrics are portable enough to identify in a snapshot. A cache
using a custom distance callback returns
`FUTCACHE_ERROR_UNSUPPORTED_PLATFORM` from serialization because a process
function pointer and its context cannot be recovered safely. A custom nearest-
neighbour backend is derived state rather than decision state, so its snapshot
restores with the exact linear scan. The built-in VP-tree is identified and
rebuilt from representative coordinates.

The CRC detects torn or corrupted snapshot bytes, but atomic replacement of a
disk file remains the application's responsibility: write the complete bytes
to a temporary file in the destination directory, flush it, then rename it
over the prior checkpoint. Keeping the previous successfully written snapshot
provides deterministic restart after a process or machine crash.
