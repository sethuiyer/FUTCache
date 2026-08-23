# Interval-cache snapshot format, version 1

The packing engine has its own format, documented in
[pack-serialization.md](pack-serialization.md).

All multi-byte fields are little-endian. Floating-point fields are IEEE-754
binary64 bit patterns. Offsets are bytes from the start of the snapshot.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | ASCII magic `FUTCACHE` |
| 8 | 2 | format version (`1`) |
| 10 | 2 | header size (`72`) |
| 12 | 4 | flags; must be zero in version 1 |
| 16 | 8 | epsilon |
| 24 | 8 | domain minimum |
| 32 | 8 | domain maximum |
| 40 | 8 | successful observation count |
| 48 | 8 | novel observation count |
| 56 | 8 | generation |
| 64 | 8 | interval count `n` |
| 72 | `16*n` | `(lower, upper)` interval pairs |
| `72+16*n` | 4 | CRC32 of every preceding byte |

Intervals must be finite, within the domain, sorted by their lower endpoint,
and strictly disjoint. Closed intervals that overlap or touch are canonicalized
into one interval, so a serialized pair must satisfy
`previous.upper < next.lower`.

Readers must reject unknown versions or flags. Version 1 readers also reject
trailing bytes; this prevents ambiguous framing when snapshots are transported
without an outer length prefix.

Telemetry is validated as well: interval count cannot exceed novel-observation
count, novel observations cannot exceed total observations, and generation
cannot be lower than the observation count. These relations hold for every
state produced by the public API, including after counter saturation and clear.
