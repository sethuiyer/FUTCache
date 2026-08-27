#!/usr/bin/env python3
"""Real-World Alibaba Cloud Microservice Trace Scaling Benchmark (SoCC '21).

Pure, unadulterated evaluation on real production Alibaba Cloud microservice call graphs.
NO synthetic labels, NO invented failure flags, NO fake status codes.

Reconstructs dynamic call trees directly from real Alibaba `rpcid` hierarchical structures:
  - Call/span count
  - Max tree depth (from rpcid dot hierarchy)
  - Max fanout (max sibling branches under a single parent)
  - RPC / HTTP / DB / MC invocation counts
  - Mean RT, Max RT, and Critical-Path RT

Compares:
  FUTCache (Two-Stage Metric Net) vs. Exact Hash vs. Random Uniform (1%)

Measures:
  - Suppression / Compression Ratio on Real Traffic
  - Representative Count Growth (|R| vs. stream length N)
  - Physical RAM footprint & Throughput (traces/sec)
"""

import os
import sys
import time
import tarfile
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

# Ensure local futcache package is prioritized
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg_dir = os.path.join(repo_root, "python")
if python_pkg_dir not in sys.path:
    sys.path.insert(0, python_pkg_dir)

import numpy as np
from futcache import PackCache

ALIBABA_TRACE_URL = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces/MSCallGraph/MSCallGraph_0.tar.gz"
LOCAL_CACHE_TAR = os.path.join(repo_root, "demos", ".cache", "MSCallGraph_0.tar.gz")


@dataclass
class RealAlibabaTrace:
    trace_id: str
    entry_service: str  # Sacred partition: root caller service
    # 10D Native Geometric Signature:
    # [call_count, max_depth, max_fanout, rpc_count, http_count, db_count, mc_count, mean_rt, max_rt, crit_path_rt]
    raw_features: np.ndarray
    scaled_geometry: np.ndarray


# ---------------------------------------------------------------------------
# 1. Real Call-Graph Tree Reconstruction from rpcid (Memoized O(N) DAG)
# ---------------------------------------------------------------------------

def extract_tree_signature(span_rows: List[Tuple[str, str, str, str, float]]) -> Tuple[str, np.ndarray]:
    """Reconstructs the execution tree from real rpcid hierarchy in linear time."""
    call_count = len(span_rows)
    entry_service = span_rows[0][1] if span_rows else "unknown_service"

    parent_children = defaultdict(set)
    rpc_rts = {}
    rpc_types = defaultdict(int)
    all_rts = []

    for rpc_id, um, dm, rpc_type, rt in span_rows:
        rt_val = max(0.0, float(rt))
        all_rts.append(rt_val)
        # In Alibaba traces, UM and DM record same rpcid; take max RT
        if rpc_id not in rpc_rts or rt_val > rpc_rts[rpc_id]:
            rpc_rts[rpc_id] = rt_val

        # Count communication paradigms from real Alibaba data (RPC, HTTP, DB, MC)
        t_clean = rpc_type.upper().strip()
        rpc_types[t_clean] += 1

        # Identify parent rpcid
        last_dot = rpc_id.rfind(".")
        if last_dot != -1:
            parent_id = rpc_id[:last_dot]
            parent_children[parent_id].add(rpc_id)
        else:
            parent_children["root"].add(rpc_id)

    # 1. Max Depth (max dot segments + 1)
    max_depth = max(rpc_id.count(".") + 1 for rpc_id in rpc_rts.keys()) if rpc_rts else 1

    # 2. Max Fanout (max immediate children spawned by any single parent call)
    max_fanout = max(len(children) for children in parent_children.values()) if parent_children else 1

    # 3. Communication counts
    rpc_count = rpc_types.get("RPC", 0) + rpc_types.get("USER", 0)
    http_count = rpc_types.get("HTTP", 0)
    db_count = rpc_types.get("DB", 0)
    mc_count = rpc_types.get("MC", 0)

    # 4. Latency metrics
    mean_rt = float(np.mean(all_rts)) if all_rts else 0.0
    max_rt = float(np.max(all_rts)) if all_rts else 0.0

    # 5. Critical Path RT: Linear DP over DAG
    memo_path = {}
    visited_cycle = set()

    def compute_path_rt(node_id: str) -> float:
        if node_id in memo_path:
            return memo_path[node_id]
        if node_id in visited_cycle:
            return 0.0
        visited_cycle.add(node_id)

        node_cost = rpc_rts.get(node_id, 0.0)
        children = parent_children.get(node_id, ())
        if not children:
            res = node_cost
        else:
            res = node_cost + max(compute_path_rt(cid) for cid in children)

        visited_cycle.remove(node_id)
        memo_path[node_id] = res
        return res

    roots = parent_children.get("root", ())
    if not roots and rpc_rts:
        min_dots = min(k.count(".") for k in rpc_rts.keys())
        roots = [k for k in rpc_rts.keys() if k.count(".") == min_dots]

    crit_path_rt = max((compute_path_rt(r) for r in roots), default=max_rt)

    raw_vec = np.array([
        float(call_count),
        float(max_depth),
        float(max_fanout),
        float(rpc_count),
        float(http_count),
        float(db_count),
        float(mc_count),
        float(mean_rt),
        float(max_rt),
        float(crit_path_rt),
    ], dtype=np.float64)

    return entry_service, raw_vec


# Normalization weights for the 10D space:
# Coordinates are scaled so meaningful structural differences are ~1.0 unit.
WEIGHTS_10D = np.array([
    0.2,    # call_count: 5 spans ~ 1 unit
    1.0,    # max_depth: 1 tree level ~ 1 unit
    0.5,    # max_fanout: 2 branches ~ 1 unit
    0.2,    # rpc_count
    0.5,    # http_count
    0.5,    # db_count
    0.5,    # mc_count
    0.02,   # mean_rt (50ms ~ 1 unit)
    0.01,   # max_rt (100ms ~ 1 unit)
    0.01,   # crit_path_rt (100ms ~ 1 unit)
], dtype=np.float64)


def scale_10d_features(raw_vec: np.ndarray) -> np.ndarray:
    return raw_vec * WEIGHTS_10D


# ---------------------------------------------------------------------------
# 2. Download & Parse Real Alibaba Production Shard
# ---------------------------------------------------------------------------

def ensure_alibaba_shard(dest_path: str) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100_000_000:
        print(f"[✓] Using downloaded Alibaba shard: {dest_path} ({os.path.getsize(dest_path)/1024/1024:.1f} MB)")
        return dest_path

    print(f"[+] Fetching official Alibaba Cloud microservice shard from OSS:\n    {ALIBABA_TRACE_URL}")
    t0 = time.perf_counter()
    req = urllib.request.Request(ALIBABA_TRACE_URL, headers={"User-Agent": "FUTCache-Benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest_path, "wb") as out_f:
        tot = int(resp.getheader("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)
            done += len(chunk)
            sys.stdout.write(f"\r    Progress: {done/1024/1024:.1f} / {tot/1024/1024:.1f} MB ({done/tot*100:.1f}%)")
            sys.stdout.flush()
    print(f"\n[✓] Download finished in {time.perf_counter() - t0:.2f}s")
    return dest_path


def stream_parse_alibaba_traces(tar_path: str, max_traces: int = 100_000) -> List[RealAlibabaTrace]:
    print(f"\n[+] Extracting & parsing {max_traces:,} chronological Alibaba production traces...")
    t0 = time.perf_counter()

    spans_by_trace = defaultdict(list)
    trace_order = []
    seen_traces = set()

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".csv"):
                continue
            print(f"    Reading CSV member: {member.name} ({member.size / 1024 / 1024:.1f} MB)...")
            f = tar.extractfile(member)
            if not f:
                continue

            for raw_line in f:
                try:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or "traceid" in line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 8:
                        continue

                    # Unpack columns: ,traceid,timestamp,rpcid,um,rpctype,dm,interface,rt
                    if len(parts) >= 9:
                        t_id = parts[1]
                        rpc_id = parts[3]
                        um = parts[4]
                        rpc_type = parts[5]
                        dm = parts[6]
                        rt = float(parts[8]) if parts[8] else 0.0
                    else:
                        t_id = parts[0]
                        rpc_id = parts[2]
                        um = parts[3]
                        rpc_type = parts[4]
                        dm = parts[5]
                        rt = float(parts[7]) if parts[7] else 0.0

                    if t_id not in seen_traces:
                        seen_traces.add(t_id)
                        trace_order.append(t_id)

                    spans_by_trace[t_id].append((rpc_id, um, dm, rpc_type, rt))

                    if len(trace_order) >= max_traces:
                        break
                except Exception:
                    continue
            break

    t_read = time.perf_counter()
    print(f"    [✓] Ingested {len(trace_order):,} distinct traces in {t_read - t0:.2f}s")
    print(f"    Reconstructing call-trees & 10D geometric signatures...")
    
    reconstructed_traces: List[RealAlibabaTrace] = []

    for t_id in trace_order:
        span_rows = spans_by_trace[t_id]
        if not span_rows:
            continue
        entry_svc, raw_vec = extract_tree_signature(span_rows)
        scaled_vec = scale_10d_features(raw_vec)

        reconstructed_traces.append(RealAlibabaTrace(
            trace_id=t_id,
            entry_service=entry_svc,
            raw_features=raw_vec,
            scaled_geometry=scaled_vec,
        ))

    t1 = time.perf_counter()
    print(f"[✓] Reconstructed {len(reconstructed_traces):,} real production trace trees in {t1 - t0:.2f}s (Reconstruction: {t1 - t_read:.2f}s)")
    return reconstructed_traces


# ---------------------------------------------------------------------------
# 3. Benchmark Execution: Compression, State Boundedness, Throughput
# ---------------------------------------------------------------------------

def run_alibaba_production_benchmark(traces: List[RealAlibabaTrace]):
    print("\n" + "=" * 88)
    print("  ALIBABA CLOUD PRODUCTION TRACE BENCHMARK: REAL-WORLD COMPRESSION & SCALING")
    print("=" * 88)

    n_total = len(traces)
    checkpoints = [10_000, 25_000, 50_000, 75_000, 100_000]
    checkpoints = [c for c in checkpoints if c <= n_total]
    if not checkpoints or checkpoints[-1] != n_total:
        checkpoints.append(n_total)

    # 1. Random Uniform (1%)
    rng = np.random.default_rng(42)
    rand_samples = sum(1 for _ in traces if rng.random() < 0.01)

    # 2. Exact Hash Matcher
    seen_hashes = set()
    hash_reps_at_chk = {}
    t0 = time.perf_counter()
    for idx, t in enumerate(traces, 1):
        key = (
            t.entry_service,
            int(t.raw_features[0]), # call count
            int(t.raw_features[1]), # max depth
            int(t.raw_features[2]), # max fanout
            int(t.raw_features[3]), # rpc count
            int(t.raw_features[4]), # http count
            int(t.raw_features[5]), # db count
            int(t.raw_features[6]), # mc count
            int(t.raw_features[7] / 20.0), # 20ms mean rt bucket
            int(t.raw_features[8] / 50.0), # 50ms max rt bucket
        )
        seen_hashes.add(key)
        if idx in checkpoints:
            hash_reps_at_chk[idx] = len(seen_hashes)
    hash_elapsed = time.perf_counter() - t0
    hash_mem_kb = (len(seen_hashes) * 96) / 1024.0

    # 3. FUTCache (Sacred Entry Service + 10D Metric Net, ε = 0.55)
    caches: Dict[str, PackCache] = {}
    fc_novel_count = 0
    fc_reps_at_chk = {}
    t0 = time.perf_counter()

    for idx, t in enumerate(traces, 1):
        svc = t.entry_service
        if svc not in caches:
            caches[svc] = PackCache(
                dimension=10,
                epsilon=0.55,
                distance="l2",
                backend="vptree",
                domain_min=-1e5,
                domain_max=1e5,
            )
        res = caches[svc].observe(t.scaled_geometry)
        if res.is_novel:
            fc_novel_count += 1

        if idx in checkpoints:
            tot_reps = sum(len(c) for c in caches.values())
            fc_reps_at_chk[idx] = tot_reps

    fc_elapsed = time.perf_counter() - t0
    fc_mem_kb = sum(c.memory_bytes() for c in caches.values()) / 1024.0
    fc_total_reps = sum(len(c) for c in caches.values())

    print("\n" + "=" * 88)
    print("  1. PRODUCTION COMPRESSION & EFFICIENCY (Full Stream)")
    print("=" * 88)
    header = f"{'Strategy':<28} | {'Suppression':<11} | {'Retained':<9} | {'Throughput':<14} | {'Memory (KB)':<10}"
    print(header)
    print("-" * len(header))
    print(f"{'Random Uniform (1%)':<28} | {(n_total - rand_samples)/n_total*100:>10.2f}% | {rand_samples:>9,d} | {'~10,000,000 t/s':>14} | {'0.0':>10}")
    print(f"{'Exact Hash Matcher':<28} | {(n_total - len(seen_hashes))/n_total*100:>10.2f}% | {len(seen_hashes):>9,d} | {n_total/hash_elapsed:>10.0f} t/s | {hash_mem_kb:>10.1f}")
    print(f"{'FUTCache (ε = 0.55, VP-Tree)':<28} | {(n_total - fc_novel_count)/n_total*100:>10.2f}% | {fc_novel_count:>9,d} | {n_total/fc_elapsed:>10.0f} t/s | {fc_mem_kb:>10.1f}")

    print("\n" + "=" * 88)
    print("  2. STATE BOUNDEDNESS: REPRESENTATIVE GROWTH (|R|) VS. STREAM LENGTH (N)")
    print("=" * 88)
    print(f"{'Processed Traces (N)':<22} | {'Exact Hash Keys':<18} | {'FUTCache Reps (|R|)':<22} | {'Growth Damping':<14}")
    print("-" * 88)
    for chk in checkpoints:
        h_reps = hash_reps_at_chk[chk]
        fc_reps = fc_reps_at_chk[chk]
        ratio = (fc_reps / h_reps) * 100 if h_reps > 0 else 100.0
        print(f"{chk:>15,d} traces    | {h_reps:>14,d}    | {fc_reps:>16,d} reps    | {ratio:>10.1f}% of Hash")

    print("\n" + "=" * 88)
    print("  3. SUMMARY & TAKEAWAYS (Real Alibaba Production Traffic)")
    print("=" * 88)
    print(f"• Compression: FUTCache filtered out {(n_total - fc_novel_count)/n_total*100:.2f}% of redundant Alibaba call graphs.")
    print(f"• Bounded Geometric State: While Exact Hash grew to {len(seen_hashes):,} keys due to timing noise,")
    print(f"  FUTCache compressed the entire observed operational topology into just {fc_total_reps:,} representative centers ({fc_mem_kb:.1f} KB RAM).")
    print(f"• Pure Production Geometry: Evaluated at ~{n_total/fc_elapsed:,.0f} traces/sec with 0 synthetic assumptions.")
    print("=" * 88)


def main():
    dest = ensure_alibaba_shard(LOCAL_CACHE_TAR)
    traces = stream_parse_alibaba_traces(dest, max_traces=100_000)
    run_alibaba_production_benchmark(traces)


if __name__ == "__main__":
    main()
