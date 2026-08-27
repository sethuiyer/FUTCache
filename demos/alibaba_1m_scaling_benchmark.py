#!/usr/bin/env python3
"""1,000,000-Trace Real Production Scaling Benchmark on Alibaba Cloud (SoCC '21).

The Million-Trace Empirical Growth Law Experiment:
  Tests whether geometric state |R_N| sublinearizes / bends relative to exact hash
  states |H_N| as observation stream length N scales from 100k -> 250k -> 500k -> 750k -> 1,000,000.

Maintains strict experimental consistency:
  - Exact same 10D native geometry derived from `rpcid` tree structure
  - Exact same normalization weights
  - Exact same sacred partition (entry_service)
  - Exact same epsilon = 0.55, distance = "l2", backend = "vptree"
  - Chronological ordering across Alibaba shards (MSCallGraph_0 to MSCallGraph_7)
"""

import os
import sys
import time
import tarfile
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Generator, List, Set, Tuple

# Ensure local futcache package is prioritized
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg_dir = os.path.join(repo_root, "python")
if python_pkg_dir not in sys.path:
    sys.path.insert(0, python_pkg_dir)

import numpy as np
from futcache import PackCache

BASE_OSS_URL = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces/MSCallGraph"
CACHE_DIR = os.path.join(repo_root, "demos", ".cache")


@dataclass
class RealAlibabaTrace:
    trace_id: str
    entry_service: str
    raw_features: np.ndarray
    scaled_geometry: np.ndarray


# ---------------------------------------------------------------------------
# 1. 10D Native Call-Tree Geometry (Linear DP Reconstruction)
# ---------------------------------------------------------------------------

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


def extract_tree_signature(span_rows: List[Tuple[str, str, str, str, float]]) -> Tuple[str, np.ndarray]:
    call_count = len(span_rows)
    entry_service = span_rows[0][1] if span_rows else "unknown_service"

    parent_children = defaultdict(set)
    rpc_rts = {}
    rpc_types = defaultdict(int)
    all_rts = []

    for rpc_id, um, dm, rpc_type, rt in span_rows:
        rt_val = max(0.0, float(rt))
        all_rts.append(rt_val)
        if rpc_id not in rpc_rts or rt_val > rpc_rts[rpc_id]:
            rpc_rts[rpc_id] = rt_val

        t_clean = rpc_type.upper().strip()
        rpc_types[t_clean] += 1

        last_dot = rpc_id.rfind(".")
        if last_dot != -1:
            parent_id = rpc_id[:last_dot]
            parent_children[parent_id].add(rpc_id)
        else:
            parent_children["root"].add(rpc_id)

    max_depth = max(rpc_id.count(".") + 1 for rpc_id in rpc_rts.keys()) if rpc_rts else 1
    max_fanout = max(len(children) for children in parent_children.values()) if parent_children else 1

    rpc_count = rpc_types.get("RPC", 0) + rpc_types.get("USER", 0)
    http_count = rpc_types.get("HTTP", 0)
    db_count = rpc_types.get("DB", 0)
    mc_count = rpc_types.get("MC", 0)

    mean_rt = float(np.mean(all_rts)) if all_rts else 0.0
    max_rt = float(np.max(all_rts)) if all_rts else 0.0

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
        res = node_cost if not children else node_cost + max(compute_path_rt(cid) for cid in children)

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


# ---------------------------------------------------------------------------
# 2. Multi-Shard Streaming Trace Generator
# ---------------------------------------------------------------------------

def ensure_shard_file(shard_idx: int) -> str:
    filename = f"MSCallGraph_{shard_idx}.tar.gz"
    dest_path = os.path.join(CACHE_DIR, filename)
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100_000_000:
        return dest_path

    url = f"{BASE_OSS_URL}/{filename}"
    print(f"\n[+] Fetching Alibaba shard #{shard_idx} ({url})...")
    t0 = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "FUTCache-Benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest_path, "wb") as out_f:
        tot = int(resp.getheader("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)
            done += len(chunk)
            sys.stdout.write(f"\r    Downloading {filename}: {done/1024/1024:.1f}/{tot/1024/1024:.1f} MB ({done/tot*100:.1f}%)")
            sys.stdout.flush()
    print(f"\n[✓] Shard #{shard_idx} saved in {time.perf_counter() - t0:.2f}s")
    return dest_path


def stream_real_alibaba_traces(target_count: int = 1_000_000) -> Generator[RealAlibabaTrace, None, None]:
    """Streams up to target_count complete trace trees across consecutive Alibaba shards."""
    yielded = 0
    shard_idx = 0

    while yielded < target_count:
        shard_tar = ensure_shard_file(shard_idx)
        print(f"\n[+] Parsing shard #{shard_idx} ({os.path.basename(shard_tar)})...")
        t0 = time.perf_counter()

        spans_by_trace = defaultdict(list)
        trace_order = []
        seen_in_shard = set()

        with tarfile.open(shard_tar, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".csv"):
                    continue
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

                        if t_id not in seen_in_shard:
                            seen_in_shard.add(t_id)
                            trace_order.append(t_id)

                        spans_by_trace[t_id].append((rpc_id, um, dm, rpc_type, rt))
                    except Exception:
                        continue
                break

        print(f"    [✓] Shard #{shard_idx} yielded {len(trace_order):,} raw trace trees in {time.perf_counter() - t0:.2f}s")
        
        for t_id in trace_order:
            if yielded >= target_count:
                break
            span_rows = spans_by_trace[t_id]
            if not span_rows:
                continue
            entry_svc, raw_vec = extract_tree_signature(span_rows)
            scaled_vec = raw_vec * WEIGHTS_10D

            yield RealAlibabaTrace(
                trace_id=t_id,
                entry_service=entry_svc,
                raw_features=raw_vec,
                scaled_geometry=scaled_vec,
            )
            yielded += 1

        shard_idx += 1


# ---------------------------------------------------------------------------
# 3. 1-Million Trace Scaling Benchmark
# ---------------------------------------------------------------------------

def run_1m_scaling_benchmark(target_count: int = 1_000_000):
    print("=" * 96)
    print("  ALIBABA CLOUD MILLION-TRACE EMPIRICAL SCALING BENCHMARK (SoCC '21)")
    print("  Testing Empirical Growth Law: |R_N| vs. |H_N| across 1,000,000 real production traces")
    print("=" * 96)

    checkpoints = [50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000]
    
    seen_hashes: Set[Tuple] = set()
    caches: Dict[str, PackCache] = {}
    fc_novel_count = 0

    chk_stats = []
    t_global_start = time.perf_counter()

    print(f"\n[+] Streaming {target_count:,} real production traces through FUTCache & Exact Hash...")

    idx = 0
    for t in stream_real_alibaba_traces(target_count=target_count):
        idx += 1

        # 1. Exact Hash Matcher
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

        # 2. FUTCache (ε = 0.55, VP-Tree)
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

        # Checkpoint Reporting
        if idx in checkpoints or idx == target_count:
            now_sec = time.perf_counter() - t_global_start
            h_len = len(seen_hashes)
            fc_len = sum(len(c) for c in caches.values())
            fc_mem_mb = sum(c.memory_bytes() for c in caches.values()) / (1024 * 1024)
            hash_mem_mb = (h_len * 96) / (1024 * 1024)
            ratio = (fc_len / h_len) * 100 if h_len > 0 else 100.0
            suppression = ((idx - fc_novel_count) / idx) * 100
            rate = idx / now_sec if now_sec > 0 else 0

            chk_stats.append((idx, h_len, fc_len, ratio, suppression, fc_mem_mb, hash_mem_mb, rate))
            
            print(f"  >>> Checkpoint N={idx:>9,d} | Hash={h_len:>7,d} | FUTCache={fc_len:>7,d} reps ({ratio:>5.1f}% of Hash) | Suppr={suppression:>5.1f}% | Rate={rate:>7.0f} t/s")

    t_total = time.perf_counter() - t_global_start

    print("\n" + "=" * 96)
    print("  1. EMPIRICAL GROWTH LAW: STATE SIZE VS. STREAM LENGTH (N = 50k -> 1,000,000)")
    print("=" * 96)
    header = f"{'Stream (N)':<16} | {'Exact Hash (|H|)':<18} | {'FUTCache (|R|)':<18} | {'Damping (|R|/|H|)':<18} | {'Suppression':<12} | {'FUTCache RAM':<12}"
    print(header)
    print("-" * len(header))
    for c_n, c_h, c_fc, c_ratio, c_supp, c_mem, _, _ in chk_stats:
        print(
            f"{c_n:>12,d}    | "
            f"{c_h:>14,d}    | "
            f"{c_fc:>14,d}    | "
            f"{c_ratio:>14.2f}%    | "
            f"{c_supp:>10.2f}% | "
            f"{c_mem:>9.2f} MB"
        )

    print("\n" + "=" * 96)
    print("  2. OVERALL 1-MILLION TRACE SUMMARY")
    print("=" * 96)
    last = chk_stats[-1]
    print(f"• Processed {last[0]:,} Real Production Alibaba Traces in {t_total:.2f}s (~{last[7]:,.0f} traces/sec)")
    print(f"• Total Redundant Call-Graphs Suppressed: {last[0] - last[2]:,} ({last[4]:.2f}%)")
    print(f"• State Comparison at 1M Traces:")
    print(f"    - Exact Hash Matcher : {last[1]:,} discrete keys ({last[6]:.2f} MB)")
    print(f"    - FUTCache (ε = 0.55): {last[2]:,} geometric representatives ({last[5]:.2f} MB)")
    print(f"• Empirical Growth Trajectory (|R| / |H|):")
    for c_n, _, _, c_ratio, _, _, _, _ in chk_stats:
        bar_len = int(c_ratio / 2)
        print(f"    N = {c_n:>9,d} : [{ '#' * bar_len }{ ' ' * (50 - bar_len) }] {c_ratio:.2f}%")
    print("=" * 96)


if __name__ == "__main__":
    count = 1_000_000
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    run_1m_scaling_benchmark(target_count=count)
