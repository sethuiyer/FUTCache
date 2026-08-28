#!/usr/bin/env python3
"""OpenTelemetry (OTel) & Jaeger Geometric Tail-Sampling Pipeline.

Ingests standard OTLP JSON or Jaeger JSON trace dumps, extracts call graph
topological geometry, and uses FUTCache to make zero-overhead tail-sampling
decisions:
  - 100% of structurally novel traces are retained and exported.
  - Redundant traces (happy-path jitter) are suppressed with a counter increment.

Usage:
  # Generate a sample OTLP JSON dataset with normal & anomalous traces
  python3 demos/otel_tail_sampler.py --generate-sample otlp_sample.json --count 5000

  # Run FUTCache tail-sampling on the dataset
  python3 demos/otel_tail_sampler.py --input otlp_sample.json --output-novel novel_traces.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure local futcache package is prioritized
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg_dir = os.path.join(repo_root, "python")
if python_pkg_dir not in sys.path:
    sys.path.insert(0, python_pkg_dir)

import numpy as np
# Ensure local python/futcache is loaded before any site-packages copy.
import os as _os_demo, sys as _sys_demo
_repo_root_demo = _os_demo.path.dirname(_os_demo.path.dirname(_os_demo.path.abspath(__file__)))
_python_pkg_demo = _os_demo.path.join(_repo_root_demo, 'python')
if _python_pkg_demo not in _sys_demo.path:
    _sys_demo.path.insert(0, _python_pkg_demo)


from futcache import PackCache

# ---------------------------------------------------------------------------
# 1. Span & Trace Representation
# ---------------------------------------------------------------------------

@dataclass
class OTelSpan:
    trace_id: str
    span_id: str
    parent_span_id: str
    name: str
    service_name: str
    service_version: str
    start_time_ns: int
    end_time_ns: int
    status_code: int  # 0=UNSET, 1=OK, 2=ERROR
    attributes: Dict[str, Any]
    raw_span: Dict[str, Any]


@dataclass
class ParsedTrace:
    trace_id: str
    root_span: Optional[OTelSpan]
    spans: List[OTelSpan]
    sacred_key: Tuple[str, str, int]
    # Geometric signature: [depth, fanout_spans, crit_path_ms, error_spans, db_calls]
    geometry: np.ndarray


# ---------------------------------------------------------------------------
# 2. OTLP & Jaeger JSON Parsers
# ---------------------------------------------------------------------------

def _extract_attr_val(val_dict: Dict[str, Any]) -> Any:
    """Unpack OTLP AnyValue protobuf-style dict."""
    if not isinstance(val_dict, dict):
        return val_dict
    for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if k in val_dict:
            return val_dict[k]
    if "arrayValue" in val_dict:
        return [_extract_attr_val(v) for v in val_dict["arrayValue"].get("values", [])]
    return str(val_dict)


def parse_otlp_json(data: Dict[str, Any]) -> List[ParsedTrace]:
    """Parse standard OTLP JSON (ExportTraceServiceRequest)."""
    spans_by_trace: Dict[str, List[OTelSpan]] = defaultdict(list)

    resource_spans = data.get("resourceSpans", [])
    for rs in resource_spans:
        res_attrs = {}
        for attr in rs.get("resource", {}).get("attributes", []):
            res_attrs[attr["key"]] = _extract_attr_val(attr.get("value", {}))

        service_name = res_attrs.get("service.name", "unknown_service")
        service_ver = res_attrs.get("service.version", "v1.0.0")

        for scope_span in rs.get("scopeSpans", []):
            for sp in scope_span.get("spans", []):
                span_attrs = {}
                for attr in sp.get("attributes", []):
                    span_attrs[attr["key"]] = _extract_attr_val(attr.get("value", {}))

                status = sp.get("status", {})
                status_code = status.get("code", 1)  # Default OK

                otel_span = OTelSpan(
                    trace_id=sp["traceId"],
                    span_id=sp["spanId"],
                    parent_span_id=sp.get("parentSpanId", ""),
                    name=sp.get("name", "unnamed_span"),
                    service_name=service_name,
                    service_version=service_ver,
                    start_time_ns=int(sp.get("startTimeUnixNano", 0)),
                    end_time_ns=int(sp.get("endTimeUnixNano", 0)),
                    status_code=status_code,
                    attributes=span_attrs,
                    raw_span=sp,
                )
                spans_by_trace[sp["traceId"]].append(otel_span)

    return [_build_trace_geometry(t_id, span_list) for t_id, span_list in spans_by_trace.items()]


def parse_jaeger_json(data: Dict[str, Any]) -> List[ParsedTrace]:
    """Parse Jaeger UI JSON export format."""
    parsed = []
    traces = data.get("data", []) if "data" in data else [data]

    for t in traces:
        t_id = t.get("traceID", "")
        processes = t.get("processes", {})
        span_list = []

        for sp in t.get("spans", []):
            p_id = sp.get("processID", "")
            proc = processes.get(p_id, {})
            service_name = proc.get("serviceName", "unknown_service")

            # Extract parent span from references
            parent_id = ""
            for ref in sp.get("references", []):
                if ref.get("refType") == "CHILD_OF":
                    parent_id = ref.get("spanID", "")
                    break

            span_attrs = {}
            for tag in sp.get("tags", []):
                span_attrs[tag["key"]] = tag.get("value")

            start_us = int(sp.get("startTime", 0))
            duration_us = int(sp.get("duration", 0))

            otel_span = OTelSpan(
                trace_id=t_id,
                span_id=sp.get("spanID", ""),
                parent_span_id=parent_id,
                name=sp.get("operationName", ""),
                service_name=service_name,
                service_version="v1.0.0",
                start_time_ns=start_us * 1000,
                end_time_ns=(start_us + duration_us) * 1000,
                status_code=2 if span_attrs.get("error") else 1,
                attributes=span_attrs,
                raw_span=sp,
            )
            span_list.append(otel_span)

        if span_list:
            parsed.append(_build_trace_geometry(t_id, span_list))

    return parsed


# ---------------------------------------------------------------------------
# 3. Geometric Feature Extraction from Call Graph
# ---------------------------------------------------------------------------

def _build_trace_geometry(trace_id: str, spans: List[OTelSpan]) -> ParsedTrace:
    """Calculates tree depth, critical path duration, and structural signature."""
    span_map: Dict[str, OTelSpan] = {s.span_id: s for s in spans}
    children_map: Dict[str, List[str]] = defaultdict(list)

    root_span: Optional[OTelSpan] = None
    for s in spans:
        if not s.parent_span_id or s.parent_span_id not in span_map:
            root_span = s
        else:
            children_map[s.parent_span_id].append(s.span_id)

    if root_span is None and spans:
        root_span = spans[0]

    # Calculate tree depth via BFS/DFS
    def get_depth(span_id: str) -> int:
        children = children_map.get(span_id, [])
        if not children:
            return 1
        return 1 + max(get_depth(cid) for cid in children)

    max_depth = get_depth(root_span.span_id) if root_span else 1
    total_spans = len(spans)

    # Calculate total duration in ms
    min_start = min(s.start_time_ns for s in spans)
    max_end = max(s.end_time_ns for s in spans)
    duration_ms = max(0.0, (max_end - min_start) / 1_000_000.0)

    # Count DB queries & Error spans
    db_calls = sum(1 for s in spans if "db.system" in s.attributes or "db.statement" in s.attributes)
    error_spans = sum(1 for s in spans if s.status_code == 2 or s.attributes.get("error"))

    # Stage 1: Sacred Partition Key
    service = root_span.service_name if root_span else "unknown"
    route = root_span.name if root_span else "unknown"
    status_class = 5 if error_spans > 0 else 2
    sacred_key = (service, route, status_class)

    # Stage 2: Scaled Geometric Vector
    # Weights: [depth=1.0, fanout=0.25, duration_ms=0.02 (50ms=1.0), error=2.0, db=0.5]
    geometry = np.array([
        float(max_depth) * 1.0,
        float(total_spans) * 0.25,
        float(duration_ms) * 0.02,
        float(error_spans) * 2.0,
        float(db_calls) * 0.5,
    ], dtype=np.float64)

    return ParsedTrace(
        trace_id=trace_id,
        root_span=root_span,
        spans=spans,
        sacred_key=sacred_key,
        geometry=geometry,
    )


# ---------------------------------------------------------------------------
# 4. FUTCache OTel Tail-Sampler Engine
# ---------------------------------------------------------------------------

class OTelFUTCacheSampler:
    def __init__(self, epsilon: float = 0.50):
        self.epsilon = epsilon
        self._caches: Dict[Tuple, PackCache] = {}

    def _get_cache(self, sacred_key: Tuple) -> PackCache:
        if sacred_key not in self._caches:
            self._caches[sacred_key] = PackCache(
                dimension=5,
                epsilon=self.epsilon,
                distance="l2",
                backend="vptree",
                domain_min=-1e5,
                domain_max=1e5,
            )
        return self._caches[sacred_key]

    def observe(self, trace: ParsedTrace) -> Tuple[bool, float, int]:
        """Returns (is_novel, distance, rep_id)."""
        cache = self._get_cache(trace.sacred_key)
        res = cache.observe(trace.geometry)
        return res.is_novel, res.distance, res.representative_id

    def memory_bytes(self) -> int:
        return sum(c.memory_bytes() for c in self._caches.values())

    def rep_count(self) -> int:
        return sum(len(c) for c in self._caches.values())


# ---------------------------------------------------------------------------
# 5. Synthetic OTLP Generator (for testing)
# ---------------------------------------------------------------------------

def generate_sample_otlp(filepath: str, count: int = 5000) -> None:
    """Generates realistic OTLP JSON with happy-path traffic and injected anomalies."""
    print(f"[+] Generating realistic OTLP JSON with {count:,} traces -> {filepath}")
    rng = np.random.default_rng(42)

    services = ["frontend-proxy", "order-service", "payment-service", "inventory-service"]
    routes = ["/api/v1/checkout", "/api/v1/cart", "/api/v1/products"]

    resource_spans = []

    base_time_ns = int(time.time() * 1_000_000_000)

    for i in range(count):
        t_id = f"{i:032x}"
        root_s_id = f"{i:016x}"
        t_start = base_time_ns + i * 50_000_000 # 50ms interval

        # 98% Happy, 2% Injected anomalies
        is_anomaly = (i % 50 == 0) and (i > 0)
        route = routes[i % len(routes)]

        if not is_anomaly:
            # Happy path
            duration_ms = float(rng.lognormal(2.5, 0.2)) # ~12-15ms
            depth = 3
            fanout = 4
            status_code = 1
            has_error = False
            db_count = 2
        else:
            # Structural anomaly (e.g. cascading retry or N+1 query burst)
            anomaly_type = (i // 50) % 3
            if anomaly_type == 0: # DB Starvation
                duration_ms = float(rng.uniform(300.0, 600.0))
                depth = 4
                fanout = 8
                status_code = 2
                has_error = True
                db_count = 8
            elif anomaly_type == 1: # N+1 query canary
                duration_ms = float(rng.uniform(50.0, 90.0))
                depth = 3
                fanout = 32
                status_code = 1
                has_error = False
                db_count = 30
            else: # Deep recursion / Timeout
                duration_ms = float(rng.uniform(900.0, 1200.0))
                depth = 10
                fanout = 15
                status_code = 2
                has_error = True
                db_count = 5

        t_end = t_start + int(duration_ms * 1_000_000)

        # Build root span
        spans = [{
            "traceId": t_id,
            "spanId": root_s_id,
            "parentSpanId": "",
            "name": route,
            "kind": 1,
            "startTimeUnixNano": str(t_start),
            "endTimeUnixNano": str(t_end),
            "attributes": [
                {"key": "http.route", "value": {"stringValue": route}},
                {"key": "http.status_code", "value": {"intValue": 500 if has_error else 200}},
            ],
            "status": {"code": status_code},
        }]

        # Build child spans
        for f in range(1, fanout):
            child_s_id = f"{(i*1000 + f):016x}"
            is_db = f <= db_count
            s_name = "SELECT * FROM db" if is_db else f"subcall-{f}"
            c_start = t_start + int((f / fanout) * (duration_ms * 0.7) * 1_000_000)
            c_end = c_start + int(duration_ms * 0.2 * 1_000_000)

            attrs = []
            if is_db:
                attrs.append({"key": "db.system", "value": {"stringValue": "postgresql"}})
                attrs.append({"key": "db.statement", "value": {"stringValue": "SELECT * FROM orders"}})

            spans.append({
                "traceId": t_id,
                "spanId": child_s_id,
                "parentSpanId": root_s_id,
                "name": s_name,
                "kind": 3 if is_db else 2,
                "startTimeUnixNano": str(c_start),
                "endTimeUnixNano": str(c_end),
                "attributes": attrs,
                "status": {"code": status_code if has_error and f == fanout - 1 else 1},
            })

        resource_spans.append({
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": services[i % len(services)]}},
                    {"key": "service.version", "value": {"stringValue": "v1.2.0"}},
                ]
            },
            "scopeSpans": [{"spans": spans}]
        })

    with open(filepath, "w") as fp:
        json.dump({"resourceSpans": resource_spans}, fp, indent=2)
    print(f"[✓] Saved {count:,} OTLP traces ({os.path.getsize(filepath) / 1024 / 1024:.2f} MB)")


# ---------------------------------------------------------------------------
# 6. Main Execution CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FUTCache OpenTelemetry Geometric Tail-Sampling Pipeline")
    parser.add_argument("--input", "-i", type=str, help="Input OTLP JSON or Jaeger JSON file")
    parser.add_argument("--output-novel", "-o", type=str, default="novel_traces.json", help="Output novel traces JSON")
    parser.add_argument("--epsilon", "-e", type=float, default=0.50, help="Geometric novelty threshold epsilon")
    parser.add_argument("--generate-sample", type=str, help="Generate synthetic OTLP JSON dataset to file")
    parser.add_argument("--count", type=int, default=5000, help="Number of traces to generate")

    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_otlp(args.generate_sample, args.count)
        if not args.input:
            args.input = args.generate_sample

    if not args.input:
        parser.print_help()
        sys.exit(1)

    print(f"\n[+] Loading & parsing traces from: {args.input}")
    t0 = time.perf_counter()
    with open(args.input, "r") as fp:
        raw_data = json.load(fp)

    # Detect format
    if "resourceSpans" in raw_data:
        parsed_traces = parse_otlp_json(raw_data)
        fmt = "OTLP JSON"
    else:
        parsed_traces = parse_jaeger_json(raw_data)
        fmt = "Jaeger JSON"

    t1 = time.perf_counter()
    print(f"[✓] Parsed {len(parsed_traces):,} traces ({fmt}) in {t1 - t0:.3f}s")

    # Run FUTCache Tail-Sampling
    print(f"\n[+] Running FUTCache Geometric Tail-Sampling (ε = {args.epsilon:.2f})...")
    sampler = OTelFUTCacheSampler(epsilon=args.epsilon)

    novel_traces: List[ParsedTrace] = []
    suppressed_count = 0
    t_start = time.perf_counter()

    for trace in parsed_traces:
        is_novel, dist, rep_id = sampler.observe(trace)
        if is_novel:
            novel_traces.append(trace)
        else:
            suppressed_count += 1

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    tps = len(parsed_traces) / elapsed if elapsed > 0 else 0

    suppression_pct = (suppressed_count / len(parsed_traces)) * 100 if parsed_traces else 0

    print("\n" + "=" * 70)
    print("  FUTCACHE OTEL TAIL-SAMPLING RESULTS")
    print("=" * 70)
    print(f"Total Traces Ingested    : {len(parsed_traces):,}")
    print(f"Novel Traces Retained    : {len(novel_traces):,} ({100 - suppression_pct:.2f}%)")
    print(f"Redundant Suppressed     : {suppressed_count:,} ({suppression_pct:.2f}%) -> 💰 Bandwidth Saved!")
    print(f"Distinct Reps Retained   : {sampler.rep_count():,} representative Voronoi centers")
    print(f"Sampling Throughput      : {tps:,.0f} traces/second")
    print(f"Active Memory Footprint  : {sampler.memory_bytes() / 1024:.2f} KB")
    print("=" * 70)

    # Export novel traces
    if args.output_novel:
        out_data = {
            "summary": {
                "total_ingested": len(parsed_traces),
                "novel_retained": len(novel_traces),
                "suppressed": suppressed_count,
                "suppression_percentage": round(suppression_pct, 2),
            },
            "novel_traces": [
                {
                    "trace_id": t.trace_id,
                    "service": t.sacred_key[0],
                    "route": t.sacred_key[1],
                    "status_class": f"{t.sacred_key[2]}xx",
                    "geometry": [round(float(x), 3) for x in t.geometry],
                    "spans_count": len(t.spans),
                }
                for t in novel_traces
            ]
        }
        with open(args.output_novel, "w") as fp:
            json.dump(out_data, fp, indent=2)
        print(f"\n[✓] Exported {len(novel_traces):,} novel traces to: {args.output_novel}")


if __name__ == "__main__":
    main()
