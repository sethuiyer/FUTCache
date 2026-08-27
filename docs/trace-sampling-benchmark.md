# Empirical Benchmark: Geometric Tail-Sampling for Distributed Traces

> **FUTCache vs. Industry Observability Baselines on a 100,000-Trace Production Stream**

Distributed tracing (OpenTelemetry, Jaeger, Datadog, Honeycomb) faces an expensive dilemma:
* **Ingesting 100% of traces** costs fortunes in network bandwidth, storage, and SaaS APM bills.
* **Random sampling (e.g. 1% or 5%)** saves money but **misses rare, catastrophic incident traces**.
* **Fixed latency / 5xx rules** fail on masked fallbacks, fast structural regressions (like N+1 queries), and subtle topology shifts.

This benchmark evaluates **FUTCache as a Geometric Tail-Sampler** on a realistic microservice fleet under 6 real-world incident archetypes.

---

## 1. The Synthetic Production Fleet

The benchmark simulates a 10-endpoint e-commerce microservice fleet (`/checkout`, `/search`, `/cart`, `/login`, etc.) processing **100,000 traces** with natural log-normal execution jitter and 6 distinct operational anomalies:

| Incident Archetype | Trace Shape & Mechanism | Frequency | Real-World Impact |
|---|---|---|---|
| **`happy`** (Normal Traffic) | Normal call depth (3-5), fanout (2-6), median latency ~15ms, 1-4 DB calls. | 99.54% (99,545 traces) | Standard baseline traffic. |
| **`db_pool_exhaustion`** | High latency (350-750ms), 2-4 retries, 8-14 DB queries, 503 errors. | 0.09% (89 traces) | Connection starvation during high load. |
| **`circuit_breaker_fallback`** | Masked **200 OK** status! Deeper fallback path (depth 6-8, fanout 8-12), static cache served in 80-140ms. | 0.05% (47 traces) | Downstream service outage hidden by degraded static response. |
| **`n_plus_one_regression`** | Introduced in canary `v2.5.0`. Fast execution (45-95ms), but **fanout explodes to 45 DB queries in a loop**. | 0.12% (118 traces) | Silent database killer introduced in code deployment. |
| **`microservice_circular_loop`** | Circular RPC recursion (depth jumps to 12-14), gateway timeout (504 cap at 1500ms). | 0.03% (33 traces) | Deadlock / misconfigured routing rule. |
| **`p99_silent_degradation`** | Identical call structure, but critical path is **15x slower** (280-420ms). | 0.15% (152 traces) | Resource contention / lock wait. |
| **`auth_unicode_panic`** | Rare 1-in-50,000 malformed payload crashes auth middleware early (depth 2, fanout 1, 500 error). | 0.02% (16 traces) | Edge-case parser panic. |

---

## 2. The Evaluated Strategies

1. **Random Uniform (1%)**: Standard default in most distributed trace collectors.
2. **Random Uniform (5%)**: Aggressive random sampling.
3. **Latency & Error Gate (>200ms / 5xx)**: Static rules (sample only if slow or 5xx).
4. **Exact Hash Matcher**: Discrete hash map of `(route, version, status, depth, fanout, bucketed_latency, db_queries)`.
5. **FUTCache ($\varepsilon = 0.55$, VP-Tree)**: Two-stage architecture:
   - **Stage 1 (Sacred Partition):** `(route, service_version, status_class)` exact match.
   - **Stage 2 (Geometric $\varepsilon$-Net):** Continuous packing over `[depth, fanout, duration_norm, retries, db_queries]`.

---

## 3. Benchmark Results

### Overall Performance & Efficiency

| Strategy | Ingestion Suppression | Traces Exported | Incident Archetypes Discovered | First-Sighting Accuracy (Trace #1) | Throughput | RAM Footprint |
|---|---|---|---|---|---|---|
| **Random Uniform (1%)** | 99.03% | 966 | 50.0% (3/6) | 16.7% (1/6) | ~10.8M t/s | 0.0 KB |
| **Random Uniform (5%)** | 94.93% | 5,070 | 83.3% (5/6) | 16.7% (1/6) | ~10.2M t/s | 0.0 KB |
| **Latency Gate (>200ms / 5xx)** | 99.71% | 290 | 66.7% (4/6) | 66.7% (4/6) | ~6.2M t/s | 0.0 KB |
| **Exact Hash Matcher** | 98.51% | 1,489 | **100.0% (6/6)** | **100.0% (6/6)** | ~1.3M t/s | 93.1 KB |
| **FUTCache ($\varepsilon=0.55$)** | **98.76%** | **1,238** | **100.0% (6/6)** | **100.0% (6/6)** | **~262,000 t/s** | **263.8 KB** |

---

### Incident Capture Breakdown & First-Sighting Latency

```
======================================================================================
Strategy: Random Uniform (1%)
  - db_pool_exhaustion         :   1/ 89 (  1.1%) | ✗ MISSED FIRST SIGHTING
  - circuit_breaker_fallback   :   0/ 47 (  0.0%) | ✗ MISSED FIRST SIGHTING (NEVER CAUGHT)
  - n_plus_one_regression      :   0/118 (  0.0%) | ✗ MISSED FIRST SIGHTING (NEVER CAUGHT)
  - microservice_circular_loop :   1/ 33 (  3.0%) | ✓ CAUGHT ON TRACE #1
  - p99_silent_degradation     :   2/152 (  1.3%) | ✗ MISSED FIRST SIGHTING
  - auth_unicode_panic         :   0/ 16 (  0.0%) | ✗ MISSED FIRST SIGHTING (NEVER CAUGHT)

Strategy: Latency Gate (>200ms / 5xx)
  - db_pool_exhaustion         :  89/ 89 (100.0%) | ✓ CAUGHT ON TRACE #1
  - circuit_breaker_fallback   :   0/ 47 (  0.0%) | ✗ MISSED FIRST SIGHTING (100% BLIND)
  - n_plus_one_regression      :   0/118 (  0.0%) | ✗ MISSED FIRST SIGHTING (100% BLIND)
  - microservice_circular_loop :  33/ 33 (100.0%) | ✓ CAUGHT ON TRACE #1
  - p99_silent_degradation     : 152/152 (100.0%) | ✓ CAUGHT ON TRACE #1
  - auth_unicode_panic         :  16/ 16 (100.0%) | ✓ CAUGHT ON TRACE #1

Strategy: FUTCache (ε = 0.55, VP-Tree)
  - db_pool_exhaustion         :  83/ 89 ( 93.3%) | ✓ CAUGHT ON TRACE #1
  - circuit_breaker_fallback   :  18/ 47 ( 38.3%) | ✓ CAUGHT ON TRACE #1
  - n_plus_one_regression      :  90/118 ( 76.3%) | ✓ CAUGHT ON TRACE #1
  - microservice_circular_loop :  32/ 33 ( 97.0%) | ✓ CAUGHT ON TRACE #1
  - p99_silent_degradation     :   4/152 (  2.6%) | ✓ CAUGHT ON TRACE #1
  - auth_unicode_panic         :   1/ 16 (  6.2%) | ✓ CAUGHT ON TRACE #1
======================================================================================
```

---

## 4. Key Takeaways & Technical Insights

### 1. Why Latency Gates Fail (The False Safety of Static Rules)
Static rules (`duration > 200ms`) missed **100% of two critical production incidents**:
* **Circuit Breaker Fallbacks:** The service was returning degraded static content, masking the failure behind a fast 80ms `200 OK`.
* **N+1 Query Canary Regressions:** The loop executed 45 queries in 60ms. Because 60ms is under 200ms, the static rule allowed the database to be hammered unnoticed until production collapsed.
* **FUTCache caught both on Trace #1** because their structural geometry (depth and fanout) moved outside the $\varepsilon$-neighborhood of normal traffic.

### 2. Why Random Sampling Loses Crucial Incidents
Random 1% sampling completely missed `circuit_breaker_fallback`, `n_plus_one_regression`, and `auth_unicode_panic` entirely ($0$ traces retained). For incidents with low volume (e.g. 16 crashes), random sampling has an **85%+ probability of never storing a single trace**.

### 3. FUTCache's Auto-Deduplication of Repeat Crashes
Notice how for `auth_unicode_panic`, FUTCache captured `1/16` traces with **100% Trace #1 accuracy**:
* The **first time** the crash happened: $d(\text{trace}, R) > \varepsilon \implies \textbf{NOVEL}$ (Instantly retained and alerted to SREs).
* The **next 15 times** the identical crash happened: $d(\text{trace}, R) \le \varepsilon \implies \textbf{REDUNDANT}$ (Suppressed to save storage; metric counter incremented).
* This provides **ideal operational behavior**: immediate detection of new failure shapes without spamming storage with duplicate payloads.

---

## 5. How to Reproduce

Run the standalone benchmark script:

```bash
python3 demos/distributed_trace_benchmark.py
```
