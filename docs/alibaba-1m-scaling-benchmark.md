# Empirical Scaling Benchmark: 1,000,000 Real Alibaba Cloud Traces

> **Testing the Empirical Growth Law of Operational Memory on Real Production Microservices**

A central theoretical claim of FUTCache is that **memory state is bounded by geometric coverage ($P(K, \varepsilon)$), not by observation stream length ($N$)**.

To test this thesis on uncurated, large-scale enterprise traffic, we benchmarked FUTCache across **1,000,000 real production microservice traces** from the **Alibaba Cloud 2021 Trace Dataset (SoCC '21)**.

---

## 1. Methodology & Data Ingestion

* **Dataset:** Official Alibaba Cloud Microservices Shards (`MSCallGraph_0.tar.gz` through `MSCallGraph_7.tar.gz`).
* **Scale:** 1,000,000 complete, chronological execution trees processed across thousands of enterprise microservices.
* **Call Tree Reconstruction:** Execution DAGs reconstructed directly from hierarchical `rpcid` dot structures (e.g. `0.1.1` and `0.1.2` as sibling child calls beneath `0.1`).
* **Native 10D Geometric Feature Vector (Zero Synthetic Assumptions):**

$$x(T) = \Big(\text{call\_count},\, \text{max\_depth},\, \text{max\_fanout},\, \text{rpc\_count},\, \text{http\_count},\, \text{db\_count},\, \text{mc\_count},\, \text{mean\_rt},\, \text{max\_rt},\, \text{crit\_path\_rt}\Big)$$

* **Sacred Symbolic Partition:** Entry caller microservice (`um` of root call).
* **FUTCache Configuration:** $\varepsilon = 0.55$, Metric = Euclidean ($L_2$ on normalized space), Backend = VP-Tree.

---

## 2. Empirical Growth Law Results ($N = 50\text{k} \to 1,000,000$)

```
=============================================================================================================
Stream (N)       | Exact Hash (|H|)   | FUTCache (|R|)     | Damping (|R|/|H|)  | Suppression  | FUTCache RAM
-------------------------------------------------------------------------------------------------------------
      50,000    |         19,335    |         17,394    |          89.96%    |      65.21% |      4.34 MB
     100,000    |         36,121    |         32,631    |          90.34%    |      67.37% |      7.99 MB
     250,000    |         84,359    |         76,278    |          90.42%    |      69.49% |     18.40 MB
     500,000    |        161,204    |        145,352    |          90.17%    |      70.93% |     34.82 MB
     750,000    |        235,965    |        212,238    |          89.94%    |      71.70% |     50.69 MB
   1,000,000    |        308,210    |        276,592    |          89.74%    |      72.34% |     65.95 MB
=============================================================================================================
```

---

## 3. Key Scientific Findings

### 1. Monotonically Increasing Production Suppression
As the representative net fills up, suppression rate monotonically increases:

$$\mathbf{65.21\%} \longrightarrow \mathbf{67.37\%} \longrightarrow \mathbf{69.49\%} \longrightarrow \mathbf{70.93\%} \longrightarrow \mathbf{71.70\%} \longrightarrow \mathbf{72.34\%}$$

Across 1,000,000 production requests, **723,408 redundant trace call graphs were suppressed** without storing duplicate payloads.

### 2. Downward Bending of the Damping Ratio ($|R_N| / |H_N|$)
* From $N = 50\text{k}$ to $250\text{k}$, the damping ratio peaked at **$90.42\%$**.
* From $N = 250\text{k}$ to $1,000,000$, the ratio steadily bent downward:

$$90.42\% \longrightarrow 90.17\% \longrightarrow 89.94\% \longrightarrow \mathbf{89.74\%}$$

This demonstrates that geometric $\varepsilon$-ball coverage collapses more marginal states than discrete bucket hashing as scale increases.

### 3. Sublinear State Growth & Low Memory Footprint
* The marginal state addition rate dropped from **$34.79\%$** at $50\text{k}$ to **$27.66\%$** at $1\text{M}$.
* Storing the entire geometric representative set for **1,000,000 enterprise traces** took only **$65.95\text{ MB}$ of RAM**.

---

## 4. How to Reproduce

To run the benchmark on any subset or the full 1-million trace stream:

```bash
# Run on 100,000 traces (uses 1 shard, ~150MB download)
python3 demos/alibaba_1m_scaling_benchmark.py 100000

# Run full 1,000,000 trace benchmark (streams shards 0-7)
python3 demos/alibaba_1m_scaling_benchmark.py 1000000
```
