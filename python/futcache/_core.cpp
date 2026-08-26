// futcache/_core.cpp — Python bindings for the FUTCache packing cache.
//
// Wraps futcache_pack.h via nanobind. Payloads (LLM responses, etc.) are
// stored in a Python dict keyed by representative slot index; the C cache
// itself manages only novelty semantics and the Voronoi seed set.
//
// Thread-safety mirrors the C API: queries are non-mutating and safe to
// call concurrently; observe is atomic. The Python object holds a
// reference to the C cache and releases it in __del__.

#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "futcache/futcache.h"
#include "futcache/pack.h"
#include "futcache/embed.h"
#include "futcache/select.h"
#include "futcache/persist.h"
#include "futcache/persist_nd.h"

namespace nb = nanobind;

namespace {

constexpr int FUTCACHE_PY_VERSION_MAJOR = 1;
constexpr int FUTCACHE_PY_VERSION_MINOR = 4;
constexpr int FUTCACHE_PY_VERSION_PATCH = 0;

futcache_distance_fn resolve_distance(const std::string &name) {
    if (name == "linf" || name == "L_inf" || name == "Linf" ||
        name == "chebyshev") {
        return futcache_distance_linf;
    }
    if (name == "l1" || name == "L1") {
        return futcache_distance_l1;
    }
    if (name == "l2" || name == "L2" || name == "euclidean") {
        return futcache_distance_l2;
    }
    if (name == "cosine" || name == "cos") {
        return futcache_distance_cosine;
    }
    if (name == "poincare" || name == "poincare_ball" ||
        name == "hyperbolic") {
        return futcache_distance_poincare;
    }
    throw std::invalid_argument(
        "unknown distance '" + name +
        "', expected one of: linf, l1, l2, cosine, poincare");
}

struct NoveltyResult {
    int representative_id;
    bool is_novel;
    double distance;
    bool inserted;
};

class PackCache {
public:
    PackCache(int dimension,
              double epsilon,
              std::string distance,
              nb::object domain_min,
              nb::object domain_max,
              std::string backend = "linear",
              size_t max_memory_bytes = 0,
              size_t max_entries = 0,
              double ttl_seconds = 0.0)
        : dimension_(static_cast<size_t>(dimension)),
          epsilon_(epsilon),
          poincare_(distance == "poincare" || distance == "poincare_ball" ||
                    distance == "hyperbolic"),
          payload_mutex_(),
          payloads_(),
          access_time_(),
          insert_time_(),
          max_entries_(max_entries),
          ttl_seconds_(ttl_seconds) {
        if (dimension <= 0) {
            throw std::invalid_argument("dimension must be >= 1");
        }
        if (!std::isfinite(epsilon) || epsilon < 0.0) {
            throw std::invalid_argument("epsilon must be a finite non-negative double");
        }
        if (!std::isfinite(ttl_seconds) || ttl_seconds < 0.0) {
            throw std::invalid_argument("ttl must be a finite non-negative seconds");
        }

        std::vector<double> lo(dimension, -1.0);
        std::vector<double> hi(dimension, 1.0);
        if (!domain_min.is_none()) {
            copy_array(domain_min, lo);
        }
        if (!domain_max.is_none()) {
            copy_array(domain_max, hi);
        }
        for (size_t i = 0; i < lo.size(); ++i) {
            if (!std::isfinite(lo[i]) || !std::isfinite(hi[i]) ||
                hi[i] <= lo[i]) {
                throw std::invalid_argument(
                    "domain bounds must be finite with min < max per coordinate");
            }
        }

        futcache_pack_config_t cfg;
        futcache_pack_config_init(&cfg);
        cfg.dimension = dimension_;
        cfg.epsilon = epsilon;
        cfg.distance = resolve_distance(distance);
        distance_name_ = distance;
        distance_fn_ = cfg.distance;
        distance_context_ = nullptr;
        cfg.distance_context = nullptr;
        cfg.domain_min = lo.data();
        cfg.domain_max = hi.data();
        cfg.allocator.allocate = nullptr;
        cfg.allocator.deallocate = nullptr;
        cfg.allocator.context = nullptr;
        if (backend == "linear") {
            cfg.backend = nullptr;
        } else if (backend == "vptree") {
            cfg.backend = &futcache_pack_vptree_backend;
        } else {
            throw std::invalid_argument(
                "backend must be \"linear\" or \"vptree\"");
        }
        cfg.backend_context = nullptr;
        cfg.max_memory_bytes = max_memory_bytes;

        futcache_pack_t *raw = nullptr;
        futcache_status_t st = futcache_pack_create(&cfg, &raw);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error("futcache_pack_create failed");
        }
        cache_ = raw;
        domain_lo_ = std::move(lo);
        domain_hi_ = std::move(hi);
    }

    PackCache(const PackCache &) = delete;
    PackCache &operator=(const PackCache &) = delete;

    ~PackCache() {
        if (cache_ != nullptr) {
            futcache_pack_destroy(cache_);
            cache_ = nullptr;
        }
    }

    NoveltyResult query(nb::ndarray<double, nb::ndim<1>> point) {
        check_point(point);

        bool found = false;
        double matched_distance = INFINITY;
        size_t matched_idx = SIZE_MAX;
        futcache_status_t st = futcache_pack_lookup(
            cache_, point.data(), &found, &matched_distance, &matched_idx);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error("query failed with status " +
                                     std::to_string(static_cast<int>(st)));
        }
        if (!found) {
            /* Keep the historical NoveltyResult contract on misses: report
             * the nearest-centre distance, even though adaptive decisions are
             * based on containment in representative-owned balls. */
            size_t nearest_idx = SIZE_MAX;
            st = futcache_pack_nearest(
                cache_, point.data(), &matched_distance, &nearest_idx);
            if (st != FUTCACHE_OK) {
                throw std::runtime_error(
                    "nearest lookup failed with status " +
                    std::to_string(static_cast<int>(st)));
            }
        }
        NoveltyResult r;
        r.representative_id = found ? static_cast<int>(matched_idx) : -1;
        r.is_novel = !found;
        r.distance = matched_distance;
        r.inserted = false;
        return r;
    }

    /* Combined query + insert. On novel observations, attaches the
     * supplied payload (if any) to the newly created representative.
     * On redundant observations, the existing payload (if any) is left
     * untouched. Returns a NoveltyResult. */
    NoveltyResult observe(nb::ndarray<double, nb::ndim<1>> point,
                           nb::object payload) {
        return observe_impl(point, epsilon_, payload);
    }

    NoveltyResult observe_with_radius(
        nb::ndarray<double, nb::ndim<1>> point,
        double radius,
        nb::object payload) {
        if (!std::isfinite(radius) || radius < 0.0) {
            throw std::invalid_argument(
                "radius must be a finite non-negative double");
        }
        return observe_impl(point, radius, payload);
    }

    NoveltyResult observe_impl(nb::ndarray<double, nb::ndim<1>> point,
                               double radius,
                               nb::object payload) {
        check_point(point);

        const bool has_payload = !payload.is_none();
        std::string payload_value;
        if (has_payload) {
            nb::bytes b = nb::cast<nb::bytes>(payload);
            payload_value.assign(b.c_str(), b.size());
        }

        /* Serialise Python-level observe bookkeeping with C observe.  This is
         * required when pressure eviction shifts every representative id. */
        std::lock_guard<std::mutex> payload_lock(payload_mutex_);

        size_t before_count = 0;
        {
            futcache_pack_stats_t stats;
            futcache_pack_get_stats(cache_, &stats);
            before_count = stats.representative_count;
        }

        bool novel = false;
        double matched_distance = INFINITY;
        size_t matched_index = SIZE_MAX;
        futcache_status_t st = futcache_pack_observe_with_radius(
            cache_, point.data(), radius, &novel, &matched_distance,
            &matched_index);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                "observe failed with status " +
                std::to_string(static_cast<int>(st)));
        }

        NoveltyResult r;
        r.is_novel = novel;
        r.inserted = novel;
        r.distance = matched_distance;

        if (novel) {
            size_t after_count = 0;
            {
                futcache_pack_stats_t stats;
                futcache_pack_get_stats(cache_, &stats);
                after_count = stats.representative_count;
            }
            const bool appended = after_count == before_count + 1;
            const bool evicted = before_count > 0 &&
                                 after_count == before_count;
            if (!appended && !evicted) {
                throw std::runtime_error(
                    "internal: unexpected representative count after novel observe");
            }
            if (evicted) {
                shift_payloads_locked(before_count);
            }
            size_t rep_id = matched_index;
            if (rep_id != after_count - 1U) {
                throw std::runtime_error(
                    "internal: inserted representative id mismatch");
            }
            r.representative_id = static_cast<int>(rep_id);
            if (has_payload) {
                payloads_[rep_id] = std::move(payload_value);
                touch_locked(rep_id);
                evict_lru_locked();
            }
        } else {
            r.representative_id = static_cast<int>(matched_index);
            /* A geometric hit counts as an access: refresh LRU recency. */
            if (payloads_.find(matched_index) != payloads_.end()) {
                access_time_[matched_index] = now_secs();
            }
        }

        return r;
    }

    nb::object get_payload(int rep_id) {
        if (rep_id < 0) {
            throw std::invalid_argument("rep_id must be >= 0");
        }
        std::lock_guard<std::mutex> lock(payload_mutex_);
        auto it = payloads_.find(static_cast<size_t>(rep_id));
        if (it == payloads_.end()) {
            return nb::none();
        }
        if (payload_expired_locked(static_cast<size_t>(rep_id))) {
            /* Lazy expiry: drop the stale entry, treat as a miss. */
            access_time_.erase(static_cast<size_t>(rep_id));
            insert_time_.erase(static_cast<size_t>(rep_id));
            payloads_.erase(it);
            return nb::none();
        }
        access_time_[static_cast<size_t>(rep_id)] = now_secs();
        return nb::bytes(it->second.data(), it->second.size());
    }

    void set_payload(int rep_id, nb::object payload) {
        if (rep_id < 0) {
            throw std::invalid_argument("rep_id must be >= 0");
        }
        if (payload.is_none()) {
            std::lock_guard<std::mutex> lock(payload_mutex_);
            size_t slot = static_cast<size_t>(rep_id);
            payloads_.erase(slot);
            access_time_.erase(slot);
            insert_time_.erase(slot);
            return;
        }
        nb::bytes b = nb::cast<nb::bytes>(payload);
        std::lock_guard<std::mutex> lock(payload_mutex_);
        size_t slot = static_cast<size_t>(rep_id);
        payloads_[slot] = std::string(b.c_str(), b.size());
        touch_locked(slot);
        evict_lru_locked();
    }

    size_t __len__() const {
        futcache_pack_stats_t stats;
        futcache_pack_get_stats(cache_, &stats);
        return stats.representative_count;
    }

    size_t peak_count() const {
        futcache_pack_stats_t stats;
        futcache_pack_get_stats(cache_, &stats);
        return stats.peak_count;
    }

    size_t memory_bytes() const {
        futcache_pack_stats_t stats;
        futcache_pack_get_stats(cache_, &stats);
        return stats.memory_bytes;
    }

    uint64_t observations() const {
        futcache_pack_stats_t stats;
        futcache_pack_get_stats(cache_, &stats);
        return stats.observations;
    }

    uint64_t novel_observations() const {
        futcache_pack_stats_t stats;
        futcache_pack_get_stats(cache_, &stats);
        return stats.novel_observations;
    }

    uint64_t evictions() const {
        futcache_pack_stats_t stats;
        futcache_pack_get_stats(cache_, &stats);
        return stats.evictions;
    }

    size_t peak_memory_bytes() const {
        futcache_pack_stats_t stats;
        futcache_pack_get_stats(cache_, &stats);
        return stats.peak_memory_bytes;
    }

    size_t memory_limit_bytes() const {
        futcache_pack_stats_t stats;
        futcache_pack_get_stats(cache_, &stats);
        return stats.memory_limit_bytes;
    }

    nb::list copy_representatives() {
        /* Returns a Python list of lists, one per representative.
         * The Python wrapper reshapes to a numpy ndarray of shape
         * (count, dimension) for ergonomic downstream use. Avoids
         * nanobind's static-shape ndarray constructor signature. */
        size_t n = 0;
        futcache_status_t st = futcache_pack_copy_representatives(
            cache_, nullptr, &n);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error("copy_representatives query failed");
        }
        size_t rows = static_cast<size_t>(n);
        size_t cols = dimension_;
        std::vector<double> buf(rows * cols);
        size_t written = rows;
        st = futcache_pack_copy_representatives(
            cache_, buf.data(), &written);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error("copy_representatives copy failed");
        }
        if (written != rows) {
            throw std::runtime_error("copy_representatives size mismatch");
        }
        nb::list out;
        for (size_t i = 0; i < rows; ++i) {
            nb::list row;
            for (size_t j = 0; j < cols; ++j) {
                row.append(buf[i * cols + j]);
            }
            out.append(row);
        }
        return out;
    }

    std::vector<double> copy_radii() {
        size_t count = 0U;
        futcache_status_t status = futcache_pack_copy_radii(
            cache_, nullptr, &count);
        if (status != FUTCACHE_OK) {
            throw std::runtime_error("copy_radii query failed");
        }
        std::vector<double> radii(count);
        size_t written = count;
        status = futcache_pack_copy_radii(cache_, radii.data(), &written);
        if (status != FUTCACHE_OK || written != count) {
            throw std::runtime_error("copy_radii copy failed");
        }
        return radii;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(payload_mutex_);
        futcache_status_t st = futcache_pack_clear(cache_);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error("clear failed");
        }
        payloads_.clear();
        access_time_.clear();
        insert_time_.clear();
    }

    size_t payload_count() {
        std::lock_guard<std::mutex> lock(payload_mutex_);
        return payloads_.size();
    }

    size_t purge() {
        std::lock_guard<std::mutex> lock(payload_mutex_);
        return purge_expired_locked();
    }

    double ttl_seconds() const { return ttl_seconds_; }
    size_t max_entries() const { return max_entries_; }

    /* W1-optimal eviction: evict the representative with the smallest
     * distance to its nearest neighbour (the "most crowded" rep).
     * Returns the slot index that was evicted. */
    size_t evict_w1() {
        std::lock_guard<std::mutex> lock(payload_mutex_);
        size_t evicted = 0;
        futcache_status_t st = futcache_pack_evict_w1(cache_, &evicted);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error("evict_w1 failed with status " +
                                     std::to_string(static_cast<int>(st)));
        }
        /* After eviction, slot indices shift down by 1 for all reps
         * that were after the evicted one. Re-key the payload maps. */
        size_t before_count = evicted + 1; /* rep count before eviction */
        shift_payloads_from(evicted, before_count);
        return evicted;
    }

    /* Returns the W1 "importance" of each rep: its distance to the
     * nearest other rep. Lower = more redundant = evicted first. */
    std::vector<double> rep_importance() {
        size_t n = 0;
        futcache_status_t st = futcache_pack_copy_representatives(
            cache_, nullptr, &n);
        if (st != FUTCACHE_OK) throw std::runtime_error("copy reps failed");
        if (n == 0) return {};
        size_t cols = dimension_;
        std::vector<double> buf(n * cols);
        size_t written = n;
        st = futcache_pack_copy_representatives(cache_, buf.data(), &written);
        if (st != FUTCACHE_OK) throw std::runtime_error("copy reps failed");

        /* Compute nearest-neighbour distance for each rep. */
        std::vector<double> importance(n, INFINITY);
        for (size_t i = 0; i < n; ++i) {
            double best = INFINITY;
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                double d = distance_fn_(
                    buf.data() + i * cols,
                    buf.data() + j * cols,
                    cols, distance_context_);
                if (d < best) best = d;
            }
            importance[i] = best;
        }
        return importance;
    }

    static int version_major() { return FUTCACHE_PY_VERSION_MAJOR; }
    static int version_minor() { return FUTCACHE_PY_VERSION_MINOR; }
    static int version_patch() { return FUTCACHE_PY_VERSION_PATCH; }

private:
    void check_point(nb::ndarray<double, nb::ndim<1>> &point) {
        if (point.shape(0) != dimension_) {
            throw std::invalid_argument(
                "point dimension mismatch: cache has " +
                std::to_string(dimension_) + ", got " +
                std::to_string(point.shape(0)));
        }
        const double *data = point.data();
        double squared_norm = 0.0;
        for (size_t i = 0U; i < dimension_; ++i) {
            if (!std::isfinite(data[i])) {
                throw std::invalid_argument(
                    "point coordinates must be finite; got non-finite at index " +
                    std::to_string(i));
            }
            if (data[i] < domain_lo_[i] || data[i] > domain_hi_[i]) {
                throw std::out_of_range(
                    "point out of domain at index " + std::to_string(i));
            }
            if (poincare_) squared_norm += data[i] * data[i];
        }
        if (poincare_ && !(squared_norm < 1.0)) {
            throw std::out_of_range(
                "Poincare points must have Euclidean norm strictly below 1");
        }
    }

    static void copy_array(const nb::object &src, std::vector<double> &dst) {
        nb::ndarray<double, nb::ndim<1>> arr =
            nb::cast<nb::ndarray<double, nb::ndim<1>>>(src);
        if (static_cast<int>(arr.shape(0)) != static_cast<int>(dst.size())) {
            throw std::invalid_argument(
                "domain array length must equal dimension");
        }
        const double *data = arr.data();
        for (size_t i = 0; i < dst.size(); ++i) dst[i] = data[i];
    }

    size_t dimension_;
    double epsilon_;
    bool poincare_;
    futcache_pack_t *cache_;
    std::vector<double> domain_lo_;
    std::vector<double> domain_hi_;

    /* Payloads are owned by Python (the C cache holds only novelty
     * state). Mutex guards the dict because observe() can be called
     * from multiple Python threads even though the C observe() itself
     * is serialised. Access/insert times feed the optional LRU + TTL
     * eviction policies. */
    std::mutex payload_mutex_;
    std::unordered_map<size_t, std::string> payloads_;
    std::unordered_map<size_t, double> access_time_;
    std::unordered_map<size_t, double> insert_time_;
    size_t max_entries_;   /* 0 = unlimited payload entries            */
    double ttl_seconds_;   /* 0.0 = no expiry                          */
    std::string distance_name_;
    futcache_distance_fn distance_fn_;
    void *distance_context_ = nullptr;

    static double now_secs() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    bool payload_expired_locked(size_t slot) const {
        if (ttl_seconds_ <= 0.0) return false;
        auto it = insert_time_.find(slot);
        if (it == insert_time_.end()) return true; /* stale: no known birth */
        return now_secs() - it->second > ttl_seconds_;
    }

    /* Shift payload/timestamp keys down by one for slots >= evicted_index,
     * exactly as the C W1 eviction renumbers representatives. */
    void shift_payloads_from(size_t evicted_index, size_t before_count) {
        std::unordered_map<size_t, std::string> new_p;
        std::unordered_map<size_t, double> new_a;
        std::unordered_map<size_t, double> new_i;
        new_p.reserve(payloads_.size());
        for (auto &e : payloads_) {
            if (e.first == evicted_index) continue;
            size_t new_key = e.first;
            if (e.first > evicted_index) new_key = e.first - 1U;
            new_p.emplace(new_key, std::move(e.second));
        }
        for (auto &e : access_time_) {
            if (e.first == evicted_index) continue;
            size_t new_key = e.first;
            if (e.first > evicted_index) new_key = e.first - 1U;
            new_a.emplace(new_key, e.second);
        }
        for (auto &e : insert_time_) {
            if (e.first == evicted_index) continue;
            size_t new_key = e.first;
            if (e.first > evicted_index) new_key = e.first - 1U;
            new_i.emplace(new_key, e.second);
        }
        payloads_.swap(new_p);
        access_time_.swap(new_a);
        insert_time_.swap(new_i);
    }

    /* Shift every payload/id/timestamp map down by one slot (drop slot 0)
     * exactly as the C FIFO pressure eviction renumbers representatives. */
    void shift_payloads_locked(size_t before_count) {
        std::unordered_map<size_t, std::string> new_p;
        std::unordered_map<size_t, double> new_a;
        std::unordered_map<size_t, double> new_i;
        new_p.reserve(payloads_.size());
        for (auto &e : payloads_) {
            if (e.first > 0 && e.first < before_count)
                new_p.emplace(e.first - 1U, std::move(e.second));
        }
        for (auto &e : access_time_) {
            if (e.first > 0 && e.first < before_count)
                new_a.emplace(e.first - 1U, e.second);
        }
        for (auto &e : insert_time_) {
            if (e.first > 0 && e.first < before_count)
                new_i.emplace(e.first - 1U, e.second);
        }
        payloads_.swap(new_p);
        access_time_.swap(new_a);
        insert_time_.swap(new_i);
    }

    /* Enforce the LRU payload cap (0 = unlimited). Calls the size check
     * after any growth; no-op otherwise. */
    void evict_lru_locked() {
        if (max_entries_ == 0 || payloads_.size() <= max_entries_) return;
        while (payloads_.size() > max_entries_) {
            size_t victim = SIZE_MAX;
            double least = INFINITY;
            for (auto &kv : payloads_) {
                auto it = access_time_.find(kv.first);
                double t = it != access_time_.end() ? it->second : 0.0;
                if (t < least) { least = t; victim = kv.first; }
            }
            if (victim == SIZE_MAX) return;
            payloads_.erase(victim);
            access_time_.erase(victim);
            insert_time_.erase(victim);
        }
    }

    size_t purge_expired_locked() {
        if (ttl_seconds_ <= 0.0) return 0;
        size_t removed = 0;
        for (auto it = payloads_.begin(); it != payloads_.end(); ) {
            if (payload_expired_locked(it->first)) {
                access_time_.erase(it->first);
                insert_time_.erase(it->first);
                it = payloads_.erase(it);
                ++removed;
            } else {
                ++it;
            }
        }
        return removed;
    }

    void touch_locked(size_t slot) {
        access_time_[slot] = now_secs();
        insert_time_[slot] = now_secs();
    }
};

class AnchorEmbedding {
public:
    AnchorEmbedding(int dimension,
                    std::vector<double> anchors,
                    std::string distance,
                    std::vector<double> domain_min,
                    std::vector<double> domain_max)
        : dimension_(static_cast<size_t>(dimension)),
          distance_name_(distance) {
        if (dimension_ == 0U) {
            throw std::invalid_argument("dimension must be >= 1");
        }
        if (anchors.empty()) {
            throw std::invalid_argument("anchors must be non-empty");
        }
        if (anchors.size() % dimension_ != 0U) {
            throw std::invalid_argument(
                "anchors.size must be a multiple of dimension");
        }
        anchor_count_ = anchors.size() / dimension_;

        if (domain_min.size() != dimension_ || domain_max.size() != dimension_) {
            throw std::invalid_argument(
                "domain bounds must have length == dimension");
        }
        for (size_t i = 0; i < dimension_; ++i) {
            if (domain_max[i] <= domain_min[i]) {
                throw std::invalid_argument(
                    "domain bounds: min must be < max per coordinate");
            }
        }

        futcache_distance_fn dist_fn = resolve_distance(distance);

        futcache_embed_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.dimension = dimension_;
        cfg.anchor_count = anchor_count_;
        cfg.anchors = anchors.data();
        cfg.distance = dist_fn;
        cfg.distance_context = nullptr;
        cfg.domain_min = domain_min.data();
        cfg.domain_max = domain_max.data();

        futcache_status_t st = futcache_embed_create(&cfg, &embed_);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("embed create failed: ") +
                futcache_status_string(st));
        }

        covering_radius_ = futcache_embed_covering_radius(embed_);
        anchors_ = std::move(anchors);
        domain_min_ = std::move(domain_min);
        domain_max_ = std::move(domain_max);
    }

    ~AnchorEmbedding() {
        if (embed_ != nullptr) {
            futcache_embed_destroy(embed_);
        }
    }

    std::vector<double> embed(const std::vector<double> &point) const {
        if (point.size() != dimension_) {
            throw std::invalid_argument(
                "point dimension mismatch: expected " +
                std::to_string(dimension_) +
                ", got " + std::to_string(point.size()));
        }
        std::vector<double> result(anchor_count_);
        futcache_status_t st = futcache_embed_point(
            embed_, point.data(), result.data());
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("embed_point failed: ") +
                futcache_status_string(st));
        }
        return result;
    }

    double covering_radius() const { return covering_radius_; }
    size_t anchor_count() const { return anchor_count_; }
    size_t dimension() const { return dimension_; }

    double adjusted_epsilon(double epsilon_original) const {
        double out;
        futcache_status_t st = futcache_embed_adjusted_epsilon(
            embed_, epsilon_original, &out);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("adjusted_epsilon failed: ") +
                futcache_status_string(st));
        }
        return out;
    }

private:
    size_t dimension_;
    size_t anchor_count_;
    futcache_embed_t *embed_ = nullptr;
    double covering_radius_;
    std::string distance_name_;
    std::vector<double> anchors_;
    std::vector<double> domain_min_;
    std::vector<double> domain_max_;
};

/* PersistentNovelty wrapper */
class PersistentNovelty {
   public:
    PersistentNovelty() {
        futcache_persist_config_t cfg = {};
        futcache_status_t st =
            futcache_persist_create(&cfg, &engine_);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("futcache_persist_create failed: ") +
                futcache_status_string(st));
        }
    }

    ~PersistentNovelty() {
        if (engine_ != nullptr) {
            futcache_persist_destroy(engine_);
            engine_ = nullptr;
        }
    }

    /* Non-copyable */
    PersistentNovelty(const PersistentNovelty &) = delete;
    PersistentNovelty &operator=(const PersistentNovelty &) = delete;

    void observe(double x) {
        futcache_status_t st = futcache_persist_observe(engine_, x);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("futcache_persist_observe failed: ") +
                futcache_status_string(st));
        }
    }

    bool is_novel_at(double x, double t) const {
        bool novel;
        futcache_status_t st =
            futcache_persist_is_novel_at(engine_, x, t, &novel);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("futcache_persist_is_novel_at failed: ") +
                futcache_status_string(st));
        }
        return novel;
    }

    std::vector<double> novelty_spectrum(double x) const {
        size_t count = 0;
        futcache_status_t st = futcache_persist_novelty_spectrum(
            engine_, x, nullptr, &count);
        if (st != FUTCACHE_OK && st != FUTCACHE_ERROR_BUFFER_TOO_SMALL) {
            throw std::runtime_error(
                std::string("novelty_spectrum query failed: ") +
                futcache_status_string(st));
        }
        std::vector<double> buf(count * 2);
        st = futcache_persist_novelty_spectrum(engine_, x, buf.data(), &count);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("novelty_spectrum fill failed: ") +
                futcache_status_string(st));
        }
        buf.resize(count * 2);
        return buf;
    }

    std::vector<std::vector<double>> copy_diagram() const {
        size_t count = 0;
        futcache_status_t st = futcache_persist_copy_diagram(
            engine_, nullptr, &count);
        if (st != FUTCACHE_OK && st != FUTCACHE_ERROR_BUFFER_TOO_SMALL) {
            throw std::runtime_error(
                std::string("copy_diagram query failed: ") +
                futcache_status_string(st));
        }
        std::vector<futcache_persist_feature_t> buf(count);
        st = futcache_persist_copy_diagram(engine_, buf.data(), &count);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("copy_diagram fill failed: ") +
                futcache_status_string(st));
        }
        // Convert to vector of vectors:
        // [birth, death, birth_prime, death_prime, birth_value, death_value, persistence]
        std::vector<std::vector<double>> result;
        result.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            result.push_back({
                (double)buf[i].birth,
                (double)buf[i].death,
                (double)buf[i].birth_prime,
                (double)buf[i].death_prime,
                buf[i].birth_value,
                buf[i].death_value,
                buf[i].persistence
            });
        }
        return result;
    }

    std::vector<futcache_persist_feature_t> copy_diagram_raw() const {
        size_t count = 0;
        futcache_status_t st = futcache_persist_copy_diagram(
            engine_, nullptr, &count);
        if (st != FUTCACHE_OK && st != FUTCACHE_ERROR_BUFFER_TOO_SMALL) {
            throw std::runtime_error(
                std::string("copy_diagram query failed: ") +
                futcache_status_string(st));
        }
        std::vector<futcache_persist_feature_t> buf(count);
        st = futcache_persist_copy_diagram(engine_, buf.data(), &count);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("copy_diagram fill failed: ") +
                futcache_status_string(st));
        }
        buf.resize(count);
        return buf;
    }

    std::vector<futcache_persist_feature_t> merge_with(
        const PersistentNovelty &other) const {
        auto a = copy_diagram_raw();
        auto b = other.copy_diagram_raw();
        size_t out_count = a.size() + b.size();
        std::vector<futcache_persist_feature_t> out(out_count);
        futcache_status_t st = futcache_persist_merge_features(
            a.data(), a.size(), b.data(), b.size(), out.data(), &out_count);
        if (st != FUTCACHE_OK && st != FUTCACHE_ERROR_BUFFER_TOO_SMALL) {
            throw std::runtime_error(
                std::string("merge_features failed: ") +
                futcache_status_string(st));
        }
        out.resize(out_count);
        return out;
    }

    double selberg_zeta(double s) const {
        double zeta;
        futcache_status_t st =
            futcache_persist_selberg_zeta(engine_, s, &zeta);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("selberg_zeta failed: ") +
                futcache_status_string(st));
        }
        return zeta;
    }

    size_t prime_cycle_count(double tau = 0.0) const {
        size_t count;
        futcache_status_t st =
            futcache_persist_prime_cycle_count(engine_, tau, &count);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("prime_cycle_count failed: ") +
                futcache_status_string(st));
        }
        return count;
    }

    size_t feature_count(double tau = 0.0) const {
        size_t count;
        futcache_status_t st =
            futcache_persist_feature_count(engine_, tau, &count);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("feature_count failed: ") +
                futcache_status_string(st));
        }
        return count;
    }

    void clear() { futcache_persist_clear(engine_); }

    nb::dict stats() const {
        futcache_persist_stats_t s;
        futcache_status_t st = futcache_persist_get_stats(engine_, &s);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("get_stats failed: ") +
                futcache_status_string(st));
        }
        nb::dict d;
        d["observations"] =
            (uint64_t)s.observations;
        d["feature_count"] = (size_t)s.feature_count;
        d["prime_cycle_count"] = (size_t)s.prime_cycle_count;
        d["max_persistence"] = (double)s.max_persistence;
        d["min_persistence"] = (double)s.min_persistence;
        d["total_persistence"] = (double)s.total_persistence;
        d["memory_bytes"] = (size_t)s.memory_bytes;
        return d;
    }

    uint64_t observations() const {
        futcache_persist_stats_t s;
        if (futcache_persist_get_stats(engine_, &s) == FUTCACHE_OK) {
            return s.observations;
        }
        return 0;
    }

   private:
    futcache_persist_t *engine_ = nullptr;
};

/* PersistentNoveltyND wrapper (d-D persistent packing, Design Sketch 01 Phase 3) */
class PersistentNoveltyND {
   public:
    PersistentNoveltyND(int dimension,
                        double epsilon,
                        std::string distance,
                        std::vector<double> domain_min,
                        std::vector<double> domain_max)
        : dimension_(static_cast<size_t>(dimension)),
          distance_name_(distance) {
        if (dimension_ == 0U) {
            throw std::invalid_argument("dimension must be >= 1");
        }
        if (domain_min.size() != dimension_ || domain_max.size() != dimension_) {
            throw std::invalid_argument("domain bounds must have length == dimension");
        }
        for (size_t i = 0; i < dimension_; ++i) {
            if (!std::isfinite(domain_min[i]) || !std::isfinite(domain_max[i]) ||
                domain_max[i] <= domain_min[i]) {
                throw std::invalid_argument("domain bounds: finite, min < max per coord");
            }
        }
        futcache_distance_fn dist_fn = resolve_distance(distance);
        futcache_status_t st = futcache_persist_nd_create(
            dimension_, epsilon, dist_fn, nullptr,
            domain_min.data(), domain_max.data(), 0, nullptr, &engine_);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("persist_nd create failed: ") +
                futcache_status_string(st));
        }
    }

    ~PersistentNoveltyND() {
        if (engine_ != nullptr) {
            futcache_persist_nd_destroy(engine_);
            engine_ = nullptr;
        }
    }

    PersistentNoveltyND(const PersistentNoveltyND &) = delete;
    PersistentNoveltyND &operator=(const PersistentNoveltyND &) = delete;

    bool observe(std::vector<double> x) {
        check_dim(x);
        bool novel;
        futcache_status_t st = futcache_persist_nd_observe(
            engine_, x.data(), &novel);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("observe failed: ") + futcache_status_string(st));
        }
        return novel;
    }

    bool is_novel_at(std::vector<double> x, double t) const {
        check_dim(x);
        bool novel;
        futcache_status_t st = futcache_persist_nd_is_novel_at(
            engine_, x.data(), t, &novel);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("is_novel_at failed: ") + futcache_status_string(st));
        }
        return novel;
    }

    std::vector<double> nearest_distances() const {
        size_t count = 0;
        futcache_status_t st = futcache_persist_nd_nearest_distances(
            engine_, nullptr, &count);
        if (st != FUTCACHE_OK && st != FUTCACHE_ERROR_BUFFER_TOO_SMALL) {
            throw std::runtime_error(
                std::string("nearest_distances query failed: ") +
                futcache_status_string(st));
        }
        std::vector<double> buf(count);
        st = futcache_persist_nd_nearest_distances(engine_, buf.data(), &count);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("nearest_distances fill failed: ") +
                futcache_status_string(st));
        }
        buf.resize(count);
        return buf;
    }

    std::vector<double> persistences() const {
        size_t count = 0;
        futcache_status_t st = futcache_persist_nd_persistences(
            engine_, nullptr, &count);
        if (st != FUTCACHE_OK && st != FUTCACHE_ERROR_BUFFER_TOO_SMALL) {
            throw std::runtime_error(
                std::string("persistences query failed: ") +
                futcache_status_string(st));
        }
        std::vector<double> buf(count);
        st = futcache_persist_nd_persistences(engine_, buf.data(), &count);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("persistences fill failed: ") +
                futcache_status_string(st));
        }
        buf.resize(count);
        return buf;
    }

    size_t evict_lowest() {
        size_t evicted;
        futcache_status_t st = futcache_persist_nd_evict_lowest(engine_, &evicted);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("evict_lowest failed: ") + futcache_status_string(st));
        }
        return evicted;
    }

    size_t count_above(double tau) const {
        size_t count;
        futcache_status_t st = futcache_persist_nd_count_above(engine_, tau, &count);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("count_above failed: ") + futcache_status_string(st));
        }
        return count;
    }

    void clear() {
        futcache_persist_nd_clear(engine_);
    }

    nb::dict stats() const {
        futcache_persist_nd_stats_t s;
        futcache_status_t st = futcache_persist_nd_get_stats(engine_, &s);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                std::string("get_stats failed: ") + futcache_status_string(st));
        }
        nb::dict d;
        d["observations"] = (uint64_t)s.observations;
        d["rep_count"] = (size_t)s.rep_count;
        d["max_persistence"] = (double)s.max_persistence;
        d["min_persistence"] = (double)s.min_persistence;
        d["avg_persistence"] = (double)s.avg_persistence;
        d["prime_birth_count"] = (size_t)s.prime_birth_count;
        d["memory_bytes"] = (size_t)s.memory_bytes;
        return d;
    }

    size_t dimension() const { return dimension_; }
    size_t rep_count() const {
        futcache_persist_nd_stats_t s;
        if (futcache_persist_nd_get_stats(engine_, &s) == FUTCACHE_OK) {
            return s.rep_count;
        }
        return 0;
    }

   private:
    void check_dim(const std::vector<double> &x) const {
        if (x.size() != dimension_) {
            throw std::invalid_argument(
                "point dimension mismatch: engine has " +
                std::to_string(dimension_) +
                ", got " + std::to_string(x.size()));
        }
    }

    size_t dimension_;
    std::string distance_name_;
    futcache_persist_nd_t *engine_ = nullptr;
};

}  // namespace

NB_MODULE(futcache_ext, m) {
    m.doc() = "Low-level Python bindings for FUTCache packing cache.";

    nb::class_<NoveltyResult>(m, "NoveltyResult")
        .def_rw("representative_id", &NoveltyResult::representative_id,
                "Matched/new slot index; -1 only for a novel non-mutating query")
        .def_rw("is_novel", &NoveltyResult::is_novel,
                "True when the point lies outside every stored representative ball")
        .def_rw("distance", &NoveltyResult::distance,
                "Closest containing distance on a hit; nearest-centre distance on a query miss; 0.0 for insertion")
        .def_rw("inserted", &NoveltyResult::inserted,
                "True when observe() added a new representative");

    nb::class_<PackCache>(m, "PackCache")
        .def(nb::init<int, double, std::string, nb::object, nb::object,
                      std::string, size_t, size_t, double>(),
             nb::arg("dimension"),
             nb::arg("epsilon"),
             nb::arg("distance") = std::string("linf"),
             nb::arg("domain_min") = nb::none(),
             nb::arg("domain_max") = nb::none(),
             nb::arg("backend") = std::string("linear"),
             nb::arg("max_memory_bytes") = 0U,
             nb::arg("max_entries") = 0U,
             nb::arg("ttl") = 0.0)
        .def("query", &PackCache::query, nb::arg("point"))
        .def("observe", &PackCache::observe,
             nb::arg("point"),
             nb::arg("payload") = nb::none())
        .def("observe_with_radius", &PackCache::observe_with_radius,
             nb::arg("point"),
             nb::arg("radius"),
             nb::arg("payload") = nb::none())
        .def("get_payload", &PackCache::get_payload, nb::arg("rep_id"))
        .def("set_payload", &PackCache::set_payload,
             nb::arg("rep_id"),
             nb::arg("payload"))
        .def("payload_count", &PackCache::payload_count)
        .def("purge", &PackCache::purge)
        .def("ttl_seconds", &PackCache::ttl_seconds)
        .def("max_entries", &PackCache::max_entries)
        .def("__len__", &PackCache::__len__)
        .def("peak_count", &PackCache::peak_count)
        .def("memory_bytes", &PackCache::memory_bytes)
        .def("observations", &PackCache::observations)
        .def("novel_observations", &PackCache::novel_observations)
        .def("evictions", &PackCache::evictions)
        .def("peak_memory_bytes", &PackCache::peak_memory_bytes)
        .def("memory_limit_bytes", &PackCache::memory_limit_bytes)
        .def("copy_representatives", &PackCache::copy_representatives)
        .def("copy_radii", &PackCache::copy_radii)
        .def("evict_w1", &PackCache::evict_w1,
             "W1-optimal eviction: remove the rep with the smallest nearest-neighbour distance.")
        .def("rep_importance", &PackCache::rep_importance,
             "Per-rep nearest-neighbour distance (lower = more redundant).")
        .def("clear", &PackCache::clear)
        .def_static("version_major", &PackCache::version_major)
        .def_static("version_minor", &PackCache::version_minor)
        .def_static("version_patch", &PackCache::version_patch);

    nb::class_<AnchorEmbedding>(m, "AnchorEmbedding")
        .def(nb::init<int, std::vector<double>, std::string,
                      std::vector<double>, std::vector<double>>(),
             nb::arg("dimension"),
             nb::arg("anchors"),
             nb::arg("distance") = std::string("linf"),
             nb::arg("domain_min"),
             nb::arg("domain_max"))
        .def("embed", &AnchorEmbedding::embed, nb::arg("point"),
             "phi(x) = (d(x, a_1), ..., d(x, a_m)) — one coordinate per anchor.")
        .def("covering_radius", &AnchorEmbedding::covering_radius,
             "Estimated covering radius delta (lower bound). Distortion is 2*delta.")
        .def("anchor_count", &AnchorEmbedding::anchor_count,
             "Number of anchors (embedded dimension m).")
        .def("dimension", &AnchorEmbedding::dimension,
             "Original-space dimension d.")
        .def("adjusted_epsilon", &AnchorEmbedding::adjusted_epsilon,
             nb::arg("epsilon"),
             "Conservative embedded epsilon = epsilon - 2*delta (no false positives at original epsilon).");

    /* --- Submodular representative selection (Design Sketch 03) --- */

    /* Max-coverage selection: given n points and a budget k, select k reps
     * that maximize the number of points within epsilon of at least one rep.
     * Returns a dict with 'indices', 'coverage', 'total', 'ratio',
     * 'marginal_gains', 'opt_coverage', 'approx_ratio'. */
    m.def("select_max_coverage",
          [](nb::ndarray<double, nb::ndim<1>> points,
             size_t n, size_t dimension, double epsilon, size_t k,
             std::string distance) {
              if (points.size() != n * dimension) {
                  throw std::invalid_argument(
                      "points.size must be n * dimension");
              }
              futcache_distance_fn dist_fn = resolve_distance(distance);
              futcache_select_result_t *res = nullptr;
              futcache_status_t st = futcache_select_max_coverage(
                  points.data(), n, dimension, epsilon, k,
                  dist_fn, nullptr, &res);
              if (st != FUTCACHE_OK) {
                  throw std::runtime_error(
                      std::string("select_max_coverage failed: ") +
                      futcache_status_string(st));
              }
              nb::dict d;
              std::vector<size_t> idx(res->selected_indices,
                                       res->selected_indices + res->selected_count);
              d["indices"] = nb::cast(idx);
              d["selected_count"] = nb::cast(res->selected_count);
              d["total_covered"] = nb::cast(res->total_covered);
              d["total_points"] = nb::cast(res->total_points);
              d["coverage_ratio"] = nb::cast(res->coverage_ratio);
              std::vector<double> mg(res->marginal_gains,
                                     res->marginal_gains + res->selected_count);
              d["marginal_gains"] = nb::cast(mg);
              d["opt_coverage"] = nb::cast(res->opt_coverage);
              d["approx_ratio"] = nb::cast(res->approximation_ratio);
              futcache_select_free_result(res);
              return d;
          },
          nb::arg("points"), nb::arg("n"), nb::arg("dimension"),
          nb::arg("epsilon"), nb::arg("k"),
          nb::arg("distance") = std::string("linf"),
          "Submodular max-coverage selection with 1-1/e guarantee.");

    /* Coverage of a given rep set over observed points. */
    m.def("select_coverage",
          [](nb::ndarray<double, nb::ndim<1>> points,
             size_t n, size_t dimension, double epsilon,
             nb::ndarray<double, nb::ndim<2>> reps,
             std::string distance) {
              size_t rep_count = reps.shape(0);
              futcache_distance_fn dist_fn = resolve_distance(distance);
              double coverage;
              futcache_status_t st = futcache_select_coverage(
                  points.data(), n,
                  reps.data(), rep_count, dimension, epsilon,
                  dist_fn, nullptr, &coverage);
              if (st != FUTCACHE_OK) {
                  throw std::runtime_error(
                      std::string("select_coverage failed: ") +
                      futcache_status_string(st));
              }
              return coverage;
          },
          nb::arg("points"), nb::arg("n"), nb::arg("dimension"),
          nb::arg("epsilon"), nb::arg("reps"),
          nb::arg("distance") = std::string("linf"),
          "Fraction of observed points within epsilon of at least one rep.");

    /* Streaming swap eviction: find the rep with lowest marginal coverage. */
    m.def("select_evict_worst",
          [](nb::ndarray<double, nb::ndim<1>> points,
             size_t n, size_t dimension, double epsilon,
             nb::ndarray<double, nb::ndim<2>> reps,
             std::string distance) {
              size_t rep_count = reps.shape(0);
              futcache_distance_fn dist_fn = resolve_distance(distance);
              size_t evict_idx;
              double marginal_loss;
              futcache_status_t st = futcache_select_evict_worst(
                  points.data(), n,
                  reps.data(), rep_count, dimension, epsilon,
                  dist_fn, nullptr, &evict_idx, &marginal_loss);
              if (st != FUTCACHE_OK) {
                  throw std::runtime_error(
                      std::string("select_evict_worst failed: ") +
                      futcache_status_string(st));
              }
              nb::dict d;
              d["evict_index"] = nb::cast(evict_idx);
              d["marginal_loss"] = nb::cast(marginal_loss);
              return d;
          },
          nb::arg("points"), nb::arg("n"), nb::arg("dimension"),
          nb::arg("epsilon"), nb::arg("reps"),
          nb::arg("distance") = std::string("linf"),
          "Find the rep with lowest marginal coverage (streaming swap). Returns 'evict_index' and 'marginal_loss'.");

    /* --- Persistent novelty (Design Sketch 01) --- */

    nb::class_<PersistentNovelty>(m, "PersistentNovelty")
        .def(nb::init<>())
        .def("observe", &PersistentNovelty::observe, nb::arg("x"))
        .def("is_novel_at", &PersistentNovelty::is_novel_at,
             nb::arg("x"), nb::arg("t"))
        .def("novelty_spectrum", &PersistentNovelty::novelty_spectrum,
             nb::arg("x"),
             "Return [0, t_max] or [] if x is observed.")
        .def("copy_diagram", &PersistentNovelty::copy_diagram,
             "Return list of [birth, death, birth_prime, death_prime, birth_value, death_value, persistence].")
        .def("merge", [](const PersistentNovelty &self,
                         const PersistentNovelty &other) {
             auto a = self.copy_diagram_raw();
             auto b = other.copy_diagram_raw();
             size_t out_count = a.size() + b.size();
             std::vector<futcache_persist_feature_t> out(out_count);
             futcache_status_t st = futcache_persist_merge_features(
                 a.data(), a.size(), b.data(), b.size(),
                 out.data(), &out_count);
             if (st != FUTCACHE_OK &&
                 st != FUTCACHE_ERROR_BUFFER_TOO_SMALL) {
                 throw std::runtime_error(std::string("merge failed: ") +
                                          futcache_status_string(st));
             }
             out.resize(out_count);
             // Convert to vector<vector<double>> for Python
             std::vector<std::vector<double>> result;
             result.reserve(out.size());
             for (size_t i = 0; i < out.size(); ++i) {
                 result.push_back({
                     (double)out[i].birth, (double)out[i].death,
                     (double)out[i].birth_prime, (double)out[i].death_prime,
                     out[i].birth_value, out[i].death_value,
                     out[i].persistence
                 });
             }
             return result;
         }, nb::arg("other"),
             "CRDT merge: union of features (idempotent, commutative).")
        .def("selberg_zeta", &PersistentNovelty::selberg_zeta,
             nb::arg("s"))
        .def("prime_cycle_count", &PersistentNovelty::prime_cycle_count,
             nb::arg("tau") = 0.0)
        .def("feature_count", &PersistentNovelty::feature_count,
             nb::arg("tau") = 0.0)
        .def("clear", &PersistentNovelty::clear)
        .def("stats", &PersistentNovelty::stats)
        .def("observations", &PersistentNovelty::observations)
        .def("__repr__", [](const PersistentNovelty &e) {
             char buf[256];
             snprintf(buf, sizeof(buf),
                      "PersistentNovelty(observations=%llu)",
                      (unsigned long long)e.observations());
             return std::string(buf);
         });

    /* Prime table accessors */
    m.def("nth_prime", [](size_t i) {
        return (size_t)futcache_persist_nth_prime(i);
    }, nb::arg("i"), "Return the i-th prime (0-indexed). p_0=2, p_1=3, ...");

    /* CRDT merge of two diagrams (free function) */
    m.def("merge_persistence_diagrams",
          [](const std::vector<futcache_persist_feature_t> &a,
             const std::vector<futcache_persist_feature_t> &b) {
              size_t out_count = a.size() + b.size();
              std::vector<futcache_persist_feature_t> out(out_count);
              futcache_status_t st = futcache_persist_merge_features(
                  a.data(), a.size(), b.data(), b.size(),
                  out.data(), &out_count);
              if (st != FUTCACHE_OK &&
                  st != FUTCACHE_ERROR_BUFFER_TOO_SMALL) {
                  throw std::runtime_error(std::string("merge failed: ") +
                                           futcache_status_string(st));
              }
              out.resize(out_count);
              return out;
          },
          nb::arg("diagram_a"), nb::arg("diagram_b"),
          "CRDT merge: union of two persistence diagrams.");

    /* --- d-D persistent novelty (Design Sketch 01, Phase 3) --- */
    nb::class_<PersistentNoveltyND>(m, "PersistentNoveltyND")
        .def(nb::init<int, double, std::string, std::vector<double>, std::vector<double>>(),
             nb::arg("dimension"), nb::arg("epsilon"),
             nb::arg("distance") = std::string("linf"),
             nb::arg("domain_min"), nb::arg("domain_max"),
             "d-D persistent novelty engine. Tracks per-rep birth, nearest distance, and persistence.")
        .def("observe", &PersistentNoveltyND::observe,
             nb::arg("point"), "Observe a point. Returns true if novel.")
        .def("is_novel_at", &PersistentNoveltyND::is_novel_at,
             nb::arg("point"), nb::arg("t"),
             "Is the point novel at scale t (distance threshold)?")
        .def("nearest_distances", &PersistentNoveltyND::nearest_distances,
             "Nearest-neighbour distance for each rep.")
        .def("persistences", &PersistentNoveltyND::persistences,
             "Persistence = nearest_dist - radius for each rep.")
        .def("evict_lowest", &PersistentNoveltyND::evict_lowest,
             "Evict the rep with the lowest persistence. Returns evicted index.")
        .def("count_above", &PersistentNoveltyND::count_above,
             nb::arg("tau"), "Count reps with persistence >= tau.")
        .def("rep_count", &PersistentNoveltyND::rep_count)
        .def("stats", &PersistentNoveltyND::stats)
        .def("clear", &PersistentNoveltyND::clear)
        .def("__repr__", [](const PersistentNoveltyND &e) {
             char buf[256];
             snprintf(buf, sizeof(buf),
                      "PersistentNoveltyND(dim=%zu, reps=%zu)",
                      e.dimension(), e.rep_count());
             return std::string(buf);
         });

    m.attr("__version__") =
        std::to_string(FUTCACHE_PY_VERSION_MAJOR) + "." +
        std::to_string(FUTCACHE_PY_VERSION_MINOR) + "." +
        std::to_string(FUTCACHE_PY_VERSION_PATCH);
}
