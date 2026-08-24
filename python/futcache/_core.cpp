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
        .def("clear", &PackCache::clear)
        .def_static("version_major", &PackCache::version_major)
        .def_static("version_minor", &PackCache::version_minor)
        .def_static("version_patch", &PackCache::version_patch);
}
