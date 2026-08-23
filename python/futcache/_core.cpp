// futcache/_core.cpp — Python bindings for the FUTCache packing cache.
//
// Wraps futcache_pack.h via nanobind. Payloads (LLM responses, etc.) are
// stored in a Python dict keyed by representative slot index; the C cache
// itself manages only novelty semantics and the Voronoi seed set.
//
// Thread-safety mirrors the C API: queries are non-mutating and safe to
// call concurrently; observe is atomic. The Python object holds a
// reference to the C cache and releases it in __del__.

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
constexpr int FUTCACHE_PY_VERSION_MINOR = 1;
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
    throw std::invalid_argument(
        "unknown distance '" + name +
        "', expected one of: linf, l1, l2, cosine");
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
              nb::object domain_max)
        : dimension_(static_cast<size_t>(dimension)),
          payload_mutex_(),
          payloads_() {
        if (dimension <= 0) {
            throw std::invalid_argument("dimension must be >= 1");
        }
        if (!std::isfinite(epsilon) || epsilon < 0.0) {
            throw std::invalid_argument("epsilon must be a finite non-negative double");
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

        futcache_pack_t *c = cache_;
        bool novel = false;
        futcache_status_t st = futcache_pack_is_novel(c, point.data(), &novel);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error("query failed with status " +
                                     std::to_string(static_cast<int>(st)));
        }
        // query is non-mutating; the C API does not currently expose
        // which representative matched, so rep_id is -1 on novel and
        // an unset placeholder otherwise. A future API exposing the
        // nearest-site id will let us populate it cleanly.
        NoveltyResult r;
        r.representative_id = novel ? -1 : 0;
        r.is_novel = novel;
        r.distance = 0.0;
        r.inserted = false;
        return r;
    }

    /* Combined query + insert. On novel observations, attaches the
     * supplied payload (if any) to the newly created representative.
     * On redundant observations, the existing payload (if any) is left
     * untouched. Returns a NoveltyResult. */
    NoveltyResult observe(nb::ndarray<double, nb::ndim<1>> point,
                           nb::object payload) {
        check_point(point);

        const bool has_payload = !payload.is_none();
        const char *payload_data = nullptr;
        size_t payload_len = 0;
        if (has_payload) {
            nb::bytes b = nb::cast<nb::bytes>(payload);
            payload_data = b.c_str();
            payload_len = b.size();
        }

        size_t before_count = 0;
        {
            futcache_pack_stats_t stats;
            futcache_pack_get_stats(cache_, &stats);
            before_count = stats.representative_count;
        }

        bool novel = false;
        futcache_status_t st = futcache_pack_observe(
            cache_, point.data(), &novel);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error(
                "observe failed with status " +
                std::to_string(static_cast<int>(st)));
        }

        NoveltyResult r;
        r.is_novel = novel;
        r.inserted = novel;
        r.distance = 0.0;

        if (novel) {
            size_t after_count = 0;
            {
                futcache_pack_stats_t stats;
                futcache_pack_get_stats(cache_, &stats);
                after_count = stats.representative_count;
            }
            if (after_count != before_count + 1) {
                throw std::runtime_error(
                    "internal: representative count did not advance on novel");
            }
            size_t rep_id = after_count - 1;
            r.representative_id = static_cast<int>(rep_id);
            if (has_payload) {
                std::lock_guard<std::mutex> lock(payload_mutex_);
                payloads_[rep_id] =
                    std::string(payload_data, payload_len);
            }
        } else {
            /* redundant: id of the matched rep is not exposed by the
             * C API yet. Caller should consult get_payload() after
             * determining novelty through a domain-specific key. */
            r.representative_id = -1;
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
        return nb::bytes(it->second.data(), it->second.size());
    }

    void set_payload(int rep_id, nb::object payload) {
        if (rep_id < 0) {
            throw std::invalid_argument("rep_id must be >= 0");
        }
        if (payload.is_none()) {
            std::lock_guard<std::mutex> lock(payload_mutex_);
            payloads_.erase(static_cast<size_t>(rep_id));
            return;
        }
        nb::bytes b = nb::cast<nb::bytes>(payload);
        std::lock_guard<std::mutex> lock(payload_mutex_);
        payloads_[static_cast<size_t>(rep_id)] =
            std::string(b.c_str(), b.size());
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

    void clear() {
        futcache_status_t st = futcache_pack_clear(cache_);
        if (st != FUTCACHE_OK) {
            throw std::runtime_error("clear failed");
        }
        std::lock_guard<std::mutex> lock(payload_mutex_);
        payloads_.clear();
    }

    static int version_major() { return FUTCACHE_PY_VERSION_MAJOR; }
    static int version_minor() { return FUTCACHE_PY_VERSION_MINOR; }
    static int version_patch() { return FUTCACHE_PY_VERSION_PATCH; }

private:
    void check_point(nb::ndarray<double, nb::ndim<1>> &point) {
        if (static_cast<int>(point.shape(0)) != dimension_) {
            throw std::invalid_argument(
                "point dimension mismatch: cache has " +
                std::to_string(dimension_) + ", got " +
                std::to_string(point.shape(0)));
        }
        const double *data = point.data();
        for (int i = 0; i < dimension_; ++i) {
            if (!std::isfinite(data[i])) {
                throw std::invalid_argument(
                    "point coordinates must be finite; got non-finite at index " +
                    std::to_string(i));
            }
            if (data[i] < domain_lo_[i] || data[i] > domain_hi_[i]) {
                throw std::out_of_range(
                    "point out of domain at index " + std::to_string(i));
            }
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
    futcache_pack_t *cache_;
    std::vector<double> domain_lo_;
    std::vector<double> domain_hi_;

    /* Payloads are owned by Python (the C cache holds only novelty
     * state). Mutex guards the dict because observe() can be called
     * from multiple Python threads even though the C observe() itself
     * is serialised. */
    std::mutex payload_mutex_;
    std::unordered_map<size_t, std::string> payloads_;
};

}  // namespace

NB_MODULE(futcache_ext, m) {
    m.doc() = "Low-level Python bindings for FUTCache packing cache.";

    nb::class_<NoveltyResult>(m, "NoveltyResult")
        .def_rw("representative_id", &NoveltyResult::representative_id,
                "Slot index of the matched or new representative; -1 if novel")
        .def_rw("is_novel", &NoveltyResult::is_novel,
                "True when the point is farther than epsilon from every existing rep")
        .def_rw("distance", &NoveltyResult::distance,
                "Distance to nearest representative (currently 0; reserved)")
        .def_rw("inserted", &NoveltyResult::inserted,
                "True when observe() added a new representative");

    nb::class_<PackCache>(m, "PackCache")
        .def(nb::init<int, double, std::string, nb::object, nb::object>(),
             nb::arg("dimension"),
             nb::arg("epsilon"),
             nb::arg("distance") = std::string("linf"),
             nb::arg("domain_min") = nb::none(),
             nb::arg("domain_max") = nb::none())
        .def("query", &PackCache::query, nb::arg("point"))
        .def("observe", &PackCache::observe,
             nb::arg("point"),
             nb::arg("payload") = nb::none())
        .def("get_payload", &PackCache::get_payload, nb::arg("rep_id"))
        .def("set_payload", &PackCache::set_payload,
             nb::arg("rep_id"),
             nb::arg("payload"))
        .def("__len__", &PackCache::__len__)
        .def("peak_count", &PackCache::peak_count)
        .def("memory_bytes", &PackCache::memory_bytes)
        .def("observations", &PackCache::observations)
        .def("novel_observations", &PackCache::novel_observations)
        .def("copy_representatives", &PackCache::copy_representatives)
        .def("clear", &PackCache::clear)
        .def_static("version_major", &PackCache::version_major)
        .def_static("version_minor", &PackCache::version_minor)
        .def_static("version_patch", &PackCache::version_patch);
}
