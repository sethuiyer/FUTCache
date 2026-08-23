#include <stdbool.h>
#include <stdlib.h>

#include <futcache/box.h>
#include <futcache/futcache.h>
#include <futcache/pack.h>

int main(void)
{
    futcache_config_t config;
    futcache_t *cache = 0;
    futcache_pack_config_t pack_config;
    futcache_pack_t *pack = NULL;
    futcache_pack_t *restored_pack = NULL;
    futcache_box_config_t box_config;
    futcache_box_t *box = NULL;
    const double lower[] = {0.0};
    const double upper[] = {1.0};
    double point[] = {0.5};
    bool novel = false;
    size_t snapshot_size = 0U;
    void *snapshot = NULL;

    futcache_config_init(&config);
    if (futcache_create(&config, &cache) != FUTCACHE_OK) {
        return 1;
    }
    if (futcache_observe(cache, 0.5, &novel) != FUTCACHE_OK || !novel) {
        futcache_destroy(cache);
        return 2;
    }
    futcache_destroy(cache);

    futcache_pack_config_init(&pack_config);
    pack_config.domain_min = lower;
    pack_config.domain_max = upper;
    if (futcache_pack_create(&pack_config, &pack) != FUTCACHE_OK ||
        futcache_pack_observe(pack, point, &novel) != FUTCACHE_OK || !novel) {
        futcache_pack_destroy(pack);
        return 3;
    }
    if (futcache_pack_serialize(pack, NULL, 0U, &snapshot_size) !=
        FUTCACHE_OK || (snapshot = malloc(snapshot_size)) == NULL ||
        futcache_pack_serialize(pack, snapshot, snapshot_size,
                                &snapshot_size) != FUTCACHE_OK ||
        futcache_pack_deserialize(snapshot, snapshot_size, NULL,
                                  &restored_pack) != FUTCACHE_OK) {
        free(snapshot);
        futcache_pack_destroy(pack);
        return 4;
    }
    free(snapshot);
    futcache_pack_destroy(restored_pack);
    futcache_pack_destroy(pack);

    futcache_box_config_init(&box_config);
    box_config.domain_min = lower;
    box_config.domain_max = upper;
    if (futcache_box_create(&box_config, &box) != FUTCACHE_OK ||
        futcache_box_observe(box, point, &novel) != FUTCACHE_OK || !novel) {
        futcache_box_destroy(box);
        return 5;
    }
    futcache_box_destroy(box);
    return 0;
}
