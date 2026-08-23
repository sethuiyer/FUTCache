#include <stdbool.h>

#include <futcache/box.h>
#include <futcache/futcache.h>
#include <futcache/pack.h>

int main(void)
{
    futcache_config_t config;
    futcache_t *cache = 0;
    futcache_pack_config_t pack_config;
    futcache_pack_t *pack = NULL;
    futcache_box_config_t box_config;
    futcache_box_t *box = NULL;
    const double lower[] = {0.0};
    const double upper[] = {1.0};
    double point[] = {0.5};
    bool novel = false;

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
    futcache_pack_destroy(pack);

    futcache_box_config_init(&box_config);
    box_config.domain_min = lower;
    box_config.domain_max = upper;
    if (futcache_box_create(&box_config, &box) != FUTCACHE_OK ||
        futcache_box_observe(box, point, &novel) != FUTCACHE_OK || !novel) {
        futcache_box_destroy(box);
        return 4;
    }
    futcache_box_destroy(box);
    return 0;
}
