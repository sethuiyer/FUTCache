#include <stdbool.h>

#include <futcache/futcache.h>

int main(void)
{
    futcache_config_t config;
    futcache_t *cache = 0;
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
    return 0;
}
