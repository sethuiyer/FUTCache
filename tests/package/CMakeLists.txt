cmake_minimum_required(VERSION 3.16)
project(FUTCachePackageConsumer LANGUAGES C)

find_package(FUTCache 1 CONFIG REQUIRED)

add_executable(futcache_package_consumer main.c)
target_link_libraries(futcache_package_consumer PRIVATE FUTCache::futcache)
