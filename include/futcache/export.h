#ifndef FUTCACHE_EXPORT_H
#define FUTCACHE_EXPORT_H

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(FUTCACHE_BUILDING_LIBRARY)
#    define FUTCACHE_API __declspec(dllexport)
#  elif defined(FUTCACHE_SHARED)
#    define FUTCACHE_API __declspec(dllimport)
#  else
#    define FUTCACHE_API
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define FUTCACHE_API __attribute__((visibility("default")))
#else
#  define FUTCACHE_API
#endif

#endif
