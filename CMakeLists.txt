cmake_minimum_required(VERSION 3.16)

project(
  FUTCache
  VERSION 1.0.0
  DESCRIPTION "Future-equivalence novelty caches"
  LANGUAGES C
)

include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)

option(FUTCACHE_BUILD_SHARED "Build FUTCache as a shared library" OFF)
option(FUTCACHE_BUILD_EXAMPLES "Build FUTCache examples" ON)
option(FUTCACHE_BUILD_BENCHMARKS "Build FUTCache benchmarks" OFF)
option(FUTCACHE_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(FUTCACHE_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(FUTCACHE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(FUTCACHE_ENABLE_SANITIZERS AND FUTCACHE_ENABLE_TSAN)
  message(FATAL_ERROR "ASan/UBSan and TSan must be built separately")
endif()

set(FUTCACHE_LIBRARY_TYPE STATIC)
if(FUTCACHE_BUILD_SHARED)
  set(FUTCACHE_LIBRARY_TYPE SHARED)
endif()

find_package(Threads REQUIRED)

function(futcache_enable_warnings target_name)
  if(NOT MSVC)
    target_compile_options(${target_name} PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wshadow
      -Wstrict-prototypes
      -Wformat=2
      -Wundef
      -Wcast-align
      -Wmissing-prototypes
      -Wdouble-promotion
      -Wnull-dereference
      -Werror=return-type
    )
    if(FUTCACHE_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE -Werror)
    endif()
  endif()
endfunction()

add_library(futcache ${FUTCACHE_LIBRARY_TYPE}
  src/futcache.c
  src/tower.c
)
add_library(FUTCache::futcache ALIAS futcache)

target_compile_features(futcache PUBLIC c_std_11)
target_compile_definitions(futcache PRIVATE FUTCACHE_BUILDING_LIBRARY)
if(FUTCACHE_BUILD_SHARED)
  target_compile_definitions(futcache PUBLIC FUTCACHE_SHARED)
endif()
target_include_directories(futcache
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(futcache PUBLIC Threads::Threads)

if(NOT MSVC)
  target_link_libraries(futcache PUBLIC m)
endif()
futcache_enable_warnings(futcache)

if(FUTCACHE_ENABLE_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(futcache PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(futcache PUBLIC -fsanitize=address,undefined)
endif()

if(FUTCACHE_ENABLE_TSAN AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(futcache PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(futcache PUBLIC -fsanitize=thread)
endif()

set_target_properties(futcache PROPERTIES
  C_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(BUILD_TESTING)
  add_executable(futcache_tests
    tests/test_main.c
    tests/test_futcache.c
    tests/test_tower.c
  )
  target_link_libraries(futcache_tests PRIVATE FUTCache::futcache)
  target_compile_features(futcache_tests PRIVATE c_std_11)
  futcache_enable_warnings(futcache_tests)
  add_test(NAME futcache_tests COMMAND futcache_tests)
endif()

if(FUTCACHE_BUILD_EXAMPLES)
  add_executable(futcache_example examples/basic.c)
  target_link_libraries(futcache_example PRIVATE FUTCache::futcache)
  futcache_enable_warnings(futcache_example)

  add_executable(futcache_scaling_example examples/reciprocal_scaling.c)
  target_link_libraries(futcache_scaling_example PRIVATE FUTCache::futcache)
  futcache_enable_warnings(futcache_scaling_example)
endif()

if(FUTCACHE_BUILD_BENCHMARKS)
  add_executable(futcache_benchmark bench/benchmark.c)
  target_link_libraries(futcache_benchmark PRIVATE FUTCache::futcache)
  target_compile_features(futcache_benchmark PRIVATE c_std_11)
  futcache_enable_warnings(futcache_benchmark)

  add_executable(futcache_cache_comparison bench/cache_comparison.c)
  target_link_libraries(futcache_cache_comparison PRIVATE FUTCache::futcache m)
  target_compile_features(futcache_cache_comparison PRIVATE c_std_11)
  futcache_enable_warnings(futcache_cache_comparison)
endif()

install(TARGETS futcache
  EXPORT FUTCacheTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT FUTCacheTargets
  FILE FUTCacheTargets.cmake
  NAMESPACE FUTCache::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FUTCache
)

configure_package_config_file(
  cmake/FUTCacheConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/FUTCacheConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FUTCache
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/FUTCacheConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/FUTCacheConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/FUTCacheConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FUTCache
)
install(FILES
  README.md
  docs/serialization.md
  docs/verification.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
)
