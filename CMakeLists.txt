cmake_minimum_required(VERSION 3.16)

project(
  FUTCache
  VERSION 1.1.0
  DESCRIPTION "Future-equivalence novelty caches"
  LANGUAGES C
)

include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)

option(FUTCACHE_BUILD_SHARED "Build FUTCache as a shared library" OFF)
option(FUTCACHE_BUILD_EXAMPLES "Build FUTCache examples" ON)
option(FUTCACHE_BUILD_BENCHMARKS "Build FUTCache benchmarks" OFF)
option(FUTCACHE_BUILD_PYTHON "Build the Python bindings (futcache_ext)" OFF)
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
  src/pack.c
  src/box.c
  src/crdt.c
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
    tests/test_pack.c
    tests/test_box.c
    tests/test_crdt.c
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

  add_executable(futcache_rag_embedding_example examples/rag_embedding.c)
  target_link_libraries(futcache_rag_embedding_example PRIVATE FUTCache::futcache m)
  futcache_enable_warnings(futcache_rag_embedding_example)
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

  add_executable(futcache_nd_dedup bench/nd_dedup.c)
  target_link_libraries(futcache_nd_dedup PRIVATE FUTCache::futcache m)
  target_compile_features(futcache_nd_dedup PRIVATE c_std_11)
  futcache_enable_warnings(futcache_nd_dedup)

  add_executable(futcache_bekko_semantic_cache bench/bekko_semantic_cache.c)
  target_link_libraries(futcache_bekko_semantic_cache PRIVATE FUTCache::futcache m)
  target_compile_features(futcache_bekko_semantic_cache PRIVATE c_std_11)
  futcache_enable_warnings(futcache_bekko_semantic_cache)
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

if(FUTCACHE_BUILD_PYTHON)
  # nanobind is a C++ binding library; enable C++ if the project has
  # not already done so.
  if(NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(CXX)
  endif()
  # The Python extension is a shared library and needs -fPIC on the
  # static libfutcache that it links against. Apply to the library
  # target so all its sources get PIC.
  set_target_properties(futcache PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(futcache_ext python/futcache/_core.cpp)
  target_link_libraries(futcache_ext PRIVATE futcache m)
  target_include_directories(futcache_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  if(NOT MSVC)
    target_compile_options(futcache_ext PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wno-unused-parameter
    )
  endif()
  # Install the compiled extension next to the python package so
  # wheel builds pick it up.
  install(TARGETS futcache_ext
    LIBRARY DESTINATION futcache
    RUNTIME DESTINATION futcache
  )
endif()

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
  LICENSE
  docs/serialization.md
  docs/verification.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
)
