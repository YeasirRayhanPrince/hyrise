#pragma once

#if HYRISE_NUMA_SUPPORT
#include <numa.h>
#include <numaif.h>
#endif
#ifdef __linux__
#include <x86intrin.h>
#endif
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>
#include "benchmark/benchmark.h"
#include "hdr/hdr_histogram.h"
#include "hyrise.hpp"
#include "storage/buffer/buffer_manager.hpp"
#include "utils/assert.hpp"
#include "zipfian_int_distribution.hpp"

namespace hyrise {

constexpr static auto GB = 1024 * 1024 * 1024;
constexpr static auto SEED = 123761253768512;
constexpr static auto CACHE_LINE_SIZE = 64;
static const char FAKE_DATA[512] __attribute__((aligned(512))) =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

inline void micro_benchmark_clear_cache() {
  constexpr auto ITEM_COUNT = 20 * 1000 * 1000;
  auto clear = std::vector<int>(ITEM_COUNT, 42);
  for (auto index = size_t{0}; index < ITEM_COUNT; ++index) {
    clear[index] += 1;
  }
  benchmark::ClobberMemory();
}

inline std::byte* mmap_region(const size_t num_bytes) {
#ifdef __APPLE__
  const int flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#elif __linux__
  const int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
#endif
  const auto mapped_memory = static_cast<std::byte*>(mmap(NULL, num_bytes, PROT_READ | PROT_WRITE, flags, -1, 0));

  if (mapped_memory == MAP_FAILED) {
    const auto error = errno;
    Fail("Failed to map volatile pool region: " + strerror(error));
  }

  return mapped_memory;
}

inline int get_numa_node(void* addr) {
  int numa_node = -1;

#if HYRISE_NUMA_SUPPORT
  if (get_mempolicy(&numa_node, NULL, 0, addr, MPOL_F_NODE | MPOL_F_ADDR) != 0) {
    Fail("Failed to get numa node");
  }
#endif
  return numa_node;
}

inline void munmap_region(std::byte* region, const size_t num_bytes) {
  if (munmap(region, num_bytes) < 0) {
    const auto error = errno;
    Fail("Failed to unmap volatile pool region: " + strerror(error));
  }
}

inline void explicit_move_pages(void* mem, size_t size, int node) {
#if HYRISE_NUMA_SUPPORT
  auto nodes = numa_allocate_nodemask();
  numa_bitmask_setbit(nodes, node);
  if (mbind(mem, size, MPOL_BIND, nodes ? nodes->maskp : NULL, nodes ? nodes->size + 1 : 0,
            MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    numa_bitmask_free(nodes);

    Fail("Move pages failed");
  }
  numa_bitmask_free(nodes);
#endif
}

#ifdef __linux__
static const __m512i avx_fake_data = _mm512_load_si512(FAKE_DATA);
#endif

inline void simulate_cacheline_nontemporal_store(std::byte* ptr) {
  DebugAssert(uintptr_t(ptr) % CACHE_LINE_SIZE == 0, "Pointer must be cacheline aligned");
#ifdef __APPLE__
  std::memset(ptr, 0x1, CACHE_LINE_SIZE);
#else
  // using a non-temporal memory hint
  _mm512_stream_si512(reinterpret_cast<__m512i*>(ptr), avx_fake_data);
  _mm_sfence();
#endif
}

inline void simulate_cacheline_temporal_store(std::byte* ptr) {
  // Taken from viper
  DebugAssert(uintptr_t(ptr) % CACHE_LINE_SIZE == 0, "Pointer must be cacheline aligned");
#ifdef __APPLE__
  std::memset(ptr, 0x1, CACHE_LINE_SIZE);
#else
  _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), avx_fake_data);
  _mm_clwb(ptr);
  _mm_sfence();
#endif
}

inline void simulate_cacheline_nontemporal_load(std::byte* ptr) {
  DebugAssert(uintptr_t(ptr) % CACHE_LINE_SIZE == 0, "Pointer must be cacheline aligned");
#ifdef __APPLE__
  __builtin_prefetch(ptr);
#else
  auto v = _mm512_stream_load_si512((__m512i*)(ptr));
  benchmark::DoNotOptimize(v);
#endif
}

inline void simulate_cacheline_load(std::byte* ptr) {
  DebugAssert(uintptr_t(ptr) % CACHE_LINE_SIZE == 0, "Pointer must be cacheline aligned");
#ifdef __APPLE__
  __builtin_prefetch(ptr);
#else
  auto v = _mm512_load_si512(ptr);
  benchmark::DoNotOptimize(v);
#endif
}

inline void simulate_scan(std::byte* ptr, size_t num_bytes) {
  for (size_t i = 0; i < num_bytes; i += CACHE_LINE_SIZE) {
    simulate_cacheline_load(ptr + i);
  }
}

inline void flush_cacheline(std::byte* ptr) {
  DebugAssert(uintptr_t(ptr) % CACHE_LINE_SIZE == 0, "Pointer must be cacheline aligned");
#ifdef __linux__
  _mm_clflush(ptr);
  _mm_mfence();
#endif
}

inline void flush_pipeline() {
  for (int i = 0; i < 100; i++) {
    // __nop();
    asm("nop");
  }
}

inline void init_histogram(hdr_histogram** histogram) {
  hdr_init(1, 10000000000, 3, histogram);
}

inline int open_file(std::string_view filename) {
#ifdef __APPLE__
  int flags = O_RDWR | O_CREAT | O_DSYNC;
#elif __linux__
  int flags = O_RDWR | O_CREAT | O_DIRECT | O_DSYNC;
#endif
  auto fd = open(std::string(filename).c_str(), flags, 0666);
  if (fd < 0) {
    Fail("Cannot open file");
  }
  return fd;
}

enum class YCSBWorkload {
  UpdateHeavy,  // Workload A, 50 reads / 50 updates
  ReadMostly,   // Workload B, 95% Point Lookups
  Scan          // Workload E, Short Ranges, 95% Scans
};
enum class YSCBOperationType : int { Scan, Lookup, Update };

enum class YCSBTupleSize : uint32_t {
  // Small = CACHE_LINE_SIZE,
  // Medium = 512,
  Large = 4096,
  // VeryLarge = 8192,
  // Huge = 32768
};

struct YCSBTuple {
  YCSBTupleSize size;
  std::byte* ptr;
};

using YCSBKey = uint64_t;
using YCSBTable = std::vector<YCSBTuple>;
using YSCBOperation = std::pair<YCSBKey, YSCBOperationType>;
using YCSBOperations = std::vector<YSCBOperation>;

inline YCSBTable generate_ycsb_table(boost::container::pmr::memory_resource* memory_resource,
                                     const size_t database_size) {
  Assert(database_size >= 1 * GB, "Database size must be greater than 1 GB");
  const auto load_start = std::chrono::high_resolution_clock::now();
  std::cout << "[YCSB Loader] Starting single-threaded loading. Target size: "
            << std::fixed << std::setprecision(2) << (database_size / static_cast<double>(GB)) << " GB"
            << std::endl;

  std::mt19937 generator{std::random_device{}()};
  std::uniform_int_distribution<int> distribution(0, magic_enum::enum_count<YCSBTupleSize>() - 1);
  auto table = std::vector<YCSBTuple>{};
  size_t current_size = 0;
  auto& buffer_manager = Hyrise::get().buffer_manager;
  size_t tuples_loaded = 0;
  const size_t LOG_INTERVAL = 100000;  // Log every 100k tuples

  while (true) {
    auto tuple_size = magic_enum::enum_value<YCSBTupleSize>(distribution(generator));
    auto page_size = bytes_for_size_type(find_fitting_page_size_type(static_cast<size_t>(tuple_size)));
    if (current_size + page_size > database_size) {
      break;
    }
    auto ptr = memory_resource->allocate(static_cast<size_t>(tuple_size), CACHE_LINE_SIZE);
    Assert(ptr != nullptr, "Allocation failed");
    auto page_id = buffer_manager.find_page(ptr);
    // buffer_manager.pin_exclusive(page_id);
    std::memset(ptr, 0x1, page_size);
    buffer_manager.set_dirty(page_id);
    // buffer_manager.unpin_exclusive(page_id);
    table.push_back({tuple_size, reinterpret_cast<std::byte*>(ptr)});
    current_size += page_size;
    tuples_loaded++;

    if (tuples_loaded % LOG_INTERVAL == 0) {
      std::cout << "  [Single-thread] Loaded " << tuples_loaded << " tuples (~"
                << std::fixed << std::setprecision(2) << (current_size / static_cast<double>(GB)) << " GB)\n";
    }
  }

  const auto load_end = std::chrono::high_resolution_clock::now();
  const auto load_duration = std::chrono::duration<double>(load_end - load_start).count();
  std::cout << "[YCSB Loader] Single-threaded loading complete. Loaded " << tuples_loaded << " tuples (~"
            << std::fixed << std::setprecision(2) << (current_size / static_cast<double>(GB)) << " GB) in "
            << std::fixed << std::setprecision(2) << load_duration << " seconds (~"
            << std::fixed << std::setprecision(2) << (current_size / static_cast<double>(GB) / load_duration)
            << " GB/s)" << std::endl;

  DebugAssert(current_size <= database_size, "Table size is too small");
  //
  return table;
}

inline YCSBTable generate_ycsb_table_parallel(boost::container::pmr::memory_resource* memory_resource,
                                              const size_t database_size, const size_t loader_threads) {
  Assert(database_size >= 1 * GB, "Database size must be greater than 1 GB");
  const auto load_start = std::chrono::high_resolution_clock::now();
  const auto thread_count = std::max<size_t>(1, loader_threads);

  std::cout << "[YCSB Loader] Starting parallel loading with " << thread_count << " threads. Target size: "
            << std::fixed << std::setprecision(2) << (database_size / static_cast<double>(GB)) << " GB"
            << std::endl;

  auto& buffer_manager = Hyrise::get().buffer_manager;
  std::atomic<size_t> allocated_bytes{0};
  std::atomic<size_t> tuples_loaded{0};
  std::vector<YCSBTable> thread_tables(thread_count);
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  std::mutex logging_mutex;
  const size_t LOG_INTERVAL = 100000;  // Log every 100k tuples

  for (auto thread_idx = size_t{0}; thread_idx < thread_count; ++thread_idx) {
    threads.emplace_back([&, thread_idx]() {
      std::mt19937 generator{std::random_device{}() + static_cast<unsigned>(thread_idx)};
      std::uniform_int_distribution<int> distribution(0, magic_enum::enum_count<YCSBTupleSize>() - 1);
      size_t thread_local_tuples = 0;

      while (true) {
        const auto tuple_size = magic_enum::enum_value<YCSBTupleSize>(distribution(generator));
        const auto page_size = bytes_for_size_type(find_fitting_page_size_type(static_cast<size_t>(tuple_size)));

        auto current = allocated_bytes.load(std::memory_order_relaxed);
        while (true) {
          if (current + page_size > database_size) {
            return;
          }
          if (allocated_bytes.compare_exchange_weak(current, current + page_size, std::memory_order_relaxed)) {
            break;
          }
        }

        auto ptr = memory_resource->allocate(static_cast<size_t>(tuple_size), CACHE_LINE_SIZE);
        Assert(ptr != nullptr, "Allocation failed");
        auto page_id = buffer_manager.find_page(ptr);
        buffer_manager.pin_exclusive(page_id);
        std::memset(ptr, 0x1, page_size);
        buffer_manager.set_dirty(page_id);
        buffer_manager.unpin_exclusive(page_id);
        thread_tables[thread_idx].push_back({tuple_size, reinterpret_cast<std::byte*>(ptr)});
        thread_local_tuples++;
        const auto new_total = tuples_loaded.fetch_add(1, std::memory_order_relaxed) + 1;

        if (new_total % LOG_INTERVAL == 0) {
          std::lock_guard<std::mutex> lock{logging_mutex};
          std::cout << "  [Thread " << thread_idx << "] Loaded " << new_total << " total tuples (~"
                    << std::fixed << std::setprecision(2) << (allocated_bytes.load() / static_cast<double>(GB))
                    << " GB)" << std::endl;
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  auto table = YCSBTable{};
  size_t total_tuples = 0;
  for (const auto& thread_table : thread_tables) {
    total_tuples += thread_table.size();
  }
  table.reserve(total_tuples);
  for (auto& thread_table : thread_tables) {
    table.insert(table.end(), thread_table.begin(), thread_table.end());
  }

  const auto load_end = std::chrono::high_resolution_clock::now();
  const auto load_duration = std::chrono::duration<double>(load_end - load_start).count();
  const auto final_bytes = allocated_bytes.load();
  std::cout << "[YCSB Loader] Parallel loading complete (" << thread_count << " threads). Loaded " << total_tuples
            << " tuples (~" << std::fixed << std::setprecision(2) << (final_bytes / static_cast<double>(GB))
            << " GB) in " << std::fixed << std::setprecision(2) << load_duration << " seconds (~"
            << std::fixed << std::setprecision(2) << (final_bytes / static_cast<double>(GB) / load_duration)
            << " GB/s)" << std::endl;

  DebugAssert(final_bytes <= database_size, "Table size is too small");
  return table;
}

template <YCSBWorkload workload, size_t NumOperations>
inline YCSBOperations generate_ycsb_operations(const size_t num_keys, const float zipf_skew) {
  YCSBOperations ops;
  static thread_local std::mt19937 generator{std::random_device{}()};
  std::uniform_int_distribution<int> op_distribution(0, 100);
  zipfian_int_distribution<size_t> key_distribution{0, num_keys - 1, zipf_skew};
  std::vector<YCSBKey> shuffled_keys(num_keys, 0);
  std::iota(shuffled_keys.begin(), shuffled_keys.end(), 0);
  auto rng = std::default_random_engine{};
  std::shuffle(std::begin(shuffled_keys), std::end(shuffled_keys), rng);
  for (size_t i = 0; i < NumOperations; i++) {
    auto key_idx = key_distribution(generator);
    auto key = shuffled_keys[key_idx];
    if constexpr (workload == YCSBWorkload::UpdateHeavy) {
      auto op =
          op_distribution(generator) < static_cast<int>(50) ? YSCBOperationType::Lookup : YSCBOperationType::Update;
      ops.push_back(std::make_pair(key, op));
    } else if constexpr (workload == YCSBWorkload::ReadMostly) {
      auto op =
          op_distribution(generator) < static_cast<int>(95) ? YSCBOperationType::Lookup : YSCBOperationType::Update;
      ops.push_back(std::make_pair(key, op));
    } else if constexpr (workload == YCSBWorkload::Scan) {
      auto op = op_distribution(generator) < static_cast<int>(95) ? YSCBOperationType::Scan : YSCBOperationType::Update;
      ops.push_back(std::make_pair(key, op));
    } else {
      Fail("Workload not supported");
    }
  }

  return ops;
}

inline uint64_t execute_ycsb_action(const YCSBTable& table, BufferManager& buffer_manager,
                                    const YSCBOperation operation) {
  const auto [key, op_type] = operation;
  const auto [size_type, ptr] = table[key];
  auto page_id = buffer_manager.find_page(ptr);
  auto page_size_bytes = bytes_for_size_type(page_id.size_type());
  auto num_cachelines = page_size_bytes / CACHE_LINE_SIZE;
  switch (op_type) {
    case YSCBOperationType::Lookup: {
      auto offset = (rand() % num_cachelines) * CACHE_LINE_SIZE;
      buffer_manager.pin_shared(page_id, AccessIntent::Read);
      simulate_cacheline_load(ptr + offset);
      buffer_manager.unpin_shared(page_id);
      return CACHE_LINE_SIZE;
    }
    case YSCBOperationType::Update: {
      auto offset = (rand() % num_cachelines) * CACHE_LINE_SIZE;
      buffer_manager.pin_exclusive(page_id);
      simulate_cacheline_nontemporal_store(ptr + offset);
      buffer_manager.unpin_exclusive(page_id);
      return CACHE_LINE_SIZE;
    }
    case YSCBOperationType::Scan: {
      buffer_manager.pin_shared(page_id, AccessIntent::Read);
      simulate_scan(ptr, page_size_bytes);
      buffer_manager.unpin_shared(page_id);
      return page_size_bytes;
    }
    default:
      Fail("Operation not supported");
  }
}

template <typename Fixture>
inline void run_ycsb(Fixture& fixture, benchmark::State& state) {
  micro_benchmark_clear_cache();

  auto bytes_processed = uint64_t{0};

  hdr_histogram* local_latency_histogram;
  init_histogram(&local_latency_histogram);

  std::map<YSCBOperationType, uint64_t> operation_counts;

  for (auto _ : state) {
    const auto start = state.thread_index() * fixture.operations_per_thread;
    const auto end = start + fixture.operations_per_thread;
    for (auto i = start; i < end; ++i) {
      const auto op = fixture.operations[i];
      operation_counts[op.second]++;
      const auto timer_start = std::chrono::high_resolution_clock::now();
      bytes_processed += execute_ycsb_action(fixture.table, fixture.buffer_manager, op);
      const auto timer_end = std::chrono::high_resolution_clock::now();
      const auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_end - timer_start).count();
      hdr_record_value(local_latency_histogram, latency);
    }
    {
      std::lock_guard<std::mutex> lock{fixture.latency_histogram_mutex};
      hdr_add(fixture.latency_histogram, local_latency_histogram);
      hdr_close(local_latency_histogram);
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(fixture.operations_per_thread);
  state.SetBytesProcessed(bytes_processed);

  if (state.thread_index() == 0) {
    state.counters["cache_hit_rate"] = fixture.buffer_manager.metrics()->hit_rate();
    state.counters["latency_mean"] = hdr_mean(fixture.latency_histogram);
    state.counters["latency_stddev"] = hdr_stddev(fixture.latency_histogram);
    state.counters["latency_median"] = hdr_value_at_percentile(fixture.latency_histogram, 50.0);
    state.counters["latency_min"] = hdr_min(fixture.latency_histogram);
    state.counters["latency_max"] = hdr_max(fixture.latency_histogram);
    state.counters["latency_95percentile"] = hdr_value_at_percentile(fixture.latency_histogram, 95.0);
    state.counters["bytes_written_to_ssd"] = fixture.buffer_manager.metrics()->total_bytes_copied_to_ssd.load();
    state.counters["bytes_read_from_ssd"] = fixture.buffer_manager.metrics()->total_bytes_copied_from_ssd.load();
  }
}

}  // namespace hyrise