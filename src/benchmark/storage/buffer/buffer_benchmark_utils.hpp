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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "benchmark/benchmark.h"
#include "tpcc/constants.hpp"
#include "tpcc/tpcc_random_generator.hpp"
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

// ============================================================================
// TPC-C Buffer Manager Benchmark - Row Structures and Data Generation
// ============================================================================

// ============================================================================
// TPC-C Row Structures (Packed, matching TPC-C specification)
// ============================================================================

#pragma pack(push, 1)

struct WarehouseRow {
  int32_t w_id;
  char w_name[10];
  char w_street_1[20];
  char w_street_2[20];
  char w_city[20];
  char w_state[2];
  char w_zip[9];
  float w_tax;
  float w_ytd;
};  // ~89 bytes

struct DistrictRow {
  int32_t d_id;
  int32_t d_w_id;
  char d_name[10];
  char d_street_1[20];
  char d_street_2[20];
  char d_city[20];
  char d_state[2];
  char d_zip[9];
  float d_tax;
  float d_ytd;
  int32_t d_next_o_id;
};  // ~97 bytes

struct CustomerRow {
  int32_t c_id;
  int32_t c_d_id;
  int32_t c_w_id;
  char c_first[16];
  char c_middle[2];
  char c_last[16];
  char c_street_1[20];
  char c_street_2[20];
  char c_city[20];
  char c_state[2];
  char c_zip[9];
  char c_phone[16];
  int32_t c_since;
  char c_credit[2];
  float c_credit_lim;
  float c_discount;
  float c_balance;
  float c_ytd_payment;
  int32_t c_payment_cnt;
  int32_t c_delivery_cnt;
  char c_data[500];
};  // ~655 bytes

struct ItemRow {
  int32_t i_id;
  int32_t i_im_id;
  char i_name[24];
  float i_price;
  char i_data[50];
};  // ~86 bytes

struct StockRow {
  int32_t s_i_id;
  int32_t s_w_id;
  int32_t s_quantity;
  char s_dist[10][24];  // S_DIST_01 through S_DIST_10
  int32_t s_ytd;
  int32_t s_order_cnt;
  int32_t s_remote_cnt;
  char s_data[50];
};  // ~306 bytes

struct OrderRow {
  int32_t o_id;
  int32_t o_d_id;
  int32_t o_w_id;
  int32_t o_c_id;
  int32_t o_entry_d;
  int32_t o_carrier_id;
  int32_t o_ol_cnt;
  int32_t o_all_local;
};  // ~32 bytes

struct OrderLineRow {
  int32_t ol_o_id;
  int32_t ol_d_id;
  int32_t ol_w_id;
  int32_t ol_number;
  int32_t ol_i_id;
  int32_t ol_supply_w_id;
  int32_t ol_delivery_d;
  int32_t ol_quantity;
  float ol_amount;
  char ol_dist_info[24];
};  // ~60 bytes

struct NewOrderRow {
  int32_t no_o_id;
  int32_t no_d_id;
  int32_t no_w_id;
};  // ~12 bytes

struct HistoryRow {
  int32_t h_c_id;
  int32_t h_c_d_id;
  int32_t h_c_w_id;
  int32_t h_d_id;
  int32_t h_w_id;
  int32_t h_date;
  float h_amount;
  char h_data[24];
};  // ~52 bytes

#pragma pack(pop)

// ============================================================================
// TPC-C Table Pages Structure (Buffer Manager Pages with Row Layout)
// ============================================================================

struct TPCCTablePages {
  std::string name;
  std::vector<std::byte*> pages;  // Pages allocated through buffer manager
  size_t row_size;
  size_t rows_per_page;
  size_t total_rows;
};

// Customer name index entry: list of customer indices with the same last name
// within a district, sorted by c_first for ORDER BY C_FIRST selection
struct CustomerNameIndexKey {
  int32_t w_id;
  int32_t d_id;
  std::string c_last;

  bool operator==(const CustomerNameIndexKey& other) const {
    return w_id == other.w_id && d_id == other.d_id && c_last == other.c_last;
  }
};

struct CustomerNameIndexKeyHash {
  size_t operator()(const CustomerNameIndexKey& k) const {
    size_t h1 = std::hash<int32_t>{}(k.w_id);
    size_t h2 = std::hash<int32_t>{}(k.d_id);
    size_t h3 = std::hash<std::string>{}(k.c_last);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

// Entry contains customer indices sorted by c_first
using CustomerNameIndex = std::unordered_map<CustomerNameIndexKey, std::vector<size_t>, CustomerNameIndexKeyHash>;

struct TPCCDatabasePages {
  TPCCTablePages warehouse;    // 1 row per warehouse
  TPCCTablePages district;     // 10 per warehouse
  TPCCTablePages customer;     // 3000 per district = 30,000 per warehouse
  TPCCTablePages history;      // 1 per customer initially
  TPCCTablePages new_order;    // 900 per district initially
  TPCCTablePages orders;       // 3000 per district initially
  TPCCTablePages order_line;   // ~30,000 per district initially (10 per order avg)
  TPCCTablePages item;         // 100,000 (shared)
  TPCCTablePages stock;        // 100,000 per warehouse

  // Secondary index for customer lookup by last name (40% of Payment/OrderStatus)
  CustomerNameIndex customer_name_index;

  size_t num_warehouses;
};

// Maximum ORDER_LINE slots per order (used for table allocation)
// Actual counts per order vary 5-15 as per TPC-C spec
constexpr size_t TPCC_MAX_ORDER_LINES_PER_ORDER = MAX_ORDER_LINE_COUNT;  // 15

// Pre-generated data structures for TPC-C to match TPCCTableGenerator exactly
struct TPCCPreGeneratedData {
  // Pre-selected ORIGINAL item IDs (exactly 10% = 10,000 out of 100,000)
  std::set<size_t> original_item_ids;

  // Pre-selected bad credit customer IDs per district (exactly 10% = 300 out of 3000)
  // Key: (w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE + (d_id - 1)
  std::unordered_map<size_t, std::set<size_t>> bad_credit_customer_ids;

  // Pre-generated ORDER_LINE counts per order (5-15 each)
  // order_line_counts[warehouse][district][order]
  std::vector<std::vector<std::vector<int32_t>>> order_line_counts;
};

inline TPCCPreGeneratedData generate_tpcc_pre_data(size_t num_warehouses, TPCCRandomGenerator& rng) {
  TPCCPreGeneratedData data;

  // Select exactly 10% of items for ORIGINAL
  data.original_item_ids = rng.select_unique_ids(NUM_ITEMS / 10, NUM_ITEMS);

  // Select exactly 10% of customers per district for bad credit
  for (size_t w = 0; w < num_warehouses; ++w) {
    for (size_t d = 0; d < NUM_DISTRICTS_PER_WAREHOUSE; ++d) {
      size_t key = w * NUM_DISTRICTS_PER_WAREHOUSE + d;
      data.bad_credit_customer_ids[key] =
          rng.select_unique_ids(NUM_CUSTOMERS_PER_DISTRICT / 10, NUM_CUSTOMERS_PER_DISTRICT);
    }
  }

  // Pre-generate ORDER_LINE counts per order (5-15 each)
  data.order_line_counts.resize(num_warehouses);
  for (auto& warehouse_counts : data.order_line_counts) {
    warehouse_counts.resize(NUM_DISTRICTS_PER_WAREHOUSE);
    for (auto& district_counts : warehouse_counts) {
      district_counts.resize(NUM_ORDERS_PER_DISTRICT);
      for (auto& count : district_counts) {
        count = static_cast<int32_t>(rng.random_number(MIN_ORDER_LINE_COUNT, MAX_ORDER_LINE_COUNT));
      }
    }
  }

  return data;
}

// TPC-C transaction types with standard mix
enum class TPCCTransactionType {
  Delivery,     // 4%  - Write-heavy, processed first in enum for discrete_distribution
  NewOrder,     // 45% - Multi-table write-heavy
  OrderStatus,  // 4%  - Read-only
  Payment,      // 43% - Read + write
  StockLevel    // 4%  - Read-only scan
};

// Item for NewOrder transaction
struct NewOrderItem {
  int32_t i_id;
  int32_t supply_w_id;
  int32_t quantity;
};

// Transaction parameters with all required fields
struct TPCCTransactionParams {
  TPCCTransactionType type;
  int32_t w_id;
  int32_t d_id;
  int32_t c_id;
  int32_t c_w_id;        // For Payment: customer's warehouse (may differ for remote)
  int32_t c_d_id;        // For Payment: customer's district
  std::vector<NewOrderItem> items;  // For NewOrder
  float h_amount;        // For Payment
  int32_t carrier_id;    // For Delivery
  int32_t threshold;     // For StockLevel

  // For 40% name-based customer lookup in Payment/OrderStatus
  bool lookup_by_name = false;
  std::string c_last;    // Customer last name for name-based lookup

  // For 1% rollback simulation in NewOrder
  bool has_invalid_item = false;
};

using TPCCTransactions = std::vector<TPCCTransactionParams>;

// ============================================================================
// Row Population Functions (Using TPCCRandomGenerator)
// ============================================================================

// Helper to copy string into fixed-size char array
inline void copy_to_fixed(char* dest, const std::string& src, size_t max_len) {
  size_t copy_len = std::min(src.size(), max_len);
  std::memcpy(dest, src.data(), copy_len);
  if (copy_len < max_len) {
    std::memset(dest + copy_len, ' ', max_len - copy_len);
  }
}

inline void populate_warehouse_row(std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng) {
  auto* row = reinterpret_cast<WarehouseRow*>(ptr);
  row->w_id = static_cast<int32_t>(row_idx + 1);
  copy_to_fixed(row->w_name, rng.astring(6, 10), sizeof(row->w_name));
  copy_to_fixed(row->w_street_1, rng.astring(10, 20), sizeof(row->w_street_1));
  copy_to_fixed(row->w_street_2, rng.astring(10, 20), sizeof(row->w_street_2));
  copy_to_fixed(row->w_city, rng.astring(10, 20), sizeof(row->w_city));
  copy_to_fixed(row->w_state, rng.astring(2, 2), sizeof(row->w_state));
  copy_to_fixed(row->w_zip, rng.zip_code(), sizeof(row->w_zip));
  row->w_tax = static_cast<float>(rng.random_number(0, 2000)) / 10000.0f;
  row->w_ytd = 300000.0f;
}

inline void populate_district_row(std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng,
                                   size_t num_warehouses) {
  auto* row = reinterpret_cast<DistrictRow*>(ptr);
  row->d_id = static_cast<int32_t>((row_idx % NUM_DISTRICTS_PER_WAREHOUSE) + 1);
  row->d_w_id = static_cast<int32_t>((row_idx / NUM_DISTRICTS_PER_WAREHOUSE) + 1);
  copy_to_fixed(row->d_name, rng.astring(6, 10), sizeof(row->d_name));
  copy_to_fixed(row->d_street_1, rng.astring(10, 20), sizeof(row->d_street_1));
  copy_to_fixed(row->d_street_2, rng.astring(10, 20), sizeof(row->d_street_2));
  copy_to_fixed(row->d_city, rng.astring(10, 20), sizeof(row->d_city));
  copy_to_fixed(row->d_state, rng.astring(2, 2), sizeof(row->d_state));
  copy_to_fixed(row->d_zip, rng.zip_code(), sizeof(row->d_zip));
  row->d_tax = static_cast<float>(rng.random_number(0, 2000)) / 10000.0f;
  row->d_ytd = 30000.0f;
  row->d_next_o_id = NUM_ORDERS_PER_DISTRICT + 1;  // Initial value
}

inline void populate_customer_row(std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng,
                                   size_t num_warehouses,
                                   const std::unordered_map<size_t, std::set<size_t>>& bad_credit_ids) {
  auto* row = reinterpret_cast<CustomerRow*>(ptr);
  size_t w_id = (row_idx / (NUM_DISTRICTS_PER_WAREHOUSE * NUM_CUSTOMERS_PER_DISTRICT)) + 1;
  size_t d_id = ((row_idx / NUM_CUSTOMERS_PER_DISTRICT) % NUM_DISTRICTS_PER_WAREHOUSE) + 1;
  size_t c_id = (row_idx % NUM_CUSTOMERS_PER_DISTRICT) + 1;

  row->c_id = static_cast<int32_t>(c_id);
  row->c_d_id = static_cast<int32_t>(d_id);
  row->c_w_id = static_cast<int32_t>(w_id);
  copy_to_fixed(row->c_first, rng.astring(8, 16), sizeof(row->c_first));
  copy_to_fixed(row->c_middle, "OE", sizeof(row->c_middle));
  copy_to_fixed(row->c_last, rng.last_name(c_id <= 1000 ? c_id - 1 : 1000 + c_id), sizeof(row->c_last));
  copy_to_fixed(row->c_street_1, rng.astring(10, 20), sizeof(row->c_street_1));
  copy_to_fixed(row->c_street_2, rng.astring(10, 20), sizeof(row->c_street_2));
  copy_to_fixed(row->c_city, rng.astring(10, 20), sizeof(row->c_city));
  copy_to_fixed(row->c_state, rng.astring(2, 2), sizeof(row->c_state));
  copy_to_fixed(row->c_zip, rng.zip_code(), sizeof(row->c_zip));
  copy_to_fixed(row->c_phone, rng.nstring(16, 16), sizeof(row->c_phone));
  row->c_since = 0;  // Current date placeholder

  // Use pre-selected bad credit customer IDs (exactly 10% = 300 per district)
  const size_t district_key = (w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE + (d_id - 1);
  const size_t customer_idx = c_id - 1;
  auto it = bad_credit_ids.find(district_key);
  const bool is_bad_credit = (it != bad_credit_ids.end() && it->second.find(customer_idx) != it->second.end());
  copy_to_fixed(row->c_credit, is_bad_credit ? "BC" : "GC", sizeof(row->c_credit));

  row->c_credit_lim = 50000.0f;
  row->c_discount = static_cast<float>(rng.random_number(0, 5000)) / 10000.0f;
  row->c_balance = -10.0f;
  row->c_ytd_payment = 10.0f;
  row->c_payment_cnt = 1;
  row->c_delivery_cnt = 0;
  copy_to_fixed(row->c_data, rng.astring(300, 500), sizeof(row->c_data));
}

inline void populate_item_row(std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng,
                               const std::set<size_t>& original_item_ids) {
  auto* row = reinterpret_cast<ItemRow*>(ptr);
  row->i_id = static_cast<int32_t>(row_idx + 1);
  row->i_im_id = static_cast<int32_t>(rng.random_number(1, 10000));
  copy_to_fixed(row->i_name, rng.astring(14, 24), sizeof(row->i_name));
  row->i_price = static_cast<float>(rng.random_number(100, 10000)) / 100.0f;
  auto data = rng.astring(26, 50);
  // Use pre-selected ORIGINAL IDs (exactly 10% = 10,000 items)
  const bool is_original = original_item_ids.find(row_idx) != original_item_ids.end();
  if (is_original) {
    size_t pos = rng.random_number(0, data.size() >= 8 ? data.size() - 8 : 0);
    data.replace(pos, 8, "ORIGINAL");
  }
  copy_to_fixed(row->i_data, data, sizeof(row->i_data));
}

inline void populate_stock_row(std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng,
                                size_t num_warehouses, const std::set<size_t>& original_item_ids) {
  auto* row = reinterpret_cast<StockRow*>(ptr);
  size_t w_id = (row_idx / NUM_STOCK_ITEMS_PER_WAREHOUSE) + 1;
  size_t i_id = (row_idx % NUM_STOCK_ITEMS_PER_WAREHOUSE) + 1;

  row->s_i_id = static_cast<int32_t>(i_id);
  row->s_w_id = static_cast<int32_t>(w_id);
  row->s_quantity = static_cast<int32_t>(rng.random_number(10, 100));
  for (int i = 0; i < 10; ++i) {
    copy_to_fixed(row->s_dist[i], rng.astring(24, 24), 24);
  }
  row->s_ytd = 0;
  row->s_order_cnt = 0;
  row->s_remote_cnt = 0;
  auto data = rng.astring(26, 50);
  // Use pre-selected ORIGINAL IDs (same set as ITEM table, indexed by item_id)
  const size_t item_idx = i_id - 1;
  const bool is_original = original_item_ids.find(item_idx) != original_item_ids.end();
  if (is_original) {
    size_t pos = rng.random_number(0, data.size() >= 8 ? data.size() - 8 : 0);
    data.replace(pos, 8, "ORIGINAL");
  }
  copy_to_fixed(row->s_data, data, sizeof(row->s_data));
}

inline void populate_order_row(std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng,
                                size_t num_warehouses, const std::vector<size_t>& customer_permutation,
                                const std::vector<std::vector<std::vector<int32_t>>>& order_line_counts) {
  auto* row = reinterpret_cast<OrderRow*>(ptr);
  size_t orders_per_warehouse = NUM_DISTRICTS_PER_WAREHOUSE * NUM_ORDERS_PER_DISTRICT;
  size_t w_id = (row_idx / orders_per_warehouse) + 1;
  size_t local_idx = row_idx % orders_per_warehouse;
  size_t d_id = (local_idx / NUM_ORDERS_PER_DISTRICT) + 1;
  size_t o_id = (local_idx % NUM_ORDERS_PER_DISTRICT) + 1;

  row->o_id = static_cast<int32_t>(o_id);
  row->o_d_id = static_cast<int32_t>(d_id);
  row->o_w_id = static_cast<int32_t>(w_id);
  // Use permutation for customer assignment if available
  size_t c_idx = (o_id - 1) % customer_permutation.size();
  row->o_c_id = static_cast<int32_t>(customer_permutation[c_idx] + 1);
  row->o_entry_d = 0;  // Current date placeholder
  // O_CARRIER_ID: null for new orders (2101-3000), random 1-10 for delivered orders
  row->o_carrier_id = (o_id <= NUM_ORDERS_PER_DISTRICT - NUM_NEW_ORDERS_PER_DISTRICT)
                          ? static_cast<int32_t>(rng.random_number(MIN_CARRIER_ID, MAX_CARRIER_ID))
                          : 0;
  // Use pre-generated ORDER_LINE count (5-15) from order_line_counts
  row->o_ol_cnt = order_line_counts[w_id - 1][d_id - 1][o_id - 1];
  row->o_all_local = 1;
}

inline void populate_new_order_row(std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng,
                                    size_t num_warehouses) {
  auto* row = reinterpret_cast<NewOrderRow*>(ptr);
  size_t new_orders_per_warehouse = NUM_DISTRICTS_PER_WAREHOUSE * NUM_NEW_ORDERS_PER_DISTRICT;
  size_t w_id = (row_idx / new_orders_per_warehouse) + 1;
  size_t local_idx = row_idx % new_orders_per_warehouse;
  size_t d_id = (local_idx / NUM_NEW_ORDERS_PER_DISTRICT) + 1;
  size_t no_idx = local_idx % NUM_NEW_ORDERS_PER_DISTRICT;
  // New orders are for orders 2101-3000
  size_t o_id = 2101 + no_idx;

  row->no_o_id = static_cast<int32_t>(o_id);
  row->no_d_id = static_cast<int32_t>(d_id);
  row->no_w_id = static_cast<int32_t>(w_id);
}

inline void populate_order_line_row(std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng,
                                     size_t num_warehouses, int32_t o_id, int32_t d_id, int32_t w_id,
                                     int32_t ol_number) {
  auto* row = reinterpret_cast<OrderLineRow*>(ptr);
  row->ol_o_id = o_id;
  row->ol_d_id = d_id;
  row->ol_w_id = w_id;
  row->ol_number = ol_number;
  row->ol_i_id = static_cast<int32_t>(rng.random_number(1, NUM_ITEMS));
  row->ol_supply_w_id = w_id;
  row->ol_delivery_d = (o_id < 2101) ? 0 : -1;  // Delivered orders have delivery date
  row->ol_quantity = 5;
  row->ol_amount = (o_id < 2101) ? 0.0f : static_cast<float>(rng.random_number(1, 999999)) / 100.0f;
  copy_to_fixed(row->ol_dist_info, rng.astring(24, 24), sizeof(row->ol_dist_info));
}

inline void populate_history_row(std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng,
                                  size_t num_warehouses) {
  auto* row = reinterpret_cast<HistoryRow*>(ptr);
  size_t customers_per_warehouse = NUM_DISTRICTS_PER_WAREHOUSE * NUM_CUSTOMERS_PER_DISTRICT;
  size_t w_id = (row_idx / customers_per_warehouse) + 1;
  size_t local_idx = row_idx % customers_per_warehouse;
  size_t d_id = (local_idx / NUM_CUSTOMERS_PER_DISTRICT) + 1;
  size_t c_id = (local_idx % NUM_CUSTOMERS_PER_DISTRICT) + 1;

  row->h_c_id = static_cast<int32_t>(c_id);
  row->h_c_d_id = static_cast<int32_t>(d_id);
  row->h_c_w_id = static_cast<int32_t>(w_id);
  row->h_d_id = static_cast<int32_t>(d_id);
  row->h_w_id = static_cast<int32_t>(w_id);
  row->h_date = 0;  // Current date placeholder
  row->h_amount = 10.0f;
  copy_to_fixed(row->h_data, rng.astring(12, 24), sizeof(row->h_data));
}

// ============================================================================
// Row Access Helper (forward declaration needed for generate_tpcc_database)
// ============================================================================

template <typename RowType>
inline RowType* get_row(const TPCCTablePages& table, size_t row_idx) {
  DebugAssert(table.row_size == sizeof(RowType), "Row size mismatch for table: " + table.name);
  DebugAssert(row_idx < table.total_rows, "Row index out of bounds for table: " + table.name);
  const auto page_idx = row_idx / table.rows_per_page;
  const auto offset_in_page = row_idx % table.rows_per_page;
  return reinterpret_cast<RowType*>(table.pages[page_idx] + offset_in_page * table.row_size);
}

// ============================================================================
// Table Generation Functions
// ============================================================================

inline TPCCTablePages generate_tpcc_table_generic(
    boost::container::pmr::memory_resource* memory_resource,
    BufferManager& buffer_manager,
    const std::string& name,
    size_t row_size,
    size_t total_rows,
    std::function<void(std::byte*, size_t, TPCCRandomGenerator&)> populate_row,
    size_t loader_threads = 1) {

  constexpr auto PAGE_SIZE = bytes_for_size_type(MIN_PAGE_SIZE_TYPE);
  constexpr uint32_t TPCC_TABLE_LOAD_SEED = 42;

  TPCCTablePages table;
  table.name = name;
  table.row_size = row_size;
  table.total_rows = total_rows;
  Assert(row_size <= PAGE_SIZE, "Row size exceeds page size for table: " + name);
  table.rows_per_page = PAGE_SIZE / row_size;
  DebugAssert(table.rows_per_page > 0, "No rows fit in a page for table: " + name);

  const size_t num_pages = (total_rows + table.rows_per_page - 1) / table.rows_per_page;
  table.pages.resize(num_pages);

  if (num_pages == 0) {
    return table;
  }

  const auto thread_count = std::min(std::max<size_t>(1, loader_threads), num_pages);
  const auto pages_per_thread = (num_pages + thread_count - 1) / thread_count;

  auto load_pages = [&](size_t start_page, size_t end_page, size_t thread_idx) {
    TPCCRandomGenerator rng{TPCC_TABLE_LOAD_SEED + static_cast<uint32_t>(thread_idx)};

    for (size_t page_idx = start_page; page_idx < end_page; ++page_idx) {
      auto* page = reinterpret_cast<std::byte*>(
          memory_resource->allocate(PAGE_SIZE, CACHE_LINE_SIZE));
      Assert(page != nullptr, "Allocation failed for table: " + name);

      auto page_id = buffer_manager.find_page(page);
      buffer_manager.pin_exclusive(page_id);

      // Initialize page to zero
      std::memset(page, 0, PAGE_SIZE);

      // Populate rows in this page with real TPC-C data
      const size_t rows_in_page =
          std::min(table.rows_per_page, total_rows - page_idx * table.rows_per_page);
      for (size_t row_idx = 0; row_idx < rows_in_page; ++row_idx) {
        const size_t global_row = page_idx * table.rows_per_page + row_idx;
        populate_row(page + row_idx * row_size, global_row, rng);
      }

      buffer_manager.set_dirty(page_id);
      buffer_manager.unpin_exclusive(page_id);
      table.pages[page_idx] = page;
    }
  };

  if (thread_count == 1) {
    load_pages(0, num_pages, 0);
  } else {
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      const size_t start_page = thread_idx * pages_per_thread;
      const size_t end_page = std::min(start_page + pages_per_thread, num_pages);
      if (start_page >= end_page) {
        break;
      }
      threads.emplace_back(load_pages, start_page, end_page, thread_idx);
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  return table;
}

// Generate TPC-C database through buffer manager
// Uses pre-generated data to exactly match TPCCTableGenerator behavior
inline TPCCDatabasePages generate_tpcc_database(boost::container::pmr::memory_resource* memory_resource,
                                                size_t num_warehouses, size_t loader_threads = 1) {
  const auto load_start = std::chrono::high_resolution_clock::now();
  const auto thread_count = std::max<size_t>(1, loader_threads);

  std::cout << "[TPC-C Loader] Starting loading for " << num_warehouses
            << " warehouses (configured loader threads: " << thread_count << ")" << std::endl;

  auto& buffer_manager = Hyrise::get().buffer_manager;
  TPCCDatabasePages db;
  db.num_warehouses = num_warehouses;

  // Pre-generate ORIGINAL IDs, bad credit customer IDs, and ORDER_LINE counts
  // to exactly match TPCCTableGenerator behavior
  std::cout << "  Pre-generating TPC-C data structures..." << std::endl;
  TPCCRandomGenerator pre_rng;
  auto pre_data = generate_tpcc_pre_data(num_warehouses, pre_rng);

  std::cout << "  Allocating ITEM (with " << pre_data.original_item_ids.size() << " ORIGINAL items)..." << std::endl;
  db.item = generate_tpcc_table_generic(
      memory_resource, buffer_manager, "ITEM", sizeof(ItemRow),
      static_cast<size_t>(NUM_ITEMS),
      [&pre_data](std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng) {
        populate_item_row(ptr, row_idx, rng, pre_data.original_item_ids);
      },
      thread_count);

  std::cout << "  Allocating WAREHOUSE..." << std::endl;
  db.warehouse = generate_tpcc_table_generic(memory_resource, buffer_manager, "WAREHOUSE", sizeof(WarehouseRow),
                                             num_warehouses, populate_warehouse_row, thread_count);

  std::cout << "  Allocating DISTRICT..." << std::endl;
  db.district = generate_tpcc_table_generic(
      memory_resource, buffer_manager, "DISTRICT", sizeof(DistrictRow),
      num_warehouses * static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE),
      [num_warehouses](std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng) {
        populate_district_row(ptr, row_idx, rng, num_warehouses);
      },
      thread_count);

  std::cout << "  Allocating CUSTOMER (with pre-selected bad credit IDs)..." << std::endl;
  db.customer = generate_tpcc_table_generic(
      memory_resource, buffer_manager, "CUSTOMER", sizeof(CustomerRow),
      num_warehouses * static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE) *
          static_cast<size_t>(NUM_CUSTOMERS_PER_DISTRICT),
      [num_warehouses, &pre_data](std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng) {
        populate_customer_row(ptr, row_idx, rng, num_warehouses, pre_data.bad_credit_customer_ids);
      },
      thread_count);

  // Build customer name index for 40% name-based lookups
  std::cout << "  Building customer name index..." << std::endl;
  for (size_t c_idx = 0; c_idx < db.customer.total_rows; ++c_idx) {
    auto* c_row = get_row<CustomerRow>(db.customer, c_idx);
    CustomerNameIndexKey key;
    key.w_id = c_row->c_w_id;
    key.d_id = c_row->c_d_id;
    key.c_last = std::string(c_row->c_last, sizeof(c_row->c_last));
    // Trim trailing spaces
    key.c_last.erase(key.c_last.find_last_not_of(' ') + 1);
    db.customer_name_index[key].push_back(c_idx);
  }

  // Sort each entry by c_first for ORDER BY C_FIRST requirement
  for (auto& [key, indices] : db.customer_name_index) {
    std::sort(indices.begin(), indices.end(), [&db](size_t a, size_t b) {
      auto* row_a = get_row<CustomerRow>(db.customer, a);
      auto* row_b = get_row<CustomerRow>(db.customer, b);
      return std::strncmp(row_a->c_first, row_b->c_first, sizeof(row_a->c_first)) < 0;
    });
  }

  std::cout << "  Allocating HISTORY..." << std::endl;
  db.history = generate_tpcc_table_generic(
      memory_resource, buffer_manager, "HISTORY", sizeof(HistoryRow),
      num_warehouses * static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE) *
          static_cast<size_t>(NUM_CUSTOMERS_PER_DISTRICT),
      [num_warehouses](std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng) {
        populate_history_row(ptr, row_idx, rng, num_warehouses);
      },
      thread_count);

  TPCCRandomGenerator permutation_rng;
  const auto customer_permutation = permutation_rng.permutation(0, NUM_CUSTOMERS_PER_DISTRICT);

  std::cout << "  Allocating ORDERS (with pre-generated ORDER_LINE counts)..." << std::endl;
  db.orders = generate_tpcc_table_generic(
      memory_resource, buffer_manager, "ORDERS", sizeof(OrderRow),
      num_warehouses * static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE) *
          static_cast<size_t>(NUM_ORDERS_PER_DISTRICT),
      [num_warehouses, &customer_permutation, &pre_data](std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng) {
        populate_order_row(ptr, row_idx, rng, num_warehouses, customer_permutation, pre_data.order_line_counts);
      },
      thread_count);

  std::cout << "  Allocating NEW_ORDER..." << std::endl;
  db.new_order = generate_tpcc_table_generic(
      memory_resource, buffer_manager, "NEW_ORDER", sizeof(NewOrderRow),
      num_warehouses * static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE) *
          static_cast<size_t>(NUM_NEW_ORDERS_PER_DISTRICT),
      [num_warehouses](std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng) {
        populate_new_order_row(ptr, row_idx, rng, num_warehouses);
      },
      thread_count);

  // Allocate ORDER_LINE with maximum slots (15) per order
  // Only populate actual count per order (from pre_data.order_line_counts)
  // This keeps simple indexing while matching real TPC-C variable counts
  const auto order_lines_per_district =
      static_cast<size_t>(NUM_ORDERS_PER_DISTRICT) * TPCC_MAX_ORDER_LINES_PER_ORDER;
  const auto order_lines_per_warehouse =
      static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE) * order_lines_per_district;
  const auto total_order_line_slots = num_warehouses * order_lines_per_warehouse;

  std::cout << "  Allocating ORDER_LINE (max 15 slots per order, populating actual 5-15)..." << std::endl;

  db.order_line = generate_tpcc_table_generic(
      memory_resource, buffer_manager, "ORDER_LINE", sizeof(OrderLineRow),
      total_order_line_slots,
      [num_warehouses, order_lines_per_district, order_lines_per_warehouse, &pre_data](
          std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng) {
        const auto w_idx = row_idx / order_lines_per_warehouse;
        const auto local_idx = row_idx % order_lines_per_warehouse;
        const auto d_idx = local_idx / order_lines_per_district;
        const auto order_line_idx = local_idx % order_lines_per_district;
        const auto o_idx = order_line_idx / TPCC_MAX_ORDER_LINES_PER_ORDER;
        const auto ol_number = static_cast<int32_t>((order_line_idx % TPCC_MAX_ORDER_LINES_PER_ORDER) + 1);

        // Only populate if within actual ORDER_LINE count for this order
        const int32_t actual_ol_count = pre_data.order_line_counts[w_idx][d_idx][o_idx];
        if (ol_number <= actual_ol_count) {
          populate_order_line_row(ptr, row_idx, rng, num_warehouses,
                                   static_cast<int32_t>(o_idx + 1),
                                   static_cast<int32_t>(d_idx + 1),
                                   static_cast<int32_t>(w_idx + 1),
                                   ol_number);
        }
        // Slots beyond actual count remain zeroed (from page initialization)
      },
      thread_count);

  std::cout << "  Allocating STOCK (with pre-selected ORIGINAL items)..." << std::endl;
  db.stock = generate_tpcc_table_generic(
      memory_resource, buffer_manager, "STOCK", sizeof(StockRow),
      num_warehouses * static_cast<size_t>(NUM_STOCK_ITEMS_PER_WAREHOUSE),
      [num_warehouses, &pre_data](std::byte* ptr, size_t row_idx, TPCCRandomGenerator& rng) {
        populate_stock_row(ptr, row_idx, rng, num_warehouses, pre_data.original_item_ids);
      },
      thread_count);

  const auto total_pages = db.warehouse.pages.size() + db.district.pages.size() + db.customer.pages.size() +
                           db.history.pages.size() + db.new_order.pages.size() + db.orders.pages.size() +
                           db.order_line.pages.size() + db.item.pages.size() + db.stock.pages.size();

  const auto load_end = std::chrono::high_resolution_clock::now();
  const auto load_duration = std::chrono::duration<double>(load_end - load_start).count();
  const auto page_size = bytes_for_size_type(MIN_PAGE_SIZE_TYPE);

  std::cout << "[TPC-C Loader] Complete. Allocated " << total_pages << " pages (~"
            << std::fixed << std::setprecision(2) << (total_pages * page_size / static_cast<double>(GB)) << " GB) in "
            << std::fixed << std::setprecision(2) << load_duration << " seconds" << std::endl;

  return db;
}

// ============================================================================
// Row Access Helpers
// ============================================================================

inline std::byte* align_down_to_cacheline(const std::byte* ptr) {
  const auto address = reinterpret_cast<uintptr_t>(ptr);
  return reinterpret_cast<std::byte*>(address - (address % CACHE_LINE_SIZE));
}

inline void simulate_row_load(const std::byte* ptr, size_t row_size) {
  auto* start = align_down_to_cacheline(ptr);
  auto* end = const_cast<std::byte*>(ptr) + row_size;
  for (auto* line = start; line < end; line += CACHE_LINE_SIZE) {
    simulate_cacheline_load(line);
  }
}

inline void simulate_row_store(std::byte* ptr, size_t row_size) {
  benchmark::DoNotOptimize(ptr);
  benchmark::DoNotOptimize(row_size);
  benchmark::ClobberMemory();
}

// Note: get_row is defined earlier (before generate_tpcc_database) to allow use during database generation

template <typename RowType>
inline uint64_t read_row(const TPCCTablePages& table, BufferManager& buffer_manager, size_t row_idx, RowType* out) {
  DebugAssert(row_idx < table.total_rows, "Row index out of bounds for table: " + table.name);
  const auto page_idx = row_idx / table.rows_per_page;
  auto* page = table.pages[page_idx];
  const auto page_id = buffer_manager.find_page(page);

  buffer_manager.pin_shared(page_id, AccessIntent::Read);
  auto* row = get_row<RowType>(table, row_idx);
  simulate_row_load(reinterpret_cast<std::byte*>(row), sizeof(RowType));
  if (out) {
    *out = *row;
  }
  buffer_manager.unpin_shared(page_id);

  return sizeof(RowType);
}

template <typename RowType>
inline uint64_t write_row(const TPCCTablePages& table, BufferManager& buffer_manager, size_t row_idx,
                          const RowType& data) {
  DebugAssert(row_idx < table.total_rows, "Row index out of bounds for table: " + table.name);
  const auto page_idx = row_idx / table.rows_per_page;
  auto* page = table.pages[page_idx];
  const auto page_id = buffer_manager.find_page(page);

  buffer_manager.pin_exclusive(page_id);
  auto* row = get_row<RowType>(table, row_idx);
  *row = data;
  simulate_row_store(reinterpret_cast<std::byte*>(row), sizeof(RowType));
  buffer_manager.set_dirty(page_id);
  buffer_manager.unpin_exclusive(page_id);

  return sizeof(RowType);
}

template <typename RowType, typename FieldType>
inline uint64_t update_field(const TPCCTablePages& table, BufferManager& buffer_manager, size_t row_idx,
                             FieldType RowType::*field, FieldType new_value) {
  DebugAssert(row_idx < table.total_rows, "Row index out of bounds for table: " + table.name);
  const auto page_idx = row_idx / table.rows_per_page;
  auto* page = table.pages[page_idx];
  const auto page_id = buffer_manager.find_page(page);

  buffer_manager.pin_exclusive(page_id);
  auto* row = get_row<RowType>(table, row_idx);
  simulate_row_load(reinterpret_cast<std::byte*>(row), sizeof(RowType));
  row->*field = new_value;
  simulate_row_store(reinterpret_cast<std::byte*>(row), sizeof(RowType));
  buffer_manager.set_dirty(page_id);
  buffer_manager.unpin_exclusive(page_id);

  return sizeof(RowType);
}

// ============================================================================
// Transaction Implementations
// ============================================================================

inline uint64_t execute_new_order(const TPCCDatabasePages& db, BufferManager& buffer_manager, int32_t w_id,
                                  int32_t d_id, int32_t c_id, const std::vector<NewOrderItem>& items,
                                  bool has_invalid_item = false) {
  uint64_t bytes = 0;

  // 1. SELECT W_TAX FROM WAREHOUSE
  WarehouseRow w_row;
  bytes += read_row(db.warehouse, buffer_manager, static_cast<size_t>(w_id - 1), &w_row);

  // 2. SELECT D_TAX, D_NEXT_O_ID FROM DISTRICT
  const auto d_idx =
      static_cast<size_t>((w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE + (d_id - 1));
  DistrictRow d_row;
  bytes += read_row(db.district, buffer_manager, d_idx, &d_row);
  const auto o_id = d_row.d_next_o_id;

  // 4. SELECT C_DISCOUNT, C_LAST, C_CREDIT FROM CUSTOMER
  const auto c_idx =
      static_cast<size_t>((w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE * NUM_CUSTOMERS_PER_DISTRICT +
                          (d_id - 1) * NUM_CUSTOMERS_PER_DISTRICT + (c_id - 1));
  CustomerRow c_row;
  bytes += read_row(db.customer, buffer_manager, c_idx, &c_row);

  // For 1% rollback simulation: if has_invalid_item is true, simulate item lookups and stock reads only
  // (no writes). This matches TPC-C behavior where the transaction aborts and no updates persist.
  const auto order_lines_per_district =
      static_cast<size_t>(NUM_ORDERS_PER_DISTRICT) * TPCC_MAX_ORDER_LINES_PER_ORDER;
  const auto order_lines_per_warehouse =
      static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE) * order_lines_per_district;

  if (has_invalid_item) {
    for (size_t ol_num = 0; ol_num < items.size(); ++ol_num) {
      const auto& item = items[ol_num];

      // Last item is invalid: do the lookup and abort
      if (ol_num == items.size() - 1) {
        ItemRow i_row;
        bytes += read_row(db.item, buffer_manager, 0, &i_row);  // Placeholder for invalid item lookup
        break;
      }

      ItemRow i_row;
      bytes += read_row(db.item, buffer_manager, static_cast<size_t>(item.i_id - 1), &i_row);

      const auto s_idx = static_cast<size_t>((item.supply_w_id - 1) * NUM_STOCK_ITEMS_PER_WAREHOUSE + (item.i_id - 1));
      StockRow s_row;
      bytes += read_row(db.stock, buffer_manager, s_idx, &s_row);
    }

    return bytes;
  }

  // 3. UPDATE DISTRICT SET D_NEXT_O_ID = D_NEXT_O_ID + 1 (HOTSPOT!)
  bytes += update_field(db.district, buffer_manager, d_idx, &DistrictRow::d_next_o_id, o_id + 1);

  // Process each order line item
  for (size_t ol_num = 0; ol_num < items.size(); ++ol_num) {
    const auto& item = items[ol_num];

    ItemRow i_row;
    bytes += read_row(db.item, buffer_manager, static_cast<size_t>(item.i_id - 1), &i_row);

    // SELECT/UPDATE STOCK
    const auto s_idx = static_cast<size_t>((item.supply_w_id - 1) * NUM_STOCK_ITEMS_PER_WAREHOUSE + (item.i_id - 1));
    StockRow s_row;
    bytes += read_row(db.stock, buffer_manager, s_idx, &s_row);

    // Update stock quantity
    const auto new_qty = (s_row.s_quantity >= item.quantity + 10) ? s_row.s_quantity - item.quantity
                                                                   : s_row.s_quantity - item.quantity + 91;
    bytes += update_field(db.stock, buffer_manager, s_idx, &StockRow::s_quantity, new_qty);
    bytes += update_field(db.stock, buffer_manager, s_idx, &StockRow::s_ytd, s_row.s_ytd + item.quantity);
    bytes += update_field(db.stock, buffer_manager, s_idx, &StockRow::s_order_cnt, s_row.s_order_cnt + 1);
    if (item.supply_w_id != w_id) {
      bytes += update_field(db.stock, buffer_manager, s_idx, &StockRow::s_remote_cnt, s_row.s_remote_cnt + 1);
    }

    // INSERT ORDER_LINE
    const auto order_index = static_cast<size_t>((o_id - 1) % NUM_ORDERS_PER_DISTRICT);
    const auto ol_idx = static_cast<size_t>((w_id - 1) * order_lines_per_warehouse +
                                            (d_id - 1) * order_lines_per_district +
                                            order_index * TPCC_MAX_ORDER_LINES_PER_ORDER + ol_num);
    OrderLineRow ol_row{static_cast<int32_t>(o_id), d_id, w_id, static_cast<int32_t>(ol_num + 1), item.i_id,
                        item.supply_w_id, 0, item.quantity, static_cast<float>(item.quantity) * i_row.i_price, {}};
    // Copy S_DIST_XX to OL_DIST_INFO
    std::memcpy(ol_row.ol_dist_info, s_row.s_dist[d_id - 1], sizeof(ol_row.ol_dist_info));
    bytes += write_row(db.order_line, buffer_manager, ol_idx % db.order_line.total_rows, ol_row);
  }

  // 5. INSERT NEW_ORDER
  const auto no_idx =
      static_cast<size_t>((w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE * NUM_NEW_ORDERS_PER_DISTRICT +
                          (d_id - 1) * NUM_NEW_ORDERS_PER_DISTRICT +
                          ((o_id - 1) % NUM_NEW_ORDERS_PER_DISTRICT));
  NewOrderRow no_row{static_cast<int32_t>(o_id), d_id, w_id};
  bytes += write_row(db.new_order, buffer_manager, no_idx, no_row);

  // 6. INSERT ORDER
  const auto o_idx =
      static_cast<size_t>((w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE * NUM_ORDERS_PER_DISTRICT +
                          (d_id - 1) * NUM_ORDERS_PER_DISTRICT + ((o_id - 1) % NUM_ORDERS_PER_DISTRICT));
  OrderRow o_row{static_cast<int32_t>(o_id), d_id, w_id, c_id, /* entry_d */ 0, /* carrier */ -1,
                 static_cast<int32_t>(items.size()), 1};
  bytes += write_row(db.orders, buffer_manager, o_idx, o_row);

  return bytes;
}

inline uint64_t execute_payment(const TPCCDatabasePages& db, BufferManager& buffer_manager, int32_t w_id, int32_t d_id,
                                int32_t c_w_id, int32_t c_d_id, int32_t c_id, float h_amount,
                                bool lookup_by_name = false, const std::string& c_last = "") {
  uint64_t bytes = 0;

  // 1. Read/Update WAREHOUSE
  WarehouseRow w_row;
  bytes += read_row(db.warehouse, buffer_manager, static_cast<size_t>(w_id - 1), &w_row);
  bytes += update_field(db.warehouse, buffer_manager, static_cast<size_t>(w_id - 1), &WarehouseRow::w_ytd,
                        w_row.w_ytd + h_amount);

  // 2. Read/Update DISTRICT
  const auto d_idx =
      static_cast<size_t>((w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE + (d_id - 1));
  DistrictRow d_row;
  bytes += read_row(db.district, buffer_manager, d_idx, &d_row);
  bytes += update_field(db.district, buffer_manager, d_idx, &DistrictRow::d_ytd, d_row.d_ytd + h_amount);

  // 3. Customer lookup - 60% by C_ID, 40% by C_LAST (with ORDER BY C_FIRST)
  size_t c_idx;
  if (lookup_by_name && !c_last.empty()) {
    // Name-based lookup: find customers with matching last name, select middle one
    CustomerNameIndexKey key{c_w_id, c_d_id, c_last};
    auto it = db.customer_name_index.find(key);
    if (it != db.customer_name_index.end() && !it->second.empty()) {
      const auto& matching_customers = it->second;
      // TPC-C spec: select the customer at position (n+1)/2 rounded up (middle customer)
      // For n customers sorted by C_FIRST, this is index (n-1)/2 for 0-based indexing
      size_t middle_idx = (matching_customers.size() - 1) / 2;
      c_idx = matching_customers[middle_idx];

      // Simulate scanning through matching customers (like SQL would)
      for (size_t i = 0; i < matching_customers.size(); ++i) {
        CustomerRow scan_row;
        bytes += read_row(db.customer, buffer_manager, matching_customers[i], &scan_row);
      }
    } else {
      // Fallback to ID-based if name not found
      c_idx = static_cast<size_t>((c_w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE * NUM_CUSTOMERS_PER_DISTRICT +
                                   (c_d_id - 1) * NUM_CUSTOMERS_PER_DISTRICT + (c_id - 1));
    }
  } else {
    // ID-based lookup (60% of cases)
    c_idx = static_cast<size_t>((c_w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE * NUM_CUSTOMERS_PER_DISTRICT +
                                 (c_d_id - 1) * NUM_CUSTOMERS_PER_DISTRICT + (c_id - 1));
  }

  // 4. Read CUSTOMER
  CustomerRow c_row;
  bytes += read_row(db.customer, buffer_manager, c_idx, &c_row);

  // 5. Update CUSTOMER balance, payment counters
  bytes += update_field(db.customer, buffer_manager, c_idx, &CustomerRow::c_balance, c_row.c_balance - h_amount);
  bytes += update_field(db.customer, buffer_manager, c_idx, &CustomerRow::c_ytd_payment,
                        c_row.c_ytd_payment + h_amount);
  bytes += update_field(db.customer, buffer_manager, c_idx, &CustomerRow::c_payment_cnt, c_row.c_payment_cnt + 1);

  // 6. Bad credit handling: if C_CREDIT == "BC", update C_DATA with payment history
  // 10% of customers have bad credit
  if (c_row.c_credit[0] == 'B' && c_row.c_credit[1] == 'C') {
    // Construct new C_DATA: C_ID, C_D_ID, C_W_ID, D_ID, W_ID, H_AMOUNT prepended to existing
    // This simulates the string manipulation that happens in real TPC-C
    const auto page_idx = c_idx / db.customer.rows_per_page;
    auto* page = db.customer.pages[page_idx];
    const auto page_id = buffer_manager.find_page(page);

    buffer_manager.pin_exclusive(page_id);
    auto* customer = get_row<CustomerRow>(db.customer, c_idx);
    // Shift existing data and prepend new info (simplified: just update the field)
    char new_data[500];
    int written = snprintf(new_data, sizeof(new_data), "%d %d %d %d %d %.2f | ",
                           c_row.c_id, c_row.c_d_id, c_row.c_w_id, d_id, w_id, h_amount);
    size_t remaining = sizeof(new_data) - written;
    std::memcpy(new_data + written, c_row.c_data, std::min(remaining, sizeof(c_row.c_data)));
    std::memcpy(customer->c_data, new_data, sizeof(customer->c_data));
    simulate_row_store(reinterpret_cast<std::byte*>(customer), sizeof(CustomerRow));
    buffer_manager.set_dirty(page_id);
    buffer_manager.unpin_exclusive(page_id);
    bytes += sizeof(CustomerRow);
  }

  // 7. Insert HISTORY
  const auto h_idx = c_idx % db.history.total_rows;
  HistoryRow h_row{c_row.c_id, c_row.c_d_id, c_row.c_w_id, d_id, w_id, 0, h_amount, {}};
  // H_DATA = W_NAME + "    " + D_NAME (simplified)
  snprintf(h_row.h_data, sizeof(h_row.h_data), "%.10s    %.10s", w_row.w_name, d_row.d_name);
  bytes += write_row(db.history, buffer_manager, h_idx, h_row);

  return bytes;
}

inline uint64_t execute_order_status(const TPCCDatabasePages& db, BufferManager& buffer_manager, int32_t w_id,
                                     int32_t d_id, int32_t c_id, bool lookup_by_name = false,
                                     const std::string& c_last = "") {
  uint64_t bytes = 0;

  // 1. Customer lookup - 60% by C_ID, 40% by C_LAST (with ORDER BY C_FIRST)
  size_t c_idx;
  if (lookup_by_name && !c_last.empty()) {
    // Name-based lookup
    CustomerNameIndexKey key{w_id, d_id, c_last};
    auto it = db.customer_name_index.find(key);
    if (it != db.customer_name_index.end() && !it->second.empty()) {
      const auto& matching_customers = it->second;
      // Select middle customer per TPC-C spec
      size_t middle_idx = (matching_customers.size() - 1) / 2;
      c_idx = matching_customers[middle_idx];

      // Simulate scanning through matching customers
      for (size_t i = 0; i < matching_customers.size(); ++i) {
        CustomerRow scan_row;
        bytes += read_row(db.customer, buffer_manager, matching_customers[i], &scan_row);
      }
    } else {
      c_idx = static_cast<size_t>((w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE * NUM_CUSTOMERS_PER_DISTRICT +
                                   (d_id - 1) * NUM_CUSTOMERS_PER_DISTRICT + (c_id - 1));
    }
  } else {
    c_idx = static_cast<size_t>((w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE * NUM_CUSTOMERS_PER_DISTRICT +
                                 (d_id - 1) * NUM_CUSTOMERS_PER_DISTRICT + (c_id - 1));
  }

  CustomerRow c_row;
  bytes += read_row(db.customer, buffer_manager, c_idx, &c_row);

  // 2. Find the last order for this customer (ORDER BY O_ID DESC LIMIT 1)
  // Scan all orders in the district to find the max O_ID for this customer
  const auto orders_per_warehouse = static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE) * NUM_ORDERS_PER_DISTRICT;
  const auto o_base =
      static_cast<size_t>((w_id - 1) * orders_per_warehouse + (d_id - 1) * NUM_ORDERS_PER_DISTRICT);

  OrderRow found_order{};
  int32_t found_o_id = -1;
  bool found = false;

  for (int i = 0; i < NUM_ORDERS_PER_DISTRICT; ++i) {
    const auto o_idx = o_base + i;
    OrderRow o_row;
    bytes += read_row(db.orders, buffer_manager, o_idx % db.orders.total_rows, &o_row);
    if (o_row.o_c_id == c_row.c_id) {
      if (!found || o_row.o_id > found_o_id) {
        found_order = o_row;
        found_o_id = o_row.o_id;
        found = true;
      }
    }
  }

  // 3. Read ORDER_LINEs for the found order (variable 5-15 lines based on o_ol_cnt)
  if (found) {
    const auto order_lines_per_district =
        static_cast<size_t>(NUM_ORDERS_PER_DISTRICT) * TPCC_MAX_ORDER_LINES_PER_ORDER;
    const auto order_index = static_cast<size_t>((found_order.o_id - 1) % NUM_ORDERS_PER_DISTRICT);
    const auto ol_base =
        static_cast<size_t>((w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE * order_lines_per_district +
                            (d_id - 1) * order_lines_per_district +
                            order_index * TPCC_MAX_ORDER_LINES_PER_ORDER);

    // Read actual number of order lines (o_ol_cnt ranges from 5-15)
    const auto ol_count = std::min<int32_t>(found_order.o_ol_cnt,
                                             static_cast<int32_t>(TPCC_MAX_ORDER_LINES_PER_ORDER));
    for (int32_t i = 0; i < ol_count; ++i) {
      OrderLineRow ol_row;
      bytes += read_row(db.order_line, buffer_manager, (ol_base + i) % db.order_line.total_rows, &ol_row);
    }
  }

  return bytes;
}

inline uint64_t execute_delivery(const TPCCDatabasePages& db, BufferManager& buffer_manager, int32_t w_id,
                                 int32_t carrier_id) {
  uint64_t bytes = 0;

  const auto new_orders_per_warehouse =
      static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE) * NUM_NEW_ORDERS_PER_DISTRICT;
  const auto orders_per_warehouse = static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE) * NUM_ORDERS_PER_DISTRICT;
  const auto order_lines_per_district =
      static_cast<size_t>(NUM_ORDERS_PER_DISTRICT) * TPCC_MAX_ORDER_LINES_PER_ORDER;
  const auto order_lines_per_warehouse =
      static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE) * order_lines_per_district;

  // Process all 10 districts
  for (int32_t d_id = 1; d_id <= NUM_DISTRICTS_PER_WAREHOUSE; ++d_id) {
    // 1. Find oldest NEW_ORDER (SELECT MIN(NO_O_ID)) by scanning the district's range
    const auto district_base = static_cast<size_t>((w_id - 1) * new_orders_per_warehouse +
                                                   (d_id - 1) * NUM_NEW_ORDERS_PER_DISTRICT);
    int32_t min_no_o_id = std::numeric_limits<int32_t>::max();
    size_t min_no_idx = 0;
    bool found_no = false;

    for (size_t i = 0; i < static_cast<size_t>(NUM_NEW_ORDERS_PER_DISTRICT); ++i) {
      const auto no_idx = district_base + i;
      NewOrderRow no_row;
      bytes += read_row(db.new_order, buffer_manager, no_idx, &no_row);
      if (no_row.no_o_id > 0 && no_row.no_o_id < min_no_o_id) {
        min_no_o_id = no_row.no_o_id;
        min_no_idx = no_idx;
        found_no = true;
      }
    }

    if (!found_no) {
      continue;
    }

    NewOrderRow no_row;
    bytes += read_row(db.new_order, buffer_manager, min_no_idx, &no_row);

    // 2. DELETE NEW_ORDER
    bytes += write_row(db.new_order, buffer_manager, min_no_idx, NewOrderRow{0, 0, 0});

    // 3. Read ORDER to get O_C_ID and O_OL_CNT
    const auto o_idx = static_cast<size_t>((w_id - 1) * orders_per_warehouse +
                                           (d_id - 1) * NUM_ORDERS_PER_DISTRICT +
                                           ((no_row.no_o_id - 1) % NUM_ORDERS_PER_DISTRICT));
    OrderRow o_row;
    bytes += read_row(db.orders, buffer_manager, o_idx % db.orders.total_rows, &o_row);

    // 4. UPDATE ORDER (set carrier)
    bytes += update_field(db.orders, buffer_manager, o_idx % db.orders.total_rows, &OrderRow::o_carrier_id, carrier_id);

    // 5. Read ORDER_LINEs and calculate SUM(OL_AMOUNT)
    // Use actual o_ol_cnt (variable 5-15)
    const auto ol_count = std::min<int32_t>(o_row.o_ol_cnt, static_cast<int32_t>(TPCC_MAX_ORDER_LINES_PER_ORDER));
    const auto order_index = static_cast<size_t>((no_row.no_o_id - 1) % NUM_ORDERS_PER_DISTRICT);

    float total_amount = 0.0f;  // Dynamic SUM(OL_AMOUNT)

    for (int ol = 0; ol < ol_count; ++ol) {
      const auto ol_idx = static_cast<size_t>((w_id - 1) * order_lines_per_warehouse +
                                              (d_id - 1) * order_lines_per_district +
                                              order_index * TPCC_MAX_ORDER_LINES_PER_ORDER + ol);
      OrderLineRow ol_row;
      bytes += read_row(db.order_line, buffer_manager, ol_idx % db.order_line.total_rows, &ol_row);

      // Accumulate order line amount for customer balance update
      total_amount += ol_row.ol_amount;

      // 6. UPDATE ORDER_LINE (set delivery date)
      bytes += update_field(db.order_line, buffer_manager, ol_idx % db.order_line.total_rows,
                            &OrderLineRow::ol_delivery_d, 1);
    }

    // 7. UPDATE CUSTOMER with actual order total (not hardcoded 100.0f)
    const auto c_idx =
        static_cast<size_t>((w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE * NUM_CUSTOMERS_PER_DISTRICT +
                            (d_id - 1) * NUM_CUSTOMERS_PER_DISTRICT + (o_row.o_c_id - 1));
    CustomerRow c_row;
    bytes += read_row(db.customer, buffer_manager, c_idx, &c_row);
    bytes += update_field(db.customer, buffer_manager, c_idx, &CustomerRow::c_balance, c_row.c_balance + total_amount);
    bytes += update_field(db.customer, buffer_manager, c_idx, &CustomerRow::c_delivery_cnt,
                          c_row.c_delivery_cnt + 1);
  }

  return bytes;
}

inline uint64_t execute_stock_level(const TPCCDatabasePages& db, BufferManager& buffer_manager, int32_t w_id,
                                    int32_t d_id, int32_t threshold) {
  uint64_t bytes = 0;

  const auto d_idx =
      static_cast<size_t>((w_id - 1) * NUM_DISTRICTS_PER_WAREHOUSE + (d_id - 1));
  DistrictRow d_row;
  bytes += read_row(db.district, buffer_manager, d_idx, &d_row);
  const auto next_o_id = d_row.d_next_o_id;

  std::set<int32_t> item_ids;
  const auto start_o_id = std::max<int32_t>(1, next_o_id - 20);
  const auto order_lines_per_district =
      static_cast<size_t>(NUM_ORDERS_PER_DISTRICT) * TPCC_MAX_ORDER_LINES_PER_ORDER;
  const auto order_lines_per_warehouse =
      static_cast<size_t>(NUM_DISTRICTS_PER_WAREHOUSE) * order_lines_per_district;

  for (int32_t o_id = start_o_id; o_id < next_o_id; ++o_id) {
    const auto order_index = static_cast<size_t>((o_id - 1) % NUM_ORDERS_PER_DISTRICT);
    for (size_t ol = 0; ol < TPCC_MAX_ORDER_LINES_PER_ORDER; ++ol) {
      const auto ol_idx = static_cast<size_t>((w_id - 1) * order_lines_per_warehouse +
                                              (d_id - 1) * order_lines_per_district +
                                              order_index * TPCC_MAX_ORDER_LINES_PER_ORDER + ol);
      OrderLineRow ol_row;
      bytes += read_row(db.order_line, buffer_manager, ol_idx % db.order_line.total_rows, &ol_row);
      item_ids.insert(ol_row.ol_i_id);
    }
  }

  for (const auto i_id : item_ids) {
    const auto s_idx = static_cast<size_t>((w_id - 1) * NUM_STOCK_ITEMS_PER_WAREHOUSE + (i_id - 1));
    StockRow s_row;
    bytes += read_row(db.stock, buffer_manager, s_idx % db.stock.total_rows, &s_row);
    auto below_threshold = (s_row.s_quantity < threshold);
    benchmark::DoNotOptimize(below_threshold);
  }

  return bytes;
}

// Generate TPC-C transaction workload
inline TPCCTransactions generate_tpcc_transactions(size_t num_transactions, size_t num_warehouses, float = 0.9f) {
  TPCCTransactions txns;
  txns.reserve(num_transactions);

  std::mt19937 rng{std::random_device{}()};
  std::discrete_distribution<int> type_dist({4, 45, 4, 43, 4});
  std::uniform_int_distribution<int32_t> warehouse_dist{1, static_cast<int32_t>(num_warehouses)};
  std::uniform_int_distribution<int32_t> district_dist{1, NUM_DISTRICTS_PER_WAREHOUSE};
  std::uniform_int_distribution<int32_t> carrier_dist{MIN_CARRIER_ID, MAX_CARRIER_ID};
  std::uniform_int_distribution<int32_t> threshold_dist{10, 20};
  std::uniform_int_distribution<int32_t> item_count_dist{MIN_ORDER_LINE_COUNT, MAX_ORDER_LINE_COUNT};
  std::uniform_int_distribution<int32_t> quantity_dist{1, MAX_ORDER_LINE_QUANTITY};
  std::uniform_int_distribution<int32_t> remote_roll_dist{1, 100};
  std::uniform_real_distribution<float> payment_dist{1.0f, 5000.0f};

  // For 40% name-based lookup and 1% rollback
  std::uniform_int_distribution<int> lookup_type_dist{1, 100};
  std::uniform_int_distribution<int> rollback_dist{1, 100};

  TPCCRandomGenerator tpcc_rng;

  for (size_t i = 0; i < num_transactions; ++i) {
    TPCCTransactionParams p{};
    p.type = static_cast<TPCCTransactionType>(type_dist(rng));
    p.w_id = warehouse_dist(rng);
    p.d_id = district_dist(rng);
    p.c_id = static_cast<int32_t>(tpcc_rng.nurand(1023, 1, NUM_CUSTOMERS_PER_DISTRICT));
    p.c_w_id = p.w_id;
    p.c_d_id = p.d_id;

    if (p.type == TPCCTransactionType::NewOrder) {
      const auto ol_cnt = item_count_dist(rng);
      p.items.reserve(static_cast<size_t>(ol_cnt));
      for (int32_t j = 0; j < ol_cnt; ++j) {
        NewOrderItem item;
        item.i_id = static_cast<int32_t>(tpcc_rng.nurand(8191, 1, NUM_ITEMS));
        item.supply_w_id = p.w_id;
        if (num_warehouses > 1 && remote_roll_dist(rng) == 1) {  // ~1% remote items
          do {
            item.supply_w_id = warehouse_dist(rng);
          } while (item.supply_w_id == p.w_id);
        }
        item.quantity = quantity_dist(rng);
        p.items.push_back(item);
      }
      // 1% of NewOrder transactions have an invalid item (rollback)
      p.has_invalid_item = (rollback_dist(rng) == 1);
    } else if (p.type == TPCCTransactionType::Payment) {
      p.h_amount = payment_dist(rng);
      if (num_warehouses > 1 && remote_roll_dist(rng) <= 15) {  // 15% remote customers
        do {
          p.c_w_id = warehouse_dist(rng);
        } while (p.c_w_id == p.w_id);
        p.c_d_id = district_dist(rng);
      }
      // 60% of Payment transactions use name-based lookup
      if (lookup_type_dist(rng) <= 60) {
        p.lookup_by_name = true;
        // Generate a last name using NURand for realistic distribution
        p.c_last = tpcc_rng.last_name(static_cast<size_t>(tpcc_rng.nurand(255, 0, 999)));
      }
    } else if (p.type == TPCCTransactionType::OrderStatus) {
      // 60% of OrderStatus transactions use name-based lookup
      if (lookup_type_dist(rng) <= 60) {
        p.lookup_by_name = true;
        p.c_last = tpcc_rng.last_name(static_cast<size_t>(tpcc_rng.nurand(255, 0, 999)));
      }
    } else if (p.type == TPCCTransactionType::Delivery) {
      p.carrier_id = carrier_dist(rng);
    } else if (p.type == TPCCTransactionType::StockLevel) {
      p.threshold = threshold_dist(rng);
    }

    txns.push_back(std::move(p));
  }

  return txns;
}

// Execute a single TPC-C transaction on buffer manager pages
inline uint64_t execute_tpcc_transaction(const TPCCDatabasePages& db, BufferManager& buffer_manager,
                                         const TPCCTransactionParams& txn) {
  switch (txn.type) {
    case TPCCTransactionType::NewOrder:
      return execute_new_order(db, buffer_manager, txn.w_id, txn.d_id, txn.c_id, txn.items, txn.has_invalid_item);
    case TPCCTransactionType::Payment:
      return execute_payment(db, buffer_manager, txn.w_id, txn.d_id, txn.c_w_id, txn.c_d_id, txn.c_id, txn.h_amount,
                             txn.lookup_by_name, txn.c_last);
    case TPCCTransactionType::OrderStatus:
      return execute_order_status(db, buffer_manager, txn.w_id, txn.d_id, txn.c_id, txn.lookup_by_name, txn.c_last);
    case TPCCTransactionType::Delivery:
      return execute_delivery(db, buffer_manager, txn.w_id, txn.carrier_id);
    case TPCCTransactionType::StockLevel:
      return execute_stock_level(db, buffer_manager, txn.w_id, txn.d_id, txn.threshold);
  }

  return 0;
}

// Run TPC-C benchmark
template <typename Fixture>
inline void run_tpcc(Fixture& fixture, benchmark::State& state) {
  micro_benchmark_clear_cache();

  auto bytes_processed = uint64_t{0};

  hdr_histogram* local_latency_histogram;
  init_histogram(&local_latency_histogram);

  std::map<TPCCTransactionType, uint64_t> transaction_counts;

  for (auto _ : state) {
    const auto start = state.thread_index() * fixture.operations_per_thread;
    const auto end = start + fixture.operations_per_thread;
    for (auto i = start; i < end; ++i) {
      const auto& txn = fixture.transactions[i];
      transaction_counts[txn.type]++;
      const auto timer_start = std::chrono::high_resolution_clock::now();
      bytes_processed += execute_tpcc_transaction(fixture.database, fixture.buffer_manager, txn);
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

    // TPC-C specific counters
    state.counters["new_order_txns"] = transaction_counts[TPCCTransactionType::NewOrder];
    state.counters["payment_txns"] = transaction_counts[TPCCTransactionType::Payment];
    state.counters["order_status_txns"] = transaction_counts[TPCCTransactionType::OrderStatus];
    state.counters["delivery_txns"] = transaction_counts[TPCCTransactionType::Delivery];
    state.counters["stock_level_txns"] = transaction_counts[TPCCTransactionType::StockLevel];
  }
}

}  // namespace hyrise
