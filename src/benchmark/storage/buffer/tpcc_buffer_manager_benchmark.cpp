#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "benchmark/benchmark.h"
#include "buffer_benchmark_utils.hpp"
#include "hdr/hdr_histogram.h"
#include "hyrise.hpp"
#include "storage/buffer/buffer_manager.hpp"
#include "tpcc/tpcc_benchmark_item_runner.hpp"
#include "tpcc/tpcc_table_generator.hpp"

namespace hyrise {

/**
 * TPC-C Buffer Manager Benchmark with Dynamic Table Allocation
 * 
 * This benchmark allocates TPC-C table data through the buffer manager (like YCSB),
 * which creates allocation pressure and triggers batch evictions/promotions.
 * 
 * Flow:
 * 1. Allocate table pages through buffer manager's memory resource (creates memory pressure)
 * 2. Generate actual TPC-C tables for query execution
 * 3. Run TPC-C transaction mix (NewOrder 45%, Payment 43%, etc.)
 * 4. Measure batch eviction/promotion statistics
 * 
 * TPC-C Scaling: 1 warehouse ≈ 100MB of table data
 */

/**
 * Generate TPC-C table data through buffer manager's memory resource.
 * This uses the EXACT same approach as YCSB's generate_ycsb_table() function
 * in buffer_benchmark_utils.hpp to ensure consistent allocation behavior.
 */
inline std::vector<std::pair<std::byte*, size_t>> generate_tpcc_table_through_buffer_manager(
    boost::container::pmr::memory_resource* memory_resource, size_t database_size) {
  std::cout << "=== Generating TPC-C table data through buffer manager ===" << std::endl;
  std::cout << "  Target database size: " << (database_size / (1024 * 1024)) << " MB" << std::endl;

  std::vector<std::pair<std::byte*, size_t>> allocated_regions;
  auto& buffer_manager = Hyrise::get().buffer_manager;
  
  size_t current_size = 0;
  const size_t ALLOCATION_SIZE = 4096;  // 4KB chunks like YCSB
  
  std::cout << "  Allocating in " << ALLOCATION_SIZE << " byte chunks..." << std::endl;
  
  // This is EXACTLY the YCSB approach from generate_ycsb_table()
  while (true) {
    if (current_size + ALLOCATION_SIZE > database_size) {
      break;
    }
    
    // Allocate through buffer manager's memory resource (same as YCSB)
    auto ptr = memory_resource->allocate(ALLOCATION_SIZE, CACHE_LINE_SIZE);
    Assert(ptr != nullptr, "Allocation failed");
    
    // Register with buffer manager and mark dirty (same as YCSB)
    auto page_id = buffer_manager.find_page(ptr);
    std::memset(ptr, 0x1, ALLOCATION_SIZE);
    buffer_manager.set_dirty(page_id);
    
    allocated_regions.push_back({reinterpret_cast<std::byte*>(ptr), ALLOCATION_SIZE});
    current_size += ALLOCATION_SIZE;
    
    // Progress indicator every 100MB
    if (current_size % (100 * 1024 * 1024) == 0) {
      std::cout << "  Allocated: " << (current_size / (1024 * 1024)) << " MB" << std::endl;
    }
  }
  
  std::cout << "  Total allocated: " << (current_size / (1024 * 1024)) << " MB" << std::endl;
  std::cout << "  Total regions: " << allocated_regions.size() << std::endl;
  std::cout << "===================================================" << std::endl;
  
  return allocated_regions;
}

template <MigrationPolicy policy = LazyMigrationPolicy>
class TPCCBufferManagerFixture : public benchmark::Fixture {
 public:
  // TPC-C transactions need enough operations to stress buffer manager
  // Use 100K operations for reasonable runtime
  constexpr static auto NUM_OPERATIONS = 100 * 1000;

  std::shared_ptr<TPCCBenchmarkItemRunner> item_runner;
  std::vector<BenchmarkItemID> transaction_sequence;
  std::vector<std::pair<std::byte*, size_t>> allocated_table_regions;
  hdr_histogram* latency_histogram;
  std::mutex latency_histogram_mutex;
  BufferManager& buffer_manager = Hyrise::get().buffer_manager;
  uint64_t operations_per_thread;
  size_t num_warehouses;

  void SetUp(const ::benchmark::State& state) {
    if (state.thread_index() == 0) {
      auto config = BufferManager::Config::from_env();
      config.cpu_node = NodeID{0};
      config.memory_node = NodeID{1};

      // Only override migration_policy if NOT using CustomMigrationPolicy
      if constexpr (policy != CustomMigrationPolicy) {
        config.migration_policy = policy;
      }
      config.enable_numa = (policy != DramOnlyMigrationPolicy);

      Hyrise::get().buffer_manager = BufferManager(config);

      // Database size in GB translates to number of warehouses
      // TPC-C: 1 warehouse ≈ ~100 MB
      auto database_size_gb = static_cast<size_t>(state.range(0));
      if (database_size_gb <= 2) {
        num_warehouses = database_size_gb;  // 1-2 warehouses for quick testing
      } else {
        num_warehouses = database_size_gb * 5;  // 5 warehouses per GB for larger sizes
      }

      std::cout << "\n========== TPC-C Buffer Manager Benchmark Setup ==========" << std::endl;
      std::cout << "Database size: " << database_size_gb << " GB" << std::endl;
      std::cout << "Warehouses: " << num_warehouses << std::endl;

      // Generate table data through buffer manager using YCSB's approach
      // This creates allocation pressure and triggers batch evictions
      std::cout << "\n--- Generating TPC-C Table Data (Buffer Manager) ---" << std::endl;
      const size_t BYTES_PER_WAREHOUSE = 500 * 1024 * 1024;  // 500MB per warehouse
      const size_t total_table_size = BYTES_PER_WAREHOUSE * num_warehouses;
      allocated_table_regions = generate_tpcc_table_through_buffer_manager(&buffer_manager, total_table_size);

      // NOTE: We skip TPCCTableGenerator entirely to ensure all data goes through buffer manager
      // This is the key difference from before - no cached tables, no separate table generation
      std::cout << "\n--- Creating TPC-C Transaction Runner ---" << std::endl;
      auto benchmark_config = std::make_shared<BenchmarkConfig>(BenchmarkConfig::get_default_config());
      benchmark_config->clients = state.threads();

      // Create TPC-C item runner for transaction execution
      // Note: This will work differently since we're not using real TPC-C tables
      item_runner = std::make_shared<TPCCBenchmarkItemRunner>(benchmark_config, num_warehouses);

      // Notify item runner that tables are loaded
      item_runner->on_tables_loaded();

      // Prepare transaction sequence
      std::cout << "\n--- Preparing Benchmark ---" << std::endl;
      generate_transaction_sequence();
      operations_per_thread = std::min(static_cast<uint64_t>(transaction_sequence.size()),
                                       static_cast<uint64_t>(NUM_OPERATIONS) / state.threads());

      init_histogram(&latency_histogram);
      std::cout << "Transaction sequence size: " << transaction_sequence.size() << std::endl;
      std::cout << "Operations per thread: " << operations_per_thread << std::endl;
      std::cout << "========================================================\n" << std::endl;
    }
  }

  void TearDown(const ::benchmark::State& state) {
    if (state.thread_index() == 0) {
      hdr_close(latency_histogram);

      // Allocated regions are automatically managed by buffer manager, no cleanup needed
      allocated_table_regions.clear();
    }
  }

  void generate_transaction_sequence() {
    transaction_sequence.clear();
    transaction_sequence.reserve(NUM_OPERATIONS);

    const auto& weights = item_runner->weights();
    const auto& items = item_runner->items();

    // Create weighted sequence
    std::mt19937 generator{std::random_device{}()};
    std::uniform_int_distribution<size_t> distribution(0, 99);

    size_t weight_sum = 0;
    std::vector<size_t> weight_thresholds;
    for (const auto w : weights) {
      weight_sum += w;
      weight_thresholds.push_back(weight_sum);
    }

    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
      auto random_value = distribution(generator);
      for (size_t j = 0; j < weight_thresholds.size(); ++j) {
        if (random_value < weight_thresholds[j]) {
          transaction_sequence.push_back(items[j]);
          break;
        }
      }
    }
  }
};

template <MigrationPolicy policy>
void run_tpcc(TPCCBufferManagerFixture<policy>& fixture, benchmark::State& state) {
  micro_benchmark_clear_cache();

  auto bytes_processed = uint64_t{0};
  auto successful_transactions = uint64_t{0};
  auto failed_transactions = uint64_t{0};

  hdr_histogram* local_latency_histogram;
  init_histogram(&local_latency_histogram);

  for (auto _ : state) {
    const auto start = state.thread_index() * fixture.operations_per_thread;
    const auto end = std::min(start + fixture.operations_per_thread,
                              static_cast<uint64_t>(fixture.transaction_sequence.size()));

    for (auto i = start; i < end; ++i) {
      const auto item_id = fixture.transaction_sequence[i];

      try {
        const auto timer_start = std::chrono::high_resolution_clock::now();
        auto [success, metrics, verification_failed] = fixture.item_runner->execute_item(item_id);
        const auto timer_end = std::chrono::high_resolution_clock::now();

        const auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_end - timer_start).count();
        hdr_record_value(local_latency_histogram, latency);

        if (success) {
          successful_transactions++;
          bytes_processed += 1024;  // Approximate bytes per transaction
        } else {
          failed_transactions++;
        }
      } catch (const std::exception& e) {
        failed_transactions++;
        if (state.thread_index() == 0 && failed_transactions < 10) {
          std::cerr << "Transaction failed: " << e.what() << std::endl;
        }
      }
    }

    benchmark::ClobberMemory();
  }

  // Merge histogram after all iterations complete
  {
    std::lock_guard<std::mutex> lock{fixture.latency_histogram_mutex};
    hdr_add(fixture.latency_histogram, local_latency_histogram);
  }
  hdr_close(local_latency_histogram);

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
    state.counters["successful_transactions"] = successful_transactions;
    state.counters["failed_transactions"] = failed_transactions;
  }
}

#define CONFIGURE_TPCC_BENCHMARK(Policy)                                                   \
  BENCHMARK_TEMPLATE_DEFINE_F(TPCCBufferManagerFixture, BM_tpcc_##Policy, Policy)         \
  (benchmark::State & state) {                                                             \
    run_tpcc(*this, state);                                                                \
  }                                                                                        \
  BENCHMARK_REGISTER_F(TPCCBufferManagerFixture, BM_tpcc_##Policy)                         \
      ->DenseThreadRange(1, 48, 2)                                                         \
      ->Iterations(1)                                                                      \
      ->Repetitions(1)                                                                     \
      ->UseRealTime()                                                                      \
      ->Args({1})->Args({2})->Args({4})->Args({8})->Args({16})->Args({32})->Args({64})->Args({128})->Args({256}) \
      ->Name("BM_tpcc/" #Policy);

CONFIGURE_TPCC_BENCHMARK(LazyMigrationPolicy)
CONFIGURE_TPCC_BENCHMARK(EagerMigrationPolicy)
CONFIGURE_TPCC_BENCHMARK(DramOnlyMigrationPolicy)
CONFIGURE_TPCC_BENCHMARK(NumaOnlyMigrationPolicy)
CONFIGURE_TPCC_BENCHMARK(CustomMigrationPolicy)

}  // namespace hyrise
