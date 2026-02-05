#include <atomic>
#include <chrono>
#include <condition_variable>
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
#include "storage/buffer/migration_profiler.hpp"

namespace hyrise {

/**
 * TPC-C Buffer Manager Benchmark with Synthetic Workload
 *
 * This benchmark allocates TPC-C table data through the buffer manager and executes
 * synthetic TPC-C transactions that operate directly on buffer manager pages.
 * This follows the same pattern as the YCSB benchmark.
 *
 * Flow:
 * 1. Allocate all 9 TPC-C tables through buffer manager (creates memory pressure)
 * 2. Generate TPC-C transaction workload (45% NewOrder, 43% Payment, etc.)
 * 3. Execute transactions as buffer manager page operations
 * 4. Measure batch eviction/promotion statistics
 *
 * TPC-C Scaling: 1 warehouse ≈ 100MB of table data
 */

template <MigrationPolicy policy = LazyMigrationPolicy>
class TPCCBufferManagerFixture : public benchmark::Fixture {
 public:
  constexpr static auto NUM_OPERATIONS = 1 * 1000 * 1000;  // 1M transactions

  TPCCDatabasePages database;
  TPCCTransactions transactions;
  hdr_histogram* latency_histogram;
  std::mutex latency_histogram_mutex;
  BufferManager& buffer_manager = Hyrise::get().buffer_manager;
  uint64_t operations_per_thread;
  size_t num_warehouses;

  void SetUp(const ::benchmark::State& state) {
    static std::mutex load_mutex;
    static std::condition_variable load_cv;
    static std::atomic<bool> load_started{false};
    static std::atomic<bool> load_complete{false};

    if (state.thread_index() == 0) {
      auto expected = false;
      if (load_started.compare_exchange_strong(expected, true)) {
        auto config = BufferManager::Config::from_env();
        config.cpu_node = NodeID{0};
        config.memory_node = NodeID{1};

        // Only override migration_policy if NOT using CustomMigrationPolicy
        if constexpr (policy != CustomMigrationPolicy) {
          config.migration_policy = policy;
        }
        config.enable_numa = (policy != DramOnlyMigrationPolicy);

        Hyrise::get().buffer_manager = BufferManager(config);

        // Database size in GB determines number of warehouses
        // TPC-C: 1 warehouse ≈ ~100 MB of data
        auto database_size_gb = static_cast<size_t>(state.range(0));
        if (database_size_gb <= 2) {
          num_warehouses = database_size_gb;  // 1-2 warehouses for quick testing
        } else {
          num_warehouses = database_size_gb * 5;  // 5 warehouses per GB for larger sizes
        }

        std::cout << "\n========== TPC-C Buffer Manager Benchmark Setup ==========" << std::endl;
        std::cout << "Database size: " << database_size_gb << " GB" << std::endl;
        std::cout << "Warehouses: " << num_warehouses << std::endl;
        std::cout << "Threads: " << state.threads() << std::endl;

        // Generate TPC-C database through buffer manager
        std::cout << "\n--- Generating TPC-C Database (Buffer Manager) ---" << std::endl;
        database = generate_tpcc_database(&buffer_manager, num_warehouses, config.loader_threads);

        // Generate transaction workload
        std::cout << "\n--- Generating Transaction Workload ---" << std::endl;
        transactions = generate_tpcc_transactions(NUM_OPERATIONS, num_warehouses, 0.9);
        operations_per_thread = transactions.size() / state.threads();

        init_histogram(&latency_histogram);

        // Clear profiler data for clean benchmarking
        g_migration_profiler.clear();

        std::cout << "Transaction count: " << transactions.size() << std::endl;
        std::cout << "Operations per thread: " << operations_per_thread << std::endl;
        std::cout << "========================================================\n" << std::endl;

        load_complete.store(true, std::memory_order_release);
        load_cv.notify_all();
      } else {
        std::unique_lock<std::mutex> lock(load_mutex);
        load_cv.wait(lock, [&]() { return load_complete.load(std::memory_order_acquire); });
      }
    } else {
      std::unique_lock<std::mutex> lock(load_mutex);
      load_cv.wait(lock, [&]() { return load_complete.load(std::memory_order_acquire); });
    }
  }

  void TearDown(const ::benchmark::State& state) {
    if (state.thread_index() == 0) {
      hdr_close(latency_histogram);
    }
  }
};

#define CONFIGURE_TPCC_BENCHMARK(Policy)                                                  \
  BENCHMARK_TEMPLATE_DEFINE_F(TPCCBufferManagerFixture, BM_tpcc_##Policy, Policy)         \
  (benchmark::State & state) {                                                            \
    run_tpcc(*this, state);                                                               \
  }                                                                                       \
  BENCHMARK_REGISTER_F(TPCCBufferManagerFixture, BM_tpcc_##Policy)                        \
      ->DenseThreadRange(1, 48, 2)                                                        \
      ->Iterations(1)                                                                     \
      ->Repetitions(1)                                                                    \
      ->UseRealTime()                                                                     \
      ->Args({1})->Args({2})->Args({4})->Args({8})->Args({16})->Args({32})->Args({64})->Args({128})->Args({256}) \
      ->Name("BM_tpcc/" #Policy);

CONFIGURE_TPCC_BENCHMARK(LazyMigrationPolicy)
CONFIGURE_TPCC_BENCHMARK(EagerMigrationPolicy)
CONFIGURE_TPCC_BENCHMARK(DramOnlyMigrationPolicy)
CONFIGURE_TPCC_BENCHMARK(NumaOnlyMigrationPolicy)
CONFIGURE_TPCC_BENCHMARK(CustomMigrationPolicy)

}  // namespace hyrise
