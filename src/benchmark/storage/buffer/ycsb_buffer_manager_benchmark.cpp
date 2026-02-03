#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#include <random>
#include "benchmark/benchmark.h"
#include "buffer_benchmark_utils.hpp"
#include "hdr/hdr_histogram.h"
#include "hyrise.hpp"
#include "storage/buffer/buffer_manager.hpp"
#include "storage/buffer/jemalloc_resource.hpp"
#include "storage/buffer/migration_profiler.hpp"
#include "storage/buffer/zipfian_int_distribution.hpp"

namespace hyrise {

/**
 * Bechmark idea:
 * Sacle read ops with difedderne page size
 * Scale read ops with single page size and different DRAM size ratios
 * Use zipfian skews to test different hit and miss rates for single pahe size and diffeent dram size ratios
 * 
 * Partly taken from https://github.com/hpides/viper/tree/master
 * 
*/
template <YCSBWorkload WL, MigrationPolicy policy = LazyMigrationPolicy>
class YCSBBufferManagerFixture : public benchmark::Fixture {
 public:
  constexpr static auto PAGE_SIZE_TYPE = MIN_PAGE_SIZE_TYPE;
  constexpr static auto DEFAULT_DRAM_BUFFER_POOL_SIZE = 2UL * GB;
  constexpr static auto DEFAULT_NUMA_BUFFER_POOL_SIZE = 4UL * GB;

  constexpr static auto NUM_OPERATIONS = 100 * 1000 * 1000;  // 10 million operations (~few seconds runtime)

  YCSBTable table;
  YCSBOperations operations;
  hdr_histogram* latency_histogram;
  std::mutex latency_histogram_mutex;
  BufferManager& buffer_manager = Hyrise::get().buffer_manager;
  uint64_t operations_per_thread;

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
        // Only override migration_policy if NOT using CustomMigrationPolicy.
        // CustomMigrationPolicy uses the ratios from the JSON config file.
        if constexpr (policy != CustomMigrationPolicy) {
          config.migration_policy = policy;
        }
        config.enable_numa = (policy != DramOnlyMigrationPolicy);

        Hyrise::get().buffer_manager = BufferManager(config);

        auto database_size = state.range(0) * GB;
        table = generate_ycsb_table_parallel(&buffer_manager, database_size, config.loader_threads);
        operations = generate_ycsb_operations<WL, NUM_OPERATIONS>(table.size(), 0.9);
        operations_per_thread = operations.size() / state.threads();
        init_histogram(&latency_histogram);
        
        // Clear profiler data for clean benchmarking
        g_migration_profiler.clear();
        
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

  void warmup(benchmark::State& state) {
    auto start = state.thread_index() * operations_per_thread;
    auto end = start + operations_per_thread;
    for (auto i = start; i < end; ++i) {
      const auto op = operations[i];
      execute_ycsb_action(table, buffer_manager, op);
    }
  }
};

#define CONFIGURE_BENCHMARK(WL, Policy)                                                                 \
  BENCHMARK_TEMPLATE_DEFINE_F(YCSBBufferManagerFixture, BM_ycsb_##WL##Policy, YCSBWorkload::WL, Policy) \
  (benchmark::State & state) {                                                                          \
    run_ycsb(*this, state);                                                                             \
  }                                                                                                     \
  BENCHMARK_REGISTER_F(YCSBBufferManagerFixture, BM_ycsb_##WL##Policy)                                  \
      ->DenseThreadRange(1, 48, 1)                                                                      \
      ->Iterations(1)                                                                                   \
      ->Repetitions(1)                                                                                  \
      ->UseRealTime()                                                                                   \
      ->Args({1})->Args({2})->Args({4})->Args({8})->Args({16})->Args({32})->Args({64})->Args({128})->Args({256}) \
      ->Name("BM_ycsb/" #WL "/" #Policy);

CONFIGURE_BENCHMARK(UpdateHeavy, LazyMigrationPolicy)
CONFIGURE_BENCHMARK(ReadMostly, LazyMigrationPolicy)
CONFIGURE_BENCHMARK(Scan, LazyMigrationPolicy)

CONFIGURE_BENCHMARK(UpdateHeavy, EagerMigrationPolicy)
CONFIGURE_BENCHMARK(ReadMostly, EagerMigrationPolicy)
CONFIGURE_BENCHMARK(Scan, EagerMigrationPolicy)

CONFIGURE_BENCHMARK(UpdateHeavy, DramOnlyMigrationPolicy)
CONFIGURE_BENCHMARK(ReadMostly, DramOnlyMigrationPolicy)
CONFIGURE_BENCHMARK(Scan, DramOnlyMigrationPolicy)

CONFIGURE_BENCHMARK(UpdateHeavy, NumaOnlyMigrationPolicy)
CONFIGURE_BENCHMARK(ReadMostly, NumaOnlyMigrationPolicy)
CONFIGURE_BENCHMARK(Scan, NumaOnlyMigrationPolicy)

CONFIGURE_BENCHMARK(UpdateHeavy, CustomMigrationPolicy)
CONFIGURE_BENCHMARK(ReadMostly, CustomMigrationPolicy)
CONFIGURE_BENCHMARK(Scan, CustomMigrationPolicy)

}  // namespace hyrise