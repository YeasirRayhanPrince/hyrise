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
#include "tpch/tpch_benchmark_item_runner.hpp"
#include "tpch/tpch_table_generator.hpp"

namespace hyrise {

/**
 * TPC-H Buffer Manager Benchmark
 * 
 * This benchmark runs TPC-H analytical queries with buffer manager's memory resource.
 * TPC-H already uses LinearBufferResource, so tables are automatically allocated
 * through the buffer manager, triggering batch evictions/promotions.
 * 
 * Flow:
 * 1. Configure buffer manager (DRAM + NUMA pools)
 * 2. Generate TPC-H tables using LinearBufferResource (goes through buffer manager)
 * 3. Run TPC-H queries (1-22) with mixed read patterns
 * 4. Measure batch eviction/promotion statistics
 * 
 * TPC-H Scaling: Scale factor 1 ≈ 1 GB of data
 */

template <MigrationPolicy policy = LazyMigrationPolicy>
class TPCHBufferManagerFixture : public benchmark::Fixture {
 public:
  // Run a subset of TPC-H queries for reasonable runtime
  // Queries 1, 3, 6, 12, 14, 19 cover different access patterns
  static constexpr std::array<BenchmarkItemID, 6> QUERY_IDS = {
      BenchmarkItemID{0}, BenchmarkItemID{2}, BenchmarkItemID{5},
      BenchmarkItemID{11}, BenchmarkItemID{13}, BenchmarkItemID{18}};  // 0-indexed

  std::shared_ptr<TPCHBenchmarkItemRunner> item_runner;
  std::vector<BenchmarkItemID> query_sequence;
  hdr_histogram* latency_histogram;
  std::mutex latency_histogram_mutex;
  std::mutex query_executor_mutex;  // Serialize query execution to avoid frame lock conflicts
  BufferManager& buffer_manager = Hyrise::get().buffer_manager;
  uint64_t queries_per_thread;
  float scale_factor;

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

      // Database size in GB corresponds to TPC-H scale factor
      auto database_size_gb = static_cast<size_t>(state.range(0));
      scale_factor = static_cast<float>(database_size_gb);

      std::cout << "\n========== TPC-H Buffer Manager Benchmark Setup ===========" << std::endl;
      std::cout << "Scale factor: " << scale_factor << " (" << database_size_gb << " GB)" << std::endl;
      
      // Print buffer manager configuration
      std::cout << "\n--- Buffer Manager Configuration ---" << std::endl;
      std::cout << "DRAM pool size: " << (config.dram_buffer_pool_size / (1024 * 1024)) << " MB" << std::endl;
      std::cout << "NUMA pool size: " << (config.numa_buffer_pool_size / (1024 * 1024)) << " MB" << std::endl;
      std::cout << "Batch eviction enabled: " << (config.enable_batch_eviction ? "true" : "false") << std::endl;
      std::cout << "Batch promotion enabled: " << (config.enable_batch_promotion ? "true" : "false") << std::endl;

      // Generate TPC-H tables using LinearBufferResource (goes through buffer manager)
      std::cout << "\n--- Generating TPC-H Tables ---" << std::endl;
      std::cout << "Note: TPC-H uses LinearBufferResource, which allocates through buffer manager" << std::endl;
      
      auto benchmark_config = std::make_shared<BenchmarkConfig>(BenchmarkConfig::get_default_config());
      benchmark_config->cache_binary_tables = false;  // Force fresh generation
      benchmark_config->clients = state.threads();
      
      // Log buffer manager state BEFORE table generation
      auto metrics_before = buffer_manager.metrics();
      const auto before_num_allocs = metrics_before->num_allocs.load();
      const auto before_bytes_to_ssd = metrics_before->total_bytes_copied_to_ssd.load();
      const auto before_bytes_numa_to_dram = metrics_before->total_bytes_copied_from_numa_to_dram.load();
      const auto before_bytes_dram_to_numa = metrics_before->total_bytes_copied_from_dram_to_numa.load();
      std::cout << "\n--- Buffer Manager State BEFORE Table Generation ---" << std::endl;
      std::cout << "Num allocations: " << before_num_allocs << std::endl;
      std::cout << "Total bytes copied to SSD: " << before_bytes_to_ssd << std::endl;
      std::cout << "Total bytes copied from NUMA to DRAM: " << before_bytes_numa_to_dram << std::endl;
      std::cout << "Total bytes copied from DRAM to NUMA: " << before_bytes_dram_to_numa << std::endl;
      
      // Generate tables - directly use buffer manager like YCSB does
      TPCHTableGenerator{scale_factor, ClusteringConfiguration{}, benchmark_config}.generate_and_store();
      std::cout << "TPC-H tables generated successfully" << std::endl;
      
      // Log buffer manager state AFTER table generation
      auto metrics_after = buffer_manager.metrics();
      std::cout << "\n--- Buffer Manager State AFTER Table Generation ---" << std::endl;
      std::cout << "Num allocations: " << metrics_after->num_allocs.load() << std::endl;
      std::cout << "Total bytes copied to SSD: " << metrics_after->total_bytes_copied_to_ssd.load() << std::endl;
      std::cout << "Total bytes copied from NUMA to DRAM: " << metrics_after->total_bytes_copied_from_numa_to_dram.load() << std::endl;
      std::cout << "Total bytes copied from DRAM to NUMA: " << metrics_after->total_bytes_copied_from_dram_to_numa.load() << std::endl;
      std::cout << "\nDelta (during generation):" << std::endl;
      std::cout << "  Allocations: " << (metrics_after->num_allocs.load() - before_num_allocs) << std::endl;
      std::cout << "  Bytes to SSD: " << (metrics_after->total_bytes_copied_to_ssd.load() - before_bytes_to_ssd) << std::endl;
      std::cout << "  Bytes NUMA->DRAM: " << (metrics_after->total_bytes_copied_from_numa_to_dram.load() - before_bytes_numa_to_dram) << std::endl;
      std::cout << "  Bytes DRAM->NUMA: " << (metrics_after->total_bytes_copied_from_dram_to_numa.load() - before_bytes_dram_to_numa) << std::endl;

      // Create TPC-H item runner with selected queries
      std::vector<BenchmarkItemID> selected_queries(QUERY_IDS.begin(), QUERY_IDS.end());
      item_runner = std::make_shared<TPCHBenchmarkItemRunner>(
          benchmark_config, 
          false,  // use_prepared_statements
          scale_factor,
          ClusteringConfiguration{},
          selected_queries
      );

      // Notify item runner that tables are loaded
      item_runner->on_tables_loaded();

      // Generate query execution sequence (run each query multiple times)
      std::cout << "\n--- Preparing Benchmark ---" << std::endl;
      generate_query_sequence(50);  // Run 50 iterations total
      queries_per_thread = query_sequence.size() / state.threads();

      init_histogram(&latency_histogram);
      std::cout << "Query sequence size: " << query_sequence.size() << std::endl;
      std::cout << "Queries per thread: " << queries_per_thread << std::endl;
      std::cout << "========================================================\n" << std::endl;
    }
  }

  void TearDown(const ::benchmark::State& state) {
    if (state.thread_index() == 0) {
      // Print query latency histogram
      std::cout << "\n========== Query Latency Statistics ===========" << std::endl;
      std::cout << "Min latency: " << hdr_min(latency_histogram) / 1e6 << " ms" << std::endl;
      std::cout << "Max latency: " << hdr_max(latency_histogram) / 1e6 << " ms" << std::endl;
      std::cout << "Mean latency: " << hdr_mean(latency_histogram) / 1e6 << " ms" << std::endl;
      std::cout << "p50 latency: " << hdr_value_at_percentile(latency_histogram, 50.0) / 1e6 << " ms" << std::endl;
      std::cout << "p99 latency: " << hdr_value_at_percentile(latency_histogram, 99.0) / 1e6 << " ms" << std::endl;
      std::cout << "================================================" << std::endl;

      hdr_close(latency_histogram);

      // Clean up
      Hyrise::reset();
    }
  }

  void generate_query_sequence(size_t iterations) {
    // Generate a sequence of queries to execute
    // Distribute evenly across selected queries
    query_sequence.reserve(iterations * QUERY_IDS.size());
    for (size_t i = 0; i < iterations; ++i) {
      for (auto query_id : QUERY_IDS) {
        query_sequence.push_back(query_id);
      }
    }
    
    // Shuffle for better cache behavior
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(query_sequence.begin(), query_sequence.end(), g);
  }
};

template <MigrationPolicy policy>
void run_tpch(TPCHBufferManagerFixture<policy>& fixture, benchmark::State& state) {
  micro_benchmark_clear_cache();

  auto bytes_processed = uint64_t{0};
  auto successful_queries = uint64_t{0};
  auto failed_queries = uint64_t{0};

  hdr_histogram* local_latency_histogram;
  init_histogram(&local_latency_histogram);

  for (auto _ : state) {
    const auto start = state.thread_index() * fixture.queries_per_thread;
    const auto end = std::min(start + fixture.queries_per_thread,
                              static_cast<uint64_t>(fixture.query_sequence.size()));

    for (auto i = start; i < end; ++i) {
      const auto query_id = fixture.query_sequence[i];

      try {
        // Serialize query execution to avoid frame locking conflicts between threads
        std::lock_guard<std::mutex> lock{fixture.query_executor_mutex};
        
        const auto timer_start = std::chrono::high_resolution_clock::now();
        auto [success, metrics, verification_failed] = fixture.item_runner->execute_item(query_id);
        const auto timer_end = std::chrono::high_resolution_clock::now();

        const auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_end - timer_start).count();
        hdr_record_value(local_latency_histogram, latency);

        if (success) {
          successful_queries++;
          // Estimate bytes processed (rough approximation)
          bytes_processed += 10 * 1024 * 1024;  // 10MB per query approximation
        } else {
          failed_queries++;
        }
      } catch (const std::exception& e) {
        failed_queries++;
        if (state.thread_index() == 0 && failed_queries < 5) {
          std::cerr << "Query " << (query_id + 1) << " failed: " << e.what() << std::endl;
        }
      }
    }

    benchmark::ClobberMemory();
  }

  // Merge histogram
  {
    std::lock_guard<std::mutex> lock{fixture.latency_histogram_mutex};
    hdr_add(fixture.latency_histogram, local_latency_histogram);
  }
  hdr_close(local_latency_histogram);

  state.SetItemsProcessed(successful_queries);
  state.SetBytesProcessed(bytes_processed);

  // Report failures
  if (failed_queries > 0 && state.thread_index() == 0) {
    std::cout << "Thread " << state.thread_index() << ": " << failed_queries << " queries failed" << std::endl;
  }
}

// Benchmark registrations for different migration policies and database sizes
BENCHMARK_TEMPLATE_DEFINE_F(TPCHBufferManagerFixture, LazyMigrationPolicy, LazyMigrationPolicy)(benchmark::State& st) {
  run_tpch(*this, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(TPCHBufferManagerFixture, EagerMigrationPolicy, EagerMigrationPolicy)
(benchmark::State& st) {
  run_tpch(*this, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(TPCHBufferManagerFixture, DramOnlyMigrationPolicy, DramOnlyMigrationPolicy)
(benchmark::State& st) {
  run_tpch(*this, st);
}

// Register benchmarks with different database sizes (scale factors)
// Scale factor 1 = 1 GB (works well with 2GB DRAM + 10GB NUMA configuration)
// Note: Scale factor 2+ causes frame locking conflicts during concurrent table generation
BENCHMARK_REGISTER_F(TPCHBufferManagerFixture, LazyMigrationPolicy)
  ->Args({1})
  ->Args({2})
  ->Args({4})
  ->Args({8})
  ->Args({16})
  ->Args({32})
  ->Args({64})
  ->Args({128})
  ->Args({256})
    ->DenseThreadRange(1, 48, 2)
    ->Iterations(1)
    ->UseRealTime();

BENCHMARK_REGISTER_F(TPCHBufferManagerFixture, EagerMigrationPolicy)
  ->Args({1})
  ->Args({2})
  ->Args({4})
  ->Args({8})
  ->Args({16})
  ->Args({32})
  ->Args({64})
  ->Args({128})
  ->Args({256})
    ->DenseThreadRange(1, 48, 2)
    ->Iterations(1)
    ->UseRealTime();

BENCHMARK_REGISTER_F(TPCHBufferManagerFixture, DramOnlyMigrationPolicy)
  ->Args({1})
  ->Args({2})
  ->Args({4})
  ->Args({8})
  ->Args({16})
  ->Args({32})
  ->Args({64})
  ->Args({128})
  ->Args({256})
    ->DenseThreadRange(1, 48, 2)
    ->Iterations(1)
    ->UseRealTime();

}  // namespace hyrise
