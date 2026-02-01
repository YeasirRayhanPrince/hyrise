# Batch Eviction Implementation

## Overview

This implementation adds **batch eviction** capability to the Hyrise buffer manager, allowing multiple pages to be evicted from DRAM to NUMA in a single operation using the `move_pages` syscall.

## Building and Running

### Building the Benchmark

```bash
# From the repository root
cd cmake-build-debug

# Build the buffer manager benchmark
make hyriseBenchmarkBufferManager -j$(nproc)

# The executable will be at: ./hyriseBenchmarkBufferManager
```

### Configuration

Create or edit `buffer_manager_config.json` in the repository root:

```json
{
  "dram_buffer_pool_size": 67108864,
  "numa_buffer_pool_size": 67108864,
  "enable_eviction_purge_worker": true,
  "enable_batching": true,
  "use_custom_syscall": false,
  "eager_migration_policy": {
    "dram_access_ratio": 0.0,
    "numa_access_ratio": 0.8
  },
  "lazy_migration_policy": {
    "dram_access_ratio": 0.0,
    "numa_access_ratio": 0.95
  }
}
```

**Key Configuration Options:**
- `enable_batching`: Enable batch eviction (set to `true` to use batching)
- `use_custom_syscall`: Reserved for future custom syscall optimization
- `dram_buffer_pool_size`: DRAM pool size in bytes (64 MiB = 67108864)
- `numa_buffer_pool_size`: NUMA pool size in bytes

### Running Benchmarks

**Run with batching enabled:**
```bash
cd cmake-build-debug
HYRISE_BUFFER_MANAGER_CONFIG_JSON_PATH=/users/yrayhan/hyrise/buffer_manager_config.json \
  ./hyriseBenchmarkBufferManager \
  --benchmark_filter="BM_ycsb/UpdateHeavy/EagerMigrationPolicy/1/iterations:1/repeats:1/real_time/threads:47"
```

**Run specific workloads:**
```bash
# UpdateHeavy workload (50% updates, 50% reads)
--benchmark_filter="BM_ycsb/UpdateHeavy/EagerMigrationPolicy/1/threads:47"

# ReadMostly workload (95% reads, 5% updates)
--benchmark_filter="BM_ycsb/ReadMostly/EagerMigrationPolicy/1/threads:47"

# Scan workload
--benchmark_filter="BM_ycsb/Scan/EagerMigrationPolicy/1/threads:47"
```

**Run with different migration policies:**
```bash
# Eager migration (migrates to NUMA aggressively)
--benchmark_filter="BM_ycsb/UpdateHeavy/EagerMigrationPolicy"

# Lazy migration (keeps in DRAM longer)
--benchmark_filter="BM_ycsb/UpdateHeavy/LazyMigrationPolicy"

# DRAM only (no NUMA tier)
--benchmark_filter="BM_ycsb/UpdateHeavy/DramOnlyMigrationPolicy"

# NUMA only (bypass DRAM)
--benchmark_filter="BM_ycsb/UpdateHeavy/NumaOnlyMigrationPolicy"
```

### Understanding the Log Output

When batching is enabled, you'll see log messages like:

```
[BufferPool] Using batch eviction path, bytes_required=4096, size_type=0
[BufferPool::evict_batch] Phase 2: Locked 5 pages, processing batch eviction by size type
[BufferPool::evict_batch] Attempting batch NUMA migration for 5 pages of size_type=0
[VolatileRegion::move_pages_to_numa_node_batch] Moving 5 Hyrise pages (5 OS pages) to NUMA node 1
[VolatileRegion::move_pages_to_numa_node_batch] Successfully migrated 5 pages to node 1
[BufferPool] Batch eviction completed: evicted=5, freed_bytes=20480
```

**Log Message Meanings:**
- **Using batch eviction path**: Batch eviction was triggered (vs single-page eviction)
- **Phase 2: Locked X pages**: Number of pages successfully locked for eviction
- **Attempting batch NUMA migration**: About to perform batch migration for pages of a specific size
- **Moving X Hyrise pages**: Actual batch migration with total OS page count
- **Successfully migrated**: Confirmation of successful batch migration
- **Batch eviction completed**: Summary showing pages evicted and bytes freed

**Performance Indicators:**
- Larger batch sizes (e.g., 10+ pages) = better performance
- Many small batches (1-2 pages) = might need larger buffer pools or different workload
- No batch messages = batching disabled or single-page eviction path used

## New Functions

### 1. `VolatileRegion::move_pages_to_numa_node_batch()`

**Location:** `src/lib/storage/buffer/volatile_region.cpp`

```cpp
void move_pages_to_numa_node_batch(const std::vector<PageID>& page_ids, 
                                   const NodeID target_memory_node);
```

**Purpose:** Migrates multiple pages to a NUMA node in a single `move_pages` syscall.

**How it works:**
- Takes a vector of PageIDs that all belong to the same region (same page size)
- Calculates total number of OS pages (each Hyrise page = multiple 4KB OS pages)
- Builds arrays for the `move_pages(2)` syscall
- Performs migration in one system call
- Updates frame metadata for all pages

**Benefits:**
- Reduces syscall overhead (1 syscall instead of N)
- Better TLB shootdown performance
- Atomic migration of related pages

### 2. `BufferPool::evict_batch()`

**Location:** `src/lib/storage/buffer/buffer_pool.cpp`

```cpp
size_t evict_batch(size_t num_pages_to_evict);
```

**Purpose:** Evicts multiple pages at once from the buffer pool.

**How it works:**
1. **Collection Phase:** Dequeues up to `num_pages_to_evict` items from eviction queue
2. **Locking Phase:** Attempts to lock each page exclusively
3. **Grouping Phase:** Groups pages by size type (4KB, 64KB, etc.)
4. **Batch Migration:** For each size type, calls `move_pages_to_numa_node_batch()`
5. **Cleanup:** Unlocks frames and adds to target pool's eviction queue

**Returns:** Number of pages successfully evicted

## Usage Example

### Option 1: Call directly from buffer pool

```cpp
// Evict 64 pages in a batch
auto& buffer_pool = buffer_manager._primary_buffer_pool;
size_t evicted = buffer_pool->evict_batch(64);
std::cout << "Evicted " << evicted << " pages in batch\n";
```

### Option 2: Modify `ensure_free_pages()` to use batching

Currently, `ensure_free_pages()` evicts pages one at a time in a loop. You could modify it to use batch eviction when many pages need to be freed:

```cpp
bool BufferPool::ensure_free_pages(const PageSizeType required_size) {
  const auto bytes_required = bytes_for_size_type(required_size);
  auto current_bytes = reserve_bytes(bytes_required);
  
  // If we need to free a lot of space, use batch eviction
  const auto bytes_to_free = (current_bytes + bytes_required) - max_bytes;
  const auto pages_to_free = bytes_to_free / bytes_for_size_type(required_size);
  
  if (pages_to_free > BATCH_EVICTION_THRESHOLD) {  // e.g., 32 pages
    auto evicted = evict_batch(pages_to_free);
    if (evicted >= pages_to_free) {
      return true;
    }
  }
  
  // Fallback to existing single-page eviction
  // ... existing code ...
}
```

## Limitations

1. **Same Size Requirement:** Batch only works for pages of the same size type
2. **Target Pool Space:** Must have space in target pool for entire batch (falls back to individual eviction if allocation fails)
3. **NUMA Only:** Only applies to DRAM→NUMA eviction (SSD eviction still per-page due to potential dirty writes)

## Runtime Configuration

Batching is now integrated into the buffer pool's `ensure_free_pages()` method and controlled via configuration:

**buffer_manager_config.json:**
```json
{
  "enable_batching": true,
  "use_custom_syscall": false
}
```

When `enable_batching` is true:
- The buffer pool uses `evict_batch()` when multiple pages need eviction
- Pages are collected, locked, and migrated in batches
- Reduces syscall overhead compared to single-page eviction

When `enable_batching` is false:
- Falls back to original single-page eviction loop
- Each page evicted individually with separate syscalls

## Testing

### Unit Tests

Tests are located in `src/test/lib/storage/buffer/batch_eviction_test.cpp`.

Run the tests:
```bash
cd cmake-build-debug
./hyriseTest --gtest_filter="*BatchEviction*"
```

All tests pass successfully:
```
[==========] Running 5 tests from 1 test suite.
[----------] 5 tests from BatchEvictionTest
[ RUN      ] BatchEvictionTest.BatchMigrationFunction
[       OK ] BatchEvictionTest.BatchMigrationFunction
[ RUN      ] BatchEvictionTest.BatchEvictionBasic
[       OK ] BatchEvictionTest.BatchEvictionBasic
[ RUN      ] BatchEvictionTest.BatchEvictionWithNuma
[       OK ] BatchEvictionTest.BatchEvictionWithNuma
[ RUN      ] BatchEvictionTest.BatchEvictionEmptyQueue
[       OK ] BatchEvictionTest.BatchEvictionEmptyQueue
[ RUN      ] BatchEvictionTest.BatchEvictionLargeCount
[       OK ] BatchEvictionTest.BatchEvictionLargeCount
[----------] 5 tests from BatchEvictionTest
[  PASSED  ] 5 tests.
```

The tests verify:
1. Basic batch eviction functionality
2. Batch eviction with NUMA enabled
3. Empty queue handling
4. Large batch counts
5. Migration function correctness

### Performance Testing

Compare batched vs non-batched performance:

**1. Run with batching enabled:**
```bash
HYRISE_BUFFER_MANAGER_CONFIG_JSON_PATH=/users/yrayhan/hyrise/buffer_manager_config.json \
  ./hyriseBenchmarkBufferManager \
  --benchmark_filter="BM_ycsb/UpdateHeavy/EagerMigrationPolicy/1/threads:47" \
  --benchmark_out=batch_enabled.json --benchmark_out_format=json
```

**2. Disable batching in config and run again:**
```json
{
  "enable_batching": false,
  ...
}
```

```bash
HYRISE_BUFFER_MANAGER_CONFIG_JSON_PATH=/users/yrayhan/hyrise/buffer_manager_config.json \
  ./hyriseBenchmarkBufferManager \
  --benchmark_filter="BM_ycsb/UpdateHeavy/EagerMigrationPolicy/1/threads:47" \
  --benchmark_out=batch_disabled.json --benchmark_out_format=json
```

**3. Compare results:**
```bash
# Using Hyrise's benchmark comparison script
python3 ../scripts/compare_benchmarks.py batch_disabled.json batch_enabled.json
```

Expected improvements with batching:
- Lower latency (fewer syscalls)
- Higher throughput (better TLB utilization)
- Reduced CPU overhead (fewer kernel transitions)


 1.5249e+10 ns            7 bytes_per_second=7.74827M/s bytes_read_from_ssd=302.162M bytes_written_to_ssd=850.702M cache_hit_rate=0.985493 items_per_second=126.948k/s latency_95percentile=2.507k latency_max=196.87M latency_mean=55.0445k latency_median=769 latency_min=152 latency_stddev=385.817k


1.4240e+10 ns            7 bytes_per_second=8.24538M/s bytes_read_from_ssd=303.129M bytes_written_to_ssd=853.111M cache_hit_rate=0.985441 items_per_second=135.092k/s latency_95percentile=2.455k latency_max=181.142M latency_mean=51.72k latency_median=778 latency_min=151 latency_stddev=533.39k

1.2012e+10 ns            7 bytes_per_second=11.5766M/s bytes_read_from_ssd=165.773M bytes_written_to_ssd=415.08M cache_hit_rate=0.992017 items_per_second=189.671k/s latency_95percentile=2.469k latency_max=205.39M latency_mean=36.7926k latency_median=756 latency_min=148 latency_stddev=361.461k


1.1183e+10 ns            7 bytes_per_second=12.4751M/s bytes_read_from_ssd=163.43M bytes_written_to_ssd=411.98M cache_hit_rate=0.992131 items_per_second=204.392k/s latency_95percentile=2.427k latency_max=185.991M latency_mean=34.1384k latency_median=716 latency_min=149 latency_stddev=419.68k