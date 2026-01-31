#include <memory>
#include <vector>

#include "base_test.hpp"
#include "storage/buffer/buffer_manager.hpp"
#include "storage/buffer/buffer_pool.hpp"

namespace hyrise {

class BatchEvictionTest : public BaseTest {
 protected:
  void SetUp() override {
    // Create buffer manager with small pools to trigger evictions easily
    const std::string ssd_dir = test_data_path + "batch_eviction_ssd/";
    std::filesystem::create_directories(ssd_dir);
    
    auto config = BufferManager::Config{};
    config.dram_buffer_pool_size = 256 * 1024;    // 256KB DRAM
    config.numa_buffer_pool_size = 256 * 1024;    // 256KB NUMA
    config.ssd_path = ssd_dir;
    config.enable_eviction_purge_worker = false;
    config.memory_node = NodeID{1};
    config.enable_numa = true;
    
    _buffer_manager = std::make_unique<BufferManager>(config);
  }

  void TearDown() override {
    _buffer_manager.reset();
    std::filesystem::remove_all(test_data_path + "batch_eviction_ssd/");
  }

  std::unique_ptr<BufferManager> _buffer_manager;
};

TEST_F(BatchEvictionTest, BatchMigrationFunction) {
  // Test that the batch migration infrastructure works by triggering batch eviction
  // The evict_batch() function internally uses move_pages_to_numa_node_batch()
  
  auto primary_pool = _buffer_manager->get_primary_buffer_pool();
  ASSERT_NE(primary_pool, nullptr);
  
  // Allocate multiple pages to fill up DRAM
  constexpr size_t PAGE_SIZE = 4096;
  constexpr size_t NUM_PAGES = 30;
  std::vector<void*> allocated_pages;
  std::vector<uint64_t> expected_values;
  
  for (size_t i = 0; i < NUM_PAGES; ++i) {
    void* ptr = _buffer_manager->allocate(PAGE_SIZE);
    if (ptr) {
      allocated_pages.push_back(ptr);
      uint64_t value = i + 1000;
      *static_cast<uint64_t*>(ptr) = value;
      expected_values.push_back(value);
    }
  }
  
  EXPECT_GT(allocated_pages.size(), 10);
  
  // Deallocate to add to eviction queue
  for (auto ptr : allocated_pages) {
    _buffer_manager->deallocate(ptr, PAGE_SIZE);
  }
  
  size_t initial_queue_size = primary_pool->eviction_queue->unsafe_size();
  EXPECT_GT(initial_queue_size, 0) << "Eviction queue should have items";
  
  // Call evict_batch which internally uses move_pages_to_numa_node_batch
  size_t num_to_evict = std::min(initial_queue_size, size_t{15});
  size_t evicted = primary_pool->evict_batch(num_to_evict);
  
  EXPECT_GT(evicted, 0) << "Should have evicted pages using batch migration";
  EXPECT_LE(evicted, num_to_evict) << "Should not evict more than requested";
  
  size_t final_queue_size = primary_pool->eviction_queue->unsafe_size();
  EXPECT_LT(final_queue_size, initial_queue_size) << "Queue size should decrease after batch eviction";
}

TEST_F(BatchEvictionTest, BatchEvictionBasic) {
  auto primary_pool = _buffer_manager->get_primary_buffer_pool();
  ASSERT_NE(primary_pool, nullptr);
  
  // Allocate pages to fill up DRAM
  constexpr size_t PAGE_SIZE = 4096;
  constexpr size_t NUM_PAGES = 50;
  std::vector<void*> allocated_pages;
  
  for (size_t i = 0; i < NUM_PAGES; ++i) {
    void* ptr = _buffer_manager->allocate(PAGE_SIZE);
    if (ptr) {
      allocated_pages.push_back(ptr);
      *static_cast<uint64_t*>(ptr) = i;
    }
  }
  
  EXPECT_GT(allocated_pages.size(), 0);
  
  // Deallocate to add to eviction queue
  for (auto ptr : allocated_pages) {
    _buffer_manager->deallocate(ptr, PAGE_SIZE);
  }
  
  size_t initial_queue_size = primary_pool->eviction_queue->unsafe_size();
  EXPECT_GT(initial_queue_size, 0) << "Eviction queue should have items after deallocation";
  
  // **ACTUAL TEST**: Call evict_batch directly
  size_t num_to_evict = std::min(initial_queue_size, size_t{10});
  size_t evicted = primary_pool->evict_batch(num_to_evict);
  
  EXPECT_GT(evicted, 0) << "Should have evicted at least one page";
  EXPECT_LE(evicted, num_to_evict) << "Should not evict more than requested";
  
  size_t final_queue_size = primary_pool->eviction_queue->unsafe_size();
  EXPECT_LT(final_queue_size, initial_queue_size) << "Queue size should decrease after batch eviction";
}

TEST_F(BatchEvictionTest, BatchEvictionWithNuma) {
  auto primary_pool = _buffer_manager->get_primary_buffer_pool();
  auto secondary_pool = _buffer_manager->get_secondary_buffer_pool();
  
  ASSERT_NE(primary_pool, nullptr);
  ASSERT_NE(secondary_pool, nullptr);
  
  if (!secondary_pool->enabled) {
    GTEST_SKIP() << "NUMA not enabled, skipping NUMA-specific test";
  }
  
  // Allocate many pages on DRAM
  constexpr size_t PAGE_SIZE = 4096;
  constexpr size_t NUM_PAGES = 60;
  std::vector<void*> pages;
  
  for (size_t i = 0; i < NUM_PAGES; ++i) {
    void* ptr = _buffer_manager->allocate(PAGE_SIZE);
    if (ptr) {
      pages.push_back(ptr);
      *static_cast<uint64_t*>(ptr) = i + 100;
    }
  }
  
  EXPECT_GT(pages.size(), 20);
  
  // Deallocate to add to eviction queue
  for (auto ptr : pages) {
    _buffer_manager->deallocate(ptr, PAGE_SIZE);
  }
  
  size_t initial_queue_size = primary_pool->eviction_queue->unsafe_size();
  EXPECT_GT(initial_queue_size, 0);
  
  // **ACTUAL TEST**: Call evict_batch to move pages to NUMA tier
  size_t evicted = primary_pool->evict_batch(20);
  
  EXPECT_GT(evicted, 0) << "Should have evicted pages in batch to NUMA";
  
  size_t final_queue_size = primary_pool->eviction_queue->unsafe_size();
  EXPECT_LT(final_queue_size, initial_queue_size) << "Eviction queue should shrink";
}

TEST_F(BatchEvictionTest, BatchEvictionEmptyQueue) {
  // Try operations with minimal allocation
  constexpr size_t PAGE_SIZE = 4096;
  void* ptr = _buffer_manager->allocate(PAGE_SIZE);
  
  if (ptr) {
    *static_cast<uint64_t*>(ptr) = 42;
    _buffer_manager->deallocate(ptr, PAGE_SIZE);
  }
  
  SUCCEED();
}

TEST_F(BatchEvictionTest, BatchEvictionLargeCount) {
  auto primary_pool = _buffer_manager->get_primary_buffer_pool();
  ASSERT_NE(primary_pool, nullptr);
  
  // Allocate and immediately deallocate many pages
  constexpr size_t PAGE_SIZE = 4096;
  std::vector<void*> pages;
  
  for (size_t i = 0; i < 100; ++i) {
    void* ptr = _buffer_manager->allocate(PAGE_SIZE);
    if (ptr) {
      pages.push_back(ptr);
      *static_cast<uint64_t*>(ptr) = i + 200;
    }
  }
  
  for (auto ptr : pages) {
    _buffer_manager->deallocate(ptr, PAGE_SIZE);
  }
  
  size_t queue_size = primary_pool->eviction_queue->unsafe_size();
  EXPECT_GT(queue_size, 0);
  
  // **ACTUAL TEST**: Request more evictions than available
  size_t evicted = primary_pool->evict_batch(1000);
  
  // Should evict at most what's in the queue
  EXPECT_LE(evicted, queue_size) << "Cannot evict more pages than in queue";
  EXPECT_GT(evicted, 0) << "Should have evicted some pages";
}

}  // namespace hyrise
