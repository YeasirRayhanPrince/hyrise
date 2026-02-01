#include "buffer_pool.hpp"
#include "metrics.hpp"
#include "storage/buffer/ssd_region.hpp"
#include "volatile_region.hpp"
#include <unordered_map>

namespace hyrise {
//TODO: properly check if disabled or not
BufferPool::BufferPool(const bool enabled, const size_t pool_size, const bool enable_eviction_purge_worker,
                       std::array<std::shared_ptr<VolatileRegion>, NUM_PAGE_SIZE_TYPES> volatile_regions,
                       MigrationPolicy migration_policy, std::shared_ptr<SSDRegion> ssd_region,
                       std::shared_ptr<BufferPool> target_buffer_pool, const NodeID numa_node,
                       std::shared_ptr<BufferPoolMetrics> metrics, const bool enable_batch_eviction,
                       const bool use_custom_syscall)
    : max_bytes(pool_size),
      used_bytes(0),
      metrics(metrics),
      enabled(enabled),
      volatile_regions(volatile_regions),
      eviction_queue(std::make_unique<EvictionQueue>()),
      promotion_queue(std::make_unique<PromotionQueue>()),
      node_id(numa_node),
      ssd_region(ssd_region),
      target_buffer_pool(target_buffer_pool),
      migration_policy(migration_policy),
      enable_batch_eviction(enable_batch_eviction),
      use_custom_syscall(use_custom_syscall),
      eviction_purge_worker(enable_eviction_purge_worker
                                ? std::make_unique<PausableLoopThread>(IDLE_EVICTION_QUEUE_PURGE,
                                                                       [&](size_t) { this->purge_eviction_queue(); })
                                : nullptr) {}

void BufferPool::purge_eviction_queue() {
  auto item = EvictionItem{};
  for (auto i = size_t{0}; 0 < MAX_EVICTION_QUEUE_PURGES; ++i) {
    if (!eviction_queue->try_pop(item)) {
      return;
    }

    auto region = volatile_regions[static_cast<uint64_t>(item.page_id.size_type())];
    auto frame = region->get_frame(item.page_id);
    auto current_state_and_version = frame->state_and_version();

    // The item is in state UNLOCKED and can be marked
    if (item.can_evict(current_state_and_version) || item.can_mark(current_state_and_version)) {
      eviction_queue->push(item);
      continue;
    }
  }
}

void BufferPool::add_to_eviction_queue(const PageID page_id, Frame* frame) {
  auto current_state_and_version = frame->state_and_version();
  DebugAssert(frame->node_id() == node_id, "Memory node mismatch");
  increment_counter(metrics->num_eviction_queue_adds);
  eviction_queue->push({page_id, Frame::version(current_state_and_version)});
}

void BufferPool::free_bytes(const uint64_t bytes) {
  used_bytes.fetch_sub(bytes);
}

uint64_t BufferPool::reserve_bytes(const uint64_t bytes) {
  return used_bytes.fetch_add(bytes);
}

bool BufferPool::ensure_free_pages(const PageSizeType required_size) {
  // TODO: Free at least 64 * PageSite bytes to reduce TLB shootdowns
  const auto bytes_required = bytes_for_size_type(required_size);
  auto freed_bytes = size_t{0};
  auto current_bytes = reserve_bytes(bytes_required);

  auto item = EvictionItem{};

  if (enable_batch_eviction) {
    // Batch eviction path: evict multiple pages at once
    size_t consecutive_failures = 0;
    const size_t MAX_CONSECUTIVE_FAILURES = 3;
    
    while ((current_bytes + bytes_required - freed_bytes) > max_bytes) {
      const auto bytes_needed = (current_bytes + bytes_required - freed_bytes) - max_bytes;
      const auto min_page_bytes = bytes_for_size_type(MIN_PAGE_SIZE_TYPE);
      
      // Calculate pages needed based on bytes, but enforce minimum batch size
      const auto pages_from_bytes = (bytes_needed + min_page_bytes - 1) / min_page_bytes;
      const auto pages_to_evict = std::max(MIN_BATCH_SIZE, pages_from_bytes);

      const auto evicted = evict_batch(pages_to_evict, &freed_bytes);
      // std::cout << "[BufferPool] Batch eviction: requested=" << pages_to_evict
      //           << ", evicted=" << evicted << ", freed_bytes=" << freed_bytes << std::endl;
      
      if (evicted == 0) {
        consecutive_failures++;
        if (consecutive_failures >= MAX_CONSECUTIVE_FAILURES) {
          // Batch eviction failed, fall back to single-page eviction
          break;
        }
        // Give queue time to accumulate more items
        std::this_thread::yield();
        continue;
      }
      
      consecutive_failures = 0;  // Reset on success
      current_bytes = used_bytes.load();
    }
  }
  
  // Fallback to single-page eviction if batching is disabled or failed
  if (!enable_batch_eviction || (current_bytes + bytes_required - freed_bytes) > max_bytes) {
    while ((current_bytes + bytes_required - freed_bytes) > max_bytes) {
      if (!eviction_queue->try_pop(item)) {
        free_bytes(bytes_required);
        return false;
      }

      auto region = volatile_regions[static_cast<uint64_t>(item.page_id.size_type())];
      auto frame = region->get_frame(item.page_id);
      auto current_state_and_version = frame->state_and_version();

      if (frame->node_id() != node_id) {
        increment_counter(metrics->num_eviction_queue_items_purged);
        continue;
      }

      // If the frame is already marked, we can evict it
      if (!item.can_evict(current_state_and_version)) {
        // If the frame is UNLOCKED, we can mark it
        if (item.can_mark(current_state_and_version)) {
          if (frame->try_mark(current_state_and_version)) {
            add_to_eviction_queue(item.page_id, frame);
            continue;
          }
        }
        increment_counter(metrics->num_eviction_queue_items_purged);
        continue;
      }

      // Try locking the frame exclusively
      if (!frame->try_lock_exclusive(current_state_and_version)) {
        increment_counter(metrics->num_eviction_queue_items_purged);
        continue;
      }

      Assert(frame->node_id() == node_id,
             "Memory node mismatch: " + std::to_string(frame->node_id()) + " != " + std::to_string(node_id));

      evict(item, frame);

      increment_counter(metrics->num_evictions);

      const auto size_type = item.page_id.size_type();
      freed_bytes += bytes_for_size_type(size_type);
      current_bytes = used_bytes.load();
    }
  }

  // TODO: Check if this is correct
  free_bytes(freed_bytes);

  return true;
}

void BufferPool::evict(EvictionItem& item, Frame* frame) {
  DebugAssert(Frame::state(frame->state_and_version()) == Frame::LOCKED, "Frame cannot be locked");
  auto region = volatile_regions[static_cast<uint64_t>(item.page_id.size_type())];
  const auto num_bytes = bytes_for_size_type(item.page_id.size_type());

  // We try to evict the current item. Based on the migration policy, we to evict the page to a lower tier.
  // If this fails, we retry and some point, we might land on SSD.
  for (auto repeat = size_t{0}; repeat < MAX_REPEAT_COUNT; ++repeat) {
    // If we have a target buffer pool and we don't want to bypass it, we move the page to the other pool
    const auto write_to_ssd =
        !target_buffer_pool || !target_buffer_pool->enabled || migration_policy.bypass_numa_during_write();

    if (write_to_ssd) {
      // Otherwise we just write the page if its dirty and free the associated pages
      if (frame->is_dirty()) {
        auto data = region->get_page(item.page_id);
        ssd_region->write_page(item.page_id, data);  // TODO: use global function
        region->protect_page(item.page_id);
        frame->reset_dirty();
      }
      region->free(item.page_id);
      frame->unlock_exclusive_and_set_evicted();

      increment_counter(metrics->total_bytes_copied_to_ssd, num_bytes);

      return;
    } else {
      // Or we just move to other numa node and unlock again
      if (!target_buffer_pool->ensure_free_pages(item.page_id.size_type())) {
        yield(repeat);
        continue;
      };
      region->mbind_to_numa_node(item.page_id, target_buffer_pool->node_id);
      frame->unlock_exclusive();
      target_buffer_pool->add_to_eviction_queue(item.page_id, frame);
      //   TODO:increment_counter(metrics.total_bytes_copied_from_dram_to_numa, num_bytes);
      return;
    }
  }
  Fail("Could not evict page after trying for " + std::to_string(MAX_REPEAT_COUNT) + " times");
}

size_t BufferPool::evict_batch(size_t num_pages_to_evict, size_t* bytes_freed) {
  // Collect pages to evict in batches by size type and destination
  std::unordered_map<PageSizeType, std::vector<EvictionItem>> pages_by_size;
  std::vector<std::pair<EvictionItem, Frame*>> locked_pages;
  
  auto item = EvictionItem{};
  size_t pages_collected = 0;
  size_t queue_items_scanned = 0;
  
  // Calculate maximum queue items to scan (lookahead depth)
  // In hot workloads, most queue entries are stale (already re-pinned),
  // so we need to scan deeper to find truly evictable pages
  const size_t max_queue_scans = num_pages_to_evict * MAX_QUEUE_SCAN_MULTIPLIER;

  // Phase 1: Collect and lock pages
  // Continue scanning until we either:
  //   - Collect num_pages_to_evict evictable pages, OR
  //   - Scan max_queue_scans items, OR
  //   - Queue is empty
  while (pages_collected < num_pages_to_evict && queue_items_scanned < max_queue_scans) {
    if (!eviction_queue->try_pop(item)) {
      break;  // No more items in queue
    }
    
    queue_items_scanned++;

    auto region = volatile_regions[static_cast<uint64_t>(item.page_id.size_type())];
    auto frame = region->get_frame(item.page_id);
    auto current_state_and_version = frame->state_and_version();

    // Skip if page is not on this node
    if (frame->node_id() != node_id) {
      increment_counter(metrics->num_eviction_queue_items_purged);
      continue;
    }

    // Skip if cannot evict
    if (!item.can_evict(current_state_and_version)) {
      if (item.can_mark(current_state_and_version)) {
        if (frame->try_mark(current_state_and_version)) {
          add_to_eviction_queue(item.page_id, frame);
        }
      }
      increment_counter(metrics->num_eviction_queue_items_purged);
      continue;
    }

    // Try locking the frame exclusively
    if (!frame->try_lock_exclusive(current_state_and_version)) {
      increment_counter(metrics->num_eviction_queue_items_purged);
      continue;
    }

    // Successfully locked - add to batch
    locked_pages.emplace_back(item, frame);
    pages_by_size[item.page_id.size_type()].push_back(item);
    pages_collected++;
  }

  if (locked_pages.empty()) {
    // std::cout << "[BufferPool::evict_batch] No pages locked for eviction after scanning " 
    //          << queue_items_scanned << " queue items (requested=" << num_pages_to_evict 
    //          << ", max_scans=" << max_queue_scans << ")" << std::endl;
    return 0;  // No pages to evict
  }

  // std::cout << "[BufferPool::evict_batch] Phase 2: Locked " << locked_pages.size() 
  //          << " pages after scanning " << queue_items_scanned << " queue items" 
  //          << " (requested=" << num_pages_to_evict << ", efficiency=" 
  //          << (100.0 * pages_collected / std::max(queue_items_scanned, size_t(1))) << "%)" << std::endl;

  // Phase 2: Batch evict by size type
  size_t evicted_count = 0;
  const auto write_to_ssd = 
      !target_buffer_pool || !target_buffer_pool->enabled || migration_policy.bypass_numa_during_write();

  for (auto& [size_type, items] : pages_by_size) {
    auto region = volatile_regions[static_cast<uint64_t>(size_type)];
    
    if (write_to_ssd) {
      // Evict to SSD (one by one for dirty pages)
      for (const auto& item : items) {
        auto frame = region->get_frame(item.page_id);
        
        if (frame->is_dirty()) {
          auto data = region->get_page(item.page_id);
          ssd_region->write_page(item.page_id, data);
          region->protect_page(item.page_id);
          frame->reset_dirty();
        }
        region->free(item.page_id);
        frame->unlock_exclusive_and_set_evicted();
        
        increment_counter(metrics->num_evictions);
        increment_counter(metrics->total_bytes_copied_to_ssd, bytes_for_size_type(size_type));
        if (bytes_freed) {
          *bytes_freed += bytes_for_size_type(size_type);
        }
        evicted_count++;
      }
    } else {
      // Batch migrate to NUMA
      // std::cout << "[BufferPool::evict_batch] Attempting batch NUMA migration for " 
      //          << items.size() << " pages of size_type=" << static_cast<int>(size_type) << std::endl;
      if (!target_buffer_pool->ensure_free_pages(size_type)) {
        // Fallback to individual eviction if batch allocation fails
        // std::cout << "[BufferPool::evict_batch] Target pool cannot allocate, fallback to queue" << std::endl;
        for (const auto& item : items) {
          auto frame = region->get_frame(item.page_id);
          frame->unlock_exclusive();
          add_to_eviction_queue(item.page_id, frame);
        }
        continue;
      }

      // Collect page IDs for batch migration
      std::vector<PageID> page_ids_to_migrate;
      page_ids_to_migrate.reserve(items.size());
      for (const auto& item : items) {
        page_ids_to_migrate.push_back(item.page_id);
      }

      // Perform batch migration using move_pages
      region->move_pages_to_numa_node_batch(page_ids_to_migrate, target_buffer_pool->node_id);

      // Unlock all frames and add to target pool's eviction queue
      for (const auto& item : items) {
        auto frame = region->get_frame(item.page_id);
        frame->unlock_exclusive();
        target_buffer_pool->add_to_eviction_queue(item.page_id, frame);
        
        increment_counter(metrics->num_evictions);
        if (bytes_freed) {
          *bytes_freed += bytes_for_size_type(size_type);
        }
        evicted_count++;
      }
    }
  }

  // Update batch metrics
  if (evicted_count > 0) {
    metrics->num_batch_evictions.fetch_add(1, std::memory_order_relaxed);
    metrics->total_pages_batched.fetch_add(evicted_count, std::memory_order_relaxed);
  }

  return evicted_count;
}

void BufferPool::add_to_promotion_queue(const PageID page_id) {
  promotion_queue->push(page_id);
}

size_t BufferPool::promote_batch(NodeID target_node_id) {
  // Collect pages to promote, grouped by size type
  std::unordered_map<PageSizeType, std::vector<PageID>> pages_by_size;
  std::vector<std::pair<PageID, Frame*>> locked_pages;
  
  PageID page_id;
  size_t pages_collected = 0;
  size_t queue_items_scanned = 0;
  
  // Calculate maximum queue items to scan (lookahead depth)
  const size_t max_queue_scans = MIN_PROMOTION_BATCH_SIZE * MAX_PROMOTION_QUEUE_SCAN_MULTIPLIER;

  // Phase 1: Collect and lock pages from promotion queue
  while (pages_collected < MIN_PROMOTION_BATCH_SIZE && queue_items_scanned < max_queue_scans) {
    if (!promotion_queue->try_pop(page_id)) {
      break;  // No more items in queue
    }
    
    queue_items_scanned++;

    auto region = volatile_regions[static_cast<uint64_t>(page_id.size_type())];
    auto frame = region->get_frame(page_id);
    auto current_state_and_version = frame->state_and_version();

    // Skip if page is not on this node (already promoted or moved elsewhere)
    if (frame->node_id() != node_id) {
      continue;  // Stale entry, skip
    }

    // Skip if page is locked (being used by another thread)
    if (Frame::state(current_state_and_version) != Frame::UNLOCKED) {
      // Re-queue for later attempt
      promotion_queue->push(page_id);
      continue;
    }

    // Try locking the frame exclusively for promotion
    if (!frame->try_lock_exclusive(current_state_and_version)) {
      // Re-queue for later attempt
      promotion_queue->push(page_id);
      continue;
    }

    // Re-check node_id after locking (might have changed)
    if (frame->node_id() != node_id) {
      frame->unlock_exclusive();
      continue;  // Stale, skip
    }

    // Successfully locked - add to batch
    locked_pages.emplace_back(page_id, frame);
    pages_by_size[page_id.size_type()].push_back(page_id);
    pages_collected++;
  }

  if (locked_pages.empty()) {
    return 0;  // No pages to promote
  }

  // Phase 2: Batch promote by size type
  size_t promoted_count = 0;

  for (auto& [size_type, page_ids] : pages_by_size) {
    auto region = volatile_regions[static_cast<uint64_t>(size_type)];
    
    // Perform batch migration using move_pages
    region->move_pages_to_numa_node_batch(page_ids, target_node_id);

    // Unlock all frames after migration
    for (const auto& pid : page_ids) {
      auto frame = region->get_frame(pid);
      frame->unlock_exclusive();
      promoted_count++;
    }
  }

  // Update batch promotion metrics
  if (promoted_count > 0) {
    metrics->num_batch_promotions.fetch_add(1, std::memory_order_relaxed);
    metrics->total_pages_promoted.fetch_add(promoted_count, std::memory_order_relaxed);
  }

  return promoted_count;
}

size_t BufferPool::memory_consumption() const {
  return sizeof(*this) + sizeof(*eviction_queue) + sizeof(EvictionQueue::value_type) * eviction_queue->unsafe_size();
}

size_t BufferPool::free_bytes_node() const {
#if HYRISE_NUMA_SUPPORT
  if (node_id == INVALID_NODE_ID) {
    return 0;
  }
  long long free_bytes;
  numa_node_size(node_id, &free_bytes);
  return free_bytes;
#else
  return 0;
#endif
};

size_t BufferPool::total_bytes_node() const {
#if HYRISE_NUMA_SUPPORT
  if (node_id == INVALID_NODE_ID) {
    return 0;
  }
  return numa_node_size(node_id, nullptr);
#else
  return 0;
#endif
};
}  // namespace hyrise