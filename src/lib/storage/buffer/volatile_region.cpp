#include "volatile_region.hpp"
#include <chrono>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include "migration_profiler.hpp"
#include "utils/assert.hpp"

#if HYRISE_NUMA_SUPPORT
#include <numa.h>
#include <numaif.h>

// Custom move_pages2 syscall number (Linux kernel extension)
#ifndef SYS_move_pages2
#define SYS_move_pages2 462
#endif

#endif

namespace hyrise {

namespace {
using Clock = std::chrono::high_resolution_clock;

inline double duration_us(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start).count();
}
}  // namespace

VolatileRegion::VolatileRegion(const PageSizeType size_type, std::byte* region_start, std::byte* region_end,
                               std::shared_ptr<BufferManagerMetrics> metrics)
    : _size_type(size_type),
      _region_start(region_start),
      _region_end(region_end),
      _frames(std::min(INITIAL_SLOTS_PER_REGION, (region_end - region_start) / bytes_for_size_type(size_type))),
      _free_slots(std::min(INITIAL_SLOTS_PER_REGION, (region_end - region_start) / bytes_for_size_type(size_type))),
      _metrics(metrics) {
  DebugAssertPageAligned(region_start);
  DebugAssert(region_start < region_end, "Region is too small");
  DebugAssert(static_cast<size_t>(region_end - region_start) < DEFAULT_RESERVED_VIRTUAL_MEMORY,
              "Region start and end dont match");
  DebugAssert(_frames.size() > 0, "Region is too small");
  DebugAssert(_free_slots.size() > 0, "Region is too small");
  _free_slots.set();
  if constexpr (ENABLE_MPROTECT) {
    if (mprotect(region_start, region_end - region_start, PROT_NONE) != 0) {
      const auto error = errno;
      Fail("Failed to mprotect: " + strerror(error));
    }
  }
}

void VolatileRegion::move_page_to_numa_node(PageID page_id, const NodeID target_memory_node,
                                            MigrationPhaseTimings* timing) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
#if HYRISE_NUMA_SUPPORT
  DebugAssert(target_memory_node != INVALID_NODE_ID, "Numa node has not been set.");
  const auto do_timing = timing != nullptr || g_migration_profiler.enabled();
  auto local_timing = MigrationPhaseTimings{};
  auto* timing_ptr = timing ? timing : &local_timing;

  if (do_timing) {
    timing_ptr->size_type = _size_type;
    timing_ptr->is_batch = false;
    timing_ptr->pages_attempted = 1;
    timing_ptr->pages_migrated = 1;
    timing_ptr->bytes_migrated = bytes_for_size_type(_size_type);
    timing_ptr->syscall_count = 1;
    timing_ptr->syscall_type = "move_pages";
  }

  static thread_local std::vector<void*> pages_to_move{bytes_for_size_type(_size_type) / OS_PAGE_SIZE};
  static thread_local std::vector<int> nodes{static_cast<int>(bytes_for_size_type(_size_type) / OS_PAGE_SIZE)};
  static thread_local std::vector<int> status{static_cast<int>(bytes_for_size_type(_size_type) / OS_PAGE_SIZE)};

  const auto array_start = do_timing ? Clock::now() : Clock::time_point{};
  for (auto i = 0u; i < pages_to_move.size(); ++i) {
    pages_to_move[i] = get_page(page_id) + i * OS_PAGE_SIZE;
    nodes[i] = target_memory_node;
  }
  if (do_timing) {
    timing_ptr->array_build_time_us += duration_us(array_start, Clock::now());
  }

  const auto syscall_start = do_timing ? Clock::now() : Clock::time_point{};
  if (move_pages(0, pages_to_move.size(), pages_to_move.data(), nodes.data(), status.data(), MPOL_MF_MOVE) < 0) {
    const auto error = errno;
    Fail("Move pages failed: " + strerror(error));
  }
  if (do_timing) {
    timing_ptr->syscall_time_us += duration_us(syscall_start, Clock::now());
  }
  _metrics->num_numa_tonode_memory_calls.fetch_add(1, std::memory_order_relaxed);

  const auto metadata_start = do_timing ? Clock::now() : Clock::time_point{};
  _frames[page_id.index].set_node_id(target_memory_node);
  if (do_timing) {
    timing_ptr->metadata_update_time_us += duration_us(metadata_start, Clock::now());
  }

  if (do_timing && timing == nullptr) {
    g_migration_profiler.record_migration(*timing_ptr);
  }
#else
  (void)timing;
  _frames[page_id.index].set_node_id(target_memory_node);
#endif
}

#if HYRISE_NUMA_SUPPORT
void VolatileRegion::mbind_to_numa_node(PageID page_id, const NodeID target_memory_node,
                                        MigrationPhaseTimings* timing) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
  DebugAssert(target_memory_node != INVALID_NODE_ID, "Numa node has not been set.");

  const auto do_timing = timing != nullptr || g_migration_profiler.enabled();
  auto local_timing = MigrationPhaseTimings{};
  auto* timing_ptr = timing ? timing : &local_timing;

  if (do_timing) {
    timing_ptr->size_type = _size_type;
    timing_ptr->is_batch = false;
    timing_ptr->pages_attempted = 1;
    timing_ptr->pages_migrated = 1;
    timing_ptr->bytes_migrated = bytes_for_size_type(_size_type);
    timing_ptr->syscall_count = 1;
    timing_ptr->syscall_type = "mbind";
  }

  const auto num_bytes = bytes_for_size_type(_size_type);
  const auto array_start = do_timing ? Clock::now() : Clock::time_point{};
  auto nodes = numa_allocate_nodemask();
  numa_bitmask_setbit(nodes, target_memory_node);
  if (do_timing) {
    timing_ptr->array_build_time_us += duration_us(array_start, Clock::now());
  }

  const auto syscall_start = do_timing ? Clock::now() : Clock::time_point{};
  if (mbind(get_page(page_id), num_bytes, MPOL_BIND, nodes ? nodes->maskp : NULL, nodes ? nodes->size + 1 : 0,
            MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    const auto error = errno;
    numa_bitmask_free(nodes);
    Fail("Mbind failed: " + std::string(strerror(error)) +
         " . Either no space is left or vm map count is exhausted. Try: \"sudo sysctl vm.max_map_count=X\"");
  }
  if (do_timing) {
    timing_ptr->syscall_time_us += duration_us(syscall_start, Clock::now());
  }
  numa_bitmask_free(nodes);
  _metrics->num_numa_tonode_memory_calls.fetch_add(1, std::memory_order_relaxed);

  const auto metadata_start = do_timing ? Clock::now() : Clock::time_point{};
  _frames[page_id.index].set_node_id(target_memory_node);
  if (do_timing) {
    timing_ptr->metadata_update_time_us += duration_us(metadata_start, Clock::now());
  }

  if (do_timing && timing == nullptr) {
    g_migration_profiler.record_migration(*timing_ptr);
  }
}
#else
void VolatileRegion::mbind_to_numa_node(PageID page_id, const NodeID target_memory_node,
                                        MigrationPhaseTimings* timing) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");

  const auto do_timing = timing != nullptr || g_migration_profiler.enabled();
  auto local_timing = MigrationPhaseTimings{};
  auto* timing_ptr = timing ? timing : &local_timing;

  if (do_timing) {
    timing_ptr->size_type = _size_type;
    timing_ptr->is_batch = false;
    timing_ptr->pages_attempted = 1;
    timing_ptr->pages_migrated = 1;
    timing_ptr->bytes_migrated = bytes_for_size_type(_size_type);
    timing_ptr->syscall_count = 0;
    timing_ptr->syscall_type = "none";
  }

  const auto metadata_start = do_timing ? Clock::now() : Clock::time_point{};
  _frames[page_id.index].set_node_id(target_memory_node);
  if (do_timing) {
    timing_ptr->metadata_update_time_us += duration_us(metadata_start, Clock::now());
  }

  if (do_timing && timing == nullptr) {
    g_migration_profiler.record_migration(*timing_ptr);
  }
}
#endif

void VolatileRegion::move_pages_to_numa_node_batch(const std::vector<PageID>& page_ids, 
                                                     const NodeID target_memory_node,
                                                     bool use_custom_syscall, int migration_mode,
                                                     int migration_max_bs, MigrationPhaseTimings* timing) {
#if HYRISE_NUMA_SUPPORT
  if (page_ids.empty()) {
    return;
  }

  DebugAssert(target_memory_node != INVALID_NODE_ID, "Numa node has not been set.");

  const auto do_timing = timing != nullptr || g_migration_profiler.enabled();
  auto local_timing = MigrationPhaseTimings{};
  auto* timing_ptr = timing ? timing : &local_timing;

  if (do_timing) {
    timing_ptr->size_type = _size_type;
    timing_ptr->is_batch = true;
    timing_ptr->pages_attempted = page_ids.size();
    timing_ptr->pages_migrated = page_ids.size();
    timing_ptr->bytes_migrated = page_ids.size() * bytes_for_size_type(_size_type);
    timing_ptr->syscall_count = 1;
    timing_ptr->syscall_type = use_custom_syscall ? "move_pages2" : "move_pages";
  }

  // Calculate total number of OS pages to move
  const auto page_size_bytes = bytes_for_size_type(_size_type);
  const auto os_pages_per_hyrise_page = page_size_bytes / OS_PAGE_SIZE;
  const auto total_os_pages = page_ids.size() * os_pages_per_hyrise_page;

  // Prepare arrays for move_pages syscall
  std::vector<void*> pages_to_move(total_os_pages);
  std::vector<int> nodes(total_os_pages);
  std::vector<int> status(total_os_pages);

  // Fill arrays with all OS pages from all Hyrise pages
  const auto array_start = do_timing ? Clock::now() : Clock::time_point{};
  size_t os_page_idx = 0;
  for (const auto& page_id : page_ids) {
    DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
    auto page_start = get_page(page_id);
    
    for (size_t i = 0; i < os_pages_per_hyrise_page; ++i) {
      pages_to_move[os_page_idx] = page_start + i * OS_PAGE_SIZE;
      nodes[os_page_idx] = target_memory_node;
      os_page_idx++;
    }
  }
  if (do_timing) {
    timing_ptr->array_build_time_us += duration_us(array_start, Clock::now());
  }

  // Perform batch migration using standard or custom syscall
  const auto syscall_start = do_timing ? Clock::now() : Clock::time_point{};
  if (use_custom_syscall) {
    // Custom move_pages2 syscall: move_pages2(count, pages[], nodes[], status[], migrate_mode, nr_max_batched_migration)
    const auto result = syscall(SYS_move_pages2, 0, total_os_pages, pages_to_move.data(), 
                                 nodes.data(), status.data(), migration_mode, migration_max_bs);
    if (result < 0) {
      const auto error = errno;
      if (error == ENOSYS) {
        Fail("Custom move_pages2 syscall (" + std::to_string(SYS_move_pages2) + 
             ") is not available. Ensure the kernel supports this syscall.");
      }
      Fail("Custom move_pages2 failed: " + std::string(strerror(error)));
    }
  } else {
    // Standard move_pages syscall
    if (move_pages(0, total_os_pages, pages_to_move.data(), nodes.data(), status.data(), MPOL_MF_MOVE) < 0) {
      const auto error = errno;
      Fail("Batch move_pages failed: " + std::string(strerror(error)));
    }
  }
  if (do_timing) {
    timing_ptr->syscall_time_us += duration_us(syscall_start, Clock::now());
  }

  // Update frame metadata for all migrated pages
  const auto metadata_start = do_timing ? Clock::now() : Clock::time_point{};
  for (const auto& page_id : page_ids) {
    _frames[page_id.index].set_node_id(target_memory_node);
  }
  if (do_timing) {
    timing_ptr->metadata_update_time_us += duration_us(metadata_start, Clock::now());
  }

  _metrics->num_numa_tonode_memory_calls.fetch_add(1, std::memory_order_relaxed);

  if (do_timing && timing == nullptr) {
    g_migration_profiler.record_migration(*timing_ptr);
  }
#else
  // Fallback for non-NUMA builds: just update metadata
  (void)use_custom_syscall;
  (void)migration_mode;
  (void)migration_max_bs;
  const auto do_timing = timing != nullptr || g_migration_profiler.enabled();
  auto local_timing = MigrationPhaseTimings{};
  auto* timing_ptr = timing ? timing : &local_timing;
  const auto metadata_start = do_timing ? Clock::now() : Clock::time_point{};
  for (const auto& page_id : page_ids) {
    _frames[page_id.index].set_node_id(target_memory_node);
  }
  if (do_timing) {
    timing_ptr->size_type = _size_type;
    timing_ptr->is_batch = true;
    timing_ptr->pages_attempted = page_ids.size();
    timing_ptr->pages_migrated = page_ids.size();
    timing_ptr->bytes_migrated = page_ids.size() * bytes_for_size_type(_size_type);
    timing_ptr->syscall_count = 0;
    timing_ptr->syscall_type = "none";
    timing_ptr->metadata_update_time_us += duration_us(metadata_start, Clock::now());
  }
  if (do_timing && timing == nullptr) {
    g_migration_profiler.record_migration(*timing_ptr);
  }
#endif
}

void VolatileRegion::free(PageID page_id) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");

  // https://bugs.chromium.org/p/chromium/issues/detail?id=823915
#ifdef __APPLE__
  const int flags = MADV_FREE_REUSABLE;
#elif __linux__
  const int flags = MADV_DONTNEED;
  // const int flags = MADV_FREE;
#endif
  const auto num_bytes = bytes_for_size_type(_size_type);
  auto ptr = get_page(page_id);
  unprotect_page(page_id);
  if (madvise(ptr, num_bytes, flags) < 0) {
    const auto error = errno;
    Fail("Failed to madvice region: " + strerror(error));
  }
  protect_page(page_id);
  _metrics->num_madvice_free_calls.fetch_add(1, std::memory_order_relaxed);
}

std::tuple<PageID, Frame*, std::byte*> VolatileRegion::allocate() {
  // TODO: Handle missing space
  auto idx = PageID::PageIDType{0};
  {
    std::lock_guard<std::mutex> lock(_mutex);
    Assert(_free_slots.any(), "No free slots available in region. TODO: Expand until end of region");
    idx = _free_slots.find_first();
    _free_slots.reset(idx);
  }

  const auto page_id = PageID{_size_type, idx};
  auto ptr = get_page(page_id);
  if constexpr (ENABLE_MPROTECT) {
    if (mprotect(ptr, bytes_for_size_type(_size_type), PROT_READ | PROT_WRITE) != 0) {
      const auto error = errno;
      Fail("Failed to mprotect: " + std::string(strerror(error)) +
           ", page_id=" + std::to_string(page_id.index) +
           ", size_type=" + std::string(magic_enum::enum_name(page_id.size_type())) +
           ", data=" + std::to_string(reinterpret_cast<uintptr_t>(ptr)) +
           ", num_bytes=" + std::to_string(bytes_for_size_type(_size_type)));
    }
  }
  return std::make_tuple(page_id, &_frames[idx], ptr);
}

std::byte* VolatileRegion::get_page(PageID page_id) {
  const auto num_bytes = bytes_for_size_type(_size_type);
  auto data = _region_start + page_id.index * num_bytes;
  DebugAssert(data >= _region_start && (data + num_bytes) <= _region_end,
              "Page out of region bounds: page_id=" + std::to_string(page_id.index) +
                  ", size_type=" + std::string(magic_enum::enum_name(_size_type)) +
                  ", data=" + std::to_string(reinterpret_cast<uintptr_t>(data)) +
                  ", region_start=" + std::to_string(reinterpret_cast<uintptr_t>(_region_start)) +
                  ", region_end=" + std::to_string(reinterpret_cast<uintptr_t>(_region_end)) +
                  ", num_bytes=" + std::to_string(num_bytes));
  DebugAssertPageAligned(data);
  return data;
}

void VolatileRegion::deallocate(const PageID page_id) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
  std::lock_guard<std::mutex> lock(_mutex);
  // TODO: Assert unlocked and clear
  _free_slots.set(page_id.index);
  protect_page(page_id);
}

Frame* VolatileRegion::get_frame(const PageID page_id) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");

  return &_frames[page_id.index];
}

size_t VolatileRegion::memory_consumption() const {
  return sizeof(*this) + sizeof(decltype(_frames)::value_type) * _frames.capacity() + _free_slots.capacity() / CHAR_BIT;
}

void VolatileRegion::clear() {
  std::lock_guard<std::mutex> lock(_mutex);
  _free_slots.set();
  _frames.clear();
}

void VolatileRegion::protect_page(const PageID page_id) {
  if constexpr (ENABLE_MPROTECT) {
    DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
    auto data = get_page(page_id);
    if (mprotect(data, page_id.num_bytes(), PROT_NONE) != 0) {
      const auto error = errno;
      Fail("Failed to mprotect: " + std::string(strerror(error)) + 
           ", page_id=" + std::to_string(page_id.index) +
           ", size_type=" + std::string(magic_enum::enum_name(page_id.size_type())) +
           ", data=" + std::to_string(reinterpret_cast<uintptr_t>(data)) +
           ", num_bytes=" + std::to_string(page_id.num_bytes()));
    }
  }
}

void VolatileRegion::unprotect_page(const PageID page_id) {
  if constexpr (ENABLE_MPROTECT) {
    DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
    auto data = get_page(page_id);
    if (mprotect(data, page_id.num_bytes(), PROT_READ | PROT_WRITE) != 0) {
      const auto error = errno;
      Fail("Failed to mprotect: " + std::string(strerror(error)) + 
           ", page_id=" + std::to_string(page_id.index) +
           ", size_type=" + std::string(magic_enum::enum_name(page_id.size_type())) +
           ", data=" + std::to_string(reinterpret_cast<uintptr_t>(data)) +
           ", num_bytes=" + std::to_string(page_id.num_bytes()));
    }
  }
}

}  // namespace hyrise
