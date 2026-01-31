#include "volatile_region.hpp"
#include <sys/mman.h>
#include <unistd.h>
#include "utils/assert.hpp"

#if HYRISE_NUMA_SUPPORT
#include <numa.h>
#include <numaif.h>
#endif

namespace hyrise {

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

void VolatileRegion::move_page_to_numa_node(PageID page_id, const NodeID target_memory_node) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
#if HYRISE_NUMA_SUPPORT
  DebugAssert(target_memory_node != INVALID_NODE_ID, "Numa node has not been set.");
  static thread_local std::vector<void*> pages_to_move{bytes_for_size_type(_size_type) / OS_PAGE_SIZE};
  static thread_local std::vector<int> nodes{static_cast<int>(bytes_for_size_type(_size_type) / OS_PAGE_SIZE)};
  static thread_local std::vector<int> status{static_cast<int>(bytes_for_size_type(_size_type) / OS_PAGE_SIZE)};

  for (auto i = 0u; i < pages_to_move.size(); ++i) {
    pages_to_move[i] = get_page(page_id) + i * OS_PAGE_SIZE;
    nodes[i] = target_memory_node;
  }
  if (move_pages(0, pages_to_move.size(), pages_to_move.data(), nodes.data(), status.data(), MPOL_MF_MOVE) < 0) {
    const auto error = errno;
    Fail("Move pages failed: " + strerror(error));
  }
  _metrics->num_numa_tonode_memory_calls.fetch_add(1, std::memory_order_relaxed);
  _frames[page_id.index].set_node_id(target_memory_node);
#endif
}

void VolatileRegion::mbind_to_numa_node(PageID page_id, const NodeID target_memory_node) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");

#if HYRISE_NUMA_SUPPORT
  DebugAssert(target_memory_node != INVALID_NODE_ID, "Numa node has not been set.");

  const auto num_bytes = bytes_for_size_type(_size_type);
  auto nodes = numa_allocate_nodemask();
  numa_bitmask_setbit(nodes, target_memory_node);
  if (mbind(get_page(page_id), num_bytes, MPOL_BIND, nodes ? nodes->maskp : NULL, nodes ? nodes->size + 1 : 0,
            MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    const auto error = errno;
    numa_bitmask_free(nodes);
    Fail("Mbind failed: " + strerror(error) +
         " . Either no space is left or vm map count is exhausted. Try: \"sudo sysctl vm.max_map_count=X\"");
  }
  numa_bitmask_free(nodes);
  _metrics->num_numa_tonode_memory_calls.fetch_add(1, std::memory_order_relaxed);
#endif
  _frames[page_id.index].set_node_id(target_memory_node);
}

void VolatileRegion::move_pages_to_numa_node_batch(const std::vector<PageID>& page_ids, 
                                                     const NodeID target_memory_node) {
#if HYRISE_NUMA_SUPPORT
  if (page_ids.empty()) {
    return;
  }

  DebugAssert(target_memory_node != INVALID_NODE_ID, "Numa node has not been set.");

  // Calculate total number of OS pages to move
  const auto page_size_bytes = bytes_for_size_type(_size_type);
  const auto os_pages_per_hyrise_page = page_size_bytes / OS_PAGE_SIZE;
  const auto total_os_pages = page_ids.size() * os_pages_per_hyrise_page;

  // std::cout << "[VolatileRegion::move_pages_to_numa_node_batch] Moving " << page_ids.size() 
  //          << " Hyrise pages (" << total_os_pages << " OS pages) to NUMA node " 
  //          << target_memory_node << std::endl;

  // Prepare arrays for move_pages syscall
  std::vector<void*> pages_to_move(total_os_pages);
  std::vector<int> nodes(total_os_pages);
  std::vector<int> status(total_os_pages);

  // Fill arrays with all OS pages from all Hyrise pages
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

  // Perform batch migration
  if (move_pages(0, total_os_pages, pages_to_move.data(), nodes.data(), status.data(), MPOL_MF_MOVE) < 0) {
    const auto error = errno;
    Fail("Batch move_pages failed: " + strerror(error));
  }

  // std::cout << "[VolatileRegion::move_pages_to_numa_node_batch] Successfully migrated " 
  //          << page_ids.size() << " pages to node " << target_memory_node << std::endl;

  // Update frame metadata for all migrated pages
  for (const auto& page_id : page_ids) {
    _frames[page_id.index].set_node_id(target_memory_node);
  }

  _metrics->num_numa_tonode_memory_calls.fetch_add(1, std::memory_order_relaxed);
#else
  // Fallback for non-NUMA builds: just update metadata
  for (const auto& page_id : page_ids) {
    _frames[page_id.index].set_node_id(target_memory_node);
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
