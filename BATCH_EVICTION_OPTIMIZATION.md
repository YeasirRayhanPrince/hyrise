# Batch Eviction Optimization Analysis

## Current Bottleneck Analysis

### Why Batches Stay Small (1-2 pages)

**Issue 1: Conservative batch size calculation in `ensure_free_pages()`**
```cpp
const auto bytes_needed = (current_bytes + bytes_required - freed_bytes) - max_bytes;
const auto min_page_bytes = bytes_for_size_type(MIN_PAGE_SIZE_TYPE);
const auto pages_to_evict = std::max<size_t>(1, (bytes_needed + min_page_bytes - 1) / min_page_bytes);
```

**Problem:** This calculates only the *minimum* pages needed. With 512MB DRAM pool and small allocations, `bytes_needed` is tiny (4-16KB), so `pages_to_evict` = 1 page.

**Solution:** Calculate a target batch size that's larger:
```cpp
const size_t MIN_BATCH_SIZE = 64;  // Pages to batch minimum
const size_t MAX_BATCH_SIZE = 256; // Pages to batch maximum
const auto pages_to_evict = std::max(MIN_BATCH_SIZE, 
                                     (bytes_needed / min_page_bytes) * 4); // 4x what we need
```

---

### Issue 2: Eviction queue may not have enough eligible pages

**Problem:** Phase 1 of `evict_batch()` collects pages from the queue:
- Many pages may be locked by active readers
- Pages may be on different NUMA nodes (skipped)
- Pages may not be in evictable state

**Solution:** Implement a prefetch/lookahead strategy:
```cpp
// Phase 0: Pre-scan queue to find pages, mark them, and prepare
// This increases the chances Phase 1 finds eligible pages
```

---

### Issue 3: Multiple size types create small batches

**Current behavior:**
- Collects pages from ALL size types into a single `evict_batch()` call
- Groups by size type: `pages_by_size[size_type]`
- Batches within each size type

**Example:** 
- Request 64 pages
- 20 are KiB4, 20 are KiB8, 24 are KiB16
- Instead of 1 batch of 64, you get 3 batches of 20, 20, 24

**Solution:** Pre-filter by size type or request specific size batches

---

## Optimization Strategies

### Strategy 1: Aggressive Pre-Collection (Simplest)

**Change in `ensure_free_pages()`:**
```cpp
// Instead of calculating minimum, request a large batch
const size_t TARGET_BATCH_SIZE = 128;  // Or 256 for even better
const size_t bytes_needed = (current_bytes + bytes_required - freed_bytes) - max_bytes;
const auto pages_to_evict = std::max(TARGET_BATCH_SIZE, 
                                     (bytes_needed / bytes_for_size_type(MIN_PAGE_SIZE_TYPE)) * 2);
```

**Pros:**
- 1 line change
- Immediately gets 64+ page batches
- No new logic needed

**Cons:**
- May evict more than needed (wastes work)
- Queue might not have that many eligible pages

---

### Strategy 2: Two-Phase Queue Scanning (Better)

**Phase 0:** Pre-scan to identify lockable pages
```cpp
// Scan ahead in queue without locking, just check state
// If we find N eligible pages, lock them in Phase 1
while (pages_scanned < LOOKAHEAD_SIZE && queue.peek(&item)) {
    if (is_evictable(item)) {
        eligible_pages++;
    }
    pages_scanned++;
}
pages_to_collect = eligible_pages;
```

**Pros:**
- Estimates real batch size before locking
- Adaptive to queue state
- Better predictability

**Cons:**
- 2-phase logic, more complex
- Peek operation might not be available in TBB queue

---

### Strategy 3: Pre-marking Strategy (Best Long-term)

**Idea:** Keep a separate "candidates for eviction" list

```cpp
// In background or during idle:
// - Scan eviction queue
// - Mark pages that are evictable
// - Store page IDs in a "batch candidates" vector

// In evict_batch():
// - Use pre-marked pages instead of scanning queue
// - Lock and evict them
```

**Pros:**
- Decouples candidate finding from eviction
- Can collect larger batches asynchronously
- Better for NUMA node filtering

**Cons:**
- Requires additional background work
- More memory for candidate list

---

## Recommended Quick Win: Aggressive Pre-Collection

Based on your 512MB DRAM + 1GB NUMA config, here's what I recommend:

### Changes to `buffer_pool.cpp`:

**1. Add batch size constants:**
```cpp
constexpr size_t BATCH_SIZE_CONSERVATIVE = 32;
constexpr size_t BATCH_SIZE_NORMAL = 64;
constexpr size_t BATCH_SIZE_AGGRESSIVE = 128;
```

**2. Modify `ensure_free_pages()` calculation:**
```cpp
if (enable_batching) {
    while ((current_bytes + bytes_required - freed_bytes) > max_bytes) {
        const auto bytes_needed = (current_bytes + bytes_required - freed_bytes) - max_bytes;
        const auto min_page_bytes = bytes_for_size_type(MIN_PAGE_SIZE_TYPE);
        
        // Try to batch more pages for better performance
        // Use BATCH_SIZE_NORMAL as minimum to ensure syscall batching benefits
        const auto pages_to_evict = std::max(BATCH_SIZE_NORMAL, 
                                             (bytes_needed + min_page_bytes - 1) / min_page_bytes);
        
        const auto evicted = evict_batch(pages_to_evict, &freed_bytes);
```

---

## Additional Optimizations to Consider

### 1. Increase NUMA migration batch threshold
Currently `move_pages_to_numa_node_batch()` works with any size, but benefit is only with 10+ pages.

**Recommendation:** Only use batch path if `> 10 pages`:
```cpp
if (page_ids_to_migrate.size() >= 10) {
    region->move_pages_to_numa_node_batch(page_ids_to_migrate, target_buffer_pool->node_id);
} else {
    // Fall back to individual migration
}
```

### 2. Track batch statistics
Add metrics:
```cpp
metrics->avg_batch_size
metrics->max_batch_size
metrics->batches_count
```

---

## Testing Plan

1. **Baseline (Current):** Run with BATCH_SIZE=1 (should see 1-2 page batches)
2. **Conservative:** BATCH_SIZE=32 (expect 32+ page batches)
3. **Normal:** BATCH_SIZE=64 (expect 64+ page batches)
4. **Aggressive:** BATCH_SIZE=128 (expect 128+ page batches)

Compare performance metrics:
- Batch sizes in logs
- TLB shootdown frequency (if available)
- Eviction latency
- Overall throughput

---

## Summary Table

| Strategy | Implementation | Batch Size | Complexity | Impact |
|----------|----------------|-----------|------------|--------|
| Current  | Min calculation | 1-2       | Low       | Poor syscall amortization |
| Aggressive Pre-Collection | Change one calculation | 64-128 | Low | 4-8x syscall reduction |
| Two-Phase Scanning | Add lookahead | 32-64 | Medium | Adaptive to queue |
| Pre-marking | Background worker | 128-256 | High | Best long-term |

---

## Page Eviction Queue Mechanics - Simulation

### Simulation Setup

**Buffer pool configuration:**
- DRAM pool: 512 MB (131,072 pages of 4KB each)
- NUMA pool: 1 GB
- Workload: YCSB UpdateHeavy with 11 threads (50% reads, 50% updates, Zipfian distribution)

Let's simulate what happens over a 100ms window:

---

### Time T=0ms: Benchmark Starts

**Buffer pool state:**
```
Pages allocated: 131,000 / 131,072 (buffer pool almost full)
Eviction queue size: 0 items
```

**11 worker threads actively accessing pages:**
- Thread 1: pins Page #500 (shared, for read)
- Thread 2: pins Page #501 (shared)
- Thread 3: pins Page #1500 (exclusive, for update)
- Thread 4: pins Page #200 (shared)
- ... (7 more threads accessing different pages)

All pages are **pinned** → None are in eviction queue yet.

---

### Time T=5ms: First Page Unpins

**Thread 1 finishes reading Page #500:**
```cpp
unpin_shared(Page #500)
  → frame->ref_count: 1 → 0
  → unlock_shared() returns TRUE
  → add_to_eviction_queue(Page #500, version=1)
```

**Eviction queue:**
```
Queue: [{page_id: 500, timestamp: 1}]
Size: 1 item
```

But immediately (microseconds later):
- Thread 5 needs Page #500 (Zipfian = hot page)
- `pin_shared(Page #500)` → removes from consideration
- Page #500 ref_count: 0 → 1

Page #500 is **back in use** before eviction happens!

---

### Time T=10ms: More Pages Unpin

**Thread 2 finishes Page #501, Thread 4 finishes Page #200:**

**Eviction queue:**
```
Queue: [{500, v:1}, {501, v:1}, {200, v:1}]
Size: 3 items
```

**But Thread 7 immediately re-pins Page #501:**
- `pin_shared(501)` → ref_count: 0 → 1
- Page #501 still in queue, but **no longer evictable** (version mismatch or locked)

---

### Time T=15ms: Allocation Pressure - Need to Evict!

**Thread 8 needs to allocate a new page (Page #131,001):**
```
Buffer pool is full → ensure_free_pages(bytes_required=4096)
  → bytes_needed = 4096
  → pages_to_evict = 1 (calculated minimum)
  → evict_batch(pages_to_request=1)
```

**evict_batch Phase 1:** Try to collect 1 page from queue:

```
Attempt 1: Pop {500, v:1}
  → Check frame state: ref_count=1 (Thread 5 has it pinned!)
  → try_lock_exclusive() = FALSE
  → Skip this page, continue

Attempt 2: Pop {501, v:1}
  → Check frame state: ref_count=1 (Thread 7 has it pinned!)
  → try_lock_exclusive() = FALSE
  → Skip this page, continue

Attempt 3: Pop {200, v:1}
  → Check frame state: ref_count=0, version=1 ✓
  → try_lock_exclusive() = TRUE ✓
  → Add to pages_by_size[KiB4] = [200]
  → Collected 1 page, done
```

**evict_batch Phase 2:** Migrate the batch
```
[BufferPool::evict_batch] Phase 2: Locked 1 pages
[BufferPool::evict_batch] Attempting batch NUMA migration for 1 pages
[VolatileRegion::move_pages_to_numa_node_batch] Moving 1 Hyrise pages to NUMA node 1
[BufferPool] Batch eviction completed: evicted=1, freed_bytes=4096
```

**Result:** Only 1 page evicted despite having 3 in queue!

---

### Time T=20ms: Try to Request Larger Batch

**Now with TARGET_BATCH_SIZE = 64:**

**Thread 9 needs Page #131,002:**
```
ensure_free_pages(bytes_required=4096)
  → pages_to_evict = max(64, 1) = 64 (request 64 pages!)
  → evict_batch(pages_to_request=64)
```

**Eviction queue state:**
```
Queue: [{1200, v:3}, {1201, v:2}, {1500, v:4}, {1600, v:1}, ..., {5000, v:2}]
Size: 28 items (accumulated since last eviction)
```

**evict_batch Phase 1:** Try to collect 64 pages:

```
Pages collected: 0
Target: 64

Pop queue 28 times (all available items):
  - {1200, v:3}: ref_count=2 → SKIP (pinned by Thread 1 & 2)
  - {1201, v:2}: ref_count=0, try_lock=TRUE → COLLECT ✓ (1 collected)
  - {1500, v:4}: ref_count=1 → SKIP (pinned)
  - {1600, v:1}: version mismatch (v:1 but frame is v:5) → SKIP (reused)
  - {1800, v:2}: ref_count=0, try_lock=TRUE → COLLECT ✓ (2 collected)
  ... (scanning 28 items)
  - {5000, v:2}: ref_count=0, try_lock=TRUE → COLLECT ✓
  
Queue exhausted. Collected: 3 pages (not 64!)
```

**evict_batch Phase 2:**
```
[BufferPool::evict_batch] Phase 2: Locked 3 pages
[BufferPool] Batch eviction completed: evicted=3, freed_bytes=12288
```

**Why only 3 pages?** 
- Queue had 28 items, but 25 were **already re-pinned** by other threads
- Hot workload (Zipfian) = same pages accessed repeatedly
- Queue entries become stale faster than new ones accumulate

---

### Time T=50ms: Steady State Pattern

**Typical cycle repeats every few milliseconds:**

```
1. Thread unpins Page X → add to queue (queue size: +1)
2. Another thread pins Page X (0.1ms later) → still in queue but not evictable
3. Eviction needs space → scan queue:
   - Find 25 stale entries (pinned or version mismatch)
   - Find 1-2 truly evictable pages
   - Evict those 1-2 pages
4. Queue size drops but refills slowly

Result: Batches stay at 1-3 pages despite requesting 64
```

---

### Visualization of Queue State

```
Time:     T=0ms      T=10ms      T=15ms      T=20ms      T=50ms
Queue:    []         [3 items]   [0 items]   [28 items]  [5 items]
                     (hot)       (drained)   (25 stale)  (rotating)

Eligible  0          0-1         1           2-3         1-2
pages:    

Batch     -          -           1           3           1
collected:
```

**Pattern:** Queue fills with entries, but most become stale before batch collection happens.

---

### Why Batches Stay Small: The Core Problem

**1. Hot pages rotate too quickly:**
```
Page #500: pin → unpin → QUEUE → re-pin (50 microseconds later)
           ↑___________________________________↓
           Still in queue but ref_count > 0 (not evictable)
```

**2. Queue is a FIFO, but eviction needs COLD pages:**
- FIFO order = pages added most recently
- Cold pages = pages NOT accessed recently
- Conflict: Recently unpinned ≠ cold (in Zipfian workload)

**3. 100% cache hit rate = all pages stay hot:**
- Your logs show `cache_hit_rate=1.0`
- Means: every page requested is already in DRAM
- No pages naturally "cool down" to become eviction candidates

---

### Solution Strategies (from simulation insights)

**Strategy 1: Scan deeper into queue**
```cpp
// Instead of stopping after 64 pops, scan 512+ items
while (collected < 64 && scanned < 512) {
  if (queue.pop(item)) { /* try to lock */ }
  scanned++;
}
```
**Trade-off:** More CPU scanning stale entries vs. larger batches

**Strategy 2: Delay eviction to let queue cool**
```cpp
// Only evict when queue has 512+ items accumulated
if (eviction_queue->size() < 512) {
  evict_single();  // Fall back to individual
} else {
  evict_batch(64);
}
```
**Trade-off:** May delay allocation vs. better batching

**Strategy 3: Age-based marking**
```cpp
// Add timestamp when entering queue
// Only evict pages that have been in queue > 10ms
if (now - item.queue_entry_time > 10ms) {
  // Likely truly cold
}
```
**Trade-off:** Increased metadata vs. better candidate selection
