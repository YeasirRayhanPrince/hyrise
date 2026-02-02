#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace hyrise {

struct MigrationPhaseTimings {
    // === COLLECTION PHASE ===
    double queue_scan_time_us = 0;         // Time scanning eviction/promotion queue
    double locking_time_us = 0;            // Time acquiring exclusive locks on frames
    
    // === PREPARATION PHASE ===
    double grouping_time_us = 0;           // Time grouping pages by size type (batch only)
    double dirty_check_time_us = 0;        // Time checking/writing dirty pages to SSD
    double array_build_time_us = 0;        // Time building syscall arrays (batch only)
    
    // === MIGRATION PHASE ===
    double syscall_time_us = 0;            // Total syscall time (move_pages OR mbind)
    std::string syscall_type;              // "move_pages" or "mbind"
    
    // === CLEANUP PHASE ===
    double metadata_update_time_us = 0;    // Time updating frame metadata
    double unlock_time_us = 0;             // Time releasing locks
    
    // === METADATA ===
    size_t pages_attempted = 0;            // Pages we tried to migrate
    size_t pages_migrated = 0;             // Pages successfully migrated
    size_t bytes_migrated = 0;             // Total bytes moved
    PageSizeType size_type = MIN_PAGE_SIZE_TYPE;  // Page size for this operation
    bool is_batch = false;                 // true = batched, false = single-page
    size_t syscall_count = 0;              // Number of syscalls (batch=1, single=N)
    std::string operation_type;            // "eviction" or "promotion"
    
    // Helper to compute total time
    double total_time_us() const {
        return queue_scan_time_us + locking_time_us + grouping_time_us + 
               dirty_check_time_us + array_build_time_us + syscall_time_us + 
               metadata_update_time_us + unlock_time_us;
    }

    nlohmann::json to_json() const {
        return nlohmann::json{
            {"queue_scan_time_us", queue_scan_time_us},
            {"locking_time_us", locking_time_us},
            {"grouping_time_us", grouping_time_us},
            {"dirty_check_time_us", dirty_check_time_us},
            {"array_build_time_us", array_build_time_us},
            {"syscall_time_us", syscall_time_us},
            {"syscall_type", syscall_type},
            {"metadata_update_time_us", metadata_update_time_us},
            {"unlock_time_us", unlock_time_us},
            {"total_time_us", total_time_us()},
            {"pages_attempted", pages_attempted},
            {"pages_migrated", pages_migrated},
            {"bytes_migrated", bytes_migrated},
            {"size_type", static_cast<int>(size_type)},
            {"is_batch", is_batch},
            {"syscall_count", syscall_count},
            {"operation_type", operation_type}
        };
    }
};

class MigrationProfiler {
 public:
    void set_enabled(const bool enabled) {
        _enabled.store(enabled, std::memory_order_relaxed);
    }

    bool enabled() const {
        return _enabled.load(std::memory_order_relaxed);
    }

    void record_migration(const MigrationPhaseTimings& timing) {
        if (!enabled()) {
            return;
        }
        std::lock_guard<std::mutex> lock(_samples_mutex);
        _samples.push_back(timing);
    }

    void print_summary() const {
        std::lock_guard<std::mutex> lock(_samples_mutex);
        
        if (_samples.empty()) {
            std::cout << "\n=== Migration Profiler: No samples recorded ===" << std::endl;
            return;
        }

        // Separate batch vs single-page
        std::vector<MigrationPhaseTimings> batch_samples;
        std::vector<MigrationPhaseTimings> single_samples;
        
        for (const auto& sample : _samples) {
            if (sample.is_batch) {
                batch_samples.push_back(sample);
            } else {
                single_samples.push_back(sample);
            }
        }

        std::cout << "\n=== Migration Profiler Summary ===" << std::endl;
        std::cout << "Total samples: " << _samples.size() 
                  << " (Batch: " << batch_samples.size() 
                  << ", Single: " << single_samples.size() << ")" << std::endl;

        if (!batch_samples.empty()) {
            print_category_summary("BATCH MIGRATION", batch_samples);
        }
        
        if (!single_samples.empty()) {
            print_category_summary("SINGLE-PAGE MIGRATION", single_samples);
        }
        
        std::cout << "====================================\n" << std::endl;
    }

    nlohmann::json to_json() const {
        std::lock_guard<std::mutex> lock(_samples_mutex);
        nlohmann::json result;
        result["total_samples"] = _samples.size();
        result["samples"] = nlohmann::json::array();
        
        for (const auto& sample : _samples) {
            result["samples"].push_back(sample.to_json());
        }
        
        return result;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(_samples_mutex);
        _samples.clear();
    }

    size_t sample_count() const {
        std::lock_guard<std::mutex> lock(_samples_mutex);
        return _samples.size();
    }

 private:
     std::atomic<bool> _enabled{false};
    mutable std::mutex _samples_mutex;
    std::vector<MigrationPhaseTimings> _samples;

    void print_category_summary(const std::string& category, 
                                 const std::vector<MigrationPhaseTimings>& samples) const {
        std::cout << "\n--- " << category << " ---" << std::endl;
        
        // Compute averages
        double avg_queue_scan = 0, avg_locking = 0, avg_grouping = 0;
        double avg_dirty_check = 0, avg_array_build = 0, avg_syscall = 0;
        double avg_metadata = 0, avg_unlock = 0, avg_total = 0;
        size_t total_pages = 0, total_bytes = 0, total_syscalls = 0;

        for (const auto& s : samples) {
            avg_queue_scan += s.queue_scan_time_us;
            avg_locking += s.locking_time_us;
            avg_grouping += s.grouping_time_us;
            avg_dirty_check += s.dirty_check_time_us;
            avg_array_build += s.array_build_time_us;
            avg_syscall += s.syscall_time_us;
            avg_metadata += s.metadata_update_time_us;
            avg_unlock += s.unlock_time_us;
            avg_total += s.total_time_us();
            total_pages += s.pages_migrated;
            total_bytes += s.bytes_migrated;
            total_syscalls += s.syscall_count;
        }

        const size_t n = samples.size();
        avg_queue_scan /= n; avg_locking /= n; avg_grouping /= n;
        avg_dirty_check /= n; avg_array_build /= n; avg_syscall /= n;
        avg_metadata /= n; avg_unlock /= n; avg_total /= n;

        std::cout << "  Samples: " << n << std::endl;
        std::cout << "  Total pages migrated: " << total_pages << std::endl;
        std::cout << "  Total bytes migrated: " << (total_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "  Total syscalls: " << total_syscalls << std::endl;
        std::cout << "  Avg pages per operation: " << (total_pages / (double)n) << std::endl;
        
        if (!samples.empty()) {
            std::cout << "  Syscall type: " << samples[0].syscall_type << std::endl;
        }

        std::cout << "\n  Average timings (microseconds):" << std::endl;
        std::cout << "    Queue scan:      " << avg_queue_scan << " us" << std::endl;
        std::cout << "    Locking:         " << avg_locking << " us" << std::endl;
        std::cout << "    Grouping:        " << avg_grouping << " us" << std::endl;
        std::cout << "    Dirty check:     " << avg_dirty_check << " us" << std::endl;
        std::cout << "    Array build:     " << avg_array_build << " us" << std::endl;
        std::cout << "    Syscall:         " << avg_syscall << " us" << std::endl;
        std::cout << "    Metadata update: " << avg_metadata << " us" << std::endl;
        std::cout << "    Unlock:          " << avg_unlock << " us" << std::endl;
        std::cout << "    TOTAL:           " << avg_total << " us" << std::endl;

        // Show percentage breakdown
        std::cout << "\n  Time breakdown (%):" << std::endl;
        std::cout << "    Syscall:         " << (avg_syscall / avg_total * 100.0) << "%" << std::endl;
        std::cout << "    Application:     " << ((avg_total - avg_syscall) / avg_total * 100.0) << "%" << std::endl;
    }
};

// Global profiler instance
extern MigrationProfiler g_migration_profiler;

}  // namespace hyrise
