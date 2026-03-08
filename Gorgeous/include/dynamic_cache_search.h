//
// DynamicCache (In-Memory First) Search
// Graph-based disk search with in-memory caching strategy
//
#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <atomic>
#include <future>
#include <cstring>
#include <algorithm>
#include "utils.h"
#include "neighbor.h"
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"

namespace diskann {

// DynamicCache search configuration
struct DynamicCacheSearchConfig {
    // Prefetch window size (number of concurrent I/O requests)
    unsigned prefetch_window = 4;  // r in paper
    
    // Search list size
    unsigned search_list_size = 100;  // L in paper
    
    // Enable statistics
    bool enable_stats = true;
    
    // FIFO cache size (number of pages)
    size_t cache_capacity = 1000;
    
    DynamicCacheSearchConfig() = default;
};

// Maximum number of hops for per-hop cache statistics
constexpr unsigned MAX_HOP_STATS = 100;

// DynamicCache search statistics
// Definition:
// - One hop = expansion of the "nearest unvisited disk node" in retset
// - Cache hit/miss is only counted when expanding this nearest node and hop_count is incremented
// - MISS = I/O request was specifically issued to expand this node (selected_node == trigger_node)
// - HIT = This node's data was hitchhiked (selected_node != trigger_node)
// - If expanding a node that is not the nearest (e.g., because the nearest node's I/O hasn't returned yet
//   and we selected another node from cache), it doesn't count as a hop and doesn't count toward cache stats
// 
// Note: During prefetch, we record which node triggered the I/O for each page (page_meta)
struct DynamicCacheSearchStats {
    // I/O statistics
    unsigned total_ios = 0;
    unsigned async_ios = 0;
    unsigned blocked_waits = 0;  // Number of CPU starvation events
    
    // Candidate statistics
    unsigned candidates_expanded = 0;
    unsigned candidates_skipped = 0;  // Candidates skipped due to incomplete I/O
    unsigned in_memory_expansions = 0;  // Number of expansions directly from in-memory navigation graph
    
    // Cache statistics
    // HIT: Expanded node is not the node that triggered I/O (hitchhiked)
    // MISS: Expanded node is the node that triggered I/O (I/O was specifically issued for it)
    unsigned cache_hits = 0;
    unsigned cache_misses = 0;
    
    // Per-hop cache statistics (first 100 hops)
    // hop definition: expanding the nearest unvisited disk node in retset counts as one hop
    unsigned hop_cache_hits[MAX_HOP_STATS] = {0};    // Cache hits per hop
    unsigned hop_cache_misses[MAX_HOP_STATS] = {0};  // Cache misses per hop
    unsigned total_hops = 0;  // Actual number of hops (= number of disk node expansions)
    
    // Time statistics
    double prefetch_us = 0.0;
    double candidate_selection_us = 0.0;
    double expansion_us = 0.0;
    double io_wait_us = 0.0;
    
    void reset() {
        total_ios = 0;
        async_ios = 0;
        blocked_waits = 0;
        candidates_expanded = 0;
        candidates_skipped = 0;
        in_memory_expansions = 0;
        cache_hits = 0;
        cache_misses = 0;
        total_hops = 0;
        std::memset(hop_cache_hits, 0, sizeof(hop_cache_hits));
        std::memset(hop_cache_misses, 0, sizeof(hop_cache_misses));
        prefetch_us = 0.0;
        candidate_selection_us = 0.0;
        expansion_us = 0.0;
        io_wait_us = 0.0;
    }
    
    // Record cache hit/miss for a specific hop
    void record_hop_cache(unsigned hop, bool is_hit) {
        if (hop < MAX_HOP_STATS) {
            if (is_hit) {
                hop_cache_hits[hop]++;
            } else {
                hop_cache_misses[hop]++;
            }
        }
        total_hops = std::max(total_hops, hop + 1);
    }
    
    // Get cache hit rate for a specific hop
    float get_hop_hit_rate(unsigned hop) const {
        if (hop >= MAX_HOP_STATS) return 0.0f;
        unsigned total = hop_cache_hits[hop] + hop_cache_misses[hop];
        if (total == 0) return 0.0f;
        return 100.0f * hop_cache_hits[hop] / total;
    }
    
    void print() const {
        std::cout << "\n=== DynamicCache Search Statistics ===" << std::endl;
        std::cout << "Total I/Os: " << total_ios << std::endl;
        std::cout << "Async I/Os: " << async_ios << std::endl;
        std::cout << "Blocked waits (CPU starvation): " << blocked_waits << std::endl;
        std::cout << "Candidates expanded: " << candidates_expanded << std::endl;
        std::cout << "Candidates skipped: " << candidates_skipped << std::endl;
        std::cout << "In-memory expansions: " << in_memory_expansions << std::endl;
        std::cout << "Cache hits (hitchhike): " << cache_hits << std::endl;
        std::cout << "Cache misses (trigger IO): " << cache_misses << std::endl;
        
        unsigned total_cache_accesses = cache_hits + cache_misses;
        if (total_cache_accesses > 0) {
            float cache_hit_rate = 100.0f * cache_hits / total_cache_accesses;
            std::cout << "Cache hit rate: " << cache_hit_rate << "%" << std::endl;
        }
        
        std::cout << "Prefetch time: " << prefetch_us << " us" << std::endl;
        std::cout << "Candidate selection time: " << candidate_selection_us << " us" << std::endl;
        std::cout << "Expansion time: " << expansion_us << " us" << std::endl;
        std::cout << "I/O wait time: " << io_wait_us << " us" << std::endl;
        
        std::cout << "\n--- Per-Hop Cache Hit Rate (first " << std::min(total_hops, MAX_HOP_STATS) << " hops) ---" << std::endl;
        std::cout << "(hop = expansion of disk node, HIT = hitchhike, MISS = triggered IO)" << std::endl;
        for (unsigned i = 0; i < std::min(total_hops, MAX_HOP_STATS); i++) {
            unsigned total = hop_cache_hits[i] + hop_cache_misses[i];
            if (total > 0) {
                float hit_rate = 100.0f * hop_cache_hits[i] / total;
                std::cout << "Hop " << i << ": " << hit_rate << "% (" 
                         << hop_cache_hits[i] << "/" << total << ")" << std::endl;
            }
        }
        std::cout << "==============================\n" << std::endl;
    }
};

// I/O request status
enum class IOStatus {
    PENDING,    // Issued but not completed
    COMPLETED,  // Completed
    FAILED      // Failed
};

// Asynchronous I/O request tracking
struct AsyncIORequest {
    unsigned node_id;      // Node that triggered this I/O request (used to determine miss/hit)
    unsigned page_id;
    char* buffer;
    IOStatus status;
    
    AsyncIORequest(unsigned nid, unsigned pid, char* buf)
        : node_id(nid), page_id(pid), buffer(buf), status(IOStatus::PENDING) {}
};

// Page metadata: records which node triggered the I/O for this page
struct PageMeta {
    unsigned trigger_node;  // Node ID that triggered the I/O
    bool is_loaded;         // Whether loaded into cache
    
    PageMeta() : trigger_node(INF), is_loaded(false) {}
    PageMeta(unsigned trigger) : trigger_node(trigger), is_loaded(false) {}
};

// FIFO cache manager (for DynamicCache)
class FIFOCache {
private:
    size_t capacity_;
    size_t page_size_;
    std::queue<unsigned> fifo_queue_;
    tsl::robin_map<unsigned, char*> cache_map_;
    
public:
    FIFOCache(size_t capacity, size_t page_size)
        : capacity_(capacity), page_size_(page_size) {}
    
    ~FIFOCache() {
        clear();
    }
    
    char* get(unsigned page_id) {
        auto it = cache_map_.find(page_id);
        if (it != cache_map_.end()) {
            return it->second;
        }
        return nullptr;
    }
    
    void put(unsigned page_id, const char* data) {
        if (capacity_ == 0) return;
        
        // If already exists, update data
        auto it = cache_map_.find(page_id);
        if (it != cache_map_.end()) {
            std::memcpy(it->second, data, page_size_);
            return;
        }
        
        // If cache is full, evict the oldest
        if (fifo_queue_.size() >= capacity_) {
            unsigned evict_page = fifo_queue_.front();
            fifo_queue_.pop();
            
            auto evict_it = cache_map_.find(evict_page);
            if (evict_it != cache_map_.end()) {
                if (evict_it->second) {
                    aligned_free(evict_it->second);
                }
                cache_map_.erase(evict_it);
            }
        }
        
        // Allocate new memory and insert
        char* new_data = nullptr;
        alloc_aligned((void**)&new_data, page_size_, page_size_);
        std::memcpy(new_data, data, page_size_);
        
        cache_map_[page_id] = new_data;
        fifo_queue_.push(page_id);
    }
    
    bool contains(unsigned page_id) const {
        return cache_map_.find(page_id) != cache_map_.end();
    }
    
    size_t size() const { return cache_map_.size(); }
    
    void clear() {
        for (auto& pair : cache_map_) {
            if (pair.second) {
                aligned_free(pair.second);
            }
        }
        cache_map_.clear();
        while (!fifo_queue_.empty()) {
            fifo_queue_.pop();
        }
    }
};

} // namespace diskann
