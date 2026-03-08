//
// HybridCache Search: Two-Phase Search with Similarity-Aware Prefetching
//
#pragma once

#include <vector>
#include <functional>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "utils.h"
#include "hybridcache_cache.h"

namespace diskann {

// HybridCache search configuration
struct HybridCacheSearchConfig {
    // Phase transition parameters
    float theta = 0.3f;  // Threshold: switch to Phase 2 when top candidates visited reach l_search * theta
    
    // Similarity-Aware Read parameters
    unsigned prefetch_pages = 3;  // Number of consecutive pages to prefetch in Phase 2
    
    // Cache configuration
    size_t static_cache_capacity = 100;   // Static cache capacity (in pages)
    size_t dynamic_cache_capacity = 500;  // Dynamic cache capacity (in pages)
    
    // Statistics toggle
    bool enable_stats = true;
    
    HybridCacheSearchConfig() = default;
};

// HybridCache search statistics
struct HybridCacheSearchStats {
    static constexpr unsigned MAX_HOP_STATS = 100;  // Statistics for first 100 hops
    
    // Phase statistics
    unsigned phase_1_ios = 0;
    unsigned phase_2_ios = 0;
    unsigned phase_transition_point = 0;
    
    // Cache statistics
    unsigned static_cache_hits = 0;
    unsigned dynamic_cache_hits = 0;
    unsigned cache_misses = 0;
    
    // Per-hop cache hit statistics (first 100 hops)
    // 0 = miss, 1 = static cache hit, 2 = dynamic cache hit
    unsigned per_hop_cache_status[MAX_HOP_STATS] = {0};
    unsigned total_hops = 0;  // Actual number of nodes expanded
    
    // Prefetch statistics
    unsigned prefetch_operations = 0;
    unsigned prefetch_pages_total = 0;
    unsigned prefetch_useful_pages = 0;  // Prefetched pages actually used
    

    double phase_detection_us = 0.0;
    double cache_lookup_us = 0.0;
    double prefetch_us = 0.0;
    
    void reset() {
        phase_1_ios = 0;
        phase_2_ios = 0;
        phase_transition_point = 0;
        static_cache_hits = 0;
        dynamic_cache_hits = 0;
        cache_misses = 0;
        prefetch_operations = 0;
        prefetch_pages_total = 0;
        prefetch_useful_pages = 0;
        phase_detection_us = 0.0;
        cache_lookup_us = 0.0;
        prefetch_us = 0.0;
        total_hops = 0;
        std::memset(per_hop_cache_status, 0, sizeof(per_hop_cache_status));
    }
    
    void record_hop(unsigned status) {
        if (total_hops < MAX_HOP_STATS) {
            per_hop_cache_status[total_hops] = status;
        }
        total_hops++;
    }
    
    void print() const {
        std::cout << "\n=== HybridCache Search Statistics ===" << std::endl;
        std::cout << "Phase 1 I/Os: " << phase_1_ios << std::endl;
        std::cout << "Phase 2 I/Os: " << phase_2_ios << std::endl;
        std::cout << "Phase transition at: " << phase_transition_point << " candidates" << std::endl;
        std::cout << "Static cache hits: " << static_cache_hits << std::endl;
        std::cout << "Dynamic cache hits: " << dynamic_cache_hits << std::endl;
        std::cout << "Cache misses: " << cache_misses << std::endl;
        std::cout << "Total hops: " << total_hops << std::endl;
        std::cout << "Prefetch operations: " << prefetch_operations << std::endl;
        std::cout << "Prefetch pages total: " << prefetch_pages_total << std::endl;
        std::cout << "Prefetch useful pages: " << prefetch_useful_pages << std::endl;
        
        if (prefetch_pages_total > 0) {
            float prefetch_efficiency = 100.0f * prefetch_useful_pages / prefetch_pages_total;
            std::cout << "Prefetch efficiency: " << prefetch_efficiency << "%" << std::endl;
        }
        
        unsigned total_cache_accesses = static_cache_hits + dynamic_cache_hits + cache_misses;
        if (total_cache_accesses > 0) {
            float cache_hit_rate = 100.0f * (static_cache_hits + dynamic_cache_hits) / total_cache_accesses;
            std::cout << "Overall cache hit rate: " << cache_hit_rate << "%" << std::endl;
        }
        
        std::cout << "Phase detection time: " << phase_detection_us << " us" << std::endl;
        std::cout << "Cache lookup time: " << cache_lookup_us << " us" << std::endl;
        std::cout << "Prefetch time: " << prefetch_us << " us" << std::endl;
        std::cout << "================================\n" << std::endl;
    }
    
    void print_per_hop_stats(unsigned n = MAX_HOP_STATS) const {
        unsigned limit = std::min(n, std::min(total_hops, MAX_HOP_STATS));
        std::cout << "\n=== Per-Hop Cache Hit Stats (first " << limit << " hops) ===" << std::endl;
        for (unsigned i = 0; i < limit; i++) {
            const char* status_str = (per_hop_cache_status[i] == 1) ? "STATIC" :
                                     (per_hop_cache_status[i] == 2) ? "DYNAMIC" : "MISS";
            std::cout << "Hop " << i << ": " << status_str << std::endl;
        }
    }
    
    static std::vector<float> aggregate_per_hop_hit_rate(
            const HybridCacheSearchStats* stats_array, 
            size_t num_queries,
            unsigned max_hops = MAX_HOP_STATS) {
        
        std::vector<unsigned> hit_count(max_hops, 0);   
        std::vector<unsigned> total_count(max_hops, 0); 
        
        for (size_t q = 0; q < num_queries; q++) {
            const auto& s = stats_array[q];
            unsigned limit = std::min(s.total_hops, max_hops);
            for (unsigned hop = 0; hop < limit; hop++) {
                total_count[hop]++;
                if (s.per_hop_cache_status[hop] == 1 || s.per_hop_cache_status[hop] == 2) {
                    hit_count[hop]++;
                }
            }
        }
        
        std::vector<float> hit_rate(max_hops, 0.0f);
        for (unsigned hop = 0; hop < max_hops; hop++) {
            if (total_count[hop] > 0) {
                hit_rate[hop] = 100.0f * hit_count[hop] / total_count[hop];
            }
        }
        return hit_rate;
    }
    

    static void print_aggregated_per_hop_stats(
            const HybridCacheSearchStats* stats_array,
            size_t num_queries,
            unsigned max_hops = MAX_HOP_STATS) {
        
        auto hit_rate = aggregate_per_hop_hit_rate(stats_array, num_queries, max_hops);
        
        unsigned valid_hops = 0;
        for (size_t q = 0; q < num_queries; q++) {
            valid_hops = std::max(valid_hops, std::min(stats_array[q].total_hops, max_hops));
        }
        
        std::cout << "\n=== Aggregated Per-Hop Cache Hit Rate (" << num_queries << " queries) ===" << std::endl;
        std::cout << "Hop\tHitRate(%)" << std::endl;
        for (unsigned hop = 0; hop < valid_hops; hop++) {
            std::cout << hop << "\t" << hit_rate[hop] << std::endl;
        }
        std::cout << "================================\n" << std::endl;
    }
    
    static void export_per_hop_stats_to_file(
            const HybridCacheSearchStats* stats_array,
            size_t num_queries,
            const std::string& filename,
            unsigned max_hops = MAX_HOP_STATS) {
        
        auto hit_rate = aggregate_per_hop_hit_rate(stats_array, num_queries, max_hops);
        
        // Find the maximum valid hop count
        unsigned valid_hops = 0;
        for (size_t q = 0; q < num_queries; q++) {
            valid_hops = std::max(valid_hops, std::min(stats_array[q].total_hops, max_hops));
        }
        
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        ofs << "hop,hit_rate" << std::endl;
        for (unsigned hop = 0; hop < valid_hops; hop++) {
            ofs << hop << "," << hit_rate[hop] << std::endl;
        }
        ofs.close();
        std::cout << "Per-hop stats exported to: " << filename << std::endl;
    }
};

class PhaseDetector {
private:
    float theta_;
    unsigned l_search_;
    unsigned candidates_checked_;
    bool is_phase_2_;
    
    float prev_best_dist_;
    unsigned stagnation_count_;
    const unsigned stagnation_threshold_ = 3;  
    const float improvement_threshold_ = 0.01f;  
    
public:
    PhaseDetector(float theta, unsigned l_search) 
        : theta_(theta), l_search_(l_search), 
          candidates_checked_(0), is_phase_2_(false),
          prev_best_dist_(std::numeric_limits<float>::max()),
          stagnation_count_(0) {}
    
    bool update_and_check(unsigned checked_count, float current_best_dist) {
        candidates_checked_ = checked_count;
        
        bool threshold_reached = (candidates_checked_ >= static_cast<unsigned>(l_search_ * theta_));
        
        bool stagnated = false;
        if (prev_best_dist_ < std::numeric_limits<float>::max()) {
            float improvement = (prev_best_dist_ - current_best_dist) / (prev_best_dist_ + 1e-10f);
            if (improvement < improvement_threshold_) {
                stagnation_count_++;
            } else {
                stagnation_count_ = 0;
            }
            stagnated = (stagnation_count_ >= stagnation_threshold_);
        }
        prev_best_dist_ = current_best_dist;
        
        if (!is_phase_2_ && (threshold_reached || stagnated)) {
            is_phase_2_ = true;
        }
        
        return is_phase_2_;
    }
    
    bool is_phase_2() const { return is_phase_2_; }
    
    unsigned get_transition_point() const { 
        return is_phase_2_ ? candidates_checked_ : 0; 
    }
    
    void reset() {
        candidates_checked_ = 0;
        is_phase_2_ = false;
        prev_best_dist_ = std::numeric_limits<float>::max();
        stagnation_count_ = 0;
    }
};

class SimilarityAwarePrefetcher {
private:
    unsigned prefetch_pages_;
    unsigned total_pages_;
    
public:
    SimilarityAwarePrefetcher(unsigned prefetch_pages, unsigned total_pages)
        : prefetch_pages_(prefetch_pages), total_pages_(total_pages) {}
    
    std::vector<unsigned> compute_prefetch_range(unsigned target_page) const {
        std::vector<unsigned> pages_to_fetch;
        
        if (prefetch_pages_ == 0) {
            pages_to_fetch.push_back(target_page);
            return pages_to_fetch;
        }
    
        int half_window = prefetch_pages_ / 2;
        int start_page = static_cast<int>(target_page) - half_window;
        int end_page = start_page + prefetch_pages_;
        
        if (start_page < 0) {
            start_page = 0;
            end_page = std::min(static_cast<int>(prefetch_pages_), static_cast<int>(total_pages_));
        }
        if (end_page > static_cast<int>(total_pages_)) {
            end_page = total_pages_;
            start_page = std::max(0, end_page - static_cast<int>(prefetch_pages_));
        }
        
        for (int page = start_page; page < end_page; page++) {
            pages_to_fetch.push_back(static_cast<unsigned>(page));
        }
        
        return pages_to_fetch;
    }
    
    std::vector<unsigned> compute_prefetch_range_optimized(
        unsigned target_page,
        const tsl::robin_set<unsigned>& already_fetched) const {
        
        auto all_pages = compute_prefetch_range(target_page);
        std::vector<unsigned> new_pages;
        
        for (unsigned page : all_pages) {
            if (already_fetched.find(page) == already_fetched.end()) {
                new_pages.push_back(page);
            }
        }
        
        return new_pages;
    }
    
    void set_prefetch_pages(unsigned n) { prefetch_pages_ = n; }
    unsigned get_prefetch_pages() const { return prefetch_pages_; }
};

} // namespace diskann
