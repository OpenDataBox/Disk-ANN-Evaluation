//
// HybridCache Cache Manager: Hybrid Static-Dynamic Caching with LFU
//
#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <queue>
#include <list>
#include <cstring>
#include "utils.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

namespace diskann {

struct LFUCacheNode {
    unsigned page_id;
    char* data;
    unsigned frequency;
    
    LFUCacheNode(unsigned pid, char* d, unsigned freq = 1) 
        : page_id(pid), data(d), frequency(freq) {}
};

class LFUCache {
private:
    size_t capacity_;
    size_t page_size_;
    
    // page_id -> {data, frequency, iterator}
    tsl::robin_map<unsigned, std::shared_ptr<LFUCacheNode>> cache_map_;
    
    // frequency -> list of page_ids with that frequency
    std::map<unsigned, std::list<unsigned>> freq_lists_;
    
    // page_id -> iterator in freq_lists_
    tsl::robin_map<unsigned, std::list<unsigned>::iterator> page_to_iter_;
    
    unsigned min_freq_;
    size_t current_size_;
    
public:
    LFUCache(size_t capacity, size_t page_size) 
        : capacity_(capacity), page_size_(page_size), 
          min_freq_(0), current_size_(0) {}
    
    ~LFUCache() {
        for (auto& pair : cache_map_) {
            if (pair.second && pair.second->data) {
                aligned_free(pair.second->data);
            }
        }
    }
    
    char* get(unsigned page_id) {
        auto it = cache_map_.find(page_id);
        if (it == cache_map_.end()) {
            return nullptr;
        }
        
        auto node = it->second;
        update_frequency(page_id, node->frequency);
        
        return node->data;
    }
    
    void put(unsigned page_id, const char* data) {
        if (capacity_ == 0) return;
        
        auto it = cache_map_.find(page_id);
        if (it != cache_map_.end()) {
            auto node = it->second;
            std::memcpy(node->data, data, page_size_);
            update_frequency(page_id, node->frequency);
            return;
        }
        
        if (current_size_ >= capacity_) {
            evict();
        }
        
        char* new_data = nullptr;
        alloc_aligned((void**)&new_data, page_size_, page_size_);
        std::memcpy(new_data, data, page_size_);
        
        auto node = std::make_shared<LFUCacheNode>(page_id, new_data, 1);
        cache_map_[page_id] = node;
        
        freq_lists_[1].push_back(page_id);
        page_to_iter_[page_id] = --freq_lists_[1].end();
        
        min_freq_ = 1;
        current_size_++;
    }
    
    void put_batch(const std::vector<unsigned>& page_ids, 
                   const std::vector<char*>& data_ptrs) {
        for (size_t i = 0; i < page_ids.size() && i < data_ptrs.size(); i++) {
            put(page_ids[i], data_ptrs[i]);
        }
    }
    
    bool contains(unsigned page_id) const {
        return cache_map_.find(page_id) != cache_map_.end();
    }
    
    size_t size() const { return current_size_; }
    
    void clear() {
        for (auto& pair : cache_map_) {
            if (pair.second && pair.second->data) {
                aligned_free(pair.second->data);
            }
        }
        cache_map_.clear();
        freq_lists_.clear();
        page_to_iter_.clear();
        current_size_ = 0;
        min_freq_ = 0;
    }
    
private:
    void update_frequency(unsigned page_id, unsigned old_freq) {
        auto iter_it = page_to_iter_.find(page_id);
        if (iter_it != page_to_iter_.end()) {
            freq_lists_[old_freq].erase(iter_it->second);
            if (freq_lists_[old_freq].empty()) {
                freq_lists_.erase(old_freq);
                if (min_freq_ == old_freq) {
                    min_freq_++;
                }
            }
        }
        
        unsigned new_freq = old_freq + 1;
        freq_lists_[new_freq].push_back(page_id);
        page_to_iter_[page_id] = --freq_lists_[new_freq].end();
        
        cache_map_[page_id]->frequency = new_freq;
    }
    
    void evict() {
        if (freq_lists_.empty()) return;
        
        auto& min_list = freq_lists_[min_freq_];
        if (min_list.empty()) return;
        
        unsigned evict_page = min_list.front();
        min_list.pop_front();
        
        if (min_list.empty()) {
            freq_lists_.erase(min_freq_);
        }
        
        // Release memory
        auto it = cache_map_.find(evict_page);
        if (it != cache_map_.end() && it->second && it->second->data) {
            aligned_free(it->second->data);
        }
        
        cache_map_.erase(evict_page);
        page_to_iter_.erase(evict_page);
        current_size_--;
    }
};

class HybridCacheManager {
private:
    tsl::robin_map<unsigned, char*> static_cache_;
    size_t static_cache_size_;
    
    std::unique_ptr<LFUCache> dynamic_cache_;
    
    size_t page_size_;
    bool phase_2_active_;
    
public:
    HybridCacheManager(size_t static_capacity, size_t dynamic_capacity, size_t page_size)
        : static_cache_size_(0), page_size_(page_size), phase_2_active_(false) {
        
        dynamic_cache_ = std::make_unique<LFUCache>(dynamic_capacity, page_size);
    }
    
    ~HybridCacheManager() {
        for (auto& pair : static_cache_) {
            if (pair.second) {
                aligned_free(pair.second);
            }
        }
    }
    
    void init_static_cache(const std::vector<unsigned>& entry_nodes,
                          const std::vector<char*>& entry_data) {
        for (size_t i = 0; i < entry_nodes.size() && i < entry_data.size(); i++) {
            char* cached_data = nullptr;
            alloc_aligned((void**)&cached_data, page_size_, page_size_);
            std::memcpy(cached_data, entry_data[i], page_size_);
            
            static_cache_[entry_nodes[i]] = cached_data;
            static_cache_size_++;
        }
    }
    
    char* get(unsigned page_id) {
        auto static_it = static_cache_.find(page_id);
        if (static_it != static_cache_.end()) {
            return static_it->second;
        }
        
        if (phase_2_active_) {
            return dynamic_cache_->get(page_id);
        }
        
        return nullptr;
    }
    
    void put_dynamic(unsigned page_id, const char* data) {
        if (phase_2_active_) {
            dynamic_cache_->put(page_id, data);
        }
    }
    
    void put_dynamic_batch(const std::vector<unsigned>& page_ids,
                          const std::vector<char*>& data_ptrs) {
        if (phase_2_active_) {
            dynamic_cache_->put_batch(page_ids, data_ptrs);
        }
    }
    
    void set_phase_2(bool active) {
        phase_2_active_ = active;
    }
    
    bool is_phase_2() const {
        return phase_2_active_;
    }
    
    bool contains(unsigned page_id) const {
        if (static_cache_.find(page_id) != static_cache_.end()) {
            return true;
        }
        if (phase_2_active_) {
            return dynamic_cache_->contains(page_id);
        }
        return false;
    }
    
    size_t get_static_size() const { return static_cache_size_; }
    size_t get_dynamic_size() const { return dynamic_cache_->size(); }
    
    void clear_dynamic() {
        dynamic_cache_->clear();
        phase_2_active_ = false;
        for (auto& pair : prefetch_buffers_) {
            if (pair.second) {
                aligned_free(pair.second);
            }
        }
        prefetch_buffers_.clear();
    }
    
    char* allocate_for_prefetch(unsigned page_id) {
        if (contains(page_id)) {
            return nullptr;
        }
        if (prefetch_buffers_.find(page_id) != prefetch_buffers_.end()) {
            return nullptr;
        }
        
        char* buffer = nullptr;
        alloc_aligned((void**)&buffer, page_size_, page_size_);
        if (buffer) {
            prefetch_buffers_[page_id] = buffer;
        }
        return buffer;
    }
    
    void commit_prefetch(unsigned page_id) {
        auto it = prefetch_buffers_.find(page_id);
        if (it != prefetch_buffers_.end() && it->second) {
            dynamic_cache_->put(page_id, it->second);
            aligned_free(it->second);
            prefetch_buffers_.erase(it);
        }
    }
    

    char* get_prefetch_buffer(unsigned page_id) {
        auto it = prefetch_buffers_.find(page_id);
        if (it != prefetch_buffers_.end()) {
            return it->second;
        }
        return nullptr;
    }

private:
    tsl::robin_map<unsigned, char*> prefetch_buffers_;
};

} // namespace diskann
