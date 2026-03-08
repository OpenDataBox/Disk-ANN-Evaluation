#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <list>
#include "logger.h"
#include "percentile_stats.h"
#include "deco_index.h"
#include "timer.h"
#include "hybridcache_cache.h"
#include "hybridcache_search.h"
#include "dynamic_cache_search.h"

namespace diskann {

  // HybridCache Search: Two-phase search based on Starling layout
  // Phase 1: DynamicCache-style asynchronous prefetch beam search
  // Phase 2: Prefetch adjacent pages, synchronously wait for I/O completion before expanding nodes
  template<typename T>
  void DecoIndex<T>::hybridcache_search(
      const T *query_ptr, const _u64 query_num, const _u64 query_aligned_dim, 
      const _u64 k_search, const _u32 mem_L,
      const _u64 l_search, std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
      const _u64 beam_width, const _u32 io_limit,
      const float pq_filter_ratio, const float emb_search_ratio, 
      const HybridCacheSearchConfig& hc_config,
      QueryStats* stats_ptr, HybridCacheSearchStats* hc_stats_ptr) {
    
    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this->data_dim - 1 : this->data_dim;
    _u64 sector_len = GR_SECTOR_LEN;

    // One FIFO cache per thread
    std::vector<std::unique_ptr<FIFOCache>> caches(this->max_nthreads);
    for (size_t i = 0; i < caches.size(); i++) {
      caches[i] = std::make_unique<FIFOCache>(
          hc_config.dynamic_cache_capacity,
          sector_len
      );
    }

    std::atomic_int cur_task = 0;
    
    pool->runTask([&, this](int tid) {
      IOContext& ctx = ctxs[tid];
      auto scratch = scratchs[tid];
      auto query_scratch = &(scratchs[tid]);
      
      const T *query = scratch.aligned_query_T;
      const float *query_float = scratch.aligned_query_float;
      
      _u64 &sector_scratch_idx = query_scratch->sector_idx;
      char *sector_scratch = query_scratch->sector_scratch;
      float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
      float *dist_scratch = query_scratch->aligned_dist_scratch;
      _u8 *pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;
      
      tsl::robin_set<_u32> &page_visited = *(query_scratch->page_visited);
      tsl::robin_set<_u32> &visited = *(query_scratch->visited);

      auto& cache = caches[tid];
      
      // Track pages loaded into cache and nodes in memory
      tsl::robin_set<_u32> loaded_pages;
      loaded_pages.reserve(2048);
      tsl::robin_set<unsigned> in_memory_nodes;
      in_memory_nodes.reserve(4096);
      
      // Asynchronous I/O request tracking
      std::list<std::shared_ptr<AsyncIORequest>> pending_ios;
      tsl::robin_map<char*, std::shared_ptr<AsyncIORequest>> buffer_to_request;
      
      // Auxiliary buffers
      std::vector<char*> tmp_bufs(MAX_N_SECTOR_READS);
      std::vector<AlignedRead> read_reqs;
      read_reqs.reserve(MAX_N_SECTOR_READS);
      std::vector<unsigned> nbr_buf(max_degree);
      
      // Result set
      std::vector<Neighbor> full_retset;
      full_retset.reserve(4096);

      while(true) {
        size_t task_id = cur_task++;
        if (task_id >= query_num) break;
        
        Timer query_timer, tmp_timer;

        const T* query1 = query_ptr + (task_id * query_aligned_dim);
        _u64* indices = indices_vec.data() + (task_id * k_search);
        float* distances = distances_vec.data() + (task_id * k_search);
        QueryStats* stats = stats_ptr + task_id;
        HybridCacheSearchStats* hc_stats = hc_stats_ptr ? (hc_stats_ptr + task_id) : nullptr;

        _mm_prefetch((char *) query1, _MM_HINT_T1);
        
        // Query Normalization
        float query_norm = 0;
        for (_u32 i = 0; i < query_dim; i++) {
          scratch.aligned_query_float[i] = query1[i];
          scratch.aligned_query_T[i] = query1[i];
          query_norm += query1[i] * query1[i];
        }
        if (metric == diskann::Metric::INNER_PRODUCT) {
          query_norm = std::sqrt(query_norm);
          scratch.aligned_query_T[this->data_dim - 1] = 0;
          scratch.aligned_query_float[this->data_dim - 1] = 0;
          for (_u32 i = 0; i < this->data_dim - 1; i++) {
            scratch.aligned_query_T[i] /= query_norm;
            scratch.aligned_query_float[i] /= query_norm;
          }
        }
        
        // Reset state
        query_scratch->reset();
        loaded_pages.clear();
        in_memory_nodes.clear();
        pending_ios.clear();
        buffer_to_request.clear();
        cache->clear();
        full_retset.clear();
        if (hc_stats) hc_stats->reset();

        pq_table.populate_chunk_distances(query_float, pq_dists);

        // retset: Candidate list sorted by PQ distance
        std::vector<Neighbor> retset(l_search + 1);
        unsigned cur_list_size = 0;

        // Phase detector
        PhaseDetector phase_detector(hc_config.theta, l_search);
        
        // Prefetcher - Calculate total number of pages
        unsigned total_pages = (num_points + nnodes_per_sector - 1) / nnodes_per_sector;
        SimilarityAwarePrefetcher prefetcher(hc_config.prefetch_pages, total_pages);

        // ===== Lambda function definitions =====
        
        auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](
            const unsigned *ids, const _u64 n_ids, float *dists_out) {
          search_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                            pq_coord_scratch);
          search_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                          dists_out);
        };

        auto compute_exact_dists_and_push = [&](const char* node_buf, const unsigned id) -> float {
          float exact_dist = dist_cmp->compare(query, (T*)node_buf, (unsigned) aligned_dim);
          if (stats != nullptr) stats->n_ext_cmps++;
          full_retset.push_back(Neighbor(id, exact_dist, true));
          return exact_dist;
        };

        auto add_to_retset = [&](const unsigned nbor_id, const float nbor_dist) -> unsigned {
          if (cur_list_size == l_search && nbor_dist >= retset[cur_list_size - 1].distance) {
            return INF;
          }
          Neighbor nn(nbor_id, nbor_dist, true);
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search) ++cur_list_size;
          return r;
        };

        // Expand node: compute exact distance + process neighbors
        auto expand_node = [&](unsigned node_id, const char* node_buf) {
          // Compute exact distance
          compute_exact_dists_and_push(node_buf, node_id);
          
          // Process neighbors
          char *graph_buf = const_cast<char*>(node_buf) + emb_node_len;
          unsigned *node_nbrs = (unsigned*)graph_buf;
          unsigned nnbrs = *(node_nbrs++);
          unsigned nbors_size = 0;
          
          for (unsigned m = 0; m < nnbrs; ++m) {
            if (visited.insert(node_nbrs[m]).second) {
              nbr_buf[nbors_size++] = node_nbrs[m];
            }
          }
          
          if (nbors_size > 0) {
            compute_pq_dists(nbr_buf.data(), nbors_size, dist_scratch);
            if (stats != nullptr) stats->n_cmps += (double) nbors_size;
            for (unsigned m = 0; m < nbors_size; ++m) {
              add_to_retset(nbr_buf[m], dist_scratch[m]);
            }
          }
        };

        // Get node data from cache (check static cache first, then dynamic cache)
        auto get_node_from_cache = [&](unsigned node_id) -> char* {
          if (node_id >= id2page_.size()) return nullptr;
          unsigned page_id = id2page_[node_id];
          
          // Check static cache first
          char* page_data = this->get_static_cache_page(page_id);
          
          // Then check dynamic cache
          if (page_data == nullptr) {
            page_data = cache->get(page_id);
          }
          
          if (page_data == nullptr) return nullptr;
          
          for (unsigned j = 0; j < gp_layout_[page_id].size(); j++) {
            if (gp_layout_[page_id][j] == node_id) {
              return page_data + j * max_node_len;
            }
          }
          return nullptr;
        };
        
        // Check if node is in static cache
        auto is_node_in_static_cache = [&](unsigned node_id) -> bool {
          return this->is_in_static_cache(node_id);
        };

        // Process completed I/O and put data into cache
        auto process_completed_ios = [&](int min_events) -> int {
          if (pending_ios.empty()) return 0;
          
          int n_completed = io_manager->get_events(ctx, min_events, pending_ios.size(), tmp_bufs);
          
          for (int i = 0; i < n_completed; i++) {
            auto buf = tmp_bufs[i];
            auto req_it = buffer_to_request.find(buf);
            if (req_it != buffer_to_request.end()) {
              auto req = req_it->second;
              req->status = IOStatus::COMPLETED;
              
              cache->put(req->page_id, buf);
              loaded_pages.insert(req->page_id);
              
              // Add all nodes in the page to in_memory_nodes
              if (req->page_id < gp_layout_.size()) {
                for (unsigned nid : gp_layout_[req->page_id]) {
                  in_memory_nodes.insert(nid);
                }
              }
              
              pending_ios.remove_if([&](const auto& r) { return r->page_id == req->page_id; });
              buffer_to_request.erase(req_it);
            }
          }
          return n_completed;
        };

        // Submit page read request
        auto submit_page_read = [&](unsigned page_id, unsigned node_id) -> bool {
          if (page_visited.find(page_id) != page_visited.end()) return false;
          if (cache->contains(page_id)) return false;
          
          // Check if there's already a pending I/O
          for (auto& req : pending_ios) {
            if (req->page_id == page_id) return false;
          }
          
          // Check if buffer is available
          auto next_buf = sector_scratch + sector_scratch_idx * sector_len;
          if (buffer_to_request.find(next_buf) != buffer_to_request.end()) return false;
          
          char* sector_buf = next_buf;
          sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
          
          auto offset = (static_cast<_u64>(page_id + 1)) * sector_len;
          auto io_req = std::make_shared<AsyncIORequest>(node_id, page_id, sector_buf);
          pending_ios.push_back(io_req);
          buffer_to_request[sector_buf] = io_req;
          read_reqs.push_back(AlignedRead(offset, sector_len, sector_buf));
          
          page_visited.insert(page_id);
          return true;
        };

        // ===== Initialize candidate set =====
        tmp_timer.reset();
        if (mem_L) {
          std::vector<unsigned> mem_tags(mem_L);
          std::vector<float> mem_dists(mem_L);
          std::vector<T*> res = std::vector<T*>();
          mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data(), nullptr, res);
          
          for (unsigned i = 0; i < std::min((unsigned)mem_L, (unsigned)l_search); i++) {
            retset[cur_list_size].id = mem_tags[i];
            retset[cur_list_size].distance = mem_dists[i];
            retset[cur_list_size].flag = true;
            cur_list_size++;
            visited.insert(mem_tags[i]);
          }
        } else {
          compute_pq_dists(&medoids[0], 1, dist_scratch);
          retset[0].id = medoids[0];
          retset[0].distance = dist_scratch[0];
          retset[0].flag = true;
          cur_list_size = 1;
          visited.insert(medoids[0]);
          
          // Record medoid as Hop 0 - check if it's in static cache
          if (hc_stats) {
            unsigned medoid_id = medoids[0];
            unsigned mem_pos = node_in_mem_pos(medoid_id);
            if (mem_pos != INF || is_node_in_static_cache(medoid_id)) {
              hc_stats->record_hop(1);  // Static cache hit
              hc_stats->static_cache_hits++;
            } else {
              hc_stats->record_hop(0);  // Cache miss
              hc_stats->cache_misses++;
            }
          }
        }
        std::sort(retset.begin(), retset.begin() + cur_list_size);
        if (stats != nullptr) stats->preprocess_us += (double) tmp_timer.elapsed();

        unsigned num_ios = 0;
        bool is_phase_2 = false;


        // ===== Main search loop =====
        while (num_ios < io_limit && cur_list_size > 0) {
          
          // Phase detection
          float current_best_dist = cur_list_size > 0 ? retset[0].distance : std::numeric_limits<float>::max();
          bool was_phase_1 = !is_phase_2;
          is_phase_2 = phase_detector.update_and_check(visited.size(), current_best_dist);
          
          if (was_phase_1 && is_phase_2 && hc_stats) {
            hc_stats->phase_transition_point = visited.size();
          }

          // ===== Phase 1: dynamic-style asynchronous prefetch =====
          if (!is_phase_2) {
            // First check completed I/O (non-blocking)
            process_completed_ios(0);
            
            // Prefetch: Submit I/O for the first beam_width unvisited nodes in retset
            tmp_timer.reset();
            read_reqs.clear();
            unsigned prefetch_count = 0;
            
            for (unsigned i = 0; i < cur_list_size && prefetch_count < beam_width; i++) {
              if (!retset[i].flag) continue;
              
              unsigned node_id = retset[i].id;
              unsigned mem_pos = node_in_mem_pos(node_id);
              
              if (mem_pos == INF && node_id < id2page_.size()) {
                unsigned page_id = id2page_[node_id];
                if (submit_page_read(page_id, node_id)) {
                  prefetch_count++;
                  num_ios++;
                  if (stats != nullptr) stats->n_ios++;
                  if (hc_stats) hc_stats->phase_1_ios++;
                }
              }
            }
            
            if (read_reqs.size() > 0) {
              io_manager->submit_read_reqs(read_reqs, index_fid, ctx);
              if (stats != nullptr) stats->n_hops++;
            }
            if (stats != nullptr) stats->dispatch_us += tmp_timer.elapsed();
            
            // Candidate selection: Prioritize nodes in memory/cache
            unsigned selected_node = INF;
            unsigned selected_idx = INF;
            bool from_static_cache = false;
            
            // First round: Memory navigation graph nodes
            for (unsigned i = 0; i < cur_list_size; i++) {
              if (!retset[i].flag) continue;
              unsigned node_id = retset[i].id;
              unsigned mem_pos = node_in_mem_pos(node_id);
              if (mem_pos != INF) {
                selected_node = node_id;
                selected_idx = i;
                break;
              }
            }
            
            // Second round: Nodes in static cache
            if (selected_node == INF) {
              for (unsigned i = 0; i < cur_list_size; i++) {
                if (!retset[i].flag) continue;
                unsigned node_id = retset[i].id;
                if (is_node_in_static_cache(node_id)) {
                  selected_node = node_id;
                  selected_idx = i;
                  from_static_cache = true;
                  if (hc_stats) hc_stats->static_cache_hits++;
                  break;
                }
              }
            }
            
            // Third round: Nodes in dynamic cache
            if (selected_node == INF) {
              for (unsigned i = 0; i < cur_list_size; i++) {
                if (!retset[i].flag) continue;
                unsigned node_id = retset[i].id;
                if (in_memory_nodes.find(node_id) != in_memory_nodes.end()) {
                  selected_node = node_id;
                  selected_idx = i;
                  if (hc_stats) hc_stats->dynamic_cache_hits++;
                  break;
                }
                // Check dynamic cache
                if (node_id < id2page_.size()) {
                  unsigned page_id = id2page_[node_id];
                  if (cache->contains(page_id)) {
                    // Add all nodes in the page to in_memory_nodes
                    for (unsigned nid : gp_layout_[page_id]) {
                      in_memory_nodes.insert(nid);
                    }
                    selected_node = node_id;
                    selected_idx = i;
                    if (hc_stats) hc_stats->dynamic_cache_hits++;
                    break;
                  }
                }
              }
            }
            
            // If no available node, wait for I/O (cache miss)
            bool waited_for_io = false;
            if (selected_node == INF && !pending_ios.empty()) {
              tmp_timer.reset();
              int n_completed = process_completed_ios(1);
              if (stats != nullptr) stats->read_disk_us += tmp_timer.elapsed();
              if (hc_stats) hc_stats->cache_misses += n_completed;
              waited_for_io = true;
              
              // Select the just-completed node
              for (unsigned i = 0; i < cur_list_size; i++) {
                if (!retset[i].flag) continue;
                unsigned node_id = retset[i].id;
                if (in_memory_nodes.find(node_id) != in_memory_nodes.end()) {
                  selected_node = node_id;
                  selected_idx = i;
                  break;
                }
              }
            }
            
            // Expand the selected node
            if (selected_node != INF) {
              retset[selected_idx].flag = false;
              
              unsigned mem_pos = node_in_mem_pos(selected_node);
              if (mem_pos != INF) {
                // Memory navigation graph node
                if (hc_stats) {
                  hc_stats->record_hop(1);  // 1 = static/mem cache hit
                }
                char* emb_buf = get_mem_emb_addr(selected_node);
                if (emb_buf != nullptr) {
                  compute_exact_dists_and_push(emb_buf, selected_node);
                }
                auto& neighbors = mem_graph_[mem_pos];
                unsigned nbors_size = 0;
                for (unsigned nbr_id : neighbors) {
                  if (visited.insert(nbr_id).second) {
                    nbr_buf[nbors_size++] = nbr_id;
                  }
                }
                if (nbors_size > 0) {
                  compute_pq_dists(nbr_buf.data(), nbors_size, dist_scratch);
                  if (stats != nullptr) stats->n_cmps += nbors_size;
                  for (unsigned m = 0; m < nbors_size; ++m) {
                    add_to_retset(nbr_buf[m], dist_scratch[m]);
                  }
                }
              } else if (from_static_cache) {
                // Static cache hit
                if (hc_stats) {
                  hc_stats->record_hop(1);  // 1 = static cache hit
                }
                char* node_buf = get_node_from_cache(selected_node);
                if (node_buf != nullptr) {
                  expand_node(selected_node, node_buf);
                }
              } else {
                // Dynamic cache or disk read
                if (hc_stats) {
                  unsigned cache_status = waited_for_io ? 0 : 2;
                  hc_stats->record_hop(cache_status);
                }
                char* node_buf = get_node_from_cache(selected_node);
                if (node_buf != nullptr) {
                  expand_node(selected_node, node_buf);
                }
              }
              
              std::sort(retset.begin(), retset.begin() + cur_list_size);
            }
            
            // Check termination condition
            bool has_unexpanded = false;
            for (unsigned i = 0; i < cur_list_size; i++) {
              if (retset[i].flag) { has_unexpanded = true; break; }
            }
            if (!has_unexpanded && pending_ios.empty()) break;
          }
          // ===== Phase 2: Prefetch adjacent pages, synchronous wait =====
          else {
            // Select the nearest unvisited node in retset
            unsigned selected_node = INF;
            unsigned selected_idx = INF;
            
            for (unsigned i = 0; i < cur_list_size; i++) {
              if (!retset[i].flag) continue;
              selected_node = retset[i].id;
              selected_idx = i;
              break;
            }
            
            if (selected_node == INF) break;  // All nodes have been visited
            
            retset[selected_idx].flag = false;
            
            unsigned mem_pos = node_in_mem_pos(selected_node);
            if (mem_pos != INF) {
              // Memory navigation graph node
              if (hc_stats) {
                hc_stats->record_hop(1);  // 1 = static cache hit
              }
              char* emb_buf = get_mem_emb_addr(selected_node);
              if (emb_buf != nullptr) {
                compute_exact_dists_and_push(emb_buf, selected_node);
              }
              auto& neighbors = mem_graph_[mem_pos];
              unsigned nbors_size = 0;
              for (unsigned nbr_id : neighbors) {
                if (visited.insert(nbr_id).second) {
                  nbr_buf[nbors_size++] = nbr_id;
                }
              }
              if (nbors_size > 0) {
                compute_pq_dists(nbr_buf.data(), nbors_size, dist_scratch);
                if (stats != nullptr) stats->n_cmps += nbors_size;
                for (unsigned m = 0; m < nbors_size; ++m) {
                  add_to_retset(nbr_buf[m], dist_scratch[m]);
                }
              }
              std::sort(retset.begin(), retset.begin() + cur_list_size);
              continue;
            }
            
            // Check if node is in static cache
            if (is_node_in_static_cache(selected_node)) {
              // Static cache hit
              if (hc_stats) {
                hc_stats->record_hop(1);  // 1 = static cache hit
                hc_stats->static_cache_hits++;
              }
              char* node_buf = get_node_from_cache(selected_node);
              if (node_buf != nullptr) {
                expand_node(selected_node, node_buf);
              }
              std::sort(retset.begin(), retset.begin() + cur_list_size);
              continue;
            }
            
            // Check if node is in dynamic cache
            if (in_memory_nodes.find(selected_node) != in_memory_nodes.end() ||
                (selected_node < id2page_.size() && cache->contains(id2page_[selected_node]))) {
              // Dynamic cache hit
              if (hc_stats) {
                hc_stats->record_hop(2);  // 2 = dynamic cache hit
                hc_stats->dynamic_cache_hits++;
              }
              if (selected_node < id2page_.size()) {
                unsigned page_id = id2page_[selected_node];
                for (unsigned nid : gp_layout_[page_id]) {
                  in_memory_nodes.insert(nid);
                }
              }
              char* node_buf = get_node_from_cache(selected_node);
              if (node_buf != nullptr) {
                expand_node(selected_node, node_buf);
              }
              std::sort(retset.begin(), retset.begin() + cur_list_size);
              continue;
            }
            
            // Need to read from disk, prefetch adjacent pages at the same time
            if (selected_node >= id2page_.size()) continue;
            unsigned target_page = id2page_[selected_node];
            
            // Calculate prefetch range
            auto pages_to_fetch = prefetcher.compute_prefetch_range(target_page);
            
            // Submit all I/O requests
            tmp_timer.reset();
            read_reqs.clear();
            
            for (unsigned page_id : pages_to_fetch) {
              if (page_id >= gp_layout_.size() || gp_layout_[page_id].empty()) continue;
              submit_page_read(page_id, selected_node);
            }
            
            unsigned new_ios = read_reqs.size();
            if (new_ios > 0) {
              io_manager->submit_read_reqs(read_reqs, index_fid, ctx);
              num_ios += new_ios;
              if (stats != nullptr) {
                stats->n_ios += new_ios;
                stats->n_hops++;
              }
              if (hc_stats) {
                hc_stats->phase_2_ios += new_ios;
                hc_stats->prefetch_operations++;
                hc_stats->prefetch_pages_total += new_ios;
              }
            }
            if (stats != nullptr) stats->dispatch_us += tmp_timer.elapsed();
            
            // Synchronously wait for all I/O to complete
            tmp_timer.reset();
            while (!pending_ios.empty()) {
              process_completed_ios(1);
            }
            if (stats != nullptr) stats->read_disk_us += tmp_timer.elapsed();
            
            // Expand target node (cache miss, need to read from disk)
            if (hc_stats) {
              hc_stats->record_hop(0);  // 0 = cache miss
            }
            char* node_buf = get_node_from_cache(selected_node);
            if (node_buf != nullptr) {
              expand_node(selected_node, node_buf);
              if (hc_stats) hc_stats->prefetch_useful_pages++;
            }
            
            std::sort(retset.begin(), retset.begin() + cur_list_size);
          }
        }
        // ===== End of Main Loop =====

        // Clean up remaining I/O
        while (!pending_ios.empty()) {
          process_completed_ios(1);
        }


        // ===== Embedding Search Phase =====
        // Compute exact distances for the first embedding_search_L nodes in retset
        tmp_timer.reset();
        
        _u32 embedding_search_L = (_u32)(cur_list_size * emb_search_ratio);
        if (embedding_search_L < k_search) embedding_search_L = k_search;
        
        tsl::robin_set<unsigned> exact_computed;
        for (auto& n : full_retset) {
          exact_computed.insert(n.id);
        }
        
        for (_u32 i = 0; i < embedding_search_L && i < cur_list_size; i++) {
          unsigned node_id = retset[i].id;
          if (exact_computed.find(node_id) != exact_computed.end()) continue;
          
          // Try to get from cache
          char* node_buf = get_node_from_cache(node_id);
          if (node_buf != nullptr) {
            compute_exact_dists_and_push(node_buf, node_id);
            exact_computed.insert(node_id);
            if (hc_stats) hc_stats->prefetch_useful_pages++;
            continue;
          }
          
          // Try to get from memory
          char* mem_buf = get_mem_emb_addr(node_id);
          if (mem_buf != nullptr) {
            compute_exact_dists_and_push(mem_buf, node_id);
            exact_computed.insert(node_id);
            continue;
          }
          
          // Need to read from disk
          if (node_id >= id2page_.size()) continue;
          unsigned page_id = id2page_[node_id];
          if (page_visited.find(page_id) != page_visited.end()) continue;
          
          read_reqs.clear();
          if (submit_page_read(page_id, node_id)) {
            io_manager->submit_read_reqs(read_reqs, index_fid, ctx);
            if (stats != nullptr) stats->n_emb_ios++;
            
            // Wait for completion
            process_completed_ios(1);
            
            // Process all nodes in the page that need exact distances
            char* page_data = cache->get(page_id);
            if (page_data != nullptr) {
              for (unsigned j = 0; j < gp_layout_[page_id].size(); j++) {
                unsigned nid = gp_layout_[page_id][j];
                if (exact_computed.find(nid) != exact_computed.end()) continue;
                if (visited.find(nid) == visited.end()) continue;
                
                char* nbuf = page_data + j * max_node_len;
                compute_exact_dists_and_push(nbuf, nid);
                exact_computed.insert(nid);
              }
            }
          }
        }
        
        if (stats != nullptr) stats->postprocess_us = tmp_timer.elapsed();

        // Cleanup
        visited.clear();
        page_visited.clear();
        loaded_pages.clear();
        in_memory_nodes.clear();
        buffer_to_request.clear();

        // Sort results
        std::sort(full_retset.begin(), full_retset.end(),
                  [](const Neighbor &left, const Neighbor &right) {
                    return left.distance < right.distance;
                  });

        // Output top-k results
        _u64 t = 0;
        for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
          if (i > 0 && full_retset[i].id == full_retset[i - 1].id) continue;
          indices[t] = full_retset[i].id;
          if (distances != nullptr) {
            distances[t] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT) {
              distances[t] = (-distances[t]);
              if (max_base_norm != 0) distances[t] *= (max_base_norm * query_norm);
            }
          }
          t++;
        }

        if (t < k_search) {
          // Fill insufficient results
          for (_u64 i = t; i < k_search; i++) {
            indices[i] = (t > 0) ? indices[t-1] : 0;
            if (distances) distances[i] = (t > 0) ? distances[t-1] : 0;
          }
        }

        if (stats != nullptr) {
          stats->total_us = (double) query_timer.elapsed();
        }
      }
    });
  }

  template class DecoIndex<_u8>;
  template class DecoIndex<_s8>;
  template class DecoIndex<float>;
  
} // namespace diskann
