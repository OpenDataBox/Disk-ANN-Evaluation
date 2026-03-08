#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <list>
#include "logger.h"
#include "percentile_stats.h"
#include "deco_index.h"
#include "timer.h"
#include "dynamic_cache_search.h"

namespace diskann {

  // DynamicCache (In-Memory First) search implementation - specifically for Starling layout
  template<typename T>
  void DecoIndex<T>::dynamic_cache_search(
      const T *query_ptr, const _u64 query_num, const _u64 query_aligned_dim, 
      const _u64 k_search, const _u32 mem_L,
      const _u64 l_search, std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
      const _u32 io_limit, const float emb_search_ratio,
      const DynamicCacheSearchConfig& dc_config,
      QueryStats* stats_ptr, DynamicCacheSearchStats* dc_stats_ptr) {
    
    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this->data_dim - 1 : this->data_dim;

    _u64 sector_len = GR_SECTOR_LEN;  // Use configured sector length
    std::vector<std::unique_ptr<FIFOCache>> caches(this->max_nthreads);
    for (size_t i = 0; i < caches.size(); i++) {
      caches[i] = std::make_unique<FIFOCache>(
          dc_config.cache_capacity,
          sector_len
      );
    }

    std::atomic_int cur_task = 0;
    
    // Process queries in parallel
    pool->runTask([&, this](int tid) {
      // Thread-local data
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
      
      tsl::robin_set<_u32> &visited = *(query_scratch->visited);
      tsl::robin_set<_u32> &page_visited = *(query_scratch->page_visited);
      
      // DynamicCache-specific data structures
      auto& cache = caches[tid];
      
      // Track pages loaded into cache
      tsl::robin_set<_u32> loaded_pages;
      loaded_pages.reserve(2048);
      
      // This allows direct access to these nodes during candidate selection without waiting for I/O
      tsl::robin_set<unsigned> in_memory_nodes;
      in_memory_nodes.reserve(4096);
      
      // New: record metadata for each page (node that triggered I/O)
      tsl::robin_map<unsigned, PageMeta> page_meta;
      page_meta.reserve(2048);
      
      // Asynchronous I/O request list
      std::list<std::shared_ptr<AsyncIORequest>> pending_ios;
      tsl::robin_map<char*, std::shared_ptr<AsyncIORequest>> buffer_to_request;
      
      // Auxiliary buffers
      std::vector<char*> tmp_bufs(MAX_N_SECTOR_READS);
      std::vector<AlignedRead> frontier_read_reqs;
      std::vector<unsigned> nbr_buf(max_degree);

      while(true) {
        size_t task_id = cur_task++;
        if (task_id >= query_num) {
          break;
        }
        
        Timer query_timer, tmp_timer;

        const T* query1 = query_ptr + (task_id * query_aligned_dim);
        _u64* indices = indices_vec.data() + (task_id * k_search);
        float* distances = distances_vec.data() + (task_id * k_search);
        QueryStats* stats = stats_ptr + task_id;
        DynamicCacheSearchStats* dc_stats = dc_stats_ptr ? (dc_stats_ptr + task_id) : nullptr;

        _mm_prefetch((char *) query1, _MM_HINT_T1);
        
        // Copy and normalize query
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
        in_memory_nodes.clear();  // Clear in-memory node tracking
        page_meta.clear();        // Clear page metadata
        pending_ios.clear();
        buffer_to_request.clear();
        cache->clear();
        if (dc_stats) dc_stats->reset();

        // Compute PQ distance table
        pq_table.populate_chunk_distances(query_float, pq_dists);

        // retset: candidate list (using PQ distances)
        std::vector<Neighbor> retset(dc_config.search_list_size + 1);
        unsigned cur_list_size = 0;
        
        // full_retset: result list (using exact distances)
        std::vector<Neighbor> full_retset;
        full_retset.reserve(4096);

        // Lambda function: compute PQ distances
        auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](
            const unsigned *ids, const _u64 n_ids, float *dists_out) {
          search_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                            pq_coord_scratch);
          search_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                          dists_out);
        };

        // Lambda: compute exact distances and push to result set
        auto compute_exact_dists_and_push = [&](const char* node_buf, const unsigned id) -> float {
          float exact_dist = dist_cmp->compare(query, (T*)node_buf,
                                                (unsigned) aligned_dim);
          if (stats != nullptr) {
            stats->n_ext_cmps++;
          }
          full_retset.push_back(Neighbor(id, exact_dist, true));
          return exact_dist;
        };

        // Lambda: add to retset candidate set
        auto add_to_retset = [&](const unsigned nbor_id, const float nbor_dist) -> unsigned {
          if (cur_list_size == dc_config.search_list_size && 
              nbor_dist >= retset[cur_list_size - 1].distance) {
            return INF;
          }
          Neighbor nn(nbor_id, nbor_dist, true);  // flag=true means not expanded
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < dc_config.search_list_size) {
            ++cur_list_size;
          }
          return r;
        };

        // ===== Initialization: start from entry point =====
        tmp_timer.reset();
        if (mem_L) {
          // Use in-memory navigation graph
          std::vector<unsigned> mem_tags(mem_L);
          std::vector<float> mem_dists(mem_L);
          std::vector<T*> res = std::vector<T*>();
          mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), 
                                       mem_dists.data(), nullptr, res);
          
          for (unsigned i = 0; i < std::min((unsigned)mem_L, (unsigned)dc_config.search_list_size); i++) {
            retset[cur_list_size].id = mem_tags[i];
            retset[cur_list_size].distance = mem_dists[i];
            retset[cur_list_size].flag = true;
            cur_list_size++;
            visited.insert(mem_tags[i]);
          }
        } else {
          // Fix: start from medoid, but ensure using PQ distance for initialization
          compute_pq_dists(&medoids[0], 1, dist_scratch);
          retset[0].id = medoids[0];
          retset[0].distance = dist_scratch[0];
          retset[0].flag = true;
          cur_list_size = 1;
          visited.insert(medoids[0]);
        }
        
        std::sort(retset.begin(), retset.begin() + cur_list_size);
        
        if (stats != nullptr) {
          stats->preprocess_us += (double) tmp_timer.elapsed();
        }

        unsigned num_ios = 0;

        // ===== DynamicCache main loop =====
        unsigned iteration = 0;
        // hop_count = number of disk node expansions (excluding in-memory navigation graph nodes)
        // Each disk node expansion counts as one hop
        unsigned hop_count = 0;
        
        while (num_ios < io_limit && cur_list_size > 0) {
          iteration++;
          
          // ===== Step A: Prefetching =====
          tmp_timer.reset();
          unsigned prefetch_count = 0;
          frontier_read_reqs.clear();
          
          // [Synchronous mode] Must wait for all pending I/O to complete before next prefetch round
          // This validates the correctness of cache hit rate statistics
          while (!pending_ios.empty()) {
            int n_ready = io_manager->get_events(ctx, 1, pending_ios.size(), tmp_bufs);
            for (int i = 0; i < n_ready; i++) {
              auto buf = tmp_bufs[i];
              auto req_it = buffer_to_request.find(buf);
              if (req_it != buffer_to_request.end()) {
                auto req = req_it->second;
                req->status = IOStatus::COMPLETED;
                
                // Put data into cache
                cache->put(req->page_id, buf);
                // Mark this page as loaded
                loaded_pages.insert(req->page_id);
                
                // Parse page layout, insert all node IDs in this page into in_memory_nodes
                if (!use_graph_rep_index_ && req->page_id < gp_layout_.size()) {
                  for (unsigned node_id : gp_layout_[req->page_id]) {
                    in_memory_nodes.insert(node_id);
                  }
                } else {
                  in_memory_nodes.insert(req->node_id);
                }
                
                // Remove corresponding request from pending_ios
                pending_ios.remove_if([&](const auto& r) { return r->page_id == req->page_id; });
                buffer_to_request.erase(req_it);
              }
            }
          }
          
          for (unsigned i = 0; i < cur_list_size && prefetch_count < dc_config.prefetch_window; i++) {
            if (!retset[i].flag) continue;  // Already expanded, skip
            
            unsigned node_id = retset[i].id;
            unsigned mem_pos = node_in_mem_pos(node_id);
            
            // Only perform I/O prefetch for nodes not in memory
            if (mem_pos == INF) {
              unsigned page_id;
                if (node_id >= id2page_.size()) {
                  diskann::cerr << "Error: node_id " << node_id << " >= id2page_.size() " << id2page_.size() << std::endl;
                  continue;
                }
                page_id = id2page_[node_id];
              
              // Check if page is already in cache - use actual cache instead of loaded_pages
              if (cache->contains(page_id)) {
                loaded_pages.insert(page_id); // Update tracking for this query
                continue;
              }
              
              // Check if there's a pending I/O
              bool already_pending = false;
              for (auto& req : pending_ios) {
                if (req->page_id == page_id) {
                  already_pending = true;
                  break;
                }
              }
              
              if (!already_pending && page_visited.insert(page_id).second) {
                // Prevent buffer overwrite
                auto next_sector_buf = sector_scratch + sector_scratch_idx * sector_len;
                if (buffer_to_request.find(next_sector_buf) != buffer_to_request.end()) {
                    break;
                }

                char* sector_buf = next_sector_buf;
                sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                
                auto offset = (static_cast<_u64>(page_id + 1)) * sector_len;
                
                auto io_req = std::make_shared<AsyncIORequest>(node_id, page_id, sector_buf);
                pending_ios.push_back(io_req);
                buffer_to_request[sector_buf] = io_req;
                
                // Record which node triggered the I/O for this page (used later to determine hit/miss)
                page_meta[page_id] = PageMeta(node_id);
                
                frontier_read_reqs.push_back(AlignedRead(offset, sector_len, sector_buf));
                
                prefetch_count++;
                num_ios++;
                if (stats != nullptr) stats->n_ios++;
                if (dc_stats) dc_stats->async_ios++;
              }
            }
          }
          
          if (frontier_read_reqs.size() > 0) {
              io_manager->submit_read_reqs(frontier_read_reqs, index_fid, ctx);
            
            if (stats != nullptr) stats->n_hops++;
          }
          
          if (dc_stats) {
            dc_stats->prefetch_us += tmp_timer.elapsed();
          }

          // ===== Step B: Candidate selection ("In-Memory First" core logic) =====
          tmp_timer.reset();
          unsigned selected_node = INF;
          unsigned selected_idx = INF;
          
          // First find the first unvisited disk node in retset (nearest unvisited disk node)
          // This node is used to determine if it's "one hop"
          unsigned nearest_unvisited_disk_node = INF;
          for (unsigned i = 0; i < cur_list_size; i++) {
            if (!retset[i].flag) continue;
            unsigned node_id = retset[i].id;
            unsigned mem_pos = node_in_mem_pos(node_id);
            if (mem_pos == INF) {
              // This is the first unvisited disk node
              nearest_unvisited_disk_node = node_id;
              break;
            }
          }
          
          // First round: look for nodes in memory, nodes in in-memory navigation graph
          for (unsigned i = 0; i < cur_list_size; i++) {
            if (!retset[i].flag) continue;
            
            unsigned node_id = retset[i].id;
            unsigned mem_pos = node_in_mem_pos(node_id);
            
            if (mem_pos != INF) {
              // Node is in memory, select directly
              selected_node = node_id;
              selected_idx = i;
              break;
            }
          }
          
          // Second round: if no memory node, look for cached nodes (tracked using in_memory_nodes), dynamic cache
          if (selected_node == INF) {
            for (unsigned i = 0; i < cur_list_size; i++) {
              if (!retset[i].flag) continue;
              
              unsigned node_id = retset[i].id;
              unsigned mem_pos = node_in_mem_pos(node_id);
              
              if (mem_pos == INF) {
                // This is a node tracked from loaded pages
                if (in_memory_nodes.find(node_id) != in_memory_nodes.end()) {
                  selected_node = node_id;
                  selected_idx = i;
                  break;
                }
              
                unsigned page_id;
                
                  if (node_id >= id2page_.size()) {
                    if (dc_stats) dc_stats->candidates_skipped++;
                    continue;  // Skip invalid nodes
                  }
                  page_id = id2page_[node_id];
                
                if (cache->contains(page_id)) {
                  loaded_pages.insert(page_id); // Update tracking for this query
                  // Add all nodes in the page to in_memory_nodes
                  if (!use_graph_rep_index_ && page_id < gp_layout_.size()) {
                    for (unsigned nid : gp_layout_[page_id]) {
                      in_memory_nodes.insert(nid);
                    }
                  }
                  selected_node = node_id;
                  selected_idx = i;
                  break;
                }
              }
              if (dc_stats) dc_stats->candidates_skipped++;
            }
          }
          
          if (dc_stats) {
            dc_stats->candidate_selection_us += tmp_timer.elapsed();
          }

          // ===== Step C: Handle "CPU starvation" =====
          // Paper logic: Wait -> Expand, i.e., immediately expand with the data that just returned after waiting ends
          if (selected_node == INF) {
            if (!pending_ios.empty()) {
              tmp_timer.reset();
              
              // Block waiting for I/O completion
              int n_completed = io_manager->get_events(ctx, 1, pending_ios.size(), tmp_bufs);
              
              // Record the first completed request for immediate expansion
              std::shared_ptr<AsyncIORequest> first_completed_req = nullptr;
              char* first_completed_buf = nullptr;
              
              for (int i = 0; i < n_completed; i++) {
                auto buf = tmp_bufs[i];
                auto req_it = buffer_to_request.find(buf);
                if (req_it != buffer_to_request.end()) {
                  auto req = req_it->second;
                  req->status = IOStatus::COMPLETED;
                  
                  // Add page to cache
                  cache->put(req->page_id, buf);
                  loaded_pages.insert(req->page_id);
                  
                  // Add all nodes in the page to in_memory_nodes
                  if (!use_graph_rep_index_ && req->page_id < gp_layout_.size()) {
                    for (unsigned node_id : gp_layout_[req->page_id]) {
                      in_memory_nodes.insert(node_id);
                    }
                  } else {
                    in_memory_nodes.insert(req->node_id);
                  }
                  
                  // Record the first completed request for immediate expansion
                  if (first_completed_req == nullptr) {
                    first_completed_req = req;
                    first_completed_buf = buf;
                  }
                  
                  // Remove from pending list
                  pending_ios.remove_if([&](const auto& r) { return r->page_id == req->page_id; });
                  buffer_to_request.erase(req_it);
                }
              }
              
              if (dc_stats) {
                dc_stats->blocked_waits++;
                dc_stats->io_wait_us += tmp_timer.elapsed();
              }
              if (stats != nullptr) {
                stats->read_disk_us += tmp_timer.elapsed();
              }
              
              // Paper logic: Wait -> Expand
              // Immediately select the node that just returned for expansion, instead of continuing back to loop start
              if (first_completed_req != nullptr) {
                // Find this node in retset and select it
                unsigned target_node = first_completed_req->node_id;
                for (unsigned i = 0; i < cur_list_size; i++) {
                  if (retset[i].id == target_node && retset[i].flag) {
                    selected_node = target_node;
                    selected_idx = i;
                    break;
                  }
                }
                
                // If the original request's node is not in retset, try to find other candidates in the same page
                if (selected_node == INF && !use_graph_rep_index_ && 
                    first_completed_req->page_id < gp_layout_.size()) {
                  for (unsigned nid : gp_layout_[first_completed_req->page_id]) {
                    for (unsigned i = 0; i < cur_list_size; i++) {
                      if (retset[i].id == nid && retset[i].flag) {
                        selected_node = nid;
                        selected_idx = i;
                        break;
                      }
                    }
                    if (selected_node != INF) break;
                  }
                }
                
              }
              
              // If still no expandable node found, continue to next round
              if (selected_node == INF) {
                continue;
              }
              // Otherwise continue to step D for expansion
            } else {
              // No pending I/O, check if there are still unexpanded candidates
              bool has_unexpanded = false;
              for (unsigned i = 0; i < cur_list_size; i++) {
                if (retset[i].flag) {
                  has_unexpanded = true;
                  break;
                }
              }
              
              if (!has_unexpanded) {
                break; // Truly no expandable nodes, end
              } else {
                // Have unexpanded candidates but no pending I/O, need to issue new I/O
                continue;
              }
            }
          }

          // ===== Step D: Expand selected node =====
          tmp_timer.reset();
          
          retset[selected_idx].flag = false;
          if (dc_stats) dc_stats->candidates_expanded++;
          
          unsigned mem_pos = node_in_mem_pos(selected_node);
          
          if (mem_pos != INF) {
            // Node is in memory (in-memory navigation graph), doesn't count toward hit/miss statistics
            if (dc_stats) dc_stats->in_memory_expansions++;
            char* emb_buf = get_mem_emb_addr(selected_node);
            if (emb_buf != nullptr) {
              compute_exact_dists_and_push(emb_buf, selected_node);
            }
            
            // Process neighbors in memory
            auto& neighbors = mem_graph_[mem_pos];
            unsigned nbors_size = 0;
            for (unsigned nbr_id : neighbors) {
              if (visited.insert(nbr_id).second) {
                nbr_buf[nbors_size++] = nbr_id;
              }
            }
            if (nbors_size > 0) {
              compute_pq_dists(nbr_buf.data(), nbors_size, dist_scratch);
              if (stats != nullptr) stats->n_cmps += (double) nbors_size;
              
              for (unsigned m = 0; m < nbors_size; ++m) {
                add_to_retset(nbr_buf[m], dist_scratch[m]);
              }
            }
          } else {
            // Node is on disk, get from cache
            unsigned page_id;
            if (use_graph_rep_index_) {
              page_id = selected_node;
            } else {
              if (selected_node >= id2page_.size()) {
                diskann::cerr << "Error: selected_node " << selected_node << " >= id2page_.size() " << id2page_.size() << std::endl;
                continue;
              }
              page_id = id2page_[selected_node];
            }
            char* page_data = cache->get(page_id);
            
            if (page_data == nullptr) {
                // Exception: page not in cache
                diskann::cerr << "Warning: Page " << page_id << " not in cache for node " << selected_node << std::endl;
                continue;
            }
            
            // ===== New hit/miss statistics logic =====
            // Only when expanding the "nearest unvisited disk node" does it count as one hop and count toward cache hit/miss
            // nearest_unvisited_disk_node was found before candidate selection
            if (dc_stats) {
              // Determine if this is the nearest unvisited disk node
              bool is_nearest_unvisited = (selected_node == nearest_unvisited_disk_node);
              
              if (is_nearest_unvisited) {
                // This is one hop: expanding the nearest unvisited disk node
                auto meta_it = page_meta.find(page_id);
                if (meta_it != page_meta.end()) {
                  unsigned trigger_node = meta_it->second.trigger_node;
                  bool is_hit = (selected_node != trigger_node);
                  
                  // Current hop count
                  unsigned current_hop = hop_count;
                  
                  if (is_hit) {
                    dc_stats->cache_hits++;
                  } else {
                    dc_stats->cache_misses++;
                  }
                  
                  // Record per-hop statistics
                  if (current_hop < MAX_HOP_STATS) {
                    dc_stats->record_hop_cache(current_hop, is_hit);
                  }
                }
                
                // Expanding the nearest unvisited disk node counts as one hop
                hop_count++;
              }
            }
            // ===== hit/miss statistics end =====
            
            char* target_node_buf = nullptr;
            
            if (use_graph_rep_index_) {
              // Graph-replicated layout: node is directly at page start position
              target_node_buf = page_data;
            } else {
              // Fix: Starling layout node lookup, learning from page_search.cpp approach
              bool found = false;
              for (unsigned j = 0; j < gp_layout_[page_id].size(); j++) {
                if (gp_layout_[page_id][j] == selected_node) {
                  target_node_buf = page_data + j * max_node_len;
                  found = true;
                  break;
                }
              }
              
              if (!found) {
                diskann::cerr << "Warning: Node " << selected_node 
                             << " not found in page " << page_id << " layout" << std::endl;
                continue;
              }
            }
            
            // Fix: learning from page_search.cpp page processing approach
            // 1. First process target node's exact distance and neighbors
            compute_exact_dists_and_push(target_node_buf, selected_node);
            
            // Process target node's neighbors
            char *graph_buf = target_node_buf + emb_node_len;
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
            
          }
          
          if (dc_stats) {
            dc_stats->expansion_us += tmp_timer.elapsed();
          }
          
          // Re-sort retset to ensure best candidates are at the front
          std::sort(retset.begin(), retset.begin() + cur_list_size);
          
          // Check termination condition
          bool has_unexpanded = false;
          for (unsigned i = 0; i < cur_list_size; i++) {
            if (retset[i].flag) {
              has_unexpanded = true;
              break;
            }
          }
          
          // Only terminate when there are no unexpanded candidates and no pending I/O
          if (!has_unexpanded && pending_ios.empty()) {
            break;
          }
        }
        // ===== End of Loop =====

        // Clean up remaining I/O
        while (!pending_ios.empty()) {
          // Fix: use unified I/O completion approach
          int n_completed = io_manager->get_events(ctx, 1, pending_ios.size(), tmp_bufs);
          for (int i = 0; i < n_completed; i++) {
            auto buf = tmp_bufs[i];
            auto req_it = buffer_to_request.find(buf);
            if (req_it != buffer_to_request.end()) {
              auto req = req_it->second;
              cache->put(req->page_id, buf);
              loaded_pages.insert(req->page_id);
              // Add all nodes in the page to in_memory_nodes
              if (!use_graph_rep_index_ && req->page_id < gp_layout_.size()) {
                for (unsigned node_id : gp_layout_[req->page_id]) {
                  in_memory_nodes.insert(node_id);
                }
              }
              pending_ios.remove_if([&](const auto& r) { return r->page_id == req->page_id; });
              buffer_to_request.erase(req_it);
            }
          }
        }

        // Re-sort and cleanup
        tmp_timer.reset();
        visited.clear();
        page_visited.clear();
        loaded_pages.clear();
        in_memory_nodes.clear();
        page_meta.clear();
        buffer_to_request.clear();

        std::sort(full_retset.begin(), full_retset.end(),
                  [](const Neighbor &left, const Neighbor &right) {
                    return left.distance < right.distance;
                  });

        _u64 t = 0;
        for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
          if (i > 0 && full_retset[i].id == full_retset[i - 1].id) continue;
          indices[t] = full_retset[i].id;
          if (distances != nullptr) {
            distances[t] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT) {
              distances[t] = (-distances[t]);
              if (max_base_norm != 0)
                distances[t] *= (max_base_norm * query_norm);
            }
          }
          t++;
        }
        
        // Fill insufficient results
        if (t < k_search) {
             for (_u64 i = t; i < k_search; i++) {
                indices[i] = (t > 0) ? indices[t-1] : 0;
                if (distances) distances[i] = (t > 0) ? distances[t-1] : 0;
             }
        }

        if (stats != nullptr) {
          stats->total_us = (double) query_timer.elapsed();
          stats->postprocess_us += tmp_timer.elapsed();
        }
        if (dc_stats) {
          dc_stats->total_ios = num_ios;
          dc_stats->total_hops = hop_count;  // Record actual hop count
        }
      }
    });
  }

  template class DecoIndex<_u8>;
  template class DecoIndex<_s8>;
  template class DecoIndex<float>;
  
} // namespace diskann