#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "logger.h"
#include "percentile_stats.h"
#include "deco_index.h"
#include "timer.h"
#include "baseline_io_stats.h"

namespace diskann {

 
  struct CacheNode {
    unsigned id;
    unsigned nb_size;
    unsigned* node_nbrs;
    bool from_disk; 

    CacheNode(unsigned id, unsigned nb_size, unsigned* node_nbrs, bool from_disk) {
      this->id = id;
      this->nb_size = nb_size;
      this->node_nbrs = node_nbrs;  
      this->from_disk = from_disk;
    }
  };

  template<typename T>
  void DecoIndex<T>::page_search_dup_graph(
      const T *query_ptr, const _u64 query_num, const _u64 query_aligned_dim, const _u64 k_search, const _u32 mem_L,
      const _u64 l_search, std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
      const _u64 beam_width, const _u32 io_limit,
      const float pq_filter_ratio, const float emb_search_ratio, QueryStats* stats_ptr) {
    
    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this-> data_dim - 1: this-> data_dim;

    const char* baseline_dir_env = std::getenv("BASELINE_DIR");
    BaselineIOStats* baseline_stats = nullptr;
    
    if (baseline_dir_env != nullptr && strlen(baseline_dir_env) > 0) {
      std::string baseline_dir(baseline_dir_env);
      diskann::cout << "Loading baseline data from: " << baseline_dir << std::endl;
      baseline_stats = new BaselineIOStats(baseline_dir, 16);
      if (!baseline_stats->is_loaded()) {
        diskann::cerr << "Failed to load baseline data!" << std::endl;
        delete baseline_stats;
        baseline_stats = nullptr;
      } else {
        diskann::cout << "Baseline loaded: " << baseline_stats->get_query_count() 
                      << " queries" << std::endl;
      }
    }

    std::atomic_int cur_task = 0;
    
    pool->runTask([&, this](int tid) {
      IOContext& ctx = ctxs[tid];
      auto scratch = scratchs[tid];
      auto query_scratch = &(scratchs[tid]);
      
      const T * query = scratch.aligned_query_T;
      const float *query_float = scratch.aligned_query_float;
      
      _u64 &sector_scratch_idx = query_scratch->sector_idx;
      char *sector_scratch = query_scratch->sector_scratch;
      float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
      
      float *dist_scratch = query_scratch->aligned_dist_scratch;
      _u8 * pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;
      
      tsl::robin_set<_u32> &page_visited = *(query_scratch->page_visited);
      tsl::robin_set<_u32> &visited = *(query_scratch->visited);
      tsl::robin_map<_u32, char*> loaded; 
      loaded.reserve(2048);

      CircleQueue<char*> sector_buffers(MAX_N_SECTOR_READS);
      std::vector<char*> tmp_bufs(MAX_N_SECTOR_READS);
      std::vector<AlignedRead> frontier_read_reqs(MAX_N_SECTOR_READS);
      std::vector<unsigned> nbr_buf(max_degree);
      std::vector<std::pair<unsigned, char*>> cached_id_bufs(MAX_N_SECTOR_READS);
      CircleQueue<std::shared_ptr<CacheNode>> cached_node(MAX_N_SECTOR_READS);

      while(true) {
        size_t task_id = cur_task++;
        if (task_id >= query_num) {
          break;
        }
        Timer query_timer, tmp_timer, part_timer;

        std::vector<unsigned> accessed_exact_nodes_vec;
        tsl::robin_set<unsigned> accessed_exact_nodes_set;
        std::vector<unsigned> accessed_adj_nodes_vec;
        tsl::robin_set<unsigned> accessed_adj_nodes_set;
        size_t num_pages_read = 0;

        std::vector<std::shared_ptr<FrontierNode>> frontier;
        size_t ftr_id = 0; 

        const T* query1 = query_ptr + (task_id * query_aligned_dim);
        _u64* indices = indices_vec.data() + (task_id * k_search);
        float* distances = distances_vec.data() + (task_id * k_search);
        QueryStats* stats = stats_ptr + task_id;

        _mm_prefetch((char *) query1, _MM_HINT_T1);
        
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
        
        query_scratch->reset();
        loaded.clear();

        pq_table.populate_chunk_distances(query_float, pq_dists);

        std::vector<Neighbor> retset(l_search + 1);
        unsigned cur_list_size = 0;
        std::vector<Neighbor> full_retset;
        full_retset.reserve(4096);
        
        auto compute_pq_dists = [this, pq_coord_scratch, pq_dists, stats](const unsigned *ids,
                                                                  const _u64 n_ids,
                                                                  float *dists_out) {
          auto pq_start = std::chrono::high_resolution_clock::now();
          search_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                            pq_coord_scratch);
          search_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                          dists_out);
          auto pq_end = std::chrono::high_resolution_clock::now();
          if (stats != nullptr) {
            stats->pq_distance_time_us += std::chrono::duration<double, std::micro>(pq_end - pq_start).count();
          }
        };

        auto compute_exact_dists_and_push = [&](const char* node_buf, const unsigned id) -> float {
          tmp_timer.reset();
          auto exact_start = std::chrono::high_resolution_clock::now();
          float cur_expanded_dist = dist_cmp->compare(query, (T*)node_buf,
                                          (unsigned) aligned_dim);
          auto exact_end = std::chrono::high_resolution_clock::now();
          if (stats != nullptr) {
            stats->n_ext_cmps++;
            stats->exact_distance_time_us += std::chrono::duration<double, std::micro>(exact_end - exact_start).count();
          }
          full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
          
          if (baseline_stats != nullptr && accessed_exact_nodes_set.insert(id).second) {
            accessed_exact_nodes_vec.push_back(id);
          }
          return cur_expanded_dist;
        };

        auto add_to_retset = [&](const int nbor_id, const float nbor_dist, const bool flag) -> unsigned {
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) {
            return INF;
          }
          Neighbor nn(nbor_id, nbor_dist, flag);
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search) ++cur_list_size;
          return r;
        };

        auto compute_and_push_nbrs = [&](const char *node_buf) {
          unsigned *node_nbrs = (unsigned*)node_buf;
          unsigned nnbrs = *(node_nbrs++);
          unsigned nbors_cand_size = 0;
          for (unsigned m = 0; m < nnbrs; ++m) {
            if (visited.insert(node_nbrs[m]).second) {
              nbr_buf[nbors_cand_size++] = node_nbrs[m];
            }
          }
          if (nbors_cand_size) {
            _mm_prefetch((char *) nbr_buf.data(), _MM_HINT_T1);
            compute_pq_dists(nbr_buf.data(), nbors_cand_size, dist_scratch);
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_cand_size;
            }
            for (unsigned m = 0; m < nbors_cand_size; ++m) {
              const unsigned nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              add_to_retset(nbor_id, nbor_dist, true);
            }
          }
        };

        auto compute_and_push_nbrs_target_update = [&](const char *node_buf, const unsigned node_id) {
          unsigned *node_nbrs = (unsigned*)node_buf;
          unsigned nnbrs = *(node_nbrs++);
          unsigned nbors_cand_size = 0;
          for (unsigned m = 0; m < nnbrs; ++m) {
            if (visited.insert(node_nbrs[m]).second) {
              nbr_buf[nbors_cand_size++] = node_nbrs[m];
            }
          }

          if (nbors_cand_size) {
            _mm_prefetch((char *) nbr_buf.data(), _MM_HINT_T1);
            compute_pq_dists(nbr_buf.data(), nbors_cand_size, dist_scratch);
            
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_cand_size;
            }
            std::vector<unsigned> expand_nb_ids;
            for (unsigned m = 0; m < nbors_cand_size; ++m) {
              const unsigned nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              auto r = add_to_retset(nbor_id, nbor_dist, true);
              
              if (dist_scratch[m] < retset[cur_list_size - 1].distance * pq_filter_ratio) {
                 if (loaded.find(nbor_id) != loaded.end()) {
                    expand_nb_ids.push_back(nbor_id);
                    if (r < cur_list_size) {
                      retset[r].flag = false;
                    }
                 }
              }
            }
            for (unsigned m = 0; m < expand_nb_ids.size(); m++) {
              compute_and_push_nbrs(loaded[expand_nb_ids[m]]);
            }
          }
        };

        auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
          compute_pq_dists(node_ids, n_ids, dist_scratch);
          for (_u64 i = 0; i < n_ids; ++i) {
            retset[cur_list_size].id = node_ids[i];
            retset[cur_list_size].distance = dist_scratch[i];
            retset[cur_list_size++].flag = true;
            visited.insert(node_ids[i]);
          }
        };

        part_timer.reset();
        
        if (mem_L) {
          std::vector<unsigned> mem_tags(mem_L);
          std::vector<float> mem_dists(mem_L);
          std::vector<T*> res = std::vector<T*>();
          mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data(), nullptr, res);
          compute_and_add_to_retset(mem_tags.data(), std::min((unsigned)mem_L, (unsigned)l_search));
        } else {
          // Use first medoid as entry point
          compute_and_add_to_retset(&medoids[0], 1);
        }
        
        std::sort(retset.begin(), retset.begin() + cur_list_size);

        if (stats != nullptr) {
          stats->preprocess_us += (double) part_timer.elapsed();
        }
        unsigned num_ios = 0;

        tsl::robin_map<char*, std::shared_ptr<FrontierNode>> sec_buf2ftr;

        _u32 n_io_in_q = 0; 
        _u32 n_cached_in_q = 0; 
        _u32 n_proc_in_q = 0;
        
        // Per-hop cache hit tracking
        size_t current_hop = 0;
        size_t hop_cache_hits = 0;     
        size_t hop_total_accesses = 0; 

        while (num_ios < io_limit) {
          if (n_proc_in_q > 0) {
            part_timer.reset();
            auto sector_buf = sector_buffers.get();
            if (sec_buf2ftr.find(sector_buf) == sec_buf2ftr.end()) {
              std::cout << "(bug) read error!" << std::endl;
              exit(-1);
            }
            
            auto fn = sec_buf2ftr[sector_buf];
            const _u32 exact_id = fn->id;
            
            // 1. Calculate Exact Distance
            compute_exact_dists_and_push(sector_buf, exact_id);
            
            char *node_buf = sector_buf + emb_node_len + sizeof(unsigned) * (1 + n_gc_node_per_sector);
            unsigned *p_layout = (unsigned*)(sector_buf + emb_node_len);
            unsigned p_size = *(p_layout++);

            if (baseline_stats != nullptr) {
                if (accessed_adj_nodes_set.insert(exact_id).second) {
                    accessed_adj_nodes_vec.push_back(exact_id);
                }
              
                for (unsigned j = 1; j < p_size; j++) {
                     if (node_in_mem_pos(p_layout[j]) == INF) {
                         
                         if (accessed_adj_nodes_set.insert(p_layout[j]).second) {
                             accessed_adj_nodes_vec.push_back(p_layout[j]);
                         }
                     }
                }
            }

            // 2. Load neighbors into loaded map
            for (unsigned j = 1; j < p_size; j++) {
              if (node_in_mem_pos(p_layout[j]) == INF) {
                char *nnbr_buf = node_buf + j * graph_node_len;
                loaded.insert({p_layout[j], nnbr_buf});
              }
            }
            // 3. Expand neighbors for target node.
            compute_and_push_nbrs_target_update(node_buf, exact_id);
            if (stats != nullptr) stats->disk_proc_us += (double) part_timer.elapsed();

            sec_buf2ftr.erase(sector_buf);
            n_proc_in_q--;
            num_pages_read++;
          }

          // calculate in memory node.
          while (n_cached_in_q > 0) {
            part_timer.reset();
            auto cn = cached_node.get();

          

            unsigned nbors_size = 0;
            for (unsigned m = 0; m < cn->nb_size; ++m) {
              if (visited.insert(cn->node_nbrs[m]).second) {
                nbr_buf[nbors_size++] = cn->node_nbrs[m];
              }
            }
            compute_pq_dists(nbr_buf.data(), nbors_size, dist_scratch);
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_size;
            }
            for (unsigned m = 0; m < nbors_size; ++m) {
              const unsigned nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              add_to_retset(nbor_id, nbor_dist, true);
            }
            n_cached_in_q--;
            if (stats != nullptr) stats->cache_proc_us += (double) part_timer.elapsed();
          }

          if (n_io_in_q > 0) {
            unsigned min_r = 0;
            if (n_proc_in_q == 0) min_r = 1;
            part_timer.reset();
            int n_read_blks = io_manager->get_events(ctx, min_r, n_io_in_q, tmp_bufs);
            for (int i = n_read_blks - 1; i >= 0; i--) {
              auto fn = sec_buf2ftr[tmp_bufs[i]];
              sector_buffers.push(tmp_bufs[i]);
            }
            if (stats != nullptr) stats->read_disk_us += (double) part_timer.elapsed();
            n_io_in_q -= n_read_blks;
            n_proc_in_q += n_read_blks;
          }
          
          _u32 disk_batch_size = beam_width;
          if (n_io_in_q == 0 && n_cached_in_q == 0 && n_proc_in_q <= beam_width / 2) {
            part_timer.reset();
            frontier_read_reqs.clear();
            _u32 marker = 0;
            _u32 num_seen = 0;
            _u32 disk_seen = 0;
            
            hop_cache_hits = 0;
            hop_total_accesses = 0;

            while (marker < cur_list_size && num_seen < beam_width && disk_seen < disk_batch_size) {
              if (retset[marker].flag) {
                auto id = retset[marker].id;
                unsigned mem_pos = node_in_mem_pos(id);
                if (mem_pos != INF) {
                  cached_node.push(std::make_shared<CacheNode>(id, mem_graph_[mem_pos].size(), mem_graph_[mem_pos].data(), false));
                  if (disk_batch_size > 0)
                    disk_seen++;
                  else
                    num_seen++;
                  n_cached_in_q++;
                  hop_cache_hits++;      
                  hop_total_accesses++;  
                } else if (loaded.find(id) != loaded.end()) {
  
                  unsigned* node_nbrs = (unsigned*)loaded[id];
                  unsigned nb_size = *(node_nbrs++);
                  cached_node.push(std::make_shared<CacheNode>(id, nb_size, node_nbrs, true));
                  num_seen++;
                  n_cached_in_q++;
                  hop_total_accesses++;
                } else {
                  if (page_visited.insert(id).second) {
                    num_seen++;
                    auto fn = std::make_shared<FrontierNode>(id, id, gc_index_fid);
                    frontier.push_back(fn);
                    hop_total_accesses++;  
                  }
                }
                retset[marker].flag = false;
              }
              marker++;
            }
            if (stats != nullptr) stats->dispatch_us += (double) part_timer.elapsed();
            
            if (stats != nullptr && current_hop < QueryStats::MAX_HOP_STATS && hop_total_accesses > 0) {
              stats->per_hop_cache_hits[current_hop] = hop_cache_hits;
              stats->per_hop_total_accesses[current_hop] = hop_total_accesses;
              stats->actual_hops = current_hop + 1;
            }
            current_hop++;

            if (ftr_id < frontier.size()) {
              part_timer.reset();
              if (stats != nullptr) stats->n_hops++;
              n_io_in_q += frontier.size() - ftr_id;
              while(ftr_id < frontier.size()) {
                auto sector_buf = sector_scratch + sector_scratch_idx * GR_SECTOR_LEN;
                sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                auto offset = (static_cast<_u64>(frontier[ftr_id]->pid)) * GR_SECTOR_LEN;
                offset += GR_SECTOR_LEN; 
                sec_buf2ftr.insert({sector_buf, frontier[ftr_id]});
                frontier_read_reqs.push_back(AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
                if (stats != nullptr) {
                  stats->n_ios++;
                }
                num_ios++;
                ftr_id++;
              }
              io_manager->submit_read_reqs(frontier_read_reqs, gc_index_fid, ctx);
              if (stats != nullptr) stats->read_disk_us += (double) part_timer.elapsed();
            }
            if (n_io_in_q == 0 && n_proc_in_q == 0 && n_cached_in_q == 0) break;
          }
        }
        
        part_timer.reset();
        frontier.clear();

        _u32 l_idx = 0;
        _u32 embedding_search_L = (_u32)(cur_list_size * emb_search_ratio);
        if (embedding_search_L < k_search) embedding_search_L = k_search;
        while (l_idx < embedding_search_L) {
          frontier_read_reqs.clear();
          cached_id_bufs.clear();
          tsl::robin_map<char*, unsigned> sec_buf2pid;
          for (_u32 ord_idx = l_idx; l_idx - ord_idx < MAX_N_SECTOR_READS && l_idx < embedding_search_L; l_idx++) {
            auto pid = retset[l_idx].id;
            if (page_visited.find(pid) == page_visited.end()) {
              char* cached_emb_buf = get_mem_emb_addr(retset[l_idx].id);
              if (cached_emb_buf != nullptr) {
                cached_id_bufs.push_back(std::make_pair(retset[l_idx].id, cached_emb_buf));
              } else {
                auto sector_buf = sector_scratch + sector_scratch_idx * GR_SECTOR_LEN;
                sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                auto offset = (static_cast<_u64>(pid + 1)) * GR_SECTOR_LEN; 
                frontier_read_reqs.push_back(AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
                page_visited.insert(pid);
                sec_buf2pid.insert({sector_buf, pid});
              }
            }
          }
          int n_ops = 0;
          if (frontier_read_reqs.size() != 0) {
            n_ops = io_manager->submit_read_reqs(frontier_read_reqs, gc_index_fid, ctx);
            if (stats != nullptr) {
              stats-> n_emb_ios += n_ops;
            }
          }
          if (cached_id_bufs.size() != 0) {
            for (_u64 i = 0; i < cached_id_bufs.size(); i++) {
              _mm_prefetch((char *) cached_id_bufs[i].second, _MM_HINT_T0);
              compute_exact_dists_and_push(cached_id_bufs[i].second, cached_id_bufs[i].first);
            }
          }
          while (n_ops > 0) {
            int n_read_blks = io_manager->get_events(ctx, 1, n_ops, tmp_bufs);
            n_ops -= n_read_blks;
            for (int i = 0; i < n_read_blks; i++) {
              auto sector_buf = tmp_bufs[i];
              auto pid = sec_buf2pid[sector_buf];
              
              compute_exact_dists_and_push(sector_buf, pid);
              num_pages_read++; 
            }
          }
        }

        frontier_read_reqs.clear();
        visited.clear();
        page_visited.clear();
        
        std::sort(full_retset.begin(), full_retset.end(),
                  [](const Neighbor &left, const Neighbor &right) {
                    return left.distance < right.distance;
                  });

        _u64 t = 0;
        for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
          if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
            continue;
          }
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

        if (t < k_search) {
          diskann::cerr << "The number of unique ids is less than topk" << std::endl;
          exit(1);
        }

        if (baseline_stats != nullptr && baseline_stats->is_loaded() && stats != nullptr) {
          size_t exact_vector_bytes = this->disk_bytes_per_point;
          size_t adj_list_bytes = this->max_degree * sizeof(unsigned);
          
          auto disk_util = baseline_stats->calculate_disk_utilization(
              accessed_exact_nodes_vec, accessed_adj_nodes_vec, task_id, num_pages_read, 
              exact_vector_bytes, adj_list_bytes);
          
          stats->baseline_total_disk_bytes = disk_util.total_disk_bytes;
          stats->baseline_hit_disk_bytes = disk_util.hit_disk_bytes;
          
          if (disk_util.total_disk_bytes > 0) {
            stats->io_utilization = static_cast<float>(disk_util.hit_disk_bytes) / 
                                    static_cast<float>(disk_util.total_disk_bytes);
          } else {
            stats->io_utilization = 0.0f;
          }
          
          stats->useful_io_nodes = accessed_exact_nodes_vec.size() + accessed_adj_nodes_vec.size();
          stats->total_io_nodes = stats->useful_io_nodes;
        }

        if (stats != nullptr) {
          stats->total_us = (double) query_timer.elapsed();
          stats->postprocess_us = (double) part_timer.elapsed();
        }
      }
    });

    if (baseline_stats != nullptr) {
      delete baseline_stats;
    }
  }

  template class DecoIndex<_u8>;
  template class DecoIndex<_s8>;
  template class DecoIndex<float>;
} // namespace diskann