#include "aligned_file_reader.h"
#include "linux_aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "neighbor.h"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "timer.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/page_cache.h"
#include "global_stats.h"
#include <math.h>

#include <unistd.h>
#include <sys/syscall.h>

#ifndef USE_AIO
#include "liburing.h"
#endif

namespace pipeann {
  struct rerank_io_t {
    Neighbor nbr;
    unsigned page_id;
    unsigned loc;
    IORequest *read_req;
    int cb_idx;
    int rank;
    int timestamp;
    bool need_unlock;
    rerank_io_t(Neighbor nbr, unsigned int pid, unsigned int loc, IORequest *req,
      int cb_idx, int rank, int timestamp, bool need_unlock = true){
      this->nbr = nbr;
      this->page_id = pid;
      this->loc = loc;
      this->read_req = req;
      this->cb_idx = cb_idx;
      this->rank = rank;
      this->timestamp = timestamp;
      this->need_unlock = need_unlock;
    }
    bool operator>(const rerank_io_t &rhs) const {
      return rhs < *this;
    }
  
    bool operator<(const rerank_io_t &rhs) const {
      if (rank != rhs.rank) {
        return rank < rhs.rank;
      }
      return timestamp < rhs.timestamp;
    }
    bool finished() const {
      // std::cerr << "8uck\n";
      // std::cerr << (void *) read_req << "\n";
      // std::cerr << "finish is " << read_req->finished << std::endl;
      // std::cerr << "8uck you\n";
      return read_req->finished;
    }
  };

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_rerank_search(const T *query1, const _u32 mem_L, const _u64 l_search, const _u64 beam_width,
                                            std::vector<Neighbor> &expanded_nodes_info,
                                            tsl::robin_map<uint32_t, T *> *coord_map, 
                                            tsl::robin_map<uint32_t, char *> *topo_page_map, 
                                            QueryStats *stats,
                                            tsl::robin_set<uint32_t> *exclude_nodes /* tags */,
                                            QueryBuffer<T> * passthrough_data,
                                            tsl::robin_map<uint32_t, int> *passthrough_page_ref,
                                            int strategy_mask) {

    QueryBuffer<T> *query_buf = nullptr;
    if (passthrough_data == nullptr) {    
      query_buf = pop_query_buf(query1);
    } else {
      query_buf = passthrough_data;
    }
    tsl::robin_map<uint32_t, int> * page_ref = nullptr;
    if(passthrough_page_ref == nullptr){
      page_ref = new tsl::robin_map<uint32_t, int>();
    } else {
      page_ref = passthrough_page_ref;
    }
#ifdef USE_AIO
    void *ctx = reader->get_ctx();
#else
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.
#endif

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }
    // std::cerr << "0\n";

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    const T *query = query_buf->aligned_query_T;
    const T *query_2 = query_buf->aligned_query_T_2;

    // reset query
    query_buf->reset();

    // pointers to buffers for data
    T *data_buf = query_buf->coord_scratch;
    _u64 &data_buf_idx = query_buf->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;

    // query <-> neighbor list
    float *dist_scratch = query_buf->aligned_dist_scratch;
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;
    _u8 *pq_coord_scratch_2 = query_buf->aligned_pq_coord_scratch_2;
    auto p_reader = (LinuxAlignedFileReader *)this->reader.get();
    int strategy = p_reader->strategy & ~strategy_mask;
    FileHandle coord_fd = p_reader->coord_file_desc;
    FileHandle topo_fd = p_reader->topo_file_desc;
    std::unordered_map<unsigned, char *> page_buf_map;
    std::unordered_map<unsigned, int> page_rank_map;

    auto & loc2phy_topo = p_reader->loc2phy_topo;
    auto & loc2phy_coord = p_reader->loc2phy_coord;
    auto & block_cache = p_reader->block_cache;
    int use_rerank = (strategy >> 0) & 0x1;
    int use_topo_reorder = (strategy >> 1) & 0x1;
    int use_double_pq = (strategy >> 2) & 0x1;
    int use_coord_reorder = (strategy >> 3) & 0x1;
    int use_topo_buffer = (strategy >> 4) & 0x1;
    int use_nhood_cache = (strategy >> 5) & 0x1;
    int use_truncate = (strategy >> 6) & 0x1;
    int use_triple_pq = (strategy >> 7) & 0x1;
    int use_page_compute = (strategy >> 8) & 0x1;
    int rank = 0, ts = 0;
    // LOG(DEBUG) << "1";
    // std::cerr << "ntopo_per_sector: " << ntopo_per_sector << std::endl;

    Timer query_timer;
    std::vector<Neighbor> retset(mem_L + l_search * 10);
    auto &visited = *(query_buf->visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    full_retset.reserve(l_search * 10);

    // query <-> PQ chunk centers distances
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;
    float *pq_dists2_ptr = query_buf->aligned_pqtable_dist_scratch_2;
    // float *pq_dists = nullptr;
    // pipeann::alloc_aligned((void **) &pq_dists, 256 * MAX_PQ_CHUNKS * sizeof(float), 256);
#ifndef OVERLAP_INIT
    pq_table.populate_chunk_distances(query, pq_dists);  // overlap with the first I/O.
#endif

    auto pq_dist_lookup_dual_branching = [this](const unsigned *ids, const _u64 n_pts, const _u64 pq_nchunks,
                                            const _u8 *pq_ids,       // candidate codes: [n_pts * pq_nchunks]
                                            const _u8 *pq_ids_2,
                                            const float *pq_dists1,  // [256 * pq_nchunks]
                                            const float *pq_dists2,  // [256 * pq_nchunks]
                                            float *dists_out) {      // output [n_pts]
      // map_bytes 是全局 std::vector<uint8_t>
      const uint8_t *map_ptr = map_bytes.data();

      memset(dists_out, 0, (size_t) n_pts * sizeof(float));
      for (_u64 chunk = 0; chunk < pq_nchunks; ++chunk) {
        const float *chunk1 = pq_dists1 + (size_t) 256 * (size_t) chunk;
        const float *chunk2 = pq_dists2 + (size_t) 256 * (size_t) chunk;
        _mm_prefetch((char *) (chunk1), _MM_HINT_T0);
        _mm_prefetch((char *) (chunk2), _MM_HINT_T0);
        for (_u64 idx = 0; idx < n_pts; ++idx) {
          unsigned pid = ids[idx];  // 全局 id：必须与 map_bytes 的索引对齐
          uint8_t use_second = map_ptr[(size_t) pid * (size_t) pq_nchunks + (size_t) chunk];
          _u8 code = pq_ids[pq_nchunks * idx + chunk];
          // _u8 code2 = pq_ids_2[pq_nchunks * idx + chunk];
          dists_out[idx] += use_second ? chunk2[code] : chunk1[code];
        }
      }
    };

    // compute_pq_dists（保持签名不变）
    auto compute_pq_dists = [this, pq_coord_scratch, pq_coord_scratch_2, pq_dists, pq_dists2_ptr, &pq_dist_lookup_dual_branching, stats]
    (const unsigned *ids, const _u64 n_ids, float *dists_out) {
      if (stats != nullptr) {
        stats->pq_cmps += n_ids;
      }
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      // ::aggregate_coords(ids, n_ids, this->data_2.data(), this->n_chunks, pq_coord_scratch_2);
      pq_dist_lookup_dual_branching(ids, n_ids, this->n_chunks,
                                    (_u8 *)pq_coord_scratch,
                                    (_u8 *)pq_coord_scratch_2,
                                    pq_dists,
                                    pq_dists2_ptr,
                                    dists_out);
    };

    auto compute_exact_dists_and_push = [&](const char *node_buf, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      return cur_expanded_dist;
    };

    uint64_t n_computes = 0;
    auto compute_and_push_nbrs = [&](char *node_buf, unsigned &nk) {
      unsigned *node_nbrs = use_rerank ? (unsigned *)node_buf : offset_to_node_nhood(node_buf);
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;
      // if(t)
      // std::cerr << " has " << nnbrs << "\n";
      
      for (unsigned m = 0; m < nnbrs; ++m) {
      // if(t);
      // std::cerr << node_nbrs[m] << " ";
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }
      // if(t);
      // std::cerr << "\n";

      n_computes += nbors_cand_size;
      if (nbors_cand_size) {
        // auto cpu1_st = std::chrono::high_resolution_clock::now();
        // std::cerr << "push " << nbors_cand_size << "\n";
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
          if (cur_list_size < l_search) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }
          // nk logs the best position in the retset that was updated due to
          // neighbors of n.
          if (r < nk)
            nk = r;
        }

      }
    };

    auto add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids, float *dists) {
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size++] = Neighbor(node_ids[i], dists[i], true);
        visited.insert(node_ids[i]);
      }
    };  

    // stats.
    if(stats != nullptr){
      stats->io_us = 0;
      stats->non_io_us = 0;
      stats->io_us1 = 0;
      stats->cpu_us = 0;
      stats->cpu_us1 = 0;
      stats->cpu_us2 = 0;
      stats->pq_cmps = 0;
    }
    // search in in-memory index.
    // std::cerr << "1\n";

#ifdef DYN_PIPE_WIDTH
    int64_t cur_beam_width = 4;  // before converge.
#else
    int64_t cur_beam_width = beam_width;  // before converge.
#endif

    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dists(mem_L);

#ifdef OVERLAP_INIT
    if (mem_L) {
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      add_to_retset(mem_tags.data(), std::min((_u64) mem_L, l_search), mem_dists.data());
    } else {
      // cannot overlap.
      pq_table.populate_chunk_distances_nt(query, pq_dists);
      // 生成第二套 LUT —— **用相同的 query（不是 query_2）**
      pq_table_2.populate_chunk_distances_nt(query, pq_dists2_ptr);
      compute_pq_dists(&medoids[0], 1, dist_scratch);
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
#else
    if (mem_L) {
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch);
      add_to_retset(mem_tags.data(), std::min((_u64) mem_L, l_search), dist_scratch);
    } else {
      compute_pq_dists(&medoids[0], 1, dist_scratch);
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);
#endif

    std::priority_queue<rerank_io_t, std::vector<rerank_io_t>, std::greater<rerank_io_t>> on_flight_ios;
    std::unordered_map<unsigned, char *> id_buf_map;

    auto send_read_req = [&](Neighbor &item) -> bool {
      item.flag = false;

      // lock the corresponding page.
      this->lock_idx(idx_lock_table, item.id, std::vector<uint32_t>(), true);
#ifdef USE_NHOOD_CACHE
      if(nhood_cache.find(item.id) != nhood_cache.end()){
        id_buf_map.insert(std::make_pair(item.id, (char *)nhood_cache[item.id]));
        this->unlock_idx(idx_lock_table, item.id);
        return true;
      }
#endif
#ifdef USE_TOPO_DISK
      unsigned loc = id2loc_topo(item.id), pid = loc_sector_no_topo(loc);
#else
      unsigned loc = id2loc(item.id), pid = loc_sector_no(loc);
#endif
      if(loc == kInvalidID){
        LOG(ERROR) << "id " << item.id << " not found in id2loc";
        exit(-1);
      }

      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      // std::cerr << "cur buf addr " << (void *) &(query_buf->reqs[cur_buf_idx]) << "\n";
      int io_cb_idx = -1;
      if (use_rerank) {
#ifndef  USE_TOPO_DISK
        pid = loc_sector_no_topo(loc);
        if(use_topo_reorder){
          loc = loc2phy_topo[loc];
          pid = loc_sector_no_topo(loc);
        }
#endif
        if(page_buf_map.find(pid) != page_buf_map.end()){
          buf = page_buf_map[pid];
          req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset_topo(loc), topo_len);
          req.finished = true;
          this->unlock_idx(idx_lock_table, item.id);
          on_flight_ios.push(rerank_io_t(item, pid, loc, &req, -1, page_rank_map[pid], ts++, false));
          cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
          if (topo_page_map != nullptr) {
            topo_page_map->insert(std::make_pair(pid, buf));
          }
          return true;//return true ?
        }

        req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset_topo(loc), topo_len);
        page_buf_map.insert({pid, buf});
        if (topo_page_map != nullptr) {
          topo_page_map->insert(std::make_pair(pid, buf));
        }
        page_rank_map.insert({pid, rank});

        if(use_topo_buffer){
          io_cb_idx = block_cache->request_block(pid);
          int status = block_cache->cache_status[io_cb_idx];
          // if(status != 1 && status != 0) {
          //   std::cerr << "cache error!" << std::endl;
          //   exit(0);
          // }
          // std::cerr << "**3\n";

          if(status == 1){// hit
            // std::cerr << "hit " << "\n";
            req.finished = true;
            this->unlock_idx(idx_lock_table, item.id);
            on_flight_ios.push(rerank_io_t(item, pid, loc, &req, io_cb_idx, 0, ts++, false)); // push front
            cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
            return true;
          } else 
          {
            reader->send_io(req, ctx, false, topo_fd);
          }
        } else {
          reader->send_io(req, ctx, false, topo_fd);
        }
      } else {
        req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), max_node_len);
        reader->send_read_no_alloc(req, ctx);
      }
      on_flight_ios.push(rerank_io_t{item, pid, loc, &req, io_cb_idx, rank++, ts++});

      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

      if (stats != nullptr) {
        stats->n_ios++;
      }
      return true;
    };
    
    auto print_state = [&]() {
      LOG(INFO) << "cur_list_size: " << cur_list_size;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        LOG(INFO) << "retset[" << i << "]: " << retset[i].id << ", " << retset[i].distance << ", " << retset[i].flag
                  << ", " << retset[i].visited << ", " << (id_buf_map.find(retset[i].id) != id_buf_map.end());
      }
      LOG(INFO) << "On flight IOs: " << on_flight_ios.size();
      if (on_flight_ios.size() != 0) {
        auto &io = on_flight_ios.top();
        LOG(INFO) << "on_flight_io: " << io.nbr.id << ", " << io.nbr.distance << ", " << io.nbr.flag << ", "
                  << io.page_id << ", " << io.loc << ", " << io.finished();
      }
    };

    // std::cerr << "1\n";
    auto poll_all = [&]() -> std::pair<int, int> {
      // poll once.
      reader->poll_all(ctx);
      // print_state();
      // LOG(ERROR) << "2";
      unsigned n_in = 0, n_out = 0;
      while (!on_flight_ios.empty() && on_flight_ios.top().finished()) {
        const rerank_io_t &io = on_flight_ios.top();
        if(use_topo_buffer && io.cb_idx != -1){
          int cb_idx = io.cb_idx;
          int status = block_cache->cache_status[cb_idx];
          if(status == 1){
            // std::cerr << "status == 1\n";
            std::memcpy(io.read_req->buf, &block_cache->cache_block_vec[cb_idx], size_per_io);
          } 
          else if(status == 0)
          {
            // std::cerr << "status == 0\n";
            std::memcpy(&block_cache->cache_block_vec[cb_idx], io.read_req->buf, size_per_io);
            // std::cerr << "cb_idx " << cb_idx << " set 1\n";
            block_cache->cache_status[cb_idx] = 1;
          }
          page_ref->insert(std::make_pair(io.read_req->offset / SECTOR_LEN, cb_idx));
        }
        
        if (use_rerank) {
          char * buf = (char *) io.read_req->buf;
          
          id_buf_map.insert(std::make_pair(io.nbr.id, offset_to_loc_topo(buf, io.loc)));
        } else {
          id_buf_map.insert(std::make_pair(io.nbr.id,  offset_to_loc((char *) io.read_req->buf, io.loc)));
        }
        io.nbr.distance <= retset[cur_list_size - 1].distance ? ++n_in : ++n_out;
        if(io.need_unlock)
          this->unlock_idx(idx_lock_table, io.nbr.id);
        on_flight_ios.pop();
        // LOG(ERROR) << "#@(())";

      }
      // LOG(ERROR) << "3";
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      // auto io_st = std::chrono::high_resolution_clock::now();
      unsigned n_sent = 0, marker = 0;
      while (marker < cur_list_size && n_sent < n) {
        while (marker < cur_list_size /* pool size */ &&
               (retset[marker].flag == false /* on flight */ ||
                id_buf_map.find(retset[marker].id) != id_buf_map.end() /* already read */)) {
          retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
          ++marker;
        }
        if (marker >= cur_list_size) {
          break;  // nothing to send.
        }
        n_sent += send_read_req(retset[marker]);
      }
      return n_sent != 0;  // nothing to send.
    };

    char * calc_buf = nullptr;
    pipeann::alloc_aligned((void **)&calc_buf, size_per_io, SECTOR_LEN);
    auto calc_best_node = [&]() -> int {  // if converged.
      // auto cpu_st = std::chrono::high_resolution_clock::now();
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      /* calculate one from "already read" */
      // std::cerr << "**1\n";
      for (marker = 0; marker < cur_list_size; ++marker) {
        // std::cerr << "***1\n";

        if (!retset[marker].visited && id_buf_map.find(retset[marker].id) != id_buf_map.end()) {
          retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
          retset[marker].visited = true;
          auto it = id_buf_map.find(retset[marker].id);
          auto [id, buf] = *it;
          // std::cerr << "***2\n";

          if (use_rerank){// just push pq dist
            // do nothing
          }else {
            compute_exact_dists_and_push(buf, id);
          }

          // std::cerr << "pop " << id << " " << retset[marker].distance << std::endl;
          
          memcpy(calc_buf, buf, topo_len);
          compute_and_push_nbrs(calc_buf, nk);
          
          break;
        }

      }

      /* guess the first unvisited vector (eager) */
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited && retset[i].flag /* not on-fly */
            && id_buf_map.find(retset[i].id) == id_buf_map.end() /* not already read */) {
          first_unvisited_eager = i;
          break;
        }
      }

      return first_unvisited_eager;
      // auto cpu_ed = std::chrono::high_resolution_clock::now();
      // if(stats != nullptr)
      // stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
    };

    auto get_first_unvisited = [&]() -> int {
      int ret = -1;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          ret = i;
          break;
        }
      }
      return ret;
    };


    std::ignore = print_state;

    // std::cerr << "2\n";
    auto io1_st = std::chrono::high_resolution_clock::now();
    send_best_read_req(cur_beam_width - on_flight_ios.size());
    unsigned marker = 0, max_marker = 0;
#ifdef OVERLAP_INIT
    if (likely(mem_L != 0)) {
      pq_table.populate_chunk_distances_nt(query, pq_dists);  // overlap with the first I/O.
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch);
      for (unsigned i = 0; i < cur_list_size; ++i) {
        retset[i].distance = dist_scratch[i];
      }
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    }
#endif

#ifndef STATIC_POLICY
    int cur_n_in = 0, cur_tot = 0;
#endif    


    while (get_first_unvisited() != -1) {
      // poll to heap (best-effort) -> calc best from heap (skip if heap is empty) -> send IO (if can send) -> ...
      auto [n_in, n_out] = poll_all();
      auto io1_ed = std::chrono::high_resolution_clock::now();
      if(stats != nullptr){
        stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io1_ed - io1_st).count();
        stats->io_us1 += std::chrono::duration_cast<std::chrono::microseconds>(io1_ed - io1_st).count();
      }
      std::ignore = n_in;
      std::ignore = n_out;

#ifdef DYN_PIPE_WIDTH
#ifdef STATIC_POLICY
      constexpr int kBeamWidths[] = {4, 4, 8, 8, 16, 16, 24, 24, 32};
      cur_beam_width = kBeamWidths[std::min(max_marker / 5, 8u)];
#else
      if (max_marker >= 5 && n_in + n_out > 0) {//flop第一轮
        cur_n_in += n_in;
        cur_tot += n_in + n_out;
        // converged, tune beam width.
        constexpr double kWasteThreshold = 0.1;  // 0.1 * 10
        if ((cur_tot - cur_n_in) * 1.0 / cur_tot <= kWasteThreshold) {
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }
      }
#endif
#endif
      io1_st = std::chrono::high_resolution_clock::now();
      if ((int64_t) on_flight_ios.size() < cur_beam_width) {
#ifdef NAIVE_PIPE
        send_best_read_req(cur_beam_width - on_flight_ios.size());
#else
        send_best_read_req(1);
#endif
      }
      marker = calc_best_node();

      max_marker = std::max(max_marker, marker);
    }

    pipeann::aligned_free((void *)calc_buf);
    
    while(!on_flight_ios.empty()){
      auto io = on_flight_ios.top();
      // counter --;
      this->unlock_idx(idx_lock_table, io.nbr.id);
      if(io.cb_idx != -1)
        block_cache->release_cache_block(io.cb_idx, 0);
      on_flight_ios.pop();
    }
    
    Timer rerank_timer;
//进行rerank 第二阶段
    if(use_rerank){
      
      int p = p_reader->trunc_len;
      int trunc_len;
      // trunc_len = p;
      trunc_len = std::max(p, (int)std::ceil(p * (1 + log10((double)l_search / (double)p))));

      retset.resize(std::min(trunc_len, (int)l_search));
      //去重
      for(auto node : retset){
        TagT tag = id2tag(node.id);
        if (exclude_nodes != nullptr && exclude_nodes->find(tag) != exclude_nodes->end()) {
          continue;
        }
        full_retset.push_back(node);
      }

      Timer rerank_timer_2;
      std::vector<std::pair<int, char *>> idx_buf;
      std::vector<IORequest> read_reqs;
      idx_buf.reserve(full_retset.size());
      read_reqs.reserve(full_retset.size());
      page_buf_map.clear();

      std::vector<uint32_t> to_lock(full_retset.size());
      for(size_t i = 0; i < full_retset.size(); i++){
        to_lock[i] = full_retset[i].id;
      }
      sort(to_lock.begin(), to_lock.end());

      auto last = std::unique(to_lock.begin(), to_lock.end());
      if(last != to_lock.end()){
        LOG(ERROR) << "Rerank, some id are duplicated";
        exit(-1);
      }
      
      for (auto &id : to_lock) {
        vec_lock_table.rdlock(id);
      }
      
      for(size_t i = 0; i < full_retset.size(); i++){
        auto id = full_retset[i].id;
#ifdef USE_TOPO_DISK
        unsigned loc = id2loc_coord(id), pid = loc_sector_no_coord(loc);
#else
        unsigned loc = id2loc(id), pid = loc_sector_no_coord(loc);
#endif
        if(loc == kInvalidID){
          LOG(ERROR) << "id " << id << " not found in id2loc_coord";
          exit(-1);
        }

#ifndef USE_TOPO_DISK
        if(use_coord_reorder){
          loc = loc2phy_coord[loc];
          pid = loc_sector_no_coord(loc);
        }
#endif
        if (page_buf_map.find(pid) != page_buf_map.end()) {
          idx_buf.push_back({i, offset_to_loc_coord(page_buf_map[pid], loc)});
          continue;
        }
        uint64_t &cur_buf_idx = query_buf->sector_idx;
        auto buf = sector_scratch + cur_buf_idx * size_per_io;
        auto io2_st = std::chrono::high_resolution_clock::now();
        read_reqs.push_back(IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset_coord(loc), coord_len));
        page_buf_map.insert({pid, buf});
        idx_buf.push_back({i, offset_to_loc_coord(buf, loc)});
        cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
      }
      // Batch coordinate reads to avoid submitting an oversized request list at once.
      auto io2_st = std::chrono::high_resolution_clock::now();
      constexpr size_t kRerankReadBatchSize = 32;
      std::vector<IORequest> batched_read_reqs;
      batched_read_reqs.reserve(kRerankReadBatchSize);
      for (size_t req_st = 0; req_st < read_reqs.size(); req_st += kRerankReadBatchSize) {
        const size_t req_ed = std::min(req_st + kRerankReadBatchSize, read_reqs.size());
        batched_read_reqs.assign(read_reqs.begin() + req_st, read_reqs.begin() + req_ed);
        reader->read_fd(coord_fd, batched_read_reqs, ctx);
      }
      auto io2_ed = std::chrono::high_resolution_clock::now();
      //计算读取时间
      if(stats != nullptr){
        stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io2_ed - io2_st).count();
        stats->io_us2 += std::chrono::duration_cast<std::chrono::microseconds>(io2_ed - io2_st).count();
        stats->rerank_ios += read_reqs.size();
      }
      for (auto &id : to_lock) {
        vec_lock_table.unlock(id);
      }

      for(auto [idx, node_buf] : idx_buf){
        unsigned id = full_retset[idx].id;
        T *node_fp_coords_copy = data_buf + data_buf_idx * aligned_dim;
        memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));
        data_buf_idx ++; 
        if (coord_map != nullptr){
          coord_map->insert(std::make_pair(id, node_fp_coords_copy));
        }
        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
        full_retset[idx].distance = cur_expanded_dist;
      }
      if(stats != nullptr){
        stats->rerank_us_2 += rerank_timer_2.elapsed();
      }
    }
    if (stats != nullptr) {
      stats->rerank_us += (double) rerank_timer.elapsed();
    }
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    if(passthrough_page_ref == nullptr){
      for(auto [pid, idx] : *page_ref){
        block_cache->release_cache_block(idx, 1);
      }
      delete page_ref;
    }
    if (passthrough_data == nullptr) {
      push_query_buf(query_buf);
    }

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return ;
  }

  
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::rerank_search(const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, TagT * gt_tags) {
    // iterate to fixed point
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_rerank_search(query, mem_L, (_u32) l_search, (_u32) beam_width, expanded_nodes_info, nullptr, nullptr, stats,
                         deleted_nodes, nullptr, nullptr);
    _u64 res_count = 0;
    for (uint32_t i = 0; i < l_search && res_count < k_search && i < expanded_nodes_info.size(); i++) {
      if (i > 0 && expanded_nodes_info[i].id == expanded_nodes_info[i - 1].id) {
        continue;  // deduplicate.
      }
      res_tags[res_count] = id2tag(expanded_nodes_info[i].id);
      distances[res_count] = expanded_nodes_info[i].distance;
      res_count++;
    }
    return res_count;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann
