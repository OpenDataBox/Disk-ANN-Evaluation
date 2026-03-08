// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <algorithm>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann {
  struct QueryStats {
    float total_us = 0;  // total time to process query in micros
    float io_us = 0;     // total time spent in IO
    float cpu_us = 0;    // total time spent in CPU
    float preprocess_us = 0;    // total time spent for pre process (in-mem nev)
    float postprocess_us = 0;    // total time spent for post process (sort and lim-k)

    float dispatch_us = 0;    // total time spent for dispatch nodes
    float read_disk_us = 0;    // total time spent for read disk process
    float cache_proc_us = 0;    // total time spent for cache node process
    float disk_proc_us = 0;    // total time spent for disk node process
    float page_proc_us = 0;    // total time spent for page process in PageSearch

    float total_io_time_us = 0;   // total IO time for this query in micros

    unsigned n_ios = 0;         // total # of search IOs issued
    unsigned n_emb_ios = 0;     // total # of refine IOs issued
    unsigned n_cmps = 0;        // # cmps
    unsigned n_ext_cmps = 0;    // # exact cmps
    unsigned n_cache_hits = 0;  // # cache_hits
    unsigned n_hops = 0;        // # search hops
    
    // Distance computation time statistics (in microseconds)
    double pq_distance_time_us = 0.0;      // Total time for PQ distance computations
    double exact_distance_time_us = 0.0;   // Total time for exact distance computations
    
    // IO utilization statistics
    float io_utilization = 0.0f;      // IO utilization ratio (useful_nodes / total_nodes)
    uint64_t useful_io_nodes = 0;     // # of useful nodes read from disk
    uint64_t total_io_nodes = 0;      // # of total nodes read from disk
    
    // Baseline comparison statistics (for disk IO hit rate analysis)
    size_t baseline_total_disk_bytes = 0;     // Total disk bytes accessed
    size_t baseline_hit_disk_bytes = 0;       // Disk bytes that hit baseline
    
    // Cache hit rate statistics
    size_t cache_hit_bytes = 0;               // Total bytes hit in cache
    size_t cache_total_bytes = 0;             // Total cache size in bytes
    float cache_hit_rate = 0.0f;              // Cache hit rate (hit_bytes / (1000 * cache_size))
    
    // Per-hop cache hit rate statistics (for first 100 hops)
    static constexpr size_t MAX_HOP_STATS = 100;
    size_t per_hop_cache_hits[MAX_HOP_STATS] = {0};    // Cache hits at each hop
    size_t per_hop_total_accesses[MAX_HOP_STATS] = {0}; // Total accesses at each hop
    size_t actual_hops = 0;                             // Actual number of hops recorded
    
    // Resource overhead statistics (shared across all queries for a given configuration)
    static double init_memory_peak_mb;     // Peak memory usage during index loading and cache initialization in MB
    static double search_memory_peak_mb;   // Peak memory usage during search queries in MB
  };

  template<typename T>
  inline T get_percentile_stats(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<T(const QueryStats &)> &member_fn) {
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(),
              [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
  }

  template<typename T>
  inline double get_mean_stats(
      QueryStats *stats, uint64_t len, 
      const std::function<T(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      avg += (double) member_fn(stats[i]);
    }
    return avg / len;
  }

  template<typename T>
  inline double get_mean_stats(
      QueryStats *stats, uint64_t len, uint64_t warmup, 
      const std::function<T(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = warmup; i < len; i++) {
      avg += (double) member_fn(stats[i]);
    }
    return avg / (len - warmup);
  }

  // The following two functions are used when getting statistics while range searching on only queries with
  // non-zero gt lengths
  template<typename T>
  inline T get_percentile_stats_gt(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<T(const QueryStats &)> &member_fn, std::vector<std::vector<uint32_t>> &gt) {
    std::vector<T> vals;
    for (uint64_t i = 0; i < len; i++) {
      if (gt[i].size()) vals.push_back(member_fn(stats[i]));
    }

    std::sort(vals.begin(), vals.end(),
              [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * vals.size())];
    vals.clear();
    return retval;
  }

  template<typename T>
  inline double get_mean_stats_gt(
      QueryStats *stats, uint64_t len,
      const std::function<T(const QueryStats &)> &member_fn, std::vector<std::vector<uint32_t>> &gt) {
    uint32_t cnt = 0;
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      if (gt[i].size()) {
        ++cnt;
        avg += (double) member_fn(stats[i]);
      }
    }
    return avg / cnt;
  }
}  // namespace diskann
