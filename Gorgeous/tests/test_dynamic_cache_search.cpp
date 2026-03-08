// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <set>
#include <omp.h>
#include <boost/program_options.hpp>

#include "deco_index.h"
#include "dynamic_cache_search.h"
#include "utils.h"
#include "percentile_stats.h"
#include "file_io_manager.h"
#include "aux_utils.h"

namespace po = boost::program_options;

void print_usage() {
    std::cout << "Usage: test_dynamic_cache_search [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --data_type <type>           Data type (float/int8/uint8)" << std::endl;
    std::cout << "  --index_path_prefix <path>   Index path prefix" << std::endl;
    std::cout << "  --pq_path_prefix <path>      PQ path prefix" << std::endl;
    std::cout << "  --query_file <file>          Query file path" << std::endl;
    std::cout << "  --gt_file <file>             Ground truth file path" << std::endl;
    std::cout << "  --K <num>                    Number of neighbors to retrieve" << std::endl;
    std::cout << "  --L <num>                    Search list size (default: 100)" << std::endl;
    std::cout << "  --prefetch_window <num>      Prefetch window size (default: 4)" << std::endl;
    std::cout << "  --cache_capacity <num>       FIFO cache capacity in pages (default: 1000)" << std::endl;
    std::cout << "  --io_limit <num>             I/O limit (default: 10000)" << std::endl;
    std::cout << "  --num_threads <num>          Number of threads (default: 1)" << std::endl;
    std::cout << "  --mem_L <num>                Memory navigation graph L (default: 0)" << std::endl;
    std::cout << "  --emb_search_ratio <ratio>   Embedding search ratio (default: 1.0)" << std::endl;
    std::cout << "  --sector_len <num>           Sector length (default: 4096)" << std::endl;
}

template<typename T>
int run_dynamic_cache_search(const std::string& index_prefix,
                              const std::string& pq_prefix,
                              const std::string& query_file,
                              const std::string& gt_file,
                              const std::string& mem_index_path,
                              unsigned K, unsigned L,
                              unsigned prefetch_window,
                              size_t cache_capacity,
                              unsigned io_limit,
                   unsigned num_threads,
                   unsigned mem_L,
                   float emb_search_ratio,
                   _u64 sector_len) {
    
    diskann::Metric metric = diskann::Metric::L2;
    
    std::shared_ptr<FileIOManager> fio_manager(new FileIOManager());
    
    // Create index - for DynamicCache search, need to use Starling mode to load metadata
    // Fix: use use_graph_rep_index=false to ensure id2page_ and gp_layout_ are loaded correctly
    diskann::DecoIndex<T> index(fio_manager, metric, false, sector_len);
    
    std::cout << "Loading index from " << index_prefix << std::endl;
    
    std::string disk_index_path = index_prefix + "_disk.index";
    std::string graph_rep_prefix = index_prefix + "GRAPH_CACHE_INDEX/";
    std::string disk_graph_prefix = index_prefix + "GRAPH/";
    
    // Load index
    int res = index.load(num_threads, index_prefix.c_str(), pq_prefix.c_str(),
                         disk_index_path, graph_rep_prefix, disk_graph_prefix);
    
    if (res != 0) {
        std::cerr << "Error: Failed to load index. Return code: " << res << std::endl;
        return res;
    }
    
    std::cout << "Index loaded successfully." << std::endl;
    
    std::cout << "Loading queries from " << query_file << std::endl;
    T* queries = nullptr;
    size_t query_num, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_file, queries, query_num, query_dim, query_aligned_dim);
    
    std::cout << "Loaded " << query_num << " queries of dimension " << query_dim << std::endl;
    
    if (mem_L > 0 && !mem_index_path.empty()) {
        index.load_mem_index(metric, query_dim, mem_index_path, num_threads, mem_L);
        std::cout << "Memory navigation graph loaded from " << mem_index_path << std::endl;
    }
    
    unsigned* gt_ids = nullptr;
    float* gt_dists = nullptr;
    size_t gt_num = 0, gt_dim = 0;
    bool calc_recall = false;
    
    if (!gt_file.empty() && gt_file != "null" && gt_file != "NULL" && file_exists(gt_file)) {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num) {
            std::cerr << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        } else {
            calc_recall = true;
            std::cout << "Ground truth loaded: " << gt_num << " queries, " << gt_dim << " neighbors each" << std::endl;
        }
    }
    
    std::vector<_u64> indices(query_num * K);
    std::vector<float> distances(query_num * K);
    std::vector<diskann::QueryStats> stats(query_num);
    std::vector<diskann::DynamicCacheSearchStats> dc_stats(query_num);
    
    diskann::DynamicCacheSearchConfig dc_config;
    dc_config.prefetch_window = prefetch_window;
    dc_config.search_list_size = L;
    dc_config.cache_capacity = cache_capacity;
    dc_config.enable_stats = true;
    
    std::cout << "\n=== DynamicCache Search Configuration ===" << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "L (search list size): " << L << std::endl;
    std::cout << "Prefetch window: " << prefetch_window << std::endl;
    std::cout << "Cache capacity: " << cache_capacity << " pages" << std::endl;
    std::cout << "I/O limit: " << io_limit << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Memory navigation L: " << mem_L << std::endl;
    std::cout << "Embedding search ratio: " << emb_search_ratio << std::endl;
    std::cout << "Sector length: " << sector_len << std::endl;
    std::cout << "================================\n" << std::endl;

    omp_set_num_threads(num_threads);
    
    std::cout << "Running DynamicCache search..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    index.dynamic_cache_search(queries, query_num, query_aligned_dim, K, mem_L, L,
                               indices, distances, io_limit, emb_search_ratio,
                               dc_config, stats.data(), dc_stats.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Search completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Average latency: " << (double)duration.count() / query_num << " ms/query" << std::endl;
    if (duration.count() > 0) {
        std::cout << "QPS: " << (double)query_num * 1000.0 / duration.count() << std::endl;
    }
    
    std::cout << "\n=== Aggregated Statistics ===" << std::endl;
    
    size_t warmup_cnt = std::min((size_t)20, query_num);
    
    std::cout << "\n=== Per-Hop Cache Hit Rate (first 100 hops) ===" << std::endl;
    std::vector<unsigned> total_hop_hits(diskann::MAX_HOP_STATS, 0);
    std::vector<unsigned> total_hop_misses(diskann::MAX_HOP_STATS, 0);
    unsigned max_hops = 0;
    
    for (size_t i = warmup_cnt; i < query_num; i++) {
        max_hops = std::max(max_hops, dc_stats[i].total_hops);
        for (unsigned h = 0; h < diskann::MAX_HOP_STATS; h++) {
            total_hop_hits[h] += dc_stats[i].hop_cache_hits[h];
            total_hop_misses[h] += dc_stats[i].hop_cache_misses[h];
        }
    }
    
    // Output per-hop cache hit rate (CSV format, convenient for plotting)
    std::cout << "hop,hits,misses,total,hit_rate" << std::endl;
    for (unsigned h = 0; h < std::min(max_hops, diskann::MAX_HOP_STATS); h++) {
        unsigned total = total_hop_hits[h] + total_hop_misses[h];
        if (total > 0) {
            double hit_rate = 100.0 * total_hop_hits[h] / total;
            std::cout << h << "," << total_hop_hits[h] << "," << total_hop_misses[h] 
                     << "," << total << "," << hit_rate << std::endl;
        }
    }
    std::cout << "=== End Per-Hop Stats ===" << std::endl;
    
    if (calc_recall) {
        std::cout << "\nCalculating recall..." << std::endl;
        
        std::vector<unsigned> indices_u32(query_num * K);
        for (size_t i = 0; i < query_num * K; i++) {
            indices_u32[i] = static_cast<unsigned>(indices[i]);
        }
        
        double recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                                   indices_u32.data(), K, K);
        std::cout << "Recall@" << K << ": " << recall << std::endl;
    }
    
    // Print detailed statistics for first query
    if (query_num > warmup_cnt) {
        std::cout << "\n=== Sample Query Detailed Stats (Query " << warmup_cnt << ") ===" << std::endl;
        dc_stats[warmup_cnt].print();
    }

    diskann::aligned_free(queries);
    if (gt_ids) delete[] gt_ids;
    if (gt_dists) delete[] gt_dists;
    
    return 0;
}

int main(int argc, char** argv) {
    std::string data_type, index_prefix, pq_prefix, query_file, gt_file, mem_index_path;
    unsigned K, L, prefetch_window, io_limit, num_threads, mem_L;
    size_t cache_capacity;
    float emb_search_ratio;
    _u64 sector_len;
    
    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type",
                           po::value<std::string>(&data_type)->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("index_path_prefix",
                           po::value<std::string>(&index_prefix)->required(),
                           "Path prefix to the index");
        desc.add_options()("pq_path_prefix",
                           po::value<std::string>(&pq_prefix)->required(),
                           "Path prefix to the PQ files");
        desc.add_options()("query_file",
                           po::value<std::string>(&query_file)->required(),
                           "Query file in binary format");
        desc.add_options()("gt_file",
                           po::value<std::string>(&gt_file)->default_value("null"),
                           "Ground truth file for the queryset");
        desc.add_options()("mem_index_path",
                           po::value<std::string>(&mem_index_path)->default_value(""),
                           "Path to memory navigation index");
        desc.add_options()("K",
                           po::value<unsigned>(&K)->default_value(10),
                           "Number of neighbors to retrieve");
        desc.add_options()("L",
                           po::value<unsigned>(&L)->default_value(100),
                           "Search list size");
        desc.add_options()("prefetch_window",
                           po::value<unsigned>(&prefetch_window)->default_value(4),
                           "Prefetch window size (concurrent I/O requests)");
        desc.add_options()("cache_capacity",
                           po::value<size_t>(&cache_capacity)->default_value(1000),
                           "FIFO cache capacity in pages");
        desc.add_options()("io_limit",
                           po::value<unsigned>(&io_limit)->default_value(10000),
                           "Maximum I/O operations per query");
        desc.add_options()("num_threads,T",
                           po::value<unsigned>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads");
        desc.add_options()("mem_L",
                           po::value<unsigned>(&mem_L)->default_value(0),
                           "Memory navigation graph L (0 to disable)");
        desc.add_options()("emb_search_ratio",
                           po::value<float>(&emb_search_ratio)->default_value(1.0f),
                           "Embedding search ratio");
        desc.add_options()("sector_len",
                           po::value<_u64>(&sector_len)->default_value(4096),
                           "Sector length in bytes");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }
    
    try {
        if (data_type == "float") {
            return run_dynamic_cache_search<float>(index_prefix, pq_prefix, query_file, gt_file,
                                                   mem_index_path, K, L, prefetch_window, 
                                                   cache_capacity, io_limit, num_threads, 
                                                   mem_L, emb_search_ratio, sector_len);
        } else if (data_type == "int8") {
            return run_dynamic_cache_search<int8_t>(index_prefix, pq_prefix, query_file, gt_file,
                                                    mem_index_path, K, L, prefetch_window,
                                                    cache_capacity, io_limit, num_threads,
                                                    mem_L, emb_search_ratio, sector_len);
        } else if (data_type == "uint8") {
            return run_dynamic_cache_search<uint8_t>(index_prefix, pq_prefix, query_file, gt_file,
                                                     mem_index_path, K, L, prefetch_window,
                                                     cache_capacity, io_limit, num_threads,
                                           mem_L, emb_search_ratio, sector_len);
        } else {
            std::cerr << "Error: Unsupported data type: " << data_type << std::endl;
            return -1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
