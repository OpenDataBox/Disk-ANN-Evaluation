// HybridCache Search Test
// Demonstrates how to use HybridCache's two-phase search and hybrid cache

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <set>
#include <omp.h>
#include <boost/program_options.hpp>

#include "deco_index.h"
#include "hybridcache_search.h"
#include "hybridcache_cache.h"
#include "file_io_manager.h"
#include "utils.h"
#include "percentile_stats.h"
#include "aux_utils.h"

namespace po = boost::program_options;
using namespace diskann;

void print_usage() {
    std::cout << "Usage: test_hybridcache_search [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --data_type <type>           Data type (float/int8/uint8)" << std::endl;
    std::cout << "  --dist_fn <func>             Distance function (l2/mips/cosine)" << std::endl;
    std::cout << "  --index_path_prefix <path>   Index path prefix" << std::endl;
    std::cout << "  --pq_path_prefix <path>      PQ path prefix" << std::endl;
    std::cout << "  --query_file <file>          Query file path" << std::endl;
    std::cout << "  --gt_file <file>             Ground truth file path" << std::endl;
    std::cout << "  --result_path <path>         Result output path" << std::endl;
    std::cout << "  --stats_file <file>          Statistics output file" << std::endl;
    std::cout << "  --K <num>                    Number of neighbors to retrieve" << std::endl;
    std::cout << "  --L <num>                    Search list size (default: 100)" << std::endl;
    std::cout << "  --beamwidth <num>            Beam width (default: 4)" << std::endl;
    std::cout << "  --num_threads <num>          Number of threads (default: 1)" << std::endl;
    std::cout << "  --mem_L <num>                Memory navigation graph L (default: 0)" << std::endl;
    std::cout << "  --io_limit <num>             I/O limit (default: 10000)" << std::endl;
    std::cout << "  --pq_filter_ratio <ratio>    PQ filter ratio (default: 1.2)" << std::endl;
    std::cout << "  --emb_search_ratio <ratio>   Embedding search ratio (default: 0.4)" << std::endl;
    std::cout << "  --hybridcache_theta <ratio>     HybridCache theta (default: 0.3)" << std::endl;
    std::cout << "  --hybridcache_prefetch_pages <num>  HybridCache prefetch pages (default: 3)" << std::endl;
    std::cout << "  --hybridcache_static_cache <num>    HybridCache static cache capacity (default: 100)" << std::endl;
    std::cout << "  --hybridcache_dynamic_cache <num>   HybridCache dynamic cache capacity (default: 500)" << std::endl;
    std::cout << "  --hybridcache_enable_stats <0/1>    Enable HybridCache statistics (default: 1)" << std::endl;
    std::cout << "  --sector_len <num>           Sector length (default: 4096)" << std::endl;
}

template<typename T>
int run_hybridcache_search(const std::string& data_type,
                        const std::string& dist_fn,
                        const std::string& index_prefix,
                        const std::string& pq_prefix,
                        const std::string& query_file,
                        const std::string& gt_file,
                        const std::string& result_path,
                        const std::string& stats_file,
                        const std::string& mem_index_path,
                        const std::string& static_cache_file,
                        size_t static_cache_max_nodes,
                        unsigned K, unsigned L, unsigned beamwidth,
                        unsigned num_threads, unsigned mem_L,
                        unsigned io_limit, float pq_filter_ratio,
                        float emb_search_ratio,
                        const HybridCacheSearchConfig& hc_config,
                        _u64 sector_len) {
    
    // Parse distance function
    diskann::Metric metric;
    if (dist_fn == "mips") {
        metric = diskann::Metric::INNER_PRODUCT;
    } else if (dist_fn == "l2") {
        metric = diskann::Metric::L2;
    } else if (dist_fn == "cosine") {
        metric = diskann::Metric::COSINE;
    } else {
        std::cerr << "Unsupported distance function: " << dist_fn << std::endl;
        return -1;
    }
    
    // Create file I/O manager
    std::shared_ptr<FileIOManager> fio_manager(new FileIOManager());
    
    // Create index - HybridCache uses Starling layout (use_graph_rep_index=false)
    // because HybridCache partitioning uses index_relayout mode=0 (STARLING_INDEX)
    diskann::DecoIndex<T> index(fio_manager, metric, false, sector_len);
    
    std::cout << "Loading index from " << index_prefix << std::endl;
    
    // Build paths
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
    
    // Load static cache
    if (!static_cache_file.empty()) {
        index.load_static_cache(static_cache_file, static_cache_max_nodes);
    }
    
    // Load queries (need to load first to get query_dim)
    std::cout << "Loading queries from " << query_file << std::endl;
    T* queries = nullptr;
    size_t query_num, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_file, queries, query_num, query_dim, query_aligned_dim);
    
    std::cout << "Loaded " << query_num << " queries of dimension " << query_dim << std::endl;
    
    // If using memory navigation graph (need to load after queries because query_dim is required)
    if (mem_L > 0 && !mem_index_path.empty()) {
        index.load_mem_index(metric, query_dim, mem_index_path, num_threads, mem_L);
        std::cout << "Memory navigation graph loaded." << std::endl;
    }
    
    // Load ground truth (if provided)
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
    
    // Prepare result buffers
    std::vector<_u64> indices(query_num * K);
    std::vector<float> distances(query_num * K);
    std::vector<QueryStats> stats(query_num);
    std::vector<HybridCacheSearchStats> hc_stats(query_num);
    
    // Set thread count
    omp_set_num_threads(num_threads);
    
    // Execute search
    auto start = std::chrono::high_resolution_clock::now();
    
    index.hybridcache_search(queries, query_num, query_aligned_dim, K, mem_L, L,
                             indices, distances, beamwidth, io_limit,
                             pq_filter_ratio, emb_search_ratio, hc_config,
                             stats.data(), hc_stats.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Search completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Average latency: " << (double)duration.count() / query_num << " ms/query" << std::endl;
    if (duration.count() > 0) {
        std::cout << "QPS: " << (double)query_num * 1000.0 / duration.count() << std::endl;
    }
    
    // Aggregate HybridCache statistics
    if (hc_config.enable_stats) {
        std::cout << "\n=== Aggregated HybridCache Statistics ===" << std::endl;
        
        // Skip first few warmup queries
        size_t warmup_cnt = std::min((size_t)20, query_num);
        
        // Print per-hop hit rate statistics
        HybridCacheSearchStats::print_aggregated_per_hop_stats(
            hc_stats.data() + warmup_cnt, query_num - warmup_cnt, 100);
        
        // Export per-hop hit rate to file (for plotting)
        if (!result_path.empty()) {
            std::string per_hop_file = result_path + "_per_hop_hit_rate.csv";
            HybridCacheSearchStats::export_per_hop_stats_to_file(
                hc_stats.data() + warmup_cnt, query_num - warmup_cnt, per_hop_file, 100);
        }
    }
    
    // Calculate recall
    if (calc_recall) {
        std::cout << "\nCalculating recall..." << std::endl;
        
        // Convert _u64 to unsigned (uint32_t) for calculate_recall
        std::vector<unsigned> indices_u32(query_num * K);
        for (size_t i = 0; i < query_num * K; i++) {
            indices_u32[i] = static_cast<unsigned>(indices[i]);
        }
        
        double recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                                   indices_u32.data(), K, K);
        std::cout << "Recall@" << K << ": " << recall << std::endl;
    }
    
    // Save results
    if (!result_path.empty()) {
        std::string idx_file = result_path + "_idx_uint32.bin";
        std::string dist_file = result_path + "_dists_float.bin";
        
        // Convert and save results
        std::vector<_u32> indices_u32(query_num * K);
        for (size_t i = 0; i < query_num * K; i++) {
            indices_u32[i] = static_cast<_u32>(indices[i]);
        }
        diskann::save_bin<_u32>(idx_file, indices_u32.data(), query_num, K);
        diskann::save_bin<float>(dist_file, distances.data(), query_num, K);
        
        std::cout << "Results saved to: " << idx_file << " and " << dist_file << std::endl;
    }
    
    // Save statistics
    if (!stats_file.empty() && hc_config.enable_stats) {
        std::ofstream stats_out(stats_file);
        if (stats_out.is_open()) {
            // CSV header
            stats_out << "query_id,phase_1_ios,phase_2_ios,phase_transition_point,"
                      << "static_cache_hits,dynamic_cache_hits,cache_misses,"
                      << "prefetch_operations,prefetch_pages_total,prefetch_useful_pages,"
                      << "phase_detection_us,cache_lookup_us,prefetch_us" << std::endl;
            
            // Data rows
            for (size_t i = 0; i < query_num; i++) {
                stats_out << i << ","
                          << hc_stats[i].phase_1_ios << ","
                          << hc_stats[i].phase_2_ios << ","
                          << hc_stats[i].phase_transition_point << ","
                          << hc_stats[i].static_cache_hits << ","
                          << hc_stats[i].dynamic_cache_hits << ","
                          << hc_stats[i].cache_misses << ","
                          << hc_stats[i].prefetch_operations << ","
                          << hc_stats[i].prefetch_pages_total << ","
                          << hc_stats[i].prefetch_useful_pages << ","
                          << hc_stats[i].phase_detection_us << ","
                          << hc_stats[i].cache_lookup_us << ","
                          << hc_stats[i].prefetch_us << std::endl;
            }
            stats_out.close();
            std::cout << "Statistics saved to: " << stats_file << std::endl;
        }
    }
    
    
    // Cleanup
    diskann::aligned_free(queries);
    if (gt_ids) delete[] gt_ids;
    if (gt_dists) delete[] gt_dists;
    
    return 0;
}

int main(int argc, char** argv) {
    std::string data_type, dist_fn, index_prefix, pq_prefix, query_file, gt_file;
    std::string result_path, stats_file, mem_index_path;
    std::string static_cache_file;
    size_t static_cache_max_nodes;
    unsigned K, L, beamwidth, num_threads, mem_L, io_limit;
    float pq_filter_ratio, emb_search_ratio;
    float hybridcache_theta;
    unsigned hybridcache_prefetch_pages, hybridcache_static_cache, hybridcache_dynamic_cache;
    bool hybridcache_enable_stats;
    _u64 sector_len;
    
    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type",
                           po::value<std::string>(&data_type)->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("dist_fn",
                           po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/cosine>");
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
        desc.add_options()("result_path",
                           po::value<std::string>(&result_path)->default_value(""),
                           "Path prefix for saving results");
        desc.add_options()("stats_file",
                           po::value<std::string>(&stats_file)->default_value(""),
                           "Statistics output file (CSV format)");
        desc.add_options()("mem_index_path",
                           po::value<std::string>(&mem_index_path)->default_value(""),
                           "Path to memory navigation index");
        desc.add_options()("static_cache_file",
                           po::value<std::string>(&static_cache_file)->default_value(""),
                           "Path to static cache node list file");
        desc.add_options()("static_cache_max_nodes",
                           po::value<size_t>(&static_cache_max_nodes)->default_value(0),
                           "Maximum number of nodes to load into static cache (0 = all)");
        desc.add_options()("K",
                           po::value<unsigned>(&K)->default_value(10),
                           "Number of neighbors to retrieve");
        desc.add_options()("L",
                           po::value<unsigned>(&L)->default_value(100),
                           "Search list size");
        desc.add_options()("beamwidth",
                           po::value<unsigned>(&beamwidth)->default_value(4),
                           "Beam width for search");
        desc.add_options()("num_threads,T",
                           po::value<unsigned>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads");
        desc.add_options()("mem_L",
                           po::value<unsigned>(&mem_L)->default_value(0),
                           "Memory navigation graph L (0 to disable)");
        desc.add_options()("io_limit",
                           po::value<unsigned>(&io_limit)->default_value(10000),
                           "Maximum I/O operations per query");
        desc.add_options()("pq_filter_ratio",
                           po::value<float>(&pq_filter_ratio)->default_value(1.2f),
                           "PQ filter ratio");
        desc.add_options()("emb_search_ratio",
                           po::value<float>(&emb_search_ratio)->default_value(0.4f),
                           "Embedding search ratio");
        desc.add_options()("hybridcache_theta",
                           po::value<float>(&hybridcache_theta)->default_value(0.3f),
                           "HybridCache phase transition theta");
        desc.add_options()("hybridcache_prefetch_pages",
                           po::value<unsigned>(&hybridcache_prefetch_pages)->default_value(3),
                           "HybridCache prefetch pages in Phase 2");
        desc.add_options()("hybridcache_static_cache",
                           po::value<unsigned>(&hybridcache_static_cache)->default_value(100),
                           "HybridCache static cache capacity");
        desc.add_options()("hybridcache_dynamic_cache",
                           po::value<unsigned>(&hybridcache_dynamic_cache)->default_value(500),
                           "HybridCache dynamic cache capacity");
        desc.add_options()("hybridcache_enable_stats",
                           po::value<bool>(&hybridcache_enable_stats)->default_value(true),
                           "Enable HybridCache statistics");
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
    
    // Build HybridCache configuration
    HybridCacheSearchConfig hc_config;
    hc_config.theta = hybridcache_theta;
    hc_config.prefetch_pages = hybridcache_prefetch_pages;
    hc_config.static_cache_capacity = hybridcache_static_cache;
    hc_config.dynamic_cache_capacity = hybridcache_dynamic_cache;
    hc_config.enable_stats = hybridcache_enable_stats;
    
    try {
        if (data_type == "float") {
            return run_hybridcache_search<float>(data_type, dist_fn, index_prefix, pq_prefix,
                                                 query_file, gt_file, result_path, stats_file,
                                                 mem_index_path, static_cache_file, static_cache_max_nodes,
                                                 K, L, beamwidth, num_threads,
                                                 mem_L, io_limit, pq_filter_ratio, emb_search_ratio,
                                                 hc_config, sector_len);
        } else if (data_type == "int8") {
            return run_hybridcache_search<int8_t>(data_type, dist_fn, index_prefix, pq_prefix,
                                                  query_file, gt_file, result_path, stats_file,
                                                  mem_index_path, static_cache_file, static_cache_max_nodes,
                                                  K, L, beamwidth, num_threads,
                                                  mem_L, io_limit, pq_filter_ratio, emb_search_ratio,
                                                  hc_config, sector_len);
        } else if (data_type == "uint8") {
            return run_hybridcache_search<uint8_t>(data_type, dist_fn, index_prefix, pq_prefix,
                                                   query_file, gt_file, result_path, stats_file,
                                                   mem_index_path, static_cache_file, static_cache_max_nodes,
                                                   K, L, beamwidth, num_threads,
                                                   mem_L, io_limit, pq_filter_ratio, emb_search_ratio,
                                                   hc_config, sector_len);
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