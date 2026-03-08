// Test program for baseline IO statistics functionality
#include <iostream>
#include <vector>
#include <fstream>
#include "pq_flash_index.h"
#include "baseline_io_stats.h"

using namespace diskann;

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cout << "Usage: " << argv[0] 
              << " <index_prefix> <query_file> <k> <L> [baseline_data_dir]" 
              << std::endl;
    std::cout << "  baseline_data_dir: directory containing baseline files (optional)" << std::endl;
    return 1;
  }

  std::string index_prefix = argv[1];
  std::string query_file = argv[2];
  unsigned k = std::atoi(argv[3]);
  unsigned L = std::atoi(argv[4]);
  
  std::string baseline_dir = "";
  bool use_baseline = false;
  if (argc >= 6) {
    baseline_dir = argv[5];
    use_baseline = true;
  }

  // Load baseline data (if provided)
  BaselineIOStats* baseline_stats = nullptr;
  if (use_baseline) {
    baseline_stats = new BaselineIOStats(baseline_dir, 16);  // 16KB page
    if (!baseline_stats->is_loaded()) {
      std::cerr << "Failed to load baseline data from " << baseline_dir << std::endl;
      delete baseline_stats;
      return 1;
    }
    std::cout << "Loaded baseline data: " << baseline_stats->get_query_count() 
              << " queries" << std::endl;
  }

  // Load index
  std::cout << "Loading index from " << index_prefix << std::endl;
  // ... (actual index loading code needed here)

  // Load queries
  std::cout << "Loading queries from " << query_file << std::endl;
  // ... (actual query loading code needed here)

  // Run queries
  unsigned num_queries = 1000;  // assume 1000 queries
  std::vector<QueryStats> stats(num_queries);
  
  std::cout << "\nRunning " << num_queries << " queries..." << std::endl;
  
  // Example: run queries and collect statistics
  for (unsigned i = 0; i < num_queries; i++) {
    // ... actual query code
    // index->page_search_with_baseline(query, k, mem_L, L, indices, distances,
    //                                   beam_width, io_limit, use_ratio, &stats[i],
    //                                   baseline_stats, i);
    
    if ((i + 1) % 100 == 0) {
      std::cout << "Completed " << (i + 1) << " queries" << std::endl;
    }
  }

  // Calculate and output statistics
  if (use_baseline) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Baseline IO Hit Rate Statistics" << std::endl;
    std::cout << "========================================" << std::endl;
    
    size_t total_disk_bytes = 0;
    size_t total_hit_disk_bytes = 0;
    
    for (unsigned i = 0; i < num_queries; i++) {
      total_disk_bytes += stats[i].baseline_total_disk_bytes;
      total_hit_disk_bytes += stats[i].baseline_hit_disk_bytes;
    }
    
    double overall_hit_ratio = (total_disk_bytes > 0) ? 
                               (double)total_hit_disk_bytes / total_disk_bytes : 0.0;
    
    std::cout << "Number of queries: " << num_queries << std::endl;
    std::cout << "Total disk bytes accessed: " << total_disk_bytes 
              << " (" << (total_disk_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "Total disk bytes hit: " << total_hit_disk_bytes 
              << " (" << (total_hit_disk_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "Overall disk hit ratio: " << (overall_hit_ratio * 100.0) << "%" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Save detailed results to file
    std::ofstream result_file("baseline_hit_rate_results.txt");
    if (result_file.is_open()) {
      result_file << "query_id,total_disk_bytes,hit_disk_bytes,hit_ratio\n";
      for (unsigned i = 0; i < num_queries; i++) {
        double hit_ratio = (stats[i].baseline_total_disk_bytes > 0) ?
                          (double)stats[i].baseline_hit_disk_bytes / stats[i].baseline_total_disk_bytes : 0.0;
        result_file << i << ","
                    << stats[i].baseline_total_disk_bytes << ","
                    << stats[i].baseline_hit_disk_bytes << ","
                    << hit_ratio << "\n";
      }
      result_file.close();
      std::cout << "\nDetailed results saved to baseline_hit_rate_results.txt" << std::endl;
    }
  }

  if (baseline_stats != nullptr) {
    delete baseline_stats;
  }

  return 0;
}

