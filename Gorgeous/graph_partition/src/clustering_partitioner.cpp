//
// Clustering: Similarity-Based Disk Layout Reordering
// Simplified version - Template class definitions in header file
//

#include "clustering_partitioner.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
#include <queue>

// Include DiskANN distance functions
#include "../../include/distance.h"

namespace GP {

// Static member initialization
std::unordered_map<std::string, double> PerformanceProfiler::start_times_;
std::unordered_map<std::string, double> PerformanceProfiler::total_times_;

void PerformanceProfiler::start_timer(const std::string& operation) {
    start_times_[operation] = omp_get_wtime();
}

double PerformanceProfiler::end_timer(const std::string& operation) {
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_times_[operation];
    total_times_[operation] += elapsed;
    return elapsed;
}

void PerformanceProfiler::log_statistics(const std::string& log_file) {
    std::ofstream log(log_file, std::ios::app);
    log << "=== Clustering Performance Statistics ===" << std::endl;
    for (const auto& entry : total_times_) {
        log << entry.first << ": " << entry.second << " seconds" << std::endl;
    }
    log << "=======================================" << std::endl;
    log.close();
}

void PerformanceProfiler::reset_all_timers() {
    start_times_.clear();
    total_times_.clear();
}

// ClusteringPartitioner implementation
ClusteringPartitioner::ClusteringPartitioner(const char* indexName, 
                                       const ClusteringConfig& config,
                                       const char* data_type,
                                       bool load_disk, 
                                       unsigned BS, 
                                       bool visual,
                                       std::string freq_file, 
                                       unsigned cut,
                                       bool dist_replace_freq, 
                                       Mode mode,
                                       unsigned diskann_sector_len, 
                                       unsigned out_sector_len)
    : graph_partitioner(indexName, data_type, load_disk, BS, visual, freq_file, cut, 
                       dist_replace_freq, mode, diskann_sector_len, out_sector_len),
      config_(config), index_file_path_(indexName), data_type_(data_type) {
    
    std::cout << "Clustering partitioner initialized" << std::endl;
    
    // Set page size (number of nodes)
    if (config_.page_size == 0) {
        // Use parent class calculated C value as default page size
        config_.page_size = C;  // Directly use parent class sector capacity
    }
    std::cout << "Page size: " << config_.page_size << " nodes per page (C=" << C << ")" << std::endl;
}

ClusteringPartitioner::~ClusteringPartitioner() {
}

void ClusteringPartitioner::clustering_partition(const char* filename) {
    std::cout << "Starting Clustering partitioning..." << std::endl;
    
    // Set OpenMP thread count
    if (config_.num_threads > 0) {
        omp_set_num_threads(config_.num_threads);
        std::cout << "Using " << config_.num_threads << " threads for parallel computation" << std::endl;
    } else {
        std::cout << "Using default number of threads: " << omp_get_max_threads() << std::endl;
    }
    
    PerformanceProfiler::reset_all_timers();
    
    try {
        // Step 1: Load vector data
        std::cout << "Step 1: Loading vector data..." << std::endl;
        PerformanceProfiler::start_timer("data_loading");
        load_vector_data();
        PerformanceProfiler::end_timer("data_loading");
        
        // Step 2: K-means clustering
        std::cout << "Step 2: K-means clustering..." << std::endl;
        PerformanceProfiler::start_timer("kmeans");
        perform_kmeans_clustering();
        stats_.clustering_time = PerformanceProfiler::end_timer("kmeans");
        
        // Step 3: Page layout optimization
        std::cout << "Step 3: Page layout optimization..." << std::endl;
        PerformanceProfiler::start_timer("layout");
        optimize_page_layout();
        stats_.page_ordering_time = PerformanceProfiler::end_timer("layout");
        
        // Step 4: Validation and saving
        validate_clustering();
        compute_statistics();
        save_partition_results(filename);
        print_statistics();
        
        std::cout << "Clustering partitioning completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}

void ClusteringPartitioner::load_vector_data() {
    // Directly use parameters already loaded by parent class
    size_t total_nodes = _nd;
    size_t dimension = _dim;
    size_t max_node_len = _max_node_len;
    size_t sector_len = DISKANN_SECTOR_LEN;
    size_t nodes_per_sector = C;  // C is the number of nodes per sector
    
    std::cout << "Loading " << total_nodes << " vectors (dim=" << dimension << ")" << std::endl;
    std::cout << "Parameters from parent class:" << std::endl;
    std::cout << "  _nd (total nodes): " << _nd << std::endl;
    std::cout << "  _dim (dimension): " << _dim << std::endl;
    std::cout << "  _max_node_len: " << _max_node_len << std::endl;
    std::cout << "  DISKANN_SECTOR_LEN: " << DISKANN_SECTOR_LEN << std::endl;
    std::cout << "  C (nodes per sector): " << C << std::endl;
    
    // Create vector manager based on data type
    if (data_type_ == "uint8") {
        vector_manager_uint8_ = create_vector_manager<uint8_t>(
            index_file_path_, dimension, total_nodes, max_node_len, sector_len, nodes_per_sector);
    } else if (data_type_ == "float") {
        vector_manager_float_ = create_vector_manager<float>(
            index_file_path_, dimension, total_nodes, max_node_len, sector_len, nodes_per_sector);
    } else {
        throw std::runtime_error("Unsupported data type: " + data_type_);
    }
}

void ClusteringPartitioner::perform_kmeans_clustering() {
    size_t total_nodes = _nd;
    unsigned k_clusters = 2000;  
    
    std::cout << "K-means clustering with K=" << k_clusters << " (target ~500 nodes per cluster)" << std::endl;
    std::cout << "Expected cluster size: " << (total_nodes / k_clusters) << " nodes per cluster" << std::endl;
    
    if (data_type_ == "uint8") {
        auto kmeans = std::make_unique<KMeansClusterer<uint8_t>>(vector_manager_uint8_, k_clusters);
        kmeans->run_kmeans(clusters_);
    } else if (data_type_ == "float") {
        auto kmeans = std::make_unique<KMeansClusterer<float>>(vector_manager_float_, k_clusters);
        kmeans->run_kmeans(clusters_);
    }
    
    stats_.total_clusters = clusters_.size();
}

void ClusteringPartitioner::optimize_page_layout() {
    PageLayoutOptimizer optimizer(full_graph, config_.page_size);
    optimizer.optimize_layout(clusters_);
}

void ClusteringPartitioner::validate_clustering() {
    std::set<unsigned> assigned_nodes;
    
    for (const auto& cluster : clusters_) {
        for (unsigned node : cluster.nodes) {
            if (assigned_nodes.count(node)) {
                throw std::runtime_error("Node assigned to multiple clusters: " + std::to_string(node));
            }
            assigned_nodes.insert(node);
        }
    }
    
    if (assigned_nodes.size() != _nd) {
        throw std::runtime_error("Not all nodes assigned. Expected: " + std::to_string(_nd) + 
                                ", Got: " + std::to_string(assigned_nodes.size()));
    }
    
    std::cout << "Clustering validation passed" << std::endl;
}

void ClusteringPartitioner::compute_statistics() {
    stats_.total_nodes = _nd;
    stats_.total_clusters = clusters_.size();  // Update to optimized page count
    
    unsigned total_assigned = 0;
    unsigned full_pages = 0;
    unsigned min_size = UINT_MAX, max_size = 0;
    
    for (const auto& cluster : clusters_) {
        total_assigned += cluster.nodes.size();
        if (cluster.nodes.size() == config_.page_size) full_pages++;
        min_size = std::min(min_size, (unsigned)cluster.nodes.size());
        max_size = std::max(max_size, (unsigned)cluster.nodes.size());
    }
    
    stats_.page_utilization = (double)total_assigned / (clusters_.size() * config_.page_size);
    stats_.full_pages = full_pages;
    stats_.min_cluster_size = min_size;
    stats_.max_cluster_size = max_size;
    
    // Calculate cross-page edges
    stats_.cross_page_edges = compute_cross_page_edges();
}

size_t ClusteringPartitioner::compute_cross_page_edges() {
    // Create node-to-page mapping
    std::unordered_map<unsigned, unsigned> node_to_page;
    for (unsigned page_id = 0; page_id < clusters_.size(); page_id++) {
        for (unsigned node : clusters_[page_id].nodes) {
            node_to_page[node] = page_id;
        }
    }
    
    size_t cross_edges = 0;
    for (unsigned node = 0; node < _nd; node++) {
        unsigned node_page = node_to_page[node];
        for (unsigned neighbor : full_graph[node]) {
            unsigned neighbor_page = node_to_page[neighbor];
            if (node_page != neighbor_page) {
                cross_edges++;
            }
        }
    }
    
    return cross_edges / 2;  // Each edge is counted twice
}

void ClusteringPartitioner::save_partition_results(const char* filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open output file");
    }
    
    // Save in format expected by Starling/relayout
    _u64 C = config_.page_size;  // Nodes per page
    _u64 partition_nums = clusters_.size();  // Number of partitions
    _u64 nd = _nd;  // Total nodes (from parent class)
    
    // Write metadata
    out.write(reinterpret_cast<const char*>(&C), sizeof(_u64));
    out.write(reinterpret_cast<const char*>(&partition_nums), sizeof(_u64));
    out.write(reinterpret_cast<const char*>(&nd), sizeof(_u64));
    
    std::cout << "Saving partition with C=" << C << ", partitions=" << partition_nums 
              << ", nodes=" << nd << std::endl;
    
    // Write node list for each partition
    for (const auto& cluster : clusters_) {
        unsigned size = cluster.nodes.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(unsigned));
        out.write(reinterpret_cast<const char*>(cluster.nodes.data()), size * sizeof(unsigned));
    }
    
    // Build and write id2page mapping (node to page mapping)
    std::vector<unsigned> id2page(nd, UINT_MAX);  // Initialize to invalid value
    unsigned total_assigned = 0;
    
    for (unsigned page_id = 0; page_id < clusters_.size(); page_id++) {
        for (unsigned node_id : clusters_[page_id].nodes) {
            if (node_id < nd) {
                if (id2page[node_id] != UINT_MAX) {
                    std::cerr << "Warning: Node " << node_id << " assigned to multiple pages!" << std::endl;
                }
                id2page[node_id] = page_id;
                total_assigned++;
            } else {
                std::cerr << "Warning: Invalid node_id " << node_id << " >= " << nd << std::endl;
            }
        }
    }
    
    // Check if all nodes are assigned
    if (total_assigned != nd) {
        std::cerr << "Warning: Only " << total_assigned << "/" << nd << " nodes assigned to pages!" << std::endl;
        // Assign unassigned nodes to the last page
        for (unsigned i = 0; i < nd; i++) {
            if (id2page[i] == UINT_MAX) {
                id2page[i] = clusters_.size() - 1;  // Assign to last page
                std::cerr << "  Assigning unassigned node " << i << " to last page" << std::endl;
            }
        }
    }
    
    out.write(reinterpret_cast<const char*>(id2page.data()), nd * sizeof(unsigned));
    
    out.close();
    std::cout << "Results saved to " << filename << " (Starling format)" << std::endl;
}

void ClusteringPartitioner::print_statistics() const {
    std::cout << "\n=== Clustering Statistics ===" << std::endl;
    std::cout << "Total nodes: " << stats_.total_nodes << std::endl;
    std::cout << "Total pages: " << stats_.total_clusters << std::endl;
    std::cout << "Average cluster size: " << (double)stats_.total_nodes / stats_.total_clusters << " nodes" << std::endl;
    std::cout << "Target cluster size: ~500 nodes (K=2000)" << std::endl;
    std::cout << "Page size: " << config_.page_size << " nodes per page" << std::endl;
    std::cout << "Page utilization: " << (stats_.page_utilization * 100.0) << "%" << std::endl;
    std::cout << "Full pages: " << stats_.full_pages << "/" << stats_.total_clusters << std::endl;
    std::cout << "Page size range: [" << stats_.min_cluster_size << ", " << stats_.max_cluster_size << "]" << std::endl;
    std::cout << "Cross-page edges: " << stats_.cross_page_edges << std::endl;
    std::cout << "Cross-page edge ratio: " << (double)stats_.cross_page_edges / (stats_.total_nodes * 50) * 100.0 << "%" << std::endl;
    std::cout << "Clustering time: " << stats_.clustering_time << "s" << std::endl;
    std::cout << "Layout time: " << stats_.page_ordering_time << "s" << std::endl;
    std::cout << "=========================" << std::endl;
}

template<typename T>
std::shared_ptr<VectorManager<T>> ClusteringPartitioner::create_vector_manager(
    const std::string& index_path, size_t dimension, size_t total_nodes,
    size_t max_node_len, size_t sector_len, size_t nodes_per_sector) {
    
    return std::make_shared<VectorManager<T>>(
        index_path, dimension, total_nodes, max_node_len, 
        sector_len, nodes_per_sector, config_.distance_function);
}

template std::shared_ptr<VectorManager<uint8_t>> ClusteringPartitioner::create_vector_manager<uint8_t>(
    const std::string&, size_t, size_t, size_t, size_t, size_t);
template std::shared_ptr<VectorManager<float>> ClusteringPartitioner::create_vector_manager<float>(
    const std::string&, size_t, size_t, size_t, size_t, size_t);

} // namespace GP