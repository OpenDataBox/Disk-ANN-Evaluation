
#pragma once

#include "partitioner.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <queue>
#include <set>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

namespace GP {

struct SimilarityEntry {
    unsigned node_id;
    float similarity;
    
    SimilarityEntry(unsigned id, float sim) : node_id(id), similarity(sim) {}
    
    bool operator<(const SimilarityEntry& other) const {
        return similarity > other.similarity;  // Descending order
    }
};

// Clustering configuration structure
struct ClusteringConfig {
    float similarity_threshold = 0.7f;      // Similarity threshold
    std::string distance_function = "L2";   // Distance function type (L2 or cosine)
    unsigned page_size = 0;                 // Page size (0 means use existing C value)
    bool enable_parallel = true;            // Enable parallel computation
    unsigned num_threads = 0;               // Number of threads (0 means auto-detect)
    bool verbose_logging = false;           // Verbose logging
    unsigned kmeans_iterations = 20;        
    float kmeans_tolerance = 1e-4f;         
};

struct ClusterInfo {
    std::vector<unsigned> nodes;
    unsigned page_id;
    float internal_similarity;              
    std::set<unsigned> connected_clusters; 
    
    ClusterInfo() : page_id(0), internal_similarity(0.0f) {}
};

class PerformanceProfiler {
private:
    static std::unordered_map<std::string, double> start_times_;
    static std::unordered_map<std::string, double> total_times_;
    
public:
    static void start_timer(const std::string& operation);
    static double end_timer(const std::string& operation);
    static void log_statistics(const std::string& log_file);
    static void reset_all_timers();
};


template<typename T> class VectorManager;
template<typename T> class KMeansClusterer;
class PageLayoutOptimizer;

class ClusteringPartitioner : public graph_partitioner {
private:
    ClusteringConfig config_;
    std::vector<ClusterInfo> clusters_;
    std::vector<std::vector<float>> centroids_;    
    std::string index_file_path_;                  
    std::string data_type_;                        
    
    std::shared_ptr<VectorManager<uint8_t>> vector_manager_uint8_;
    std::shared_ptr<VectorManager<float>> vector_manager_float_;
    
    struct Statistics {
        size_t total_nodes = 0;
        size_t total_clusters = 0;
        size_t cross_page_edges = 0;
        double page_utilization = 0.0;
        unsigned full_pages = 0;
        unsigned min_cluster_size = 0;
        unsigned max_cluster_size = 0;
        double clustering_time = 0.0;
        double page_ordering_time = 0.0;
        double avg_cluster_size = 0.0;
    } stats_;

public:
    
    ClusteringPartitioner(const char* indexName, 
                       const ClusteringConfig& config = ClusteringConfig(),
                       const char* data_type = "uint8",
                       bool load_disk = true, 
                       unsigned BS = 1, 
                       bool visual = false,
                       std::string freq_file = std::string(""), 
                       unsigned cut = INF,
                       bool dist_replace_freq = false, 
                       Mode mode = Mode::ALL,
                       unsigned diskann_sector_len = 4096, 
                       unsigned out_sector_len = 4096);

    
    virtual ~ClusteringPartitioner();


    void clustering_partition(const char* filename);
    
    void load_vector_data();
    void perform_kmeans_clustering();
    void optimize_page_layout();
    
    void validate_clustering();
    void compute_statistics();
    void print_statistics() const;
    
    
    const ClusteringConfig& get_config() const { return config_; }
    void set_config(const ClusteringConfig& config) { config_ = config; }
    
    const Statistics& get_statistics() const { return stats_; }

private:
    void save_partition_results(const char* filename);
    
    size_t compute_cross_page_edges();
    template<typename T>
    std::shared_ptr<VectorManager<T>> create_vector_manager(
        const std::string& index_path, size_t dimension, size_t total_nodes,
        size_t max_node_len, size_t sector_len, size_t nodes_per_sector);
};

template<typename T>
class VectorManager {
private:
    std::vector<std::vector<T>> vectors_;
    size_t dimension_;
    std::string distance_type_;

public:
    VectorManager(const std::string& index_path, size_t dim, size_t total_nodes, 
                 size_t max_node_len, size_t sector_len, size_t nodes_per_sector,
                 const std::string& distance_type) 
        : dimension_(dim), distance_type_(distance_type) {
        
        std::cout << "VectorManager: Loading vectors from disk..." << std::endl;
        std::cout << "  Index path: " << index_path << std::endl;
        std::cout << "  Total nodes: " << total_nodes << std::endl;
        std::cout << "  Dimension: " << dimension_ << std::endl;
        std::cout << "  Max node length: " << max_node_len << std::endl;
        std::cout << "  Sector length: " << sector_len << std::endl;
        std::cout << "  Nodes per sector: " << nodes_per_sector << std::endl;
        
        load_vectors_from_disk(index_path, total_nodes, max_node_len, sector_len, nodes_per_sector);
        
        std::cout << "VectorManager: Successfully loaded " << vectors_.size() << " vectors" << std::endl;
    }
    
    const std::vector<T>& get_vector(unsigned node_id) const { 
        if (node_id >= vectors_.size()) {
            throw std::out_of_range("Node ID out of range: " + std::to_string(node_id));
        }
        return vectors_[node_id]; 
    }
    
    size_t size() const { return vectors_.size(); }
    size_t dimension() const { return dimension_; }
    
    float compute_distance_to_centroid(unsigned node_id, const std::vector<float>& centroid) const {
        if (node_id >= vectors_.size()) {
            throw std::out_of_range("Node ID out of range");
        }
        
        const auto& vec = vectors_[node_id];
        
        if (distance_type_ == "L2") {
            return compute_l2_distance(vec, centroid);
        } else if (distance_type_ == "cosine") {
            return compute_cosine_distance(vec, centroid);
        } else {
            return compute_l2_distance(vec, centroid);
        }
    }
    
    float compute_distance(unsigned node_id1, unsigned node_id2) const {
        const auto& vec1 = vectors_[node_id1];
        const auto& vec2 = vectors_[node_id2];
        
        if (distance_type_ == "L2") {
            return compute_l2_distance(vec1, vec2);
        } else if (distance_type_ == "cosine") {
            return compute_cosine_distance(vec1, vec2);
        } else {
            return compute_l2_distance(vec1, vec2);
        }
    }

private:

    void load_vectors_from_disk(const std::string& index_path, size_t total_nodes,
                                size_t max_node_len, size_t sector_len, size_t nodes_per_sector) {
        std::ifstream in(index_path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open index file: " + index_path);
        }
        
        in.seekg(sector_len, std::ios::beg);
        
        vectors_.resize(total_nodes);

        size_t num_sectors = (total_nodes + nodes_per_sector - 1) / nodes_per_sector;
        

        const size_t batch_size = 10000;
        size_t sectors_processed = 0;
        
        while (sectors_processed < num_sectors) {
            size_t current_batch = std::min(batch_size, num_sectors - sectors_processed);
            
            std::unique_ptr<char[]> sector_buffer = std::make_unique<char[]>(current_batch * sector_len);
            in.read(sector_buffer.get(), current_batch * sector_len);
            
            if (!in && !in.eof()) {
                throw std::runtime_error("Error reading from index file");
            }
            
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t i = 0; i < current_batch; i++) {
                size_t sector_idx = sectors_processed + i;
                char* sector_ptr = sector_buffer.get() + i * sector_len;
                

                for (size_t j = 0; j < nodes_per_sector; j++) {
                    size_t global_node_id = sector_idx * nodes_per_sector + j;
                    if (global_node_id >= total_nodes) break;
                    
                    char* node_ptr = sector_ptr + j * max_node_len;
                
                    T* vector_data = reinterpret_cast<T*>(node_ptr);
                    
                    vectors_[global_node_id].resize(dimension_);
                    std::memcpy(vectors_[global_node_id].data(), vector_data, dimension_ * sizeof(T));
                }
            }
            
            sectors_processed += current_batch;
            
            if (sectors_processed % 50000 == 0) {
                std::cout << "  Loaded " << sectors_processed << "/" << num_sectors << " sectors..." << std::endl;
            }
        }
        
        in.close();
    }
    
    template<typename VecType>
    float compute_l2_distance(const std::vector<VecType>& vec1, const std::vector<float>& vec2) const {
        float dist = 0.0f;
        for (size_t i = 0; i < dimension_; i++) {
            float v1 = std::is_same_v<VecType, float> ? vec1[i] : static_cast<float>(vec1[i]);
            float diff = v1 - vec2[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }
    
    float compute_l2_distance(const std::vector<T>& vec1, const std::vector<T>& vec2) const {
        float dist = 0.0f;
        for (size_t i = 0; i < dimension_; i++) {
            float v1 = std::is_same_v<T, float> ? vec1[i] : static_cast<float>(vec1[i]);
            float v2 = std::is_same_v<T, float> ? vec2[i] : static_cast<float>(vec2[i]);
            float diff = v1 - v2;
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }
    
    template<typename VecType>
    float compute_cosine_distance(const std::vector<VecType>& vec1, const std::vector<float>& vec2) const {
        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        for (size_t i = 0; i < dimension_; i++) {
            float v1 = std::is_same_v<VecType, float> ? vec1[i] : static_cast<float>(vec1[i]);
            dot += v1 * vec2[i];
            norm1 += v1 * v1;
            norm2 += vec2[i] * vec2[i];
        }
        float similarity = dot / (std::sqrt(norm1) * std::sqrt(norm2) + 1e-10f);
        return 1.0f - similarity;  
    }
    
    float compute_cosine_distance(const std::vector<T>& vec1, const std::vector<T>& vec2) const {
        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        for (size_t i = 0; i < dimension_; i++) {
            float v1 = std::is_same_v<T, float> ? vec1[i] : static_cast<float>(vec1[i]);
            float v2 = std::is_same_v<T, float> ? vec2[i] : static_cast<float>(vec2[i]);
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
        float similarity = dot / (std::sqrt(norm1) * std::sqrt(norm2) + 1e-10f);
        return 1.0f - similarity;
    }
};

template<typename T>
class KMeansClusterer {
private:
    std::shared_ptr<VectorManager<T>> vector_manager_;
    unsigned k_clusters_;
    unsigned max_iterations_;
    float tolerance_;
    std::vector<std::vector<float>> centroids_;
    std::vector<unsigned> assignments_;
    std::vector<std::vector<unsigned>> clusters_;
    bool verbose_;

public:
    KMeansClusterer(std::shared_ptr<VectorManager<T>> vm, unsigned k, 
                   unsigned max_iter = 20, float tol = 1e-4f, bool verbose = false) 
        : vector_manager_(vm), k_clusters_(k), max_iterations_(max_iter), 
          tolerance_(tol), verbose_(verbose) {
        
        if (k_clusters_ == 0 || k_clusters_ > vector_manager_->size()) {
            throw std::invalid_argument("Invalid number of clusters");
        }
        
        assignments_.resize(vector_manager_->size(), 0);
        clusters_.resize(k_clusters_);
        centroids_.resize(k_clusters_);
        
        for (auto& centroid : centroids_) {
            centroid.resize(vector_manager_->dimension(), 0.0f);
        }
    }
    
    void run_kmeans(std::vector<ClusterInfo>& result_clusters) {
        std::cout << "K-means: Starting clustering with K=" << k_clusters_ << std::endl;
        

        initialize_centroids_kmeans_plus_plus();
        
        float prev_inertia = std::numeric_limits<float>::max();
        
        for (unsigned iter = 0; iter < max_iterations_; iter++) {
            bool changed = assign_points_to_clusters();
        
            update_centroids();
            
            float inertia = compute_inertia();
            
            if (verbose_ || iter % 10 == 0) {
                std::cout << "  Iteration " << iter << ": inertia=" << inertia 
                         << ", changed=" << changed << std::endl;
            }
            
            if (!changed || std::abs(prev_inertia - inertia) < tolerance_) {
                std::cout << "K-means: Converged at iteration " << iter << std::endl;
                break;
            }
            
            prev_inertia = inertia;
        }
        
        convert_to_cluster_info(result_clusters);
        
        std::cout << "K-means: Completed. Generated " << result_clusters.size() << " clusters" << std::endl;
    }

private:
    void initialize_centroids_kmeans_plus_plus() {
        std::cout << "K-means: Initializing centroids using K-means++..." << std::endl;
        
        size_t n = vector_manager_->size();
        std::vector<float> min_distances(n, std::numeric_limits<float>::max());
    
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, n - 1);
        
        size_t first_center = dis(gen);
        copy_vector_to_centroid(first_center, 0);
        

        for (unsigned k = 1; k < k_clusters_; k++) {
            
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                float dist = vector_manager_->compute_distance_to_centroid(i, centroids_[k-1]);
                if (dist < min_distances[i]) {
                    min_distances[i] = dist;
                }
            }
            
    
            double sum_sq_distances = 0.0;
            for (float d : min_distances) {
                sum_sq_distances += d * d;
            }
            
            std::uniform_real_distribution<double> prob_dis(0.0, sum_sq_distances);
            double target = prob_dis(gen);
            double cumsum = 0.0;
            size_t next_center = 0;
            
            for (size_t i = 0; i < n; i++) {
                cumsum += min_distances[i] * min_distances[i];
                if (cumsum >= target) {
                    next_center = i;
                    break;
                }
            }
            
            copy_vector_to_centroid(next_center, k);
            
            if (k % 100 == 0) {
                std::cout << "  Initialized " << k << "/" << k_clusters_ << " centroids" << std::endl;
            }
        }
    }
    

    void copy_vector_to_centroid(size_t node_id, unsigned centroid_idx) {
        const auto& vec = vector_manager_->get_vector(node_id);
        for (size_t d = 0; d < vector_manager_->dimension(); d++) {
            centroids_[centroid_idx][d] = std::is_same_v<T, float> ? vec[d] : static_cast<float>(vec[d]);
        }
    }

    bool assign_points_to_clusters() {
        bool changed = false;
        
    
        for (auto& cluster : clusters_) {
            cluster.clear();
        }
        
    
        std::vector<bool> point_changed(vector_manager_->size(), false);
        
        #pragma omp parallel for schedule(dynamic, 1000)
        for (size_t i = 0; i < vector_manager_->size(); i++) {
            float min_dist = std::numeric_limits<float>::max();
            unsigned best_cluster = 0;
            

            for (unsigned k = 0; k < k_clusters_; k++) {
                float dist = vector_manager_->compute_distance_to_centroid(i, centroids_[k]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            
            if (assignments_[i] != best_cluster) {
                point_changed[i] = true;
                assignments_[i] = best_cluster;
            }
        }
        

        for (bool ch : point_changed) {
            if (ch) {
                changed = true;
                break;
            }
        }
        
        for (size_t i = 0; i < vector_manager_->size(); i++) {
            clusters_[assignments_[i]].push_back(i);
        }
        
        return changed;
    }
    

    void update_centroids() {
        #pragma omp parallel for
        for (unsigned k = 0; k < k_clusters_; k++) {
            if (clusters_[k].empty()) {

                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<size_t> dis(0, vector_manager_->size() - 1);
                copy_vector_to_centroid(dis(gen), k);
                continue;
            }
            
            std::fill(centroids_[k].begin(), centroids_[k].end(), 0.0f);
            
            for (unsigned node_id : clusters_[k]) {
                const auto& vec = vector_manager_->get_vector(node_id);
                for (size_t d = 0; d < vector_manager_->dimension(); d++) {
                    float val = std::is_same_v<T, float> ? vec[d] : static_cast<float>(vec[d]);
                    centroids_[k][d] += val;
                }
            }
            
            float scale = 1.0f / clusters_[k].size();
            for (float& val : centroids_[k]) {
                val *= scale;
            }
        }
    }
    
    float compute_inertia() const {
        float inertia = 0.0f;
        
        #pragma omp parallel for reduction(+:inertia)
        for (size_t i = 0; i < vector_manager_->size(); i++) {
            float dist = vector_manager_->compute_distance_to_centroid(i, centroids_[assignments_[i]]);
            inertia += dist * dist;
        }
        
        return inertia;
    }
    
    
    void convert_to_cluster_info(std::vector<ClusterInfo>& result_clusters) {
        result_clusters.clear();
        result_clusters.reserve(k_clusters_);
        
        for (unsigned k = 0; k < k_clusters_; k++) {
            if (clusters_[k].empty()) continue;
            
            ClusterInfo cluster;
            cluster.nodes = clusters_[k];
            cluster.page_id = k;
            
            
            float total_similarity = 0.0f;
            unsigned count = 0;
            
            for (size_t i = 0; i < clusters_[k].size() && i < 100; i++) {
                float dist = vector_manager_->compute_distance_to_centroid(clusters_[k][i], centroids_[k]);
                total_similarity += dist;
                count++;
            }
            
            cluster.internal_similarity = count > 0 ? total_similarity / count : 0.0f;
            result_clusters.push_back(std::move(cluster));
        }
    }
};

class PageLayoutOptimizer {
private:
    const std::vector<std::vector<unsigned>>& graph_;
    unsigned page_size_;

public:
    PageLayoutOptimizer(const std::vector<std::vector<unsigned>>& graph, unsigned page_size)
        : graph_(graph), page_size_(page_size) {}
    
    void optimize_layout(std::vector<ClusterInfo>& clusters) {
        std::cout << "PageLayoutOptimizer: Optimizing page layout..." << std::endl;
        std::cout << "  Input clusters: " << clusters.size() << std::endl;
        std::cout << "  Target page size: " << page_size_ << " nodes" << std::endl;
        
        std::vector<ClusterInfo> optimized_clusters;
        optimized_clusters.reserve(clusters.size() * 2);
        
        std::unordered_set<unsigned> assigned_nodes;
        

        std::vector<ClusterInfo> small_clusters;  
        
        for (const auto& cluster : clusters) {
            if (cluster.nodes.empty()) continue;
            
            if (cluster.nodes.size() > page_size_) {
            
                split_large_cluster(cluster, optimized_clusters, small_clusters, assigned_nodes);
            } else {
            
                small_clusters.push_back(cluster);
                for (unsigned node : cluster.nodes) {
                    assigned_nodes.insert(node);
                }
            }
        }
        
        std::cout << "  After splitting: " << optimized_clusters.size() << " full pages, " 
                  << small_clusters.size() << " partial pages" << std::endl;
        
        
        fill_partial_pages(small_clusters, optimized_clusters, assigned_nodes);
        
        for (unsigned i = 0; i < optimized_clusters.size(); i++) {
            optimized_clusters[i].page_id = i;
        }
        
        clusters = std::move(optimized_clusters);
        
        unsigned total_nodes = 0;
        unsigned full_pages = 0;
        for (const auto& cluster : clusters) {
            total_nodes += cluster.nodes.size();
            if (cluster.nodes.size() == page_size_) full_pages++;
        }
        
        std::cout << "PageLayoutOptimizer: Optimization complete" << std::endl;
        std::cout << "  Output pages: " << clusters.size() << std::endl;
        std::cout << "  Full pages: " << full_pages << std::endl;
        std::cout << "  Total nodes: " << total_nodes << std::endl;
    }

private:

    void fill_partial_pages(std::vector<ClusterInfo>& small_clusters,
                           std::vector<ClusterInfo>& output,
                           std::unordered_set<unsigned>& assigned_nodes) {
        
        if (small_clusters.empty()) return;
        
        std::cout << "  Filling partial pages..." << std::endl;
        
        std::sort(small_clusters.begin(), small_clusters.end(),
                  [](const ClusterInfo& a, const ClusterInfo& b) {
                      return a.nodes.size() > b.nodes.size();
                  });
        
        std::vector<bool> used(small_clusters.size(), false);
        unsigned filled_count = 0;
        
        for (size_t i = 0; i < small_clusters.size(); i++) {
            if (used[i]) continue;
            
            ClusterInfo current_page = std::move(small_clusters[i]);
            used[i] = true;
            
            for (size_t j = i + 1; j < small_clusters.size() && current_page.nodes.size() < page_size_; j++) {
                if (used[j]) continue;
                
                size_t space_left = page_size_ - current_page.nodes.size();
                size_t available = small_clusters[j].nodes.size();
                
                if (available <= space_left) {
                    current_page.nodes.insert(current_page.nodes.end(),
                                             small_clusters[j].nodes.begin(),
                                             small_clusters[j].nodes.end());
                    used[j] = true;
                } else {
            
                    current_page.nodes.insert(current_page.nodes.end(),
                                             small_clusters[j].nodes.begin(),
                                             small_clusters[j].nodes.begin() + space_left);
                    
                    small_clusters[j].nodes.erase(small_clusters[j].nodes.begin(),
                                                  small_clusters[j].nodes.begin() + space_left);
                
                }
            }
            
            if (current_page.nodes.size() == page_size_) {
                filled_count++;
            }
            
            output.push_back(std::move(current_page));
        }
        
        std::cout << "  Filled " << filled_count << " pages to full capacity" << std::endl;
        std::cout << "  Filling complete" << std::endl;
    }
    

    void split_large_cluster(const ClusterInfo& cluster, 
                            std::vector<ClusterInfo>& full_pages,
                            std::vector<ClusterInfo>& partial_pages,
                            std::unordered_set<unsigned>& assigned_nodes) {
        
        
        std::unordered_set<unsigned> cluster_nodes(cluster.nodes.begin(), cluster.nodes.end());
        std::unordered_set<unsigned> used_in_cluster;  
        
        size_t idx = 0;
        while (idx < cluster.nodes.size()) {
        
            while (idx < cluster.nodes.size() && used_in_cluster.count(cluster.nodes[idx])) {
                idx++;
            }
            
            if (idx >= cluster.nodes.size()) break;
            
            ClusterInfo page;
            page.nodes.reserve(page_size_);
            

            size_t current_idx = idx;
            while (page.nodes.size() < page_size_ && current_idx < cluster.nodes.size()) {
                unsigned start_node = cluster.nodes[current_idx];
                
                if (!used_in_cluster.count(start_node)) {
    
                    size_t space_left = page_size_ - page.nodes.size();
                    bfs_fill_page(start_node, cluster_nodes, used_in_cluster, page.nodes, space_left);
                    
                    for (unsigned node : page.nodes) {
                        used_in_cluster.insert(node);
                    }
                }
                
                current_idx++;
            }
            
            for (unsigned node : page.nodes) {
                assigned_nodes.insert(node);
            }
            
            if (page.nodes.size() == page_size_) {
                full_pages.push_back(std::move(page));
            } else if (!page.nodes.empty()) {
                partial_pages.push_back(std::move(page));
            }
            
            idx++;
        }
    }
    
    void bfs_fill_page(unsigned start_node,
                      const std::unordered_set<unsigned>& cluster_nodes,
                      const std::unordered_set<unsigned>& used_nodes,
                      std::vector<unsigned>& page_nodes,
                      size_t max_nodes) {
        
        if (used_nodes.count(start_node)) return;
        
        std::queue<unsigned> q;
        std::unordered_set<unsigned> visited;
        
        q.push(start_node);
        visited.insert(start_node);
        
        size_t added = 0;
        
        while (!q.empty() && added < max_nodes) {
            unsigned current = q.front();
            q.pop();
            
        
            page_nodes.push_back(current);
            added++;
            
            if (added >= max_nodes) break;
            
    
            if (current < graph_.size()) {
                for (unsigned neighbor : graph_[current]) {
                    if (cluster_nodes.count(neighbor) && 
                        !used_nodes.count(neighbor) && 
                        !visited.count(neighbor)) {
                        visited.insert(neighbor);
                        q.push(neighbor);
                    }
                }
            }
        }
    }
};

} // namespace GP
