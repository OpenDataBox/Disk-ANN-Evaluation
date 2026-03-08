#ifndef BASELINE_IO_STATS_H
#define BASELINE_IO_STATS_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "tsl/robin_set.h"

namespace diskann {

// Baseline数据加载和IO统计类
class BaselineIOStats {
public:
  // 构造函数：加载baseline文件
  BaselineIOStats(const std::string& data_dir, size_t page_size_kb = 16) 
    : page_size_bytes_(page_size_kb * 1024), loaded_(false) {
    load_baseline_data(data_dir);
  }

  // 检查是否成功加载
  bool is_loaded() const { return loaded_; }

  // 获取查询数量
  size_t get_query_count() const { return exact_data_nodes_.size(); }

  // 计算单次查询的磁盘IO利用率
  // 磁盘利用率 = 命中baseline的字节数 / 实际从磁盘读取的总字节数
  size_t calculate_hit_bytes(
      const std::vector<unsigned>& accessed_exact_nodes,
      const std::vector<unsigned>& accessed_adj_nodes,
      size_t query_id,
      size_t exact_node_bytes = 1024,  // 128维 * 4字节
      size_t adj_node_bytes = 196) const {  // 48邻居 * 4字节+邻居数量4Byte
    
    if (query_id >= exact_data_nodes_.size()) {
      return 0;
    }

    // 获取baseline中的节点集合
    const auto& baseline_exact = exact_data_nodes_[query_id];
    const auto& baseline_adj = adjacency_nodes_[query_id];
    
    tsl::robin_set<unsigned> baseline_exact_set(baseline_exact.begin(), baseline_exact.end());
    tsl::robin_set<unsigned> baseline_adj_set(baseline_adj.begin(), baseline_adj.end());

    // 计算精确向量命中数
    size_t hit_exact_nodes = 0;
    for (unsigned node_id : accessed_exact_nodes) {
      if (baseline_exact_set.find(node_id) != baseline_exact_set.end()) {
        hit_exact_nodes++;
      }
    }

    // 计算邻居表命中数
    size_t hit_adj_nodes = 0;
    for (unsigned node_id : accessed_adj_nodes) {
      if (baseline_adj_set.find(node_id) != baseline_adj_set.end()) {
        hit_adj_nodes++;
      }
    }

    // 返回命中baseline的总字节数
    return hit_exact_nodes * exact_node_bytes + hit_adj_nodes * adj_node_bytes;
  }

  // 计算磁盘IO统计（每个query）
  struct DiskUtilization {
    size_t total_disk_bytes;   // 总IO磁盘大小 = num_pages_read × 16KB
    size_t hit_disk_bytes;     // 命中baseline的字节数
  };

  DiskUtilization calculate_disk_utilization(
      const std::vector<unsigned>& accessed_exact_nodes,
      const std::vector<unsigned>& accessed_adj_nodes,
      size_t query_id,
      size_t num_pages_read,
      size_t exact_node_bytes = 1024,
      size_t adj_node_bytes = 196) const {
    
    DiskUtilization result;
    
    // 实际从磁盘读取的总字节数
    result.total_disk_bytes = num_pages_read * page_size_bytes_;
    
    // 命中baseline的字节数
    result.hit_disk_bytes = calculate_hit_bytes(
        accessed_exact_nodes, accessed_adj_nodes, query_id, 
        exact_node_bytes, adj_node_bytes);
    
    return result;
  }

private:
  size_t page_size_bytes_;  // Page大小（字节）
  bool loaded_;

  // 存储baseline数据：每个查询访问的节点列表
  std::vector<std::vector<unsigned>> exact_data_nodes_;
  std::vector<std::vector<unsigned>> adjacency_nodes_;

  // 加载baseline文件
  void load_baseline_data(const std::string& data_dir) {
    std::string exact_file = data_dir + "/exact_data_nodes.txt";
    std::string adj_file = data_dir + "/adjacency_nodes.txt";

    if (!load_node_file(exact_file, exact_data_nodes_)) {
      std::cerr << "Failed to load " << exact_file << std::endl;
      return;
    }

    if (!load_node_file(adj_file, adjacency_nodes_)) {
      std::cerr << "Failed to load " << adj_file << std::endl;
      return;
    }

    if (exact_data_nodes_.size() != adjacency_nodes_.size()) {
      std::cerr << "Mismatch in query counts: exact=" << exact_data_nodes_.size()
                << " adjacency=" << adjacency_nodes_.size() << std::endl;
      return;
    }

    loaded_ = true;
    std::cout << "Loaded baseline data: " << exact_data_nodes_.size() 
              << " queries" << std::endl;
  }

  // 加载单个节点文件
  bool load_node_file(const std::string& filename, 
                      std::vector<std::vector<unsigned>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      return false;
    }

    std::string line;
    while (std::getline(file, line)) {
      std::vector<unsigned> nodes;
      std::stringstream ss(line);
      std::string token;
      
      bool first = true;
      while (std::getline(ss, token, ',')) {
        // 去除空格
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        
        if (first) {
          // 第一个是节点数量，跳过
          first = false;
        } else {
          nodes.push_back(std::stoul(token));
        }
      }
      
      data.push_back(nodes);
    }

    file.close();
    return true;
  }
};

} // namespace diskann

#endif // BASELINE_IO_STATS_H
