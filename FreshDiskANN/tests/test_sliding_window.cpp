
#include <index.h>
#include <future>
#include <Neighbor_Tag.h>
#include <numeric>
#include <omp.h>
#include <shard.h>
#include <string.h>
#include <sync_index.h>
#include <time.h>
#include <timer.h>
#include <cstring>
#include <iomanip>
#include <fstream>

#include "aux_utils.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

#define NUM_INSERT_THREADS 9
#define NUM_DELETE_THREADS 1
#define NUM_SEARCH_THREADS 16

#define MERGE_ROUND 20
#define UPDATE_QUERY_INTERVAL_MS 200
#define MERGE_QUERY_INTERVAL_S 10

std::ofstream g_log_file;

struct TimeStats {
  double search_time = 0;       // S: Search time
  double update_time = 0;       // U: Update (insert + delete) time
  double maintenance_time = 0;  // M: Maintenance (merge) time
  double total_time = 0;        // T: Total time

  void print_summary(std::ostream& out) const {
    out << "\n========== Time Summary (seconds) ==========" << std::endl;
    out << std::fixed << std::setprecision(2);
    out << "S (Search):      " << std::setw(10) << search_time << " s"
        << std::endl;
    out << "U (Update):      " << std::setw(10) << update_time << " s"
        << std::endl;
    out << "M (Maintenance): " << std::setw(10) << maintenance_time << " s"
        << std::endl;
    out << "T (Total):       " << std::setw(10) << total_time << " s"
        << std::endl;
    out << "=============================================" << std::endl;
  }

  void print_summary_hours(std::ostream& out) const {
    out << "\n========== Time Summary (hours) ==========" << std::endl;
    out << std::fixed << std::setprecision(4);
    out << "S (Search):      " << std::setw(10) << search_time / 3600.0 << " h"
        << std::endl;
    out << "U (Update):      " << std::setw(10) << update_time / 3600.0 << " h"
        << std::endl;
    out << "M (Maintenance): " << std::setw(10) << maintenance_time / 3600.0
        << " h" << std::endl;
    out << "T (Total):       " << std::setw(10) << total_time / 3600.0 << " h"
        << std::endl;
    out << "===========================================" << std::endl;
  }

  void print_all() const {
    print_summary(std::cout);
    print_summary_hours(std::cout);
    if (g_log_file.is_open()) {
      print_summary(g_log_file);
      print_summary_hours(g_log_file);
    }
  }
};

// 双输出函数：同时输出到控制台和日志文件
void dual_print(const std::string& msg) {
  std::cout << msg;
  if (g_log_file.is_open()) {
    g_log_file << msg;
    g_log_file.flush();
  }
}

template<typename T, typename TagT>
double sync_search_kernel(T* query, size_t query_num, size_t query_aligned_dim,
                          const int recall_at, std::vector<_u64> Lvec,
                          diskann::SyncIndex<T, TagT>& sync_index,
                          const std::string&           truthset_file) {
  unsigned* gt_ids = NULL;
  float*    gt_dists = NULL;
  size_t    gt_num, gt_dim;
  diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);

  // 自动适配：如果GT数量大于query数量，只使用前query_num个GT
  size_t actual_gt_num = std::min(gt_num, query_num);
  if (gt_num > query_num) {
    std::cout << "Note: GT file has " << gt_num
              << " queries, but only using first " << query_num
              << " to match query file" << std::endl;
  } else if (gt_num < query_num) {
    std::cerr << "Warning: GT file has only " << gt_num
              << " queries, but query file has " << query_num << " queries!"
              << std::endl;
  }

  std::string       recall_string = "Recall@" + std::to_string(recall_at);
  std::stringstream header;
  header << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18)
         << "Mean Latency (ms)" << std::setw(15) << "99.9 Latency"
         << std::setw(12) << recall_string << std::endl;
  header << "==============================================================="
            "==============="
         << std::endl;
  dual_print(header.str());

  float* query_result_dists = new float[recall_at * query_num];
  TagT*  query_result_tags = new TagT[recall_at * query_num];

  std::stringstream debug_msg;
  debug_msg << "[SEARCH] Starting search with " << Lvec.size() << " L values, "
            << query_num << " queries, recall@" << recall_at << std::endl;
  dual_print(debug_msg.str());

  double total_search_time = 0;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    debug_msg.str("");
    debug_msg << "[SEARCH] Testing L=" << Lvec[test_id] << " (test "
              << (test_id + 1) << "/" << Lvec.size() << ")" << std::endl;
    dual_print(debug_msg.str());

    memset(query_result_dists, 0, sizeof(float) * recall_at * query_num);
    memset(query_result_tags, 0, sizeof(TagT) * recall_at * query_num);

    _u64                L = Lvec[test_id];
    std::vector<double> latency_stats(query_num, 0);
    auto                s = std::chrono::high_resolution_clock::now();

    try {
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {
        auto qs = std::chrono::high_resolution_clock::now();
        sync_index.search_async(query + i * query_aligned_dim, recall_at,
                                (_u32) L, query_result_tags + i * recall_at,
                                query_result_dists + i * recall_at);
        auto qe = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = qe - qs;
        latency_stats[i] = diff.count() * 1000;
      }
    } catch (const std::exception& e) {
      debug_msg.str("");
      debug_msg << "[SEARCH] ERROR during search: " << e.what() << std::endl;
      dual_print(debug_msg.str());
      throw;
    }

    debug_msg.str("");
    debug_msg << "[SEARCH] Completed all queries for L=" << L << std::endl;
    dual_print(debug_msg.str());
    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;
    float                         qps = (float) (query_num / diff.count());
    total_search_time += diff.count();

    // 计算recall，使用actual_gt_num（自动适配GT数量）
    tsl::robin_set<unsigned> empty_delete_list;
    float                    recall = (float) diskann::calculate_recall(
                           (_u32) actual_gt_num, gt_ids, gt_dists, (_u32) gt_dim,
                           query_result_tags, (_u32) recall_at, (_u32) recall_at,
                           empty_delete_list);

    std::sort(latency_stats.begin(), latency_stats.end());

    std::stringstream result;
    result << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
           << std::accumulate(latency_stats.begin(), latency_stats.end(), 0) /
                  (float) query_num
           << std::setw(15) << (float) latency_stats[(_u64) (0.999 * query_num)]
           << std::setw(12) << recall << std::endl;
    dual_print(result.str());
  }

  delete[] query_result_dists;
  delete[] query_result_tags;
  delete[] gt_ids;
  delete[] gt_dists;

  return total_search_time;
}

template<typename T, typename TagT>
double insertion_kernel(T* data_load, diskann::SyncIndex<T, TagT>& sync_index,
                        std::vector<TagT>& insert_vec, size_t aligned_dim) {
  std::stringstream msg;
  msg << "[INSERT] Starting insertion of " << insert_vec.size() << " points..."
      << std::endl;
  dual_print(msg.str());

  diskann::Timer timer;
  try {
#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
    for (_s64 i = 0; i < (_s64) insert_vec.size(); i++) {
      sync_index.insert(data_load + aligned_dim * insert_vec[i], insert_vec[i]);
      if (i > 0 && i % 500 == 0) {
#pragma omp critical
        {
          std::stringstream progress;
          progress << "[INSERT] Progress: " << i << "/" << insert_vec.size()
                   << std::endl;
          dual_print(progress.str());
        }
      }
    }
  } catch (const std::exception& e) {
    msg.str("");
    msg << "[INSERT] ERROR: " << e.what() << std::endl;
    dual_print(msg.str());
    throw;
  }

  float time_secs = timer.elapsed() / 1.0e6f;
  msg.str("");
  msg << "[INSERT] Completed " << insert_vec.size() << " points in "
      << time_secs << "s" << std::endl;
  dual_print(msg.str());
  return time_secs;
}

template<typename T, typename TagT>
double delete_kernel(diskann::SyncIndex<T, TagT>& sync_index,
                     std::vector<TagT>&           delete_vec) {
  std::stringstream msg;
  msg << "[DELETE] Starting deletion of " << delete_vec.size() << " points..."
      << std::endl;
  dual_print(msg.str());

  diskann::Timer timer;
  try {
#pragma omp parallel for num_threads(NUM_DELETE_THREADS)
    for (_s64 i = 0; i < (_s64) delete_vec.size(); ++i) {
      sync_index.lazy_delete(delete_vec[i]);
    }
  } catch (const std::exception& e) {
    msg.str("");
    msg << "[DELETE] ERROR: " << e.what() << std::endl;
    dual_print(msg.str());
    throw;
  }

  float time_secs = timer.elapsed() / 1.0e6f;
  msg.str("");
  msg << "[DELETE] Completed " << delete_vec.size() << " points in "
      << time_secs << "s" << std::endl;
  dual_print(msg.str());
  return time_secs;
}

template<typename T, typename TagT>
void test(const std::string& data_path, const unsigned L_mem,
          const unsigned R_mem, const float alpha_mem, const unsigned L_disk,
          const unsigned R_disk, const float alpha_disk,
          const size_t initial_points, const size_t batch_size,
          const size_t num_batches, const size_t num_shards,
          const unsigned num_pq_chunks, const unsigned nodes_to_cache,
          const std::string& save_path, const std::string& query_file,
          const std::string& gt_prefix, const int recall_at,
          std::vector<_u64> Lvec, const unsigned beam_width) {
  // 检测是否是gist数据集
  // gist数据集只做batch后的查询，不做并发查询
  bool is_gist = (save_path.find("gist") != std::string::npos);

  // 时间统计
  TimeStats time_stats;
  auto      total_start = std::chrono::high_resolution_clock::now();

  diskann::Parameters paras;
  paras.Set<unsigned>("L_mem", L_mem);
  paras.Set<unsigned>("R_mem", R_mem);
  paras.Set<float>("alpha_mem", alpha_mem);
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", R_disk);
  paras.Set<float>("alpha_disk", alpha_disk);
  paras.Set<unsigned>("C", 1500);
  paras.Set<unsigned>("beamwidth", beam_width);
  paras.Set<unsigned>("num_pq_chunks", num_pq_chunks);
  paras.Set<unsigned>("nodes_to_cache", nodes_to_cache);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  // 加载完整数据（需要包含所有要插入的点）
  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  std::stringstream msg;
  msg << "Loaded full data: " << num_points << " points, dim=" << dim
      << std::endl;
  dual_print(msg.str());

  // 检查数据量是否足够
  size_t required_points = initial_points + num_batches * batch_size;
  if (num_points < required_points) {
    std::cerr << "Error: data file has " << num_points << " points, but need "
              << required_points << " points for sliding window test."
              << std::endl;
    exit(-1);
  }

  // 创建SyncIndex，max_points需要足够大
  // 考虑到：1) lazy delete不会立即释放空间 2) short_term_index需要空间 3)
  // merge过程中的临时空间 保守估计：initial_points + 所有batch的插入 + 20%余量
  size_t max_points = initial_points + num_batches * batch_size;
  max_points = (size_t) (max_points * 1.3);  // 额外30%余量

  msg.str("");
  msg << "Creating SyncIndex with max_points=" << max_points
      << " (per shard: " << (max_points / num_shards) << ")" << std::endl;
  dual_print(msg.str());

  // Create SyncIndex with 4 search threads per shard (hardcoded)
  diskann::SyncIndex<T, TagT> sync_index(max_points, dim, num_shards, paras,
                                         4,  // num_search_threads_per_shard
                                         save_path);

  // 检查索引文件是否存在
  std::string shard0_path = save_path + "0_lti_disk.index";
  bool        index_exists = file_exists(shard0_path);

  size_t res = 0;
  if (index_exists) {
    // 索引存在，尝试加载
    msg.str("");
    msg << "Loading existing index from " << save_path << "..." << std::endl;
    dual_print(msg.str());

    res = sync_index.load(save_path);
    msg.str("");
    msg << "Sync index loaded " << res << " points" << std::endl;
    dual_print(msg.str());
  }

  if (!index_exists || res == 0 || res < initial_points) {
    // 索引不存在或不完整，从数据文件构建初始索引
    msg.str("");
    msg << "Building initial index from data file..." << std::endl;
    dual_print(msg.str());

    // 准备 tags
    std::vector<TagT> tags(initial_points);
    std::iota(tags.begin(), tags.end(), 0);

    diskann::Timer build_timer;
    sync_index.build(data_path.c_str(), initial_points, tags);
    msg.str("");
    msg << "Built initial index with " << initial_points << " points in "
        << build_timer.elapsed() / 1.0e6f << "s" << std::endl;
    dual_print(msg.str());

    // 注意：build()已经在内部调用了disk_index->load()，所以这里不需要再次load
  }

  // 加载query
  T*     query = NULL;
  size_t query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);
  msg.str("");
  msg << "Loaded " << query_num << " queries" << std::endl;
  dual_print(msg.str());

  // 初始搜索测试
  std::string gt_file = gt_prefix + "/gt_0.bin";
  dual_print("\n========== Initial Search (batch 0) ==========\n");

  // 所有数据集都执行Initial Search
  time_stats.search_time +=
      sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lvec,
                         sync_index, gt_file);

  // 滑动窗口循环
  size_t total_queries = 0;
  for (size_t batch = 0; batch < num_batches; batch++) {
    size_t update_queries = 0;
    size_t merge_queries = 0;

    msg.str("");
    msg << "\n========== Batch " << (batch + 1) << "/" << num_batches
        << " ==========" << std::endl;
    dual_print(msg.str());

    std::vector<TagT> insert_vec;
    std::vector<TagT> delete_vec;

    // 要删除的点: [batch * batch_size, (batch+1) * batch_size)
    for (size_t i = 0; i < batch_size; i++) {
      delete_vec.push_back(batch * batch_size + i);
    }

    // 要插入的点: [initial_points + batch * batch_size, initial_points +
    // (batch+1) * batch_size)
    for (size_t i = 0; i < batch_size; i++) {
      insert_vec.push_back(initial_points + batch * batch_size + i);
    }

    msg.str("");
    msg << "Deleting points [" << batch * batch_size << ", "
        << (batch + 1) * batch_size << ")" << std::endl;
    msg << "Inserting points [" << initial_points + batch * batch_size << ", "
        << initial_points + (batch + 1) * batch_size << ")" << std::endl;
    dual_print(msg.str());

    // 并发执行insert、delete和查询
    double batch_update_time = 0;
    double pure_update_time = 0;  // 纯粹的insert+delete时间
    {
      auto update_start = std::chrono::high_resolution_clock::now();

      std::future<double> insert_future =
          std::async(std::launch::async, insertion_kernel<T, TagT>, data_load,
                     std::ref(sync_index), std::ref(insert_vec), aligned_dim);

      std::future<double> delete_future =
          std::async(std::launch::async, delete_kernel<T, TagT>,
                     std::ref(sync_index), std::ref(delete_vec));

      // 立即开始持续查询，直到更新完成
      if (!is_gist) {
        msg.str("");
        msg << "Starting concurrent queries during update (L=50)..."
            << std::endl;
        dual_print(msg.str());

        try {
          // 使用L=50进行查询
          _u64   quick_L = 50;
          float* quick_dists = new float[recall_at * query_num];
          TagT*  quick_tags = new TagT[recall_at * query_num];

          int                query_round = 0;
          std::future_status insert_status, delete_status;

          // 持续查询直到更新完成
          do {
            query_round++;
            auto qs = std::chrono::high_resolution_clock::now();

            // 并行执行所有query
            std::atomic<bool> has_error(false);

#pragma omp parallel for num_threads(NUM_SEARCH_THREADS)
            for (int64_t i = 0; i < (int64_t) query_num; i++) {
              if (!has_error.load()) {
                try {
                  sync_index.search_async(
                      query + i * query_aligned_dim, recall_at, (_u32) quick_L,
                      quick_tags + i * recall_at, quick_dists + i * recall_at);
                } catch (...) {
                  has_error.store(true);
                }
              }
            }

            if (has_error.load()) {
              throw std::runtime_error(
                  "Query failed during parallel execution");
            }

            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> qdiff = qe - qs;
            float qps = (float) (query_num / qdiff.count());

            msg.str("");
            msg << "  Query round " << query_round << ": QPS=" << qps
                << ", Latency=" << (qdiff.count() * 1000.0 / query_num) << "ms";
            dual_print(msg.str());

            update_queries += query_num;

            // 检查更新是否完成
            insert_status =
                insert_future.wait_for(std::chrono::milliseconds(0));
            delete_status =
                delete_future.wait_for(std::chrono::milliseconds(0));

            if (insert_status != std::future_status::ready ||
                delete_status != std::future_status::ready) {
              msg.str("");
              msg << ", insert="
                  << (insert_status == std::future_status::ready ? "done"
                                                                 : "running")
                  << ", delete="
                  << (delete_status == std::future_status::ready ? "done"
                                                                 : "running")
                  << std::endl;
              dual_print(msg.str());
            } else {
              msg.str("");
              msg << ", update completed" << std::endl;
              dual_print(msg.str());
            }

          } while (insert_status != std::future_status::ready ||
                   delete_status != std::future_status::ready);

          delete[] quick_dists;
          delete[] quick_tags;

          msg.str("");
          msg << "Completed " << query_round << " query rounds during update"
              << std::endl;
          dual_print(msg.str());

        } catch (const std::exception& e) {
          msg.str("");
          msg << "ERROR during concurrent query: " << e.what() << std::endl;
          dual_print(msg.str());
        }
      }

      // 等待更新完成
      pure_update_time = insert_future.get();
      delete_future.get();

      auto update_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> update_diff = update_end - update_start;
      batch_update_time = update_diff.count();
    }
    time_stats.update_time += pure_update_time;  // 使用纯粹的update时间
    msg.str("");
    msg << "Update time: " << std::fixed << std::setprecision(2)
        << pure_update_time << "s (wall-clock: " << batch_update_time << "s)"
        << std::endl;
    dual_print(msg.str());

    // 检查是否需要 merge（每 MERGE_ROUND 个 batch）
    bool   should_merge = ((batch + 1) % MERGE_ROUND == 0);
    double batch_merge_time = 0;

    if (should_merge) {
      msg.str("");
      msg << "\n========== Triggering Merge (batch " << (batch + 1)
          << ", every " << MERGE_ROUND << " batches) ==========" << std::endl;
      dual_print(msg.str());

      auto merge_start = std::chrono::high_resolution_clock::now();

      // 异步执行 merge，期间每10秒查询一次
      std::future<int> merge_future = std::async(std::launch::async, [&]() {
        try {
          msg.str("");
          msg << "[MERGE] Starting merge_all..." << std::endl;
          dual_print(msg.str());

          int result = sync_index.merge_all(save_path);

          msg.str("");
          msg << "[MERGE] merge_all completed with result=" << result
              << std::endl;
          dual_print(msg.str());

          return result;
        } catch (const std::exception& e) {
          std::cerr << "Merge failed with exception: " << e.what() << std::endl;
          return -1;
        } catch (...) {
          std::cerr << "Merge failed with unknown exception" << std::endl;
          return -1;
        }
      });

      // 等待merge完成，期间每10秒查询一次
      std::future_status merge_status;
      do {
        merge_status =
            merge_future.wait_for(std::chrono::seconds(MERGE_QUERY_INTERVAL_S));

        if (merge_status == std::future_status::timeout) {
          if (is_gist) {
            msg.str("");
            msg << "[MERGE-QUERY] Merge in progress [SKIPPED for gist]..."
                << std::endl;
            dual_print(msg.str());
          } else {
            msg.str("");
            msg << "[MERGE-QUERY] Merge in progress, running query (L=50)..."
                << std::endl;
            dual_print(msg.str());

            try {
              // 执行一次快速查询（使用L=50）
              _u64   quick_L = 50;
              float* quick_dists = new float[recall_at * query_num];
              TagT*  quick_tags = new TagT[recall_at * query_num];

              msg.str("");
              msg << "[MERGE-QUERY] Memory allocated, starting queries..."
                  << std::endl;
              dual_print(msg.str());

              auto qs = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS)
              for (int64_t i = 0; i < (int64_t) query_num; i++) {
                sync_index.search_async(
                    query + i * query_aligned_dim, recall_at, (_u32) quick_L,
                    quick_tags + i * recall_at, quick_dists + i * recall_at);
              }
              auto qe = std::chrono::high_resolution_clock::now();
              std::chrono::duration<double> qdiff = qe - qs;
              float qps = (float) (query_num / qdiff.count());

              msg.str("");
              msg << "[MERGE-QUERY] Queries completed: L=" << quick_L
                  << ", QPS=" << qps
                  << ", Latency=" << (qdiff.count() * 1000.0 / query_num)
                  << "ms" << std::endl;
              dual_print(msg.str());

              delete[] quick_dists;
              delete[] quick_tags;
              merge_queries += query_num;

              msg.str("");
              msg << "[MERGE-QUERY] Memory freed" << std::endl;
              dual_print(msg.str());
            } catch (const std::exception& e) {
              msg.str("");
              msg << "[MERGE-QUERY] ERROR: " << e.what() << std::endl;
              dual_print(msg.str());
            } catch (...) {
              msg.str("");
              msg << "[MERGE-QUERY] ERROR: Unknown exception" << std::endl;
              dual_print(msg.str());
            }
          }
        }
      } while (merge_status != std::future_status::ready);

      // 检查merge结果
      int merge_result = merge_future.get();
      if (merge_result != 0) {
        std::cerr << "Merge failed with error code: " << merge_result
                  << std::endl;
        g_log_file.close();
        return;
      }

      auto merge_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> merge_diff = merge_end - merge_start;
      batch_merge_time = merge_diff.count();
      time_stats.maintenance_time += batch_merge_time;

      msg.str("");
      msg << "Merge completed in " << std::fixed << std::setprecision(2)
          << batch_merge_time << "s" << std::endl;
      dual_print(msg.str());
    }

    // 搜索并计算recall，统计S时间
    size_t gt_offset = (batch + 1) * batch_size;
    gt_file = gt_prefix + "/gt_" + std::to_string(gt_offset) + ".bin";
    msg.str("");
    msg << "Using ground truth: " << gt_file << std::endl;
    dual_print(msg.str());
    double batch_search_time =
        sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lvec,
                           sync_index, gt_file);
    time_stats.search_time += batch_search_time;

    msg.str("");
    msg << "Current index size: " << sync_index.return_nd() << std::endl;
    msg << "Batch " << (batch + 1) << " time breakdown: S=" << std::fixed
        << std::setprecision(2) << batch_search_time
        << "s, U=" << pure_update_time << "s, M=" << batch_merge_time << "s"
        << std::endl;
    msg << "Total queries so far: " << total_queries << std::endl;
    dual_print(msg.str());

    total_queries += query_num * Lvec.size() + update_queries + merge_queries;
  }

  auto total_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_diff = total_end - total_start;
  time_stats.total_time = total_diff.count();

  // 打印最终时间统计
  time_stats.print_all();

  delete[] data_load;
  delete[] query;
}

int main(int argc, char** argv) {
  if (argc < 21) {
    std::cout << "Sliding Window Test for Continuous Insert/Delete with QPS "
                 "and Recall\n\n";
    std::cout << "Usage: " << argv[0]
              << " <type[int8/uint8/float]> <data_file> <L_mem> <R_mem> "
                 "<alpha_mem> <L_disk> <R_disk> <alpha_disk>"
              << " <initial_points> <batch_size> <num_batches> <num_shards> "
                 "<#pq_chunks> <#nodes_to_cache>"
              << " <index_path_prefix> <query_file> <gt_path_prefix> <recall@> "
                 "<beam_width> <log_file> <L1> <L2> ..."
              << std::endl;
    std::cout << "\nExample:\n";
    std::cout
        << "  " << argv[0]
        << " float /storage/vector_data/sift1m/sift1m_base.bin \\\n"
        << "    100 64 1.2 100 64 1.2 \\\n"
        << "    800000 2000 100 2 50 10000 \\\n"
        << "    /storage/vector_data/sift1m/diskann_index_800k/disk_index \\\n"
        << "    /storage/vector_data/sift1m/sift1m_query.bin \\\n"
        << "    /storage/vector_data/sift1m/gt_sift1m_800k/ \\\n"
        << "    10 4 sliding_window.log 50 100 200\n";
    std::cout << "\nParameters:\n";
    std::cout << "  initial_points: Number of points in initial index (e.g., "
                 "800000)\n";
    std::cout
        << "  batch_size: Points to insert/delete per batch (e.g., 2000)\n";
    std::cout
        << "  num_batches: Number of sliding window iterations (e.g., 100)\n";
    std::cout << "  gt_path_prefix: Path prefix for ground truth files (e.g., "
                 "/path/to/gt_sift1m_800k/)\n";
    std::cout << "                  GT files should be named: gt_0.bin, "
                 "gt_2000.bin, gt_4000.bin, ...\n";
    std::cout << "  log_file: Output log file path\n";
    exit(-1);
  }

  unsigned ctr = 3;

  unsigned    L_mem = (unsigned) atoi(argv[ctr++]);
  unsigned    R_mem = (unsigned) atoi(argv[ctr++]);
  float       alpha_mem = (float) std::atof(argv[ctr++]);
  unsigned    L_disk = (unsigned) atoi(argv[ctr++]);
  unsigned    R_disk = (unsigned) atoi(argv[ctr++]);
  float       alpha_disk = (float) std::atof(argv[ctr++]);
  size_t      initial_points = (size_t) std::atoi(argv[ctr++]);
  size_t      batch_size = (size_t) std::atoi(argv[ctr++]);
  size_t      num_batches = (size_t) std::atoi(argv[ctr++]);
  size_t      num_shards = (size_t) std::atoi(argv[ctr++]);
  unsigned    num_pq_chunks = (unsigned) std::atoi(argv[ctr++]);
  unsigned    nodes_to_cache = (unsigned) std::atoi(argv[ctr++]);
  std::string save_path(argv[ctr++]);
  std::string query_file(argv[ctr++]);
  std::string gt_prefix(argv[ctr++]);
  int         recall_at = (int) std::atoi(argv[ctr++]);
  unsigned    beam_width = (unsigned) std::atoi(argv[ctr++]);
  std::string log_file(argv[ctr++]);

  // 打开日志文件
  g_log_file.open(log_file, std::ios::out | std::ios::trunc);
  if (!g_log_file.is_open()) {
    std::cerr << "Failed to open log file: " << log_file << std::endl;
    return -1;
  }

  std::vector<_u64> Lvec;
  for (int i = ctr; i < argc; i++) {
    _u64 curL = std::atoi(argv[i]);
    if (curL >= (_u64) recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at ("
              << recall_at << ")." << std::endl;
    g_log_file.close();
    return -1;
  }

  std::stringstream config;
  config << "========== Sliding Window Test Configuration =========="
         << std::endl;
  config << "Initial points: " << initial_points << std::endl;
  config << "Batch size: " << batch_size << std::endl;
  config << "Number of batches: " << num_batches << std::endl;
  config << "Num shards: " << num_shards << std::endl;
  config << "Merge trigger: every " << MERGE_ROUND << " batches" << std::endl;
  config << "Query intervals: " << UPDATE_QUERY_INTERVAL_MS << "ms (update), "
         << MERGE_QUERY_INTERVAL_S << "s (merge)" << std::endl;
  config << "Index path: " << save_path << std::endl;
  config << "Query file: " << query_file << std::endl;
  config << "GT prefix: " << gt_prefix << std::endl;
  config << "Recall@" << recall_at << std::endl;
  config << "Log file: " << log_file << std::endl;
  config << "L values: ";
  for (auto l : Lvec)
    config << l << " ";
  config << std::endl;
  config << "========================================================"
         << std::endl;
  dual_print(config.str());

  if (std::string(argv[1]) == std::string("int8"))
    test<int8_t, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                           alpha_disk, initial_points, batch_size, num_batches,
                           num_shards, num_pq_chunks, nodes_to_cache, save_path,
                           query_file, gt_prefix, recall_at, Lvec, beam_width);
  else if (std::string(argv[1]) == std::string("uint8"))
    test<uint8_t, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                            alpha_disk, initial_points, batch_size, num_batches,
                            num_shards, num_pq_chunks, nodes_to_cache,
                            save_path, query_file, gt_prefix, recall_at, Lvec,
                            beam_width);
  else if (std::string(argv[1]) == std::string("float"))
    test<float, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                          alpha_disk, initial_points, batch_size, num_batches,
                          num_shards, num_pq_chunks, nodes_to_cache, save_path,
                          query_file, gt_prefix, recall_at, Lvec, beam_width);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;

  g_log_file.close();
  return 0;
}
