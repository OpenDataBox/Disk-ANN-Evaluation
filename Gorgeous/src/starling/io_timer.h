#pragma once

#include <chrono>
#include <vector>
#include <numeric>

namespace diskann {

class IOTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::vector<double> io_times_;  // 存储每次IO的时间（微秒）
    bool is_timing_;

public:
    IOTimer() : is_timing_(false) {}

    // 开始计时
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        is_timing_ = true;
    }

    // 结束计时并记录
    void stop() {
        if (is_timing_) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time_).count();
            io_times_.push_back(static_cast<double>(duration));
            is_timing_ = false;
        }
    }

    // 获取总IO时间
    double get_total_io_time() const {
        return std::accumulate(io_times_.begin(), io_times_.end(), 0.0);
    }

    // 获取平均IO时间
    double get_average_io_time() const {
        if (io_times_.empty()) return 0.0;
        return get_total_io_time() / io_times_.size();
    }

    // 获取IO次数
    size_t get_io_count() const {
        return io_times_.size();
    }

    // 重置统计
    void reset() {
        io_times_.clear();
        is_timing_ = false;
    }

    // 获取所有IO时间
    const std::vector<double>& get_all_io_times() const {
        return io_times_;
    }
};

} // namespace diskann