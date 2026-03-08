#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'EOF'
Usage: memory_monitor.sh [options] -- command [args...]

Options:
  --pre-duration <seconds>    Sampling duration before cache load (default 5)
  --post-duration <seconds>   Sampling duration after cache load (default 5)
  --interval <seconds>        Sampling interval in seconds (default 0.1)
  --auto                      Skip prompts between stages and run sequentially
  --signal-mode               Drive sampling by SIGUSR1/SIGUSR2 from the monitored process
  --help                      Show this message

Example:
  ./memory_monitor.sh --pre-duration 8 --post-duration 10 -- ./bin/search ... 
EOF
}

pre_duration=5
post_duration=5
interval=0.1
interactive=true
auto_mode=false
signal_mode=false
cmd=()

stage=0
cache_peak=0
run_peak=0

# Use temporary files for inter-process communication in signal-mode
STAGE_FILE=$(mktemp)
PEAK_FILE=$(mktemp)
trap 'rm -f "$STAGE_FILE" "$PEAK_FILE"' EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pre-duration)
      pre_duration="$2"
      shift 2
      ;;
    --post-duration)
      post_duration="$2"
      shift 2
      ;;
    --interval)
      interval="$2"
      shift 2
      ;;
    --auto)
      auto_mode=true
      interactive=false
      shift
      ;;
    --signal-mode)
      signal_mode=true
      interactive=false
      shift
      ;;
    --help)
      print_usage
      exit 0
      ;;
    --)
      shift
      cmd+=("$@")
      break
      ;;
    *)
      cmd+=("$1")
      shift
      ;;
  esac
done

if [[ ${#cmd[@]} -eq 0 ]]; then
  echo "Error: missing command" >&2
  print_usage
  exit 1
fi

wait_for_stage() {
  local label="$1"
  if [[ "$interactive" == true ]]; then
    read -rp "按回车开始${label}的峰值采样... "
  elif [[ "$auto_mode" == true ]]; then
    echo "自动模式：开始 ${label} 峰值采样"
  else
    echo "开始 ${label} 峰值采样"
  fi
}

sample_stage() {
  local label="$1"
  local duration="$2"
  local peak=0
  local start
  local end

  start=$(date +%s)
  end=$((start + duration))
  while (( start < end )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Process $pid exited early" >&2
      break
    fi
    local rss
    rss=$(awk '/VmRSS/ {print $2}' /proc/"$pid"/status 2>/dev/null || echo 0)
    ((rss > peak)) && peak=$rss
    sleep "$interval"
    start=$(date +%s)
  done
  printf "%s peak memory: %d KB\n" "$label" "$peak"
}

handle_sigusr1() {
  if [[ "$signal_mode" != true ]]; then
    return
  fi
  echo "1" > "$STAGE_FILE"
  echo "Signal: 收到 SIGUSR1 -> 缓存初始化阶段开始 (Stage 1)"
}

handle_sigusr2() {
  if [[ "$signal_mode" != true ]]; then
    return
  fi
  # 收到信号后立即切换阶段
  echo "2" > "$STAGE_FILE"
  echo "Signal: 收到 SIGUSR2 -> 开始统计 Stage 2 (搜索阶段) 的全新峰值"
}

sampling_loop() {
  local c_peak=0
  local r_peak=0
  local last_stage=0
  
  while [[ -f "$STAGE_FILE" ]]; do
    if ! kill -0 "$pid" 2>/dev/null; then
        break
    fi
    
    local cur_stage
    cur_stage=$(cat "$STAGE_FILE" 2>/dev/null || echo 0)
    
    # 如果检测到 stage 切换到 2，我们可以选择在这里重置 r_peak 
    # (其实初始化已经为 0 了，只要 cur_stage 是 2，它就开始从 0 重新统计最高点)
    
    local rss
    rss=$(awk '/VmRSS/ {print $2}' /proc/"$pid"/status 2>/dev/null || echo 0)
    
    if [[ "$cur_stage" == "1" ]]; then
      ((rss > c_peak)) && c_peak=$rss
    elif [[ "$cur_stage" == "2" ]]; then
      ((rss > r_peak)) && r_peak=$rss
    fi
    
    # 将当前的实时峰值存入文件
    echo "$c_peak $r_peak" > "$PEAK_FILE"
    sleep "$interval"
  done
}

start_signal_sampling() {
  echo "0" > "$STAGE_FILE"
  sampling_loop &
  sampling_loop_pid=$!
}

stop_signal_sampling() {
  if [[ -n "${sampling_loop_pid:-}" ]]; then
    sleep "$interval"
    kill "$sampling_loop_pid" 2>/dev/null || true
    wait "$sampling_loop_pid" || true
  fi
  
  if [[ -f "$PEAK_FILE" ]]; then
    read -r cache_peak run_peak < "$PEAK_FILE" || true
  fi
}

if [[ "$signal_mode" == true ]]; then
  trap 'handle_sigusr1' SIGUSR1
  trap 'handle_sigusr2' SIGUSR2
  export PAGEANN_MONITOR_PID=$$
fi

printf "Launching command: %q (PID will follow)\n" "${cmd[@]}"
"${cmd[@]}" &
pid=$!
trap 'if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi' EXIT

echo "Tracked PID: $pid"
if [[ "$signal_mode" == true ]]; then
  start_signal_sampling
else
  wait_for_stage "Before cache"
  sample_stage "Before cache" "$pre_duration"
  wait_for_stage "After cache"
  sample_stage "After cache" "$post_duration"
fi

# 核心修改：等待进程结束，同时防止信号中断
while kill -0 "$pid" 2>/dev/null; do
  wait "$pid" || true
done

if [[ "$signal_mode" == true ]]; then
  stop_signal_sampling
  
  final_stage=$(cat "$STAGE_FILE" 2>/dev/null || echo 0)
  if [[ "$final_stage" == "0" ]]; then
    echo "警告：未接收到缓存阶段信号，采样未触发"
  else
    echo "--------------------------------------------------------"
    printf "Stage 1 (Cache Init) Peak RSS: %d KB\n" "$cache_peak"
    printf "Stage 2 (Search Stage) Peak RSS: %d KB\n" "$run_peak"
    echo "--------------------------------------------------------"
  fi
  trap - SIGUSR1 SIGUSR2
fi
trap - EXIT
