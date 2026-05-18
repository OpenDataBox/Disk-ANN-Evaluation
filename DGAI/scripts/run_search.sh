#!/bin/bash
#
# 静态 rerank_search 的搜索脚本（与 preprocess.sh 配套）。
#
# 流程：
#   1. 选数据集（dataset_xxx）
#   2. 设搜索参数（K / L / 线程 / beamwidth / strategy）
#   3. 直接读取 preprocess.sh 生成的 original_index
#   4. 调 build/tests/search_disk_index
#

set -eu
source "$(dirname "$0")/config_dataset.sh"

# ============================================================
#                          全局路径
# ============================================================
PROJECT_PATH=/storage_ssd_old/DGAI
INDEX_OUT_PATH=/storage_ssd_old/DGAI/indices          # preprocess.sh 的输出目录
LOG_PATH=/storage_ssd_old/DGAI/log

# ============================================================
#                          选择数据集
# ============================================================
DATASET_FUNCS=(
    # dataset_sift1m
     dataset_gist
    #  dataset_deep1m
    # dataset_msong
    # dataset_glove
    # dataset_tiny5m
    # dataset_OpenAI
    # dataset_sift1m_8kb
    # dataset_sift1m_16kb
    # dataset_gist_8kb
    # dataset_gist_16kb
    # dataset_deep1m_8kb
    # dataset_deep1m_16kb
    # dataset_OpenAI_300
    # dataset_OpenAI_500
    # dataset_OpenAI_700
    # dataset_OpenAI_900
    # dataset_OpenAI_1500
    # dataset_OpenAI_2000
    # dataset_OpenAI_3000
)

# ============================================================
#                          实验参数
# ============================================================
# 流水线开关
do_compile=1         # 不同 SECTOR_LEN 的索引搜索前需要重新编译
do_search=1          # 跑搜索
DYNAMIC_BEAM_WIDTH=1 # 0=全程静态 beam_width；1=启用 DYN_PIPE_WIDTH 动态调节

# rerank_search 必填
SEARCH_MODE=3        # SearchMode: 0=BEAM 1=PAGE 2=PIPE 3=RERANK 4=CORO
NUM_THREADS=(1)
# rerank 用的 pipeline_width（依次各跑一次）
BEAM_WIDTHS=(4)
K=100                 # recall@K
MEM_L=0             # 0 表示不用内存索引；非 0 会加载 ${PREFIX}_mem.index 作为内存导航图

# 搜索 L 列表（可空格分隔多个值）
# L_LIST="100 110 120 130 140 150 270 280 290 300 320 350 400 500 600 800 1000 1200 1500 1800 2000 3000 4000 5000"
L_LIST="300 500 1500"
# L_LIST="130 170 350"
# L_LIST="126 166 325"
# L_LIST="180 250 500 "
# L_LIST="300 450 1500"

# L_LIST="100 150 170 200 210 230 250 270  290 300 320 350 400 500 600 800 1000 1200 1500 1800 2000 3000 4000 5000"

# 选项位（最终拼成 strategy）
USE_RERANK=1         # bit0
USE_TOPO_REORDER=1   # bit1   建议 1，配合离线重排
USE_DOUBLE_PQ=1      # bit2   配合 -DUSE_DOUBLE_PQ 编译
USE_COORD_REORDER=1  # bit3
USE_TOPO_BUFFER=0    # bit4   topo page cache，先不开
USE_TRUNCATE=0       # bit6
USE_TRIPLE_PQ=0      # bit7   需要 _pq_*_3.bin，目前不用

# 编译宏，SECTOR_LEN 会在每个数据集里按 config_dataset.sh 自动追加。
ADDITIONAL_DEFS="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_DOUBLE_PQ"

# ============================================================
#                          下面无需改动
# ============================================================
strategy=$((
    (USE_RERANK        << 0) |
    (USE_TOPO_REORDER  << 1) |
    (USE_DOUBLE_PQ     << 2) |
    (USE_COORD_REORDER << 3) |
    (USE_TOPO_BUFFER   << 4) |
    (USE_TRUNCATE      << 6) |
    (USE_TRIPLE_PQ     << 7)
))

run_selected_dataset() {
    original_index_path="${INDEX_OUT_PATH}/${PREFIX}/original_index"
    work_prefix="${original_index_path}/${PREFIX}"
    sector_len_value=$((SECTOR_LEN))

    echo "=========================================="
    echo " dataset      : $PREFIX"
    echo " search_mode  : $SEARCH_MODE  (3 = RERANK_SEARCH)"
    echo " strategy     : $strategy"
    echo "    rerank=$USE_RERANK  topo_reorder=$USE_TOPO_REORDER  double_pq=$USE_DOUBLE_PQ"
    echo "    coord_reorder=$USE_COORD_REORDER  topo_buffer=$USE_TOPO_BUFFER"
    echo "    truncate=$USE_TRUNCATE  triple_pq=$USE_TRIPLE_PQ"
    echo " L list       : $L_LIST"
    echo " threads/BW   : (${NUM_THREADS[*]}) / (${BEAM_WIDTHS[*]})"
    echo " dynamic BW   : $DYNAMIC_BEAM_WIDTH"
    echo " sector_len   : $SECTOR_LEN"
    echo " index src    : $original_index_path"
    echo " index prefix : $work_prefix"
    echo "=========================================="

    # ---------- 0. 编译 ----------
    if [ "$do_compile" = "1" ]; then
        mkdir -p "$PROJECT_PATH/build"
        cd "$PROJECT_PATH/build"
        export ADDITIONAL_DEFINITIONS="$ADDITIONAL_DEFS -DSECTOR_LEN=${sector_len_value}ULL"
        if [ "$DYNAMIC_BEAM_WIDTH" = "1" ]; then
            cmake -DENABLE_DYN_PIPE_WIDTH=ON ..
        else
            cmake -DENABLE_DYN_PIPE_WIDTH=OFF ..
        fi
        make -j
        cd "$PROJECT_PATH"
    fi

    # ---------- 1. 跑搜索 ----------
    if [ "$do_search" = "1" ]; then
        for nt in "${NUM_THREADS[@]}"; do
            for bw in "${BEAM_WIDTHS[@]}"; do
                beam_log="${LOG_PATH}/${PREFIX}_search_t_${nt}_bm_${bw}.log"
                echo "---------- threads = ${nt}, beam_width (pipeline_width) = ${bw} -> ${beam_log} ----------"
                cmd="$PROJECT_PATH/build/tests/search_disk_index \
                    ${DATA_TYPE} \
                    ${work_prefix} \
                    ${nt} \
                    ${bw} \
                    ${QUERY_FILE} \
                    ${GT_FILE} \
                    ${K} \
                    ${DIST_FN} \
                    ${SEARCH_MODE} \
                    ${MEM_L} \
                    ${strategy} \
                    ${L_LIST}"
                echo "[search_disk_index] $cmd"
                eval "$cmd" 2>&1 | tee "$beam_log"
                eval_status=${PIPESTATUS[0]}
                if [ "$eval_status" -ne 0 ]; then
                    exit "$eval_status"
                fi
            done
        done
    fi
}

mkdir -p "$LOG_PATH"
for dataset_func in "${DATASET_FUNCS[@]}"; do
    "$dataset_func"
    log_file="${LOG_PATH}/${PREFIX}_search.log"
    echo "[start] ${PREFIX}, log: ${log_file}"
    set +e
    run_selected_dataset 2>&1 | tee "$log_file"
    search_status=${PIPESTATUS[0]}
    set -e
    if [ "$search_status" -ne 0 ]; then
        echo "[failed] ${PREFIX}, log: ${log_file}" >&2
        exit "$search_status"
    fi
    echo "[done] ${PREFIX}, log: ${log_file} (per-run: ${LOG_PATH}/${PREFIX}_search_t_*_bm_*.log)"
done
