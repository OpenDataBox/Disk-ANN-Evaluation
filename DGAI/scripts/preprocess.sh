#!/bin/bash
#
# 静态 rerank_search 的建图预处理流水线（无任何动态更新）。
#
# 用法：
#   1. 在 config_dataset.sh 里确认/添加目标数据集函数
#   2. 在 "选择数据集" 一段调用对应函数
#   3. 调好顶部开关，bash scripts/preprocess.sh
#
# 流水线 5 步，开关独立控制：
#   do_compile        编译（带 USE_TOPO_DISK / USE_DOUBLE_PQ）
#   do_build          build_disk_index：图 + PQ + reorder map
#   do_split          split_index：拆 disk_index_graph + disk_index_data
#   do_topo_reorder   按 reorder_map_graph_2 物理重排拓扑文件
#   do_coord_reorder  按 reorder_map_data_2  物理重排坐标文件
#

set -eu
source "$(dirname "$0")/config_dataset.sh"

# ============================================================
#                          全局路径
# ============================================================
PROJECT_PATH=/storage_ssd_old/DGAI
INDEX_OUT_PATH=/storage_ssd_old/DGAI/indices          # 建图输出目录（长期保存）
LOG_PATH=/storage_ssd_old/DGAI/log

# ============================================================
#                          选择数据集
# ============================================================
# 这里列出的数据集会按顺序全部构图。
DATASET_FUNCS=(
    # dataset_sift1m
    # dataset_gist
    # dataset_deep1m
    # dataset_msong
    # dataset_glove
    # dataset_tiny5m
    # dataset_sift1m_8kb
    # dataset_sift1m_16kb
    # dataset_gist_8kb
    # dataset_gist_16kb
    # dataset_deep1m_8kb
    # dataset_deep1m_16kb
    #  dataset_OpenAI_300
    # dataset_OpenAI_500
    # dataset_OpenAI_700
    # dataset_OpenAI_900
    # dataset_OpenAI_1500
    # dataset_OpenAI_2000
    # dataset_OpenAI_3000
    # dataset_OpenAI
)

# ===========================================================
#                          实验参数
# ============================================================
# 流水线开关
do_compile=1
do_build=1
do_build_mem=1
do_split=1
do_topo_reorder=1
do_coord_reorder=1

# PQ refine 类型（对应 src/utils/aux_utils.cpp 里的 int type = ?）
#   0 = 不 refine
#   1 = double PQ refine（推荐，搭配 -DUSE_DOUBLE_PQ）
#   2 = error refine
#   3 = build_dual_pq_512_from_scratch
PQ_REFINE_TYPE=1

# 编译宏
ADDITIONAL_DEFS="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_DOUBLE_PQ"

# Disk Vamana 建图参数
R=48
BUILD_L=100
B=64           # PQ 内存预算 (GB)
M=64           # 最终 RAM 预算 (GB)
BUILD_T=128    # 建图线程数
SINGLE_FILE_INDEX=0

# 内存导航图参数，先抽样 MEM_SAMPLE_RATE，再输出为 ${PREFIX}_mem.index。
MEM_R=48
MEM_BUILD_L=64
MEM_ALPHA=1.2
MEM_T=128
MEM_SAMPLE_RATE=0.001
MEM_DYNAMIC_INDEX=0
MEM_SINGLE_FILE_INDEX=0

# ============================================================
#                          下面无需改动
# ============================================================
stage_start() {
    date +%s
}

stage_elapsed() {
    local start_ts="$1"
    local end_ts
    end_ts=$(date +%s)
    echo $((end_ts - start_ts))
}

write_profile_summary() {
    local build_status="${1:-0}"
    awk \
        -v build_log="$build_profile_log" \
        -v time_log="$build_time_log" \
        -v build_status="$build_status" \
        -v compile_sec="$compile_sec" \
        -v build_wall_sec="$build_wall_sec" \
        -v split_sec="$split_sec" \
        -v topo_reorder_sec="$topo_reorder_sec" \
        -v coord_reorder_sec="$coord_reorder_sec" \
        -v profile_summary="$profile_summary" '
    function last_field_number(line,    n, a, v) {
        n = split(line, a, " ")
        v = a[n]
        sub(/s[.]?$/, "", v)
        return v + 0
    }
    BEGIN {
        pq_pivots = -1
        pq_compress = -1
        vamana_total = -1
        max_rss_kb = -1
    }
    FILENAME == build_log {
        ts = $1 + 0
        line = $0
        sub(/^[0-9]+\t/, "", line)

        if (line ~ /Pivots generated in /) {
            pq_pivots = last_field_number(line)
        }
        if (line ~ /Compressed data generated and written in:/) {
            pq_compress = last_field_number(line)
        }
        if (line ~ /Vamana index built in:/) {
            vamana_total = last_field_number(line)
        }
        next
    }
    FILENAME == time_log {
        if ($0 ~ /Maximum resident set size/) {
            max_rss_kb = $NF + 0
        }
    }
    END {
        pq_total = "N/A"
        if (pq_pivots >= 0 && pq_compress >= 0) {
            pq_total = pq_pivots + pq_compress
        }

        max_rss_gb = "N/A"
        if (max_rss_kb >= 0) {
            max_rss_gb = max_rss_kb / 1024 / 1024
        }

        print "=========================================="
        print "Preprocess profile summary"
        print "=========================================="
        print "pq_total_sec            : " pq_total
        print "vamana_total_sec        : " vamana_total
        print "max_rss_kb              : " max_rss_kb
        print "max_rss_gb              : " max_rss_gb
        print "=========================================="
    }
    ' "$build_profile_log" "$build_time_log" | tee "$profile_summary"
}

run_selected_dataset() {
    set -e

    echo "=========================================="
    echo " dataset      : $PREFIX"
    echo " base file    : $BASE_FILE"
    echo " query file   : $QUERY_FILE"
    echo " gt file      : $GT_FILE"
    echo " data_type    : $DATA_TYPE"
    echo " dim / npts   : $DATA_DIM / $DATA_N"
    echo " sector_len   : $SECTOR_LEN"
    echo " R / L / B/M  : $R / $BUILD_L / $B / $M"
    echo " PQ chunks    : $N_PQ_CHUNKS  (refine type=$PQ_REFINE_TYPE)"
    echo "=========================================="

    original_index_path="${INDEX_OUT_PATH}/${PREFIX}/original_index"
    mkdir -p "$original_index_path" "$LOG_PATH"
    index_path_prefix="${original_index_path}/${PREFIX}"
    sector_len_value=$((SECTOR_LEN))
    profile_ts=$(date +"%Y%m%d_%H%M%S")
    profile_summary="${LOG_PATH}/preprocess_profile_${PREFIX}_${profile_ts}.summary"
    build_profile_log="${LOG_PATH}/preprocess_build_${PREFIX}_${profile_ts}.log"
    build_time_log="${LOG_PATH}/preprocess_build_${PREFIX}_${profile_ts}.time"

    compile_sec="N/A"
    build_wall_sec="N/A"
    split_sec="N/A"
    topo_reorder_sec="N/A"
    coord_reorder_sec="N/A"

    # 单条记录字节数
    case "$DATA_TYPE" in
        float)        elem_size=4 ;;
        int8|uint8)   elem_size=1 ;;
        *)            echo "未知 data_type: $DATA_TYPE" >&2; exit 1 ;;
    esac
    topo_len_bytes=$(( (R + 1) * 4 ))
    coord_len_bytes=$(( DATA_DIM * elem_size ))

    # ---------- 1. 改源码：PQ refine type 与 num_pq_chunks ----------
    aux_utils_cpp_path="$PROJECT_PATH/src/utils/aux_utils.cpp"
    sed -i "s/int type = [0-9];/int type = ${PQ_REFINE_TYPE};/" "$aux_utils_cpp_path"
    sed -i -E "s/size_t num_pq_chunks[[:space:]]*=[[:space:]]*[0-9]+.*;/size_t num_pq_chunks = ${N_PQ_CHUNKS};/" "$aux_utils_cpp_path"

    # ---------- 2. 编译 ----------
    if [ "$do_compile" = "1" ]; then
        ts=$(stage_start)
        mkdir -p "$PROJECT_PATH/build"
        cd "$PROJECT_PATH/build"
        export ADDITIONAL_DEFINITIONS="$ADDITIONAL_DEFS -DSECTOR_LEN=${sector_len_value}ULL"
        cmake ..
        make -j
        cd "$PROJECT_PATH"
        compile_sec=$(stage_elapsed "$ts")
        echo "[profile] compile_wall_sec=$compile_sec"
    fi

    # ---------- 3. build_disk_index ----------
    if [ "$do_build" = "1" ]; then
        cmd="$PROJECT_PATH/build/tests/build_disk_index \
            ${DATA_TYPE} \
            ${BASE_FILE} \
            ${index_path_prefix} \
            ${R} \
            ${BUILD_L} \
            ${B} \
            ${M} \
            ${BUILD_T} \
            ${DIST_FN} \
            ${SINGLE_FILE_INDEX}"
        echo "[build_disk_index] $cmd"
        ts=$(stage_start)
        set +e
        /usr/bin/time -v -o "$build_time_log" bash -c "$cmd" 2>&1 | \
            awk '{ print systime() "\t" $0; fflush(); }' | tee "$build_profile_log"
        build_status=${PIPESTATUS[0]}
        set -e
        build_wall_sec=$(stage_elapsed "$ts")
        echo "[profile] build_wall_sec=$build_wall_sec"
        write_profile_summary "$build_status"
        if [ "$build_status" -ne 0 ]; then
            exit "$build_status"
        fi
    fi

    # ---------- 3b. build_memory_index ----------
    if [ "$do_build_mem" = "1" ]; then
        mem_sample_prefix="${index_path_prefix}_mem_sample_${MEM_SAMPLE_RATE}"
        mem_sample_data="${mem_sample_prefix}_data.bin"
        mem_sample_ids="${mem_sample_prefix}_ids.bin"
        mem_index_path="${index_path_prefix}_mem.index"
        cmd="$PROJECT_PATH/build/tests/utils/gen_random_slice \
            ${DATA_TYPE} \
            ${BASE_FILE} \
            ${mem_sample_prefix} \
            ${MEM_SAMPLE_RATE}"
        echo "[gen_random_slice mem] $cmd"
        eval "$cmd"

        cmd="$PROJECT_PATH/build/tests/build_memory_index \
            ${DATA_TYPE} \
            ${mem_sample_data} \
            ${mem_sample_ids} \
            ${mem_index_path} \
            ${MEM_DYNAMIC_INDEX} \
            ${MEM_SINGLE_FILE_INDEX} \
            ${MEM_R} \
            ${MEM_BUILD_L} \
            ${MEM_ALPHA} \
            ${MEM_T} \
            ${DIST_FN}"
        echo "[build_memory_index] $cmd"
        eval "$cmd"
    fi

    index_path="${index_path_prefix}_disk.index"
    aligned_topo_path="${original_index_path}/disk_index_graph"
    coord_path="${original_index_path}/disk_index_data"

    # ---------- 4. split_index ----------
    if [ "$do_split" = "1" ]; then
        # 第二个参数 dram_index_graph 在新版 split_index 内已不写盘，仅占位
        cmd="$PROJECT_PATH/build/tests/split_index \
            ${index_path} \
            ${original_index_path}/dram_index_graph \
            ${aligned_topo_path} \
            ${coord_path} \
            ${DATA_TYPE}"
        echo "[split_index] $cmd"
        ts=$(stage_start)
        eval "$cmd"
        split_sec=$(stage_elapsed "$ts")
        echo "[profile] split_wall_sec=$split_sec"
    fi

    # ---------- 5a. topo reorder ----------
    if [ "$do_topo_reorder" = "1" ]; then
        cmd="$PROJECT_PATH/build/tests/reorder_by_map \
            ${DATA_N} ${topo_len_bytes} \
            ${aligned_topo_path} \
            ${original_index_path}/reorder_map_graph_2 \
            ${original_index_path}/reordered_disk_index_graph_2"
        echo "[reorder topo] $cmd"
        ts=$(stage_start)
        eval "$cmd"
        topo_reorder_sec=$(stage_elapsed "$ts")
        echo "[profile] topo_reorder_wall_sec=$topo_reorder_sec"
    fi

    # ---------- 5b. coord reorder ----------
    if [ "$do_coord_reorder" = "1" ]; then
        cmd="$PROJECT_PATH/build/tests/reorder_by_map \
            ${DATA_N} ${coord_len_bytes} \
            ${coord_path} \
            ${original_index_path}/reorder_map_data_2 \
            ${original_index_path}/reordered_disk_index_data_2"
        echo "[reorder coord] $cmd"
        ts=$(stage_start)
        eval "$cmd"
        coord_reorder_sec=$(stage_elapsed "$ts")
        echo "[profile] coord_reorder_wall_sec=$coord_reorder_sec"
    fi

    write_profile_summary 0
    echo "[done] index dir: $original_index_path"
    echo "[done] profile summary: $profile_summary"
}

mkdir -p "$LOG_PATH"
for dataset_func in "${DATASET_FUNCS[@]}"; do
    "$dataset_func"
    dataset_log="${LOG_PATH}/${PREFIX}_build.log"
    echo "[start] ${PREFIX}, log: ${dataset_log}"
    set +e
    run_selected_dataset 2>&1 | tee "$dataset_log"
    dataset_status=${PIPESTATUS[0]}
    set -e
    if [ "$dataset_status" -ne 0 ]; then
        echo "[failed] ${PREFIX}, log: ${dataset_log}" >&2
        exit "$dataset_status"
    fi
done
