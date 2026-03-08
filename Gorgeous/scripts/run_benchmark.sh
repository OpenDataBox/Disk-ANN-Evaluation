#!/bin/bash

set -e
# set -x

source config_local.sh

INDEX_PREFIX_PATH="${PREFIX}/M${M}_R${R}_L${BUILD_L}/"
PQ_PREFIX_PATH="${PREFIX}/PQ/C/"
if [ ! $N_PQ_CODE -eq 0 ]; then
  PQ_PREFIX_PATH="${PREFIX}/PQ/C${N_PQ_CODE}/"
else
  echo "N_PQ_CODE should not be zero."
  exit 1
fi

MEM_SAMPLE_PATH="${PREFIX}/MEM_SAMPLE/SAMPLE_RATE_${MEM_RAND_SAMPLING_RATE}/"
MEM_INDEX_PATH="${PREFIX}/MEM_INDEX/MEM_R_${MEM_R}_L_${MEM_BUILD_L}_ALPHA_${MEM_ALPHA}_RANDOM_RATE${MEM_RAND_SAMPLING_RATE}/"
GP_PATH="${INDEX_PREFIX_PATH}GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_CUT${GP_CUT}/"

GRAPH_PATH="${DATA_DIR}/gorgeous/${INDEX_PREFIX_PATH}GRAPH/"
GRAPH_REP_INDEX_PATH="${DATA_DIR}/gorgeous/${INDEX_PREFIX_PATH}GRAPH_CACHE_INDEX/"
GRAPH_GP_PATH="${GRAPH_PATH}GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_CUT${GP_CUT}/"
GRAPH_CACHE_INDEX_GP_PATH="${GRAPH_REP_INDEX_PATH}GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_CUT${GP_CUT}/"

SUMMARY_FILE_PATH="${DATA_DIR}/gorgeous/${INDEX_PREFIX_PATH}/summary.log"
RELAYOUT_TIME_FILE="${DATA_DIR}/gorgeous/${INDEX_PREFIX_PATH}/relayout_times.log"

print_usage_and_exit() {
  echo "Usage: ./run_benchmark.sh [debug/release] [build/build_mem/gp/clustering/search/dynamic_cache_search/hybridcache_search] [knn/range]"
  exit 1
}

check_dir_and_make_if_absent() {
  local dir=$1
  if [ -d $dir ]; then
    echo "Directory $dir is already exit. Remove or rename it and then re-run."
    exit 1
  else
    mkdir -p ${dir}
  fi
}

case $1 in
  debug)
    cmake -DCMAKE_BUILD_TYPE=Debug .. -B ${DATA_DIR}/gorgeous/debug
    EXE_PATH=${DATA_DIR}/gorgeous/debug
  ;;
  release)
    cmake -DCMAKE_BUILD_TYPE=Release .. -B ${DATA_DIR}/gorgeous/release
    EXE_PATH=${DATA_DIR}/gorgeous/release
  ;;
  *)
    print_usage_and_exit
  ;;
esac
pushd $EXE_PATH
make -j
popd

mkdir -p ${DATA_DIR}/gorgeous && cd ${DATA_DIR}/gorgeous

date
case $2 in
  build)
    check_dir_and_make_if_absent ${INDEX_PREFIX_PATH}
    check_dir_and_make_if_absent ${PQ_PREFIX_PATH}
    echo "Building disk index..."
    time ${EXE_PATH}/tests/build_disk_index \
      --data_type $DATA_TYPE \
      --dist_fn $DIST_FN \
      --data_path $BASE_PATH \
      --index_path_prefix $INDEX_PREFIX_PATH \
      --pq_path_prefix $PQ_PREFIX_PATH \
      --build_pq_only 0 \
      --n_PQ_code $N_PQ_CODE \
      -R $R \
      -L $BUILD_L \
      -M $M \
      -T $BUILD_T \
      --sector_len ${SECTOR_LEN} > ${INDEX_PREFIX_PATH}build.log
    cp ${INDEX_PREFIX_PATH}_disk.index ${INDEX_PREFIX_PATH}_disk_beam_search.index
  ;;
  build_pq)
    check_dir_and_make_if_absent ${PQ_PREFIX_PATH}
    echo "Building PQ..."
    time ${EXE_PATH}/tests/build_disk_index \
      --data_type $DATA_TYPE \
      --dist_fn $DIST_FN \
      --data_path $BASE_PATH \
      --index_path_prefix $INDEX_PREFIX_PATH \
      --pq_path_prefix $PQ_PREFIX_PATH \
      --build_pq_only 1 \
      --n_PQ_code $N_PQ_CODE \
      -R $R \
      -L $BUILD_L \
      -M $M \
      -T $BUILD_T \
      --sector_len ${SECTOR_LEN} > ${PQ_PREFIX_PATH}build.log
  ;;
  build_mem)
    if [ ! -d ${MEM_SAMPLE_PATH} ]; then
      mkdir -p ${MEM_SAMPLE_PATH}
      echo "Generating random slice..."
      time ${EXE_PATH}/tests/utils/gen_random_slice $DATA_TYPE $BASE_PATH $MEM_SAMPLE_PATH $MEM_RAND_SAMPLING_RATE > ${MEM_SAMPLE_PATH}sample.log
      MEM_DATA_PATH=${MEM_SAMPLE_PATH}
    else
      echo "Random slice already generated"
    fi
    echo "Building memory index (sampling rate: ${MEM_RAND_SAMPLING_RATE})"
    check_dir_and_make_if_absent ${MEM_INDEX_PATH}
    time ${EXE_PATH}/tests/build_memory_index \
      --data_type ${DATA_TYPE} \
      --dist_fn ${DIST_FN} \
      --data_path ${MEM_DATA_PATH} \
      --index_path_prefix ${MEM_INDEX_PATH}_index \
      -R ${MEM_R} \
      -L ${MEM_BUILD_L} \
      --alpha ${MEM_ALPHA} > ${MEM_INDEX_PATH}build.log
  ;;
  gp)
    # graph partition with scaled
    if [ -d ${GP_PATH} ]; then
      echo "Directory ${GP_PATH} is already exit."
      if [ -f "${GP_PATH}_part_tmp.index" ]; then
        echo "copy ${GP_PATH}_part_tmp.index to ${INDEX_PREFIX_PATH}_disk.index."
        GP_FILE_PATH=${GP_PATH}_part.bin
        cp ${GP_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
        cp ${GP_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
        exit 0
      else
        echo "${GP_PATH}_part_tmp.index not found."
        exit 1
      fi
    else
      mkdir -p ${GP_PATH}
      # check_dir_and_make_if_absent ${GP_PATH}
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ ! -f "$OLD_INDEX_FILE" ]; then
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
      fi
      #using sq index file to gp
      GP_DATA_TYPE=$DATA_TYPE
      GP_FILE_PATH=${GP_PATH}_part.bin
      echo "Running graph partition... ${GP_FILE_PATH}.log"
      time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES \
        --in_sector_len ${SECTOR_LEN} --out_sector_len ${SECTOR_LEN} > ${GP_FILE_PATH}.log

      echo "Running relayout... ${GP_PATH}relayout.log"
      echo "$(date): GP relayout started" >> ${RELAYOUT_TIME_FILE}
      { time ${EXE_PATH}/tests/utils/index_relayout ${OLD_INDEX_FILE} ${GP_FILE_PATH} $GP_DATA_TYPE 0 ${SECTOR_LEN} ${SECTOR_LEN} > ${GP_PATH}relayout.log; } 2>> ${RELAYOUT_TIME_FILE}
      echo "$(date): GP relayout finished" >> ${RELAYOUT_TIME_FILE}
      if [ ! -f "${INDEX_PREFIX_PATH}_disk_beam_search.index" ]; then
        mv $OLD_INDEX_FILE ${INDEX_PREFIX_PATH}_disk_beam_search.index
      fi
      cp ${GP_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
      cp ${GP_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
    fi
  ;;
  clustering)
    # Clustering similarity-based partitioning
    CLUSTERING_PATH="${INDEX_PREFIX_PATH}CLUSTERING_T${CLUSTERING_SIMILARITY_THRESHOLD}_${CLUSTERING_DISTANCE_FUNC}/"
    if [ -d ${CLUSTERING_PATH} ]; then
      echo "Directory ${CLUSTERING_PATH} already exists."
      if [ -f "${CLUSTERING_PATH}_part_tmp.index" ]; then
        echo "copy ${CLUSTERING_PATH}_part_tmp.index to ${INDEX_PREFIX_PATH}_disk.index."
        CLUSTERING_FILE_PATH=${CLUSTERING_PATH}_part.bin
        cp ${CLUSTERING_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
        cp ${CLUSTERING_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
        exit 0
      else
        echo "${CLUSTERING_PATH}_part_tmp.index not found."
        exit 1
      fi
    else
      mkdir -p ${CLUSTERING_PATH}
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ ! -f "$OLD_INDEX_FILE" ]; then
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
      fi
      
      CLUSTERING_FILE_PATH=${CLUSTERING_PATH}_part.bin
      echo "Running clustering similarity-based partitioning... ${CLUSTERING_FILE_PATH}.log"
      echo "$(date): Clustering partitioning started" >> ${RELAYOUT_TIME_FILE}
      
      # Clustering parameters with defaults
      USE_CLUSTERING=${USE_CLUSTERING:-1}
      CLUSTERING_SIMILARITY_THRESHOLD=${CLUSTERING_SIMILARITY_THRESHOLD:-0.7}
      CLUSTERING_DISTANCE_FUNC=${CLUSTERING_DISTANCE_FUNC:-L2}
      CLUSTERING_PAGE_SIZE=${CLUSTERING_PAGE_SIZE:-0}
      CLUSTERING_VERBOSE=${CLUSTERING_VERBOSE:-0}
      
      CLUSTERING_ARGS=""
      if [ $USE_CLUSTERING -eq 1 ]; then
        CLUSTERING_ARGS="--clustering true --similarity_threshold ${CLUSTERING_SIMILARITY_THRESHOLD} --distance_func ${CLUSTERING_DISTANCE_FUNC}"
        if [ $CLUSTERING_PAGE_SIZE -ne 0 ]; then
          CLUSTERING_ARGS="${CLUSTERING_ARGS} --clustering_page_size ${CLUSTERING_PAGE_SIZE}"
        fi
        if [ $CLUSTERING_VERBOSE -eq 1 ]; then
          CLUSTERING_ARGS="${CLUSTERING_ARGS} --clustering_verbose true"
        fi
      fi
      
      { time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $DATA_TYPE --gp_file $CLUSTERING_FILE_PATH -T $GP_T \
        --in_sector_len ${SECTOR_LEN} --out_sector_len ${SECTOR_LEN} \
        ${CLUSTERING_ARGS} > ${CLUSTERING_FILE_PATH}.log; } 2>> ${RELAYOUT_TIME_FILE}
      
      echo "$(date): Clustering partitioning finished" >> ${RELAYOUT_TIME_FILE}
      
      echo "Running relayout... ${CLUSTERING_PATH}relayout.log"
      echo "$(date): Clustering relayout started" >> ${RELAYOUT_TIME_FILE}
      { time ${EXE_PATH}/tests/utils/index_relayout ${OLD_INDEX_FILE} ${CLUSTERING_FILE_PATH} $DATA_TYPE 0 ${SECTOR_LEN} ${SECTOR_LEN} > ${CLUSTERING_PATH}relayout.log; } 2>> ${RELAYOUT_TIME_FILE}
      echo "$(date): Clustering relayout finished" >> ${RELAYOUT_TIME_FILE}
      
      if [ ! -f "${INDEX_PREFIX_PATH}_disk_beam_search.index" ]; then
        mv $OLD_INDEX_FILE ${INDEX_PREFIX_PATH}_disk_beam_search.index
      fi
      cp ${CLUSTERING_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
      cp ${CLUSTERING_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
      
      # Copy clustering results to Starling GP directory for Starling search
      GP_PATH="${INDEX_PREFIX_PATH}GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_CUT${GP_CUT}/"
      echo "Copying clustering results to Starling directory: ${GP_PATH}"
      mkdir -p ${GP_PATH}
      cp ${CLUSTERING_PATH}_part_tmp.index ${GP_PATH}_part_tmp.index
      cp ${CLUSTERING_FILE_PATH} ${GP_PATH}_part.bin
      echo "Clustering results copied. You can now run Starling search with USE_PAGE_SEARCH=1"
    fi
  ;;
  gr_layout)
    mkdir -p ${GRAPH_REP_INDEX_PATH}
    if [ -d ${GRAPH_CACHE_INDEX_GP_PATH} ]; then
      echo "Directory ${GRAPH_CACHE_INDEX_GP_PATH} is already exit."
      if [ -f "${GRAPH_CACHE_INDEX_GP_PATH}_part_tmp.index" ]; then
        echo "copy ${GRAPH_CACHE_INDEX_GP_PATH}_part_tmp.index to ${GRAPH_REP_INDEX_PATH}_graph_rep.index."
        GP_FILE_PATH=${GRAPH_CACHE_INDEX_GP_PATH}_part.bin
        cp ${GRAPH_CACHE_INDEX_GP_PATH}_part_tmp.index ${GRAPH_REP_INDEX_PATH}_graph_rep.index
        cp ${GP_FILE_PATH} ${GRAPH_REP_INDEX_PATH}_partition.bin
        exit 0
      else
        echo "${GRAPH_CACHE_INDEX_GP_PATH}_part_tmp.index not found."
        exit 1
      fi
    else
      mkdir -p ${GRAPH_CACHE_INDEX_GP_PATH}
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ ! -f "$OLD_INDEX_FILE" ]; then
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
      fi
      GP_DATA_TYPE=$DATA_TYPE
      GP_FILE_PATH=${GRAPH_CACHE_INDEX_GP_PATH}_part.bin
      echo "Running graph partition... ${GP_FILE_PATH}.log"
      time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES\
        --mode 3 --in_sector_len ${SECTOR_LEN} --out_sector_len ${GR_SECTOR_LEN} > ${GP_FILE_PATH}.log

      echo "Running relayout... ${GRAPH_CACHE_INDEX_GP_PATH}relayout.log"
      echo "$(date): Graph cache index relayout started" >> ${RELAYOUT_TIME_FILE}
      { time ${EXE_PATH}/tests/utils/index_relayout_free_mem ${OLD_INDEX_FILE} ${GP_FILE_PATH} $GP_DATA_TYPE 3 ${SECTOR_LEN} ${GR_SECTOR_LEN} > ${GRAPH_CACHE_INDEX_GP_PATH}relayout.log; } 2>> ${RELAYOUT_TIME_FILE}
      echo "$(date): Graph cache index relayout finished" >> ${RELAYOUT_TIME_FILE}
      cp ${GRAPH_CACHE_INDEX_GP_PATH}_part_tmp.index ${GRAPH_REP_INDEX_PATH}_graph_rep.index
      cp ${GP_FILE_PATH} ${GRAPH_REP_INDEX_PATH}_partition.bin
    fi
  ;;
  split_graph)
    mkdir -p ${GRAPH_PATH}
    if [ -d ${GRAPH_GP_PATH} ]; then
      echo "Directory ${GRAPH_GP_PATH} is already exit."
      if [ -f "${GRAPH_GP_PATH}_part_tmp.index" ]; then
        echo "copy ${GRAPH_GP_PATH}_part_tmp.index to ${GRAPH_PATH}_disk_graph.index."
        GP_FILE_PATH=${GRAPH_GP_PATH}_part.bin
        cp ${GRAPH_GP_PATH}_part_tmp.index ${GRAPH_PATH}_disk_graph.index
        cp ${GP_FILE_PATH} ${GRAPH_PATH}_partition.bin
        exit 0
      else
        echo "${GRAPH_GP_PATH}_part_tmp.index not found."
        exit 1
      fi
    else
      mkdir -p ${GRAPH_GP_PATH}
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ ! -f "$OLD_INDEX_FILE" ]; then
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
      fi
      #using sq index file to gp
      GP_DATA_TYPE=$DATA_TYPE
      GP_FILE_PATH=${GRAPH_GP_PATH}_part.bin
      echo "Running graph partition... ${GP_FILE_PATH}.log"
      # the output len for graph should be 4KB only.
      time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES \
        --mode 1 --in_sector_len ${SECTOR_LEN} --out_sector_len 4096 > ${GP_FILE_PATH}.log

      echo "Running relayout... ${GRAPH_GP_PATH}relayout.log"
      echo "$(date): Split graph relayout started" >> ${RELAYOUT_TIME_FILE}
      { time ${EXE_PATH}/tests/utils/index_relayout_free_mem ${OLD_INDEX_FILE} ${GP_FILE_PATH} $GP_DATA_TYPE 1 ${SECTOR_LEN} 4096 > ${GRAPH_GP_PATH}relayout.log; } 2>> ${RELAYOUT_TIME_FILE}
      echo "$(date): Split graph relayout finished" >> ${RELAYOUT_TIME_FILE}
      #TODO: Use only one index file
      cp ${GRAPH_GP_PATH}_part_tmp.index ${GRAPH_PATH}_disk_graph.index
      cp ${GP_FILE_PATH} ${GRAPH_PATH}_partition.bin
    fi
  ;;
  search)
    mkdir -p ${INDEX_PREFIX_PATH}/search
    mkdir -p ${INDEX_PREFIX_PATH}/result
    if [ ! -d "$INDEX_PREFIX_PATH" ]; then
      echo "Directory $INDEX_PREFIX_PATH is not exist. Build it first?"
      exit 1
    fi

    # Statistics control
    if [ ${ENABLE_IO_UTILIZATION_STATS:-0} -eq 1 ]; then
      if [ -z "${BASELINE_DIR}" ]; then
        echo "Warning: ENABLE_IO_UTILIZATION_STATS=1 but BASELINE_DIR is not set"
        echo "Please set BASELINE_DIR environment variable to enable IO utilization statistics"
        echo "Example: export BASELINE_DIR=/path/to/baseline/data"
      else
        echo "IO Utilization Statistics: ENABLED (BASELINE_DIR=${BASELINE_DIR})"
      fi
    else
      # Disable IO utilization stats by unsetting BASELINE_DIR
      unset BASELINE_DIR
      echo "IO Utilization Statistics: DISABLED"
    fi
    
    # Export per-hop cache stats control
    if [ ${ENABLE_PER_HOP_CACHE_STATS:-0} -eq 1 ]; then
      export ENABLE_PER_HOP_CACHE_STATS=1
      echo "Per-Hop Cache Hit Rate Statistics: ENABLED"
    else
      export ENABLE_PER_HOP_CACHE_STATS=0
      echo "Per-Hop Cache Hit Rate Statistics: DISABLED"
    fi
    echo ""

    # choose the disk index file by settings
    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    if [ $DECO_IMPL -eq 1 ]; then
      if [ ! -f ${GRAPH_GP_PATH}_part.bin ]; then
        echo "Graph Partition file not found. Run the script with split_graph option first."
        exit 1
      fi
      SEARCH_SEC_LEN=${GR_SECTOR_LEN}
      echo "Using Gorgeous"
    else
      if [ $USE_PAGE_SEARCH -eq 1 ]; then
        if [ ! -f ${INDEX_PREFIX_PATH}_partition.bin ]; then
          echo "Partition file not found. Run the script with gp option first."
          exit 1
        fi
        echo "Using Starling"
      else
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
        if [ -f ${OLD_INDEX_FILE} ]; then
          DISK_FILE_PATH=$OLD_INDEX_FILE
        else
          echo "make sure you have not gp the index file"
        fi
        echo "Using DiskANN"
      fi
      SEARCH_SEC_LEN=${SECTOR_LEN}
    fi

    log_arr=()
    for BW in ${BM_LIST[@]}
    do
      for T in ${T_LIST[@]}
      do
        SEARCH_LOG=${INDEX_PREFIX_PATH}search/search_K${K}_CACHE${CACHE}_BW${BW}_T${T}_MEML${MEM_L}_MEMK${MEM_TOPK}_PS${USE_PAGE_SEARCH}_USE_RATIO${PS_USE_RATIO}_GP_LOCK_NUMS${GP_LOCK_NUMS}_GP_CUT${GP_CUT}.log
        echo ""
        echo "========================================="
        echo "Starting search with BW=${BW}, T=${T}"
        echo "Log file: ${SEARCH_LOG}"
        echo "========================================="
        case $1 in
          debug)
            sync; gdb ${EXE_PATH}/tests/search_disk_index --ex "run --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --pq_path_prefix $PQ_PREFIX_PATH \
              --query_file $QUERY_FILE \
              --gt_file $GT_FILE \
              -K $K \
              --result_path ${INDEX_PREFIX_PATH}result/result \
              --num_nodes_to_cache $CACHE \
              -T $T \
              -L ${LS} \
              -W $BW \
              --mem_L ${MEM_L} \
              --sector_len ${SEARCH_SEC_LEN} \
              --mem_index_path ${MEM_INDEX_PATH}_index \
              --mem_sample_path ${MEM_SAMPLE_PATH} \
              --use_page_search ${USE_PAGE_SEARCH} \
              --use_ratio ${PS_USE_RATIO} \
              --pq_ratio ${PQ_FILTER_RATIO} \
              --disk_file_path ${DISK_FILE_PATH} \
              --graph_rep_index_prefix ${GRAPH_REP_INDEX_PATH} \
              --disk_graph_prefix ${GRAPH_PATH} \
              --deco_impl ${DECO_IMPL} \
              --use_graph_rep_index ${USE_DISK_GRAPH_CACHE_INDEX} \
              --mem_graph_use_ratio ${MEM_GRAPH_USE_RATIO} \
              --mem_emb_use_ratio ${MEM_EMB_USE_RATIO} \
              --emb_search_ratio ${EMB_SEARCH_RATIO} \
              --cache_list_file "${CACHE_LIST_FILE}" > ${SEARCH_LOG}"
              log_arr+=( ${SEARCH_LOG} )
          ;;
          release)
            sync; ${EXE_PATH}/tests/search_disk_index --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --pq_path_prefix $PQ_PREFIX_PATH \
              --query_file $QUERY_FILE \
              --gt_file $GT_FILE \
              -K $K \
              --result_path ${INDEX_PREFIX_PATH}result/result \
              --num_nodes_to_cache $CACHE \
              -T $T \
              -L ${LS} \
              -W $BW \
              --mem_L ${MEM_L} \
              --sector_len ${SEARCH_SEC_LEN} \
              --mem_index_path ${MEM_INDEX_PATH}_index \
              --mem_sample_path ${MEM_SAMPLE_PATH} \
              --use_page_search ${USE_PAGE_SEARCH} \
              --use_ratio ${PS_USE_RATIO} \
              --pq_ratio ${PQ_FILTER_RATIO} \
              --disk_file_path ${DISK_FILE_PATH} \
              --graph_rep_index_prefix ${GRAPH_REP_INDEX_PATH} \
              --disk_graph_prefix ${GRAPH_PATH} \
              --deco_impl ${DECO_IMPL} \
              --use_graph_rep_index ${USE_DISK_GRAPH_CACHE_INDEX} \
              --mem_graph_use_ratio ${MEM_GRAPH_USE_RATIO} \
              --mem_emb_use_ratio ${MEM_EMB_USE_RATIO} \
              --emb_search_ratio ${EMB_SEARCH_RATIO} \
              --cache_list_file "${CACHE_LIST_FILE}" > ${SEARCH_LOG}
              log_arr+=( ${SEARCH_LOG} )
          ;;
          *)
            print_usage_and_exit
          ;;
        esac
      done
    done

    if [ ${#log_arr[@]} -ge 1 ]; then
      TITLES=$(cat ${log_arr[0]} | grep -E "^\s+L\s+")
      for f in "${log_arr[@]}"
      do
        printf "$f\n" | tee -a $SUMMARY_FILE_PATH
        printf "${TITLES}\n" | tee -a $SUMMARY_FILE_PATH
        cat $f | grep -E "([0-9]+(\.[0-9]+\s+)){5,}" | tee -a $SUMMARY_FILE_PATH
        printf "\n\n" >> $SUMMARY_FILE_PATH
      done
    fi
  ;;
  dynamic_cache_search)
    # Dynamic Cache (In-Memory First) Search
    echo "========================================="
    echo "Dynamic Cache Search Benchmark"
    echo "========================================="
    
    if [ ! -d "$INDEX_PREFIX_PATH" ]; then
      echo "Error: Index directory $INDEX_PREFIX_PATH not found."
      echo "Please build the index first: ./run_benchmark.sh release build"
      exit 1
    fi
    
    # Check executable
    DYNAMIC_CACHE_EXECUTABLE="${EXE_PATH}/tests/test_dynamic_cache_search"
    if [ ! -f "$DYNAMIC_CACHE_EXECUTABLE" ]; then
      echo "Error: Dynamic cache search executable not found at $DYNAMIC_CACHE_EXECUTABLE"
      echo "Building test_dynamic_cache_search..."
      pushd $EXE_PATH
      make test_dynamic_cache_search -j
      popd
    fi
    
    # Dynamic cache search configuration (from config_local.sh or use defaults)
    DYNAMIC_CACHE_PREFETCH_WINDOW=${DYNAMIC_CACHE_PREFETCH_WINDOW:-4}
    DYNAMIC_CACHE_CAPACITY=${DYNAMIC_CACHE_CAPACITY:-1000}
    DYNAMIC_CACHE_IO_LIMIT=${DYNAMIC_CACHE_IO_LIMIT:-10000}
    
    mkdir -p ${INDEX_PREFIX_PATH}/dynamic_cache_search
    mkdir -p ${INDEX_PREFIX_PATH}/dynamic_cache_result
    
    echo ""
    echo "Dynamic Cache Configuration:"
    echo "  Prefetch window: $DYNAMIC_CACHE_PREFETCH_WINDOW"
    echo "  Cache capacity: $DYNAMIC_CACHE_CAPACITY pages"
    echo "  I/O limit: $DYNAMIC_CACHE_IO_LIMIT"
    echo ""
    
    log_arr=()
    for BW in ${BM_LIST[@]}
    do
      for T in ${T_LIST[@]}
      do
        for L in $LS
        do
          DYNAMIC_CACHE_LOG=${INDEX_PREFIX_PATH}dynamic_cache_search/dc_K${K}_L${L}_PW${DYNAMIC_CACHE_PREFETCH_WINDOW}_CACHE${DYNAMIC_CACHE_CAPACITY}_T${T}_MEML${MEM_L}.log
          DYNAMIC_CACHE_RESULT=${INDEX_PREFIX_PATH}dynamic_cache_result/dc_K${K}_L${L}_PW${DYNAMIC_CACHE_PREFETCH_WINDOW}_CACHE${DYNAMIC_CACHE_CAPACITY}_T${T}_MEML${MEM_L}.txt
          
          echo ""
          echo "Running dynamic cache search with:"
          echo "  K: $K"
          echo "  L: $L"
          echo "  Threads: $T"
          echo "  Prefetch window: $DYNAMIC_CACHE_PREFETCH_WINDOW"
          echo "  Cache capacity: $DYNAMIC_CACHE_CAPACITY"
          echo ""
          
          sync; ${DYNAMIC_CACHE_EXECUTABLE} \
            --data_type $DATA_TYPE \
            --index_path_prefix ${INDEX_PREFIX_PATH} \
            --pq_path_prefix ${PQ_PREFIX_PATH} \
            --query_file $QUERY_FILE \
            --gt_file $GT_FILE \
            --mem_index_path ${MEM_INDEX_PATH}_index \
            --K $K \
            --L $L \
            --prefetch_window $DYNAMIC_CACHE_PREFETCH_WINDOW \
            --cache_capacity $DYNAMIC_CACHE_CAPACITY \
            --io_limit $DYNAMIC_CACHE_IO_LIMIT \
            --num_threads $T \
            --mem_L $MEM_L \
            --emb_search_ratio $EMB_SEARCH_RATIO \
            --sector_len ${GR_SECTOR_LEN} \
            > ${DYNAMIC_CACHE_LOG} 2>&1
          
          log_arr+=( ${DYNAMIC_CACHE_LOG} )
          echo "Results saved to: $DYNAMIC_CACHE_LOG"
        done
      done
    done
    
    # Summarize results
    if [ ${#log_arr[@]} -ge 1 ]; then
      DYNAMIC_CACHE_SUMMARY=${INDEX_PREFIX_PATH}dynamic_cache_search/dynamic_cache_summary.log
      echo "" | tee $DYNAMIC_CACHE_SUMMARY
      echo "=========================================" | tee -a $DYNAMIC_CACHE_SUMMARY
      echo "Dynamic Cache Search Summary" | tee -a $DYNAMIC_CACHE_SUMMARY
      echo "=========================================" | tee -a $DYNAMIC_CACHE_SUMMARY
      for f in "${log_arr[@]}"
      do
        echo "" | tee -a $DYNAMIC_CACHE_SUMMARY
        echo "Log: $f" | tee -a $DYNAMIC_CACHE_SUMMARY
        echo "---" | tee -a $DYNAMIC_CACHE_SUMMARY
        # Extract key statistics
        grep -E "Average|Recall|QPS|Latency|I/O|Cache" $f | tee -a $DYNAMIC_CACHE_SUMMARY || true
      done
      echo "" | tee -a $DYNAMIC_CACHE_SUMMARY
      echo "Summary saved to: $DYNAMIC_CACHE_SUMMARY"
    fi
    
    echo ""
    echo "========================================="
    echo "Dynamic Cache Search Completed"
    echo "========================================="
  ;;
  hybridcache_search)
    # Hybrid Cache Search with Two-Phase Strategy and Hybrid Caching
    echo "========================================="
    echo "Hybrid Cache Search Benchmark"
    echo "========================================="
    
    if [ ! -d "$INDEX_PREFIX_PATH" ]; then
      echo "Error: Index directory $INDEX_PREFIX_PATH not found."
      echo "Please build the index first: ./run_benchmark.sh release build"
      exit 1
    fi
    
    # Check executable
    HC_EXECUTABLE="${EXE_PATH}/tests/test_hybridcache_search"
    if [ ! -f "$HC_EXECUTABLE" ]; then
      echo "Error: Hybrid cache search executable not found at $HC_EXECUTABLE"
      echo "Building test_hybridcache_search..."
      pushd $EXE_PATH
      make test_hybridcache_search -j
      popd
    fi
    
    HYBRIDCACHE_THETA=${HYBRIDCACHE_THETA:-0.3}
    HYBRIDCACHE_PREFETCH_PAGES=${HYBRIDCACHE_PREFETCH_PAGES:-3}
    HYBRIDCACHE_STATIC_CACHE=${HYBRIDCACHE_STATIC_CACHE:-100}
    HYBRIDCACHE_DYNAMIC_CACHE=${HYBRIDCACHE_DYNAMIC_CACHE:-500}
    HYBRIDCACHE_ENABLE_STATS=${HYBRIDCACHE_ENABLE_STATS:-1}
    STATIC_CACHE_FILE=${STATIC_CACHE_FILE:-""}
    STATIC_CACHE_MAX_NODES=${STATIC_CACHE_MAX_NODES:-0}
    
    mkdir -p ${INDEX_PREFIX_PATH}/hybridcache_search
    mkdir -p ${INDEX_PREFIX_PATH}/hybridcache_result
    
    # Check if clustering partitioning was used
    if [ ! -f "${INDEX_PREFIX_PATH}_partition.bin" ]; then
      echo ""
      echo "Warning: Partition file not found."
      echo "Hybrid cache search works best with clustering partitioning."
      echo "Consider running: ./run_benchmark.sh release clustering"
      echo ""
    fi
    
    echo ""
    echo "Hybrid Cache Configuration:"
    echo "  Theta (phase transition): $HYBRIDCACHE_THETA"
    echo "  Prefetch pages: $HYBRIDCACHE_PREFETCH_PAGES"
    echo "  Static cache: $HYBRIDCACHE_STATIC_CACHE pages"
    echo "  Dynamic cache: $HYBRIDCACHE_DYNAMIC_CACHE pages"
    echo "  Enable statistics: $HYBRIDCACHE_ENABLE_STATS"
    echo "  Static cache file: $STATIC_CACHE_FILE"
    echo "  Static cache max nodes: $STATIC_CACHE_MAX_NODES"
    echo ""
    
    log_arr=()
    for BW in ${BM_LIST[@]}
    do
      for T in ${T_LIST[@]}
      do
        for L in $LS
        do
          HC_LOG=${INDEX_PREFIX_PATH}hybridcache_search/hc_K${K}_L${L}_THETA${HYBRIDCACHE_THETA}_PF${HYBRIDCACHE_PREFETCH_PAGES}_T${T}_MEML${MEM_L}.log
          HC_RESULT=${INDEX_PREFIX_PATH}hybridcache_result/hc_K${K}_L${L}_THETA${HYBRIDCACHE_THETA}_PF${HYBRIDCACHE_PREFETCH_PAGES}_T${T}_MEML${MEM_L}.txt
          HC_STATS=${INDEX_PREFIX_PATH}hybridcache_result/hc_stats_K${K}_L${L}_THETA${HYBRIDCACHE_THETA}_PF${HYBRIDCACHE_PREFETCH_PAGES}_T${T}_MEML${MEM_L}.csv
          
          echo ""
          echo "Running hybrid cache search with:"
          echo "  K: $K"
          echo "  L: $L"
          echo "  Beam width: $BW"
          echo "  Threads: $T"
          echo "  Theta: $HYBRIDCACHE_THETA"
          echo "  Prefetch pages: $HYBRIDCACHE_PREFETCH_PAGES"
          echo ""
          
          # Build static cache parameters
          STATIC_CACHE_ARGS=""
          if [ -n "$STATIC_CACHE_FILE" ] && [ -f "$STATIC_CACHE_FILE" ]; then
            STATIC_CACHE_ARGS="--static_cache_file $STATIC_CACHE_FILE --static_cache_max_nodes $STATIC_CACHE_MAX_NODES"
          fi
          
          sync; ${HC_EXECUTABLE} \
            --data_type $DATA_TYPE \
            --dist_fn $DIST_FN \
            --index_path_prefix ${INDEX_PREFIX_PATH} \
            --pq_path_prefix ${PQ_PREFIX_PATH} \
            --query_file $QUERY_FILE \
            --gt_file $GT_FILE \
            --mem_index_path ${MEM_INDEX_PATH}_index \
            --K $K \
            --L $L \
            --beamwidth $BW \
            --num_threads $T \
            --mem_L $MEM_L \
            --io_limit 10000 \
            --pq_filter_ratio $PQ_FILTER_RATIO \
            --emb_search_ratio $EMB_SEARCH_RATIO \
            --hybridcache_theta $HYBRIDCACHE_THETA \
            --hybridcache_prefetch_pages $HYBRIDCACHE_PREFETCH_PAGES \
            --hybridcache_static_cache $HYBRIDCACHE_STATIC_CACHE \
            --hybridcache_dynamic_cache $HYBRIDCACHE_DYNAMIC_CACHE \
            --hybridcache_enable_stats $HYBRIDCACHE_ENABLE_STATS \
            --sector_len ${GR_SECTOR_LEN} \
            --result_path $HC_RESULT \
            --stats_file $HC_STATS \
            $STATIC_CACHE_ARGS \
            > ${HC_LOG} 2>&1
          
          log_arr+=( ${HC_LOG} )
          echo "Results saved to: $HC_LOG"
          echo "Statistics saved to: $HC_STATS"
        done
      done
    done
    
    if [ ${#log_arr[@]} -ge 1 ]; then
      HC_SUMMARY=${INDEX_PREFIX_PATH}hybridcache_search/hybridcache_summary.log
      echo "" | tee $HC_SUMMARY
      echo "=========================================" | tee -a $HC_SUMMARY
      echo "Hybrid Cache Search Summary" | tee -a $HC_SUMMARY
      echo "=========================================" | tee -a $HC_SUMMARY
      for f in "${log_arr[@]}"
      do
        echo "" | tee -a $HC_SUMMARY
        echo "Log: $f" | tee -a $HC_SUMMARY
        echo "---" | tee -a $HC_SUMMARY
        # Extract key statistics
        grep -E "Average|Recall|QPS|Latency|Phase|Cache|Prefetch" $f | tee -a $HC_SUMMARY || true
      done
      echo "" | tee -a $HC_SUMMARY
      echo "Summary saved to: $HC_SUMMARY"
    fi
    
    echo ""
    echo "========================================="
    echo "Hybrid Cache Search Completed"
    echo "========================================="
  ;;
  *)
    print_usage_and_exit
  ;;
esac
