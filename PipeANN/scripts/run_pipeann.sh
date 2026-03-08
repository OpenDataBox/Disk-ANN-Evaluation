#!/bin/bash

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/config_local.sh"

print_usage_and_exit() {
    echo "Usage: ./run_pipeann.sh [build|search|clean]"
    echo ""
    echo "Commands:"
    echo "  build   - Build disk index"
    echo "  search  - Run search benchmark"
    echo "  clean   - Clean generated files"
    exit 1
}

check_dir_and_make_if_absent() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    fi
}

check_file_exists() {
    local file=$1
    local desc=$2
    if [ ! -f "$file" ]; then
        echo "Error: $desc file not found: $file"
        exit 1
    fi
}

build_index() {
    echo "=== Building PipeANN Index ==="
    print_config
    
    check_file_exists "$BASE_PATH" "Base data"
    check_dir_and_make_if_absent "$DATA_DIR"
    check_dir_and_make_if_absent "$(dirname "$INDEX_PREFIX")"
    check_dir_and_make_if_absent "$RESULTS_DIR"
    
    echo "Building disk index..."
    LOG_FILE="${RESULTS_DIR}/build_${PREFIX}_${LOG_SUFFIX}.log"
    
    { time ${BUILD_DIR}/tests/build_disk_index \
        $DATA_TYPE \
        $BASE_PATH \
        $INDEX_PREFIX \
        $BUILD_R \
        $BUILD_L \
        $N_PQ_CODE \
        2 \
        $BUILD_T \
        $DIST_FN \
        pq; } 2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "Index built: ${INDEX_PREFIX}_disk.index"
    echo "Log: $LOG_FILE"
}

run_search() {
    echo "=== Running Search Benchmark ==="
    print_config
    
    check_file_exists "$BASE_PATH" "Base data"
    check_file_exists "$QUERY_FILE" "Query"
    check_file_exists "$GT_FILE" "Ground truth"
    check_file_exists "${INDEX_PREFIX}_disk.index" "Disk index"
    check_dir_and_make_if_absent "$RESULTS_DIR"
    
    LOG_FILE="${RESULTS_DIR}/search_${PREFIX}_mode${SEARCH_MODE}_${LOG_SUFFIX}.log"
    
    echo "Testing L values: $L_VALUES"
    echo ""
    
    ${BUILD_DIR}/tests/search_disk_index \
        $DATA_TYPE \
        $INDEX_PREFIX \
        $SEARCH_THREADS \
        4 \
        $QUERY_FILE \
        $GT_FILE \
        $K \
        $DIST_FN \
        pq \
        $SEARCH_MODE \
        $MEM_L \
        $L_VALUES 2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "Results: $LOG_FILE"
}

clean_files() {
    echo "=== Cleaning Files ==="
    
    if [ -d "$DATA_DIR" ]; then
        echo "Removing: $DATA_DIR"
        rm -rf "$DATA_DIR"
    fi
    
    if [ -d "$RESULTS_DIR" ]; then
        echo "Removing: $RESULTS_DIR"
        rm -rf "$RESULTS_DIR"
    fi
    
    echo "Done."
}

# Main
case $1 in
    build)
        build_index
        ;;
    search)
        run_search
        ;;
    clean)
        clean_files
        ;;
    *)
        print_usage_and_exit
        ;;
esac