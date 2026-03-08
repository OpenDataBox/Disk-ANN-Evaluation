#!/bin/sh

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_dataset.sh"

# Build Parameters
BUILD_L=128         # Build complexity
BUILD_R=48          # Graph degree  
BUILD_T=128         # Build threads

# Memory Index (optional, for faster search)
BUILD_MEM_INDEX=0   # 1=enable, 0=disable
MEM_L=128           # Memory index search complexity

# Search Parameters
SEARCH_MODE=2       # 0=beam, 1=page, 2=pipe, 3=coro
SEARCH_THREADS=1    # Search threads
L_VALUES="20 30 40 50"  # L values to test

# Paths
BUILD_DIR="../build"
DATA_DIR="./data"
INDEX_PREFIX="${DATA_DIR}/${PREFIX}/index"
MEM_INDEX_PREFIX="${INDEX_PREFIX}_mem.index"
RESULTS_DIR="results"
LOG_SUFFIX=$(date +%Y%m%d_%H%M%S)

print_config() {
    echo "=== Configuration ==="
    echo "Dataset: ${PREFIX}"
    echo "Data: ${BASE_PATH}"
    echo "Index: ${INDEX_PREFIX}"
    echo "Search Mode: ${SEARCH_MODE}"
    echo "L Values: ${L_VALUES}"
    echo "===================="
}