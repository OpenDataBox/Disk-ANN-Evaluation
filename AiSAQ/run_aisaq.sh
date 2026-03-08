#!/bin/bash

################################################################################
# AISAQ Experiment Script
# 
# This script demonstrates how to build and search AISAQ indexes.
# Modify the parameters below to suit your dataset and requirements.
################################################################################

set -e

# ============================================================================
# CONFIGURABLE PARAMETERS - Modify these for your dataset
# ============================================================================

# Execution mode: build, search, or both
MODE=${1:-"both"}  # build, search, or both

# --- Dataset Paths (MODIFY THESE) ---
# Replace with your actual data paths
DATA_PATH="/path/to/your/dataset_base.bin"
QUERY_FILE="/path/to/your/dataset_query.bin"
GT_FILE="/path/to/your/dataset_groundtruth.bin"
DATA_TYPE="float"           # Data type: float, uint8, or int8
DIM=128                     # Vector dimension

# --- Build Parameters (MODIFY AS NEEDED) ---
BUILD_R=48                  # Max degree (graph connectivity)
BUILD_L=128                 # Build complexity (higher = better quality, slower build)
BUILD_M=2                   # Build memory budget in GB
BUILD_T=128                 # Number of threads for building

# B: Search DRAM budget in GB
# This controls how much memory is used during search for PQ data
# Formula: B = (num_pq_chunks * num_points) / (1024^3)
# Example: For 1M points with 16 chunks: B ≈ 0.015 GB
B=0.015

# QD: Number of PQ chunks (quantized dimensions)
# Common practice: DIM / 8
# If set to 0, it will be auto-calculated from B
# Recommended: Set explicitly for predictable behavior
QD=$((DIM / 8))             # Default: DIM/8 (e.g., 128/8 = 16)

# --- AISAQ-Specific Build Parameters ---
# inline_pq: Number of PQ vectors stored inline with each node
#   - 0: No inline PQ (pure all-in-disk, minimum memory)
#   - 16-32: Recommended for balanced performance
#   - -1: Auto-select based on node size
INLINE_PQ=16

# rearrange: Optimize PQ vector layout for better I/O locality
REARRANGE=true

# num_entry_points: Number of entry points for search (1-1000)
NUM_ENTRY_POINTS=512

# --- Search Parameters (MODIFY AS NEEDED) ---
# L_VALUES: List of L values to test (space-separated)
# Higher L = better recall but slower search
L_VALUES="250 300 350 400 500 600 700 800 1000 1200 1500"

# K: Number of nearest neighbors to return
K=100

# Beam width: Number of candidates to explore in parallel
BEAM_WIDTH=4

# Vector beamwidth: AISAQ-specific parameter for I/O parallelism
VECTOR_BEAMWIDTH=1

# PQ I/O engine: aio or uring (uring is faster if supported)
PQ_IO_ENGINE="aio"

# ============================================================================
# DERIVED PARAMETERS - Usually don't need to modify
# ============================================================================

INDEX_PREFIX="./indexes/aisaq_index"
RESULT_PREFIX="./results/aisaq_results"

mkdir -p ./indexes ./results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

echo "=== AISAQ Experiment ==="
echo "Mode: $MODE"
echo "Data type: $DATA_TYPE"
echo "Dimension: $DIM"
echo "PQ Chunks (QD): $QD"
echo "Search DRAM Budget (B): ${B} GB"
echo ""

# --- Build Index ---
if [[ "$MODE" == "build" || "$MODE" == "both" ]]; then
    echo "=== Building Index ==="
    echo "Parameters:"
    echo "  R (max degree): $BUILD_R"
    echo "  L (build complexity): $BUILD_L"
    echo "  M (build memory): ${BUILD_M} GB"
    echo "  B (search DRAM budget): ${B} GB"
    echo "  QD (PQ chunks): $QD"
    echo "  T (threads): $BUILD_T"
    echo "  inline_pq: $INLINE_PQ"
    echo "  rearrange: $REARRANGE"
    echo "  num_entry_points: $NUM_ENTRY_POINTS"
    echo ""
    
    BUILD_CMD="./build/apps/build_disk_index \
        --data_type $DATA_TYPE \
        --dist_fn l2 \
        --data_path $DATA_PATH \
        --index_path_prefix $INDEX_PREFIX \
        -R $BUILD_R \
        -L $BUILD_L \
        -B $B \
        -M $BUILD_M \
        -T $BUILD_T \
        --use_aisaq \
        --inline_pq $INLINE_PQ \
        --num_entry_points $NUM_ENTRY_POINTS"
    
    # Add QD if specified (non-zero)
    if [[ $QD -gt 0 ]]; then
        BUILD_CMD="$BUILD_CMD --QD $QD"
    fi
    
    # Add rearrange flag if enabled
    if [[ "$REARRANGE" == "true" ]]; then
        BUILD_CMD="$BUILD_CMD --rearrange"
    fi
    
    echo "Executing: $BUILD_CMD"
    echo ""
    eval $BUILD_CMD
    
    echo ""
    echo "Index built successfully!"
    echo "Index files:"
    ls -lh ${INDEX_PREFIX}* 2>/dev/null | awk '{print "  " $9 ": " $5}' || echo "  (no files found)"
    echo ""
fi

# --- Run Search ---
if [[ "$MODE" == "search" || "$MODE" == "both" ]]; then
    echo "=== Running Search ==="
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="./results/aisaq_experiment_${TIMESTAMP}.log"
    
    echo "Parameters:"
    echo "  K (top-K): $K"
    echo "  Beam width: $BEAM_WIDTH"
    echo "  Vector beamwidth: $VECTOR_BEAMWIDTH"
    echo "  PQ I/O engine: $PQ_IO_ENGINE"
    echo "  L values: $L_VALUES"
    echo "  Log file: $LOG_FILE"
    echo ""
    
    ./build/apps/search_disk_index \
        --data_type $DATA_TYPE \
        --dist_fn l2 \
        --index_path_prefix $INDEX_PREFIX \
        --query_file $QUERY_FILE \
        --gt_file $GT_FILE \
        --result_path $RESULT_PREFIX \
        -K $K \
        -W $BEAM_WIDTH \
        -L $L_VALUES \
        -T 1 \
        --use_aisaq \
        --pq_read_io_engine $PQ_IO_ENGINE \
        -V $VECTOR_BEAMWIDTH 2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "Search completed!"
    echo "Log saved to: $LOG_FILE"
fi

echo ""
echo "=== Done ==="
echo ""
echo "Usage:"
echo "  $0 build   - Build index only"
echo "  $0 search  - Search only (requires existing index)"
echo "  $0 both    - Build and search (default)"
echo ""
echo "Key parameters to tune:"
echo "  - B: Search DRAM budget (affects PQ memory usage)"
echo "  - QD: Number of PQ chunks (default: DIM/8)"
echo "  - inline_pq: 0 (min memory) to 32 (better performance)"
echo "  - BUILD_L: Higher = better quality, slower build"
echo "  - L_VALUES: Higher L = better recall, slower search"

