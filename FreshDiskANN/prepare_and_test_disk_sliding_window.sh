#!/bin/bash

# Complete script for disk-based sliding window test
# Step 1: Build disk index
# Step 2: Convert to SyncIndex format
# Step 3: Run sliding window test

# Memory index parameters
L_MEM=128
R_MEM=24
ALPHA_MEM=1.2

# Disk index parameters  
L_DISK=128
R_DISK=48
ALPHA_DISK=1.2

# Sliding window parameters
INITIAL_POINTS=800000
BATCH_SIZE=2000
NUM_BATCHES=100

# System parameters
NUM_SHARDS=1  
NUM_PQ_CHUNKS=8
NODES_TO_CACHE=0
BEAM_WIDTH=4
RECALL_AT=10

# Search L values
SEARCH_L_LIST="20 30 40 50 60"

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Output directory
OUTPUT_DIR="./disk_sliding_window_results"
mkdir -p ${OUTPUT_DIR}

# Master log file for the entire run
MASTER_LOG="${OUTPUT_DIR}/master_${TIMESTAMP}.log"
echo "=========================================="  | tee ${MASTER_LOG}
echo "Sliding Window Test - All Datasets"  | tee -a ${MASTER_LOG}
echo "Started at: $(date)"  | tee -a ${MASTER_LOG}
echo "=========================================="  | tee -a ${MASTER_LOG}
echo ""  | tee -a ${MASTER_LOG}

# Index directory on SSD
INDEX_DIR="./disk_indices"
mkdir -p ${INDEX_DIR}

# Function to prepare and test a dataset
prepare_and_test() {
    local DATASET_NAME=$1
    local DATA_FILE=$2
    local QUERY_FILE=$3
    local GT_DIR=$4
    local DATA_TYPE=$5
    local R_BUILD=$6
    local L_BUILD=$7
    local B=$8
    local DATASET_PQ_CHUNKS=$9  # Dataset-specific PQ chunks
    
    echo "=========================================="
    echo "Preparing and testing ${DATASET_NAME}"
    echo "=========================================="
    echo "=========================================="  | tee -a ${MASTER_LOG}
    echo "Starting ${DATASET_NAME} at $(date)"  | tee -a ${MASTER_LOG}
    echo "=========================================="  | tee -a ${MASTER_LOG}
    
    local DATASET_DIR="${OUTPUT_DIR}/${DATASET_NAME}"
    mkdir -p ${DATASET_DIR}
    
    local DISK_INDEX_PREFIX="${INDEX_DIR}/${DATASET_NAME}_disk"
    local SYNC_INDEX_PREFIX="${INDEX_DIR}/${DATASET_NAME}_index"
    local LOG_FILE="${DATASET_DIR}/${DATASET_NAME}_${TIMESTAMP}.log"
    
    echo "Step 1: Building disk index..." | tee ${LOG_FILE}
    echo "  Data: ${DATA_FILE}" | tee -a ${LOG_FILE}
    echo "  Index: ${DISK_INDEX_PREFIX}" | tee -a ${LOG_FILE}
    echo "  Parameters: R=${R_BUILD}, L=${L_BUILD}, B=${B}" | tee -a ${LOG_FILE}
    
    # Check if disk index already exists
    if [ -f "${DISK_INDEX_PREFIX}_disk.index" ]; then
        echo "  Disk index already exists, skipping build" | tee -a ${LOG_FILE}
    else
        ./build/tests/build_disk_index \
            ${DATA_TYPE} \
            ${DATA_FILE} \
            ${DISK_INDEX_PREFIX} \
            ${R_BUILD} \
            ${L_BUILD} \
            ${B} \
            2 \
            128 \
            2>&1 | tee -a ${LOG_FILE}
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "Failed to build disk index" | tee -a ${LOG_FILE}
            return 1
        fi
    fi
    
    echo "" | tee -a ${LOG_FILE}
    echo "Step 2: Converting to SyncIndex format..." | tee -a ${LOG_FILE}
    
    # 清理旧的SyncIndex文件，确保从干净状态开始
    if [ -f "${SYNC_INDEX_PREFIX}0_lti_disk.index" ]; then
        echo "  Removing old SyncIndex files..." | tee -a ${LOG_FILE}
        rm -f ${SYNC_INDEX_PREFIX}*
        echo "  Old files removed" | tee -a ${LOG_FILE}
    fi
    
    echo "  Building SyncIndex from disk index..." | tee -a ${LOG_FILE}
    ./build/tests/static_sync_index \
        ${DATA_TYPE} \
        ${DATA_FILE} \
        ${L_MEM} \
        ${R_MEM} \
        ${ALPHA_MEM} \
        ${L_DISK} \
        ${R_DISK} \
        ${ALPHA_DISK} \
        ${INITIAL_POINTS} \
        ${NUM_SHARDS} \
        ${DATASET_PQ_CHUNKS} \
        ${NODES_TO_CACHE} \
        ${SYNC_INDEX_PREFIX} \
        2>&1 | tee -a ${LOG_FILE}
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Failed to convert to SyncIndex" | tee -a ${LOG_FILE}
        return 1
    fi
    
    echo "" | tee -a ${LOG_FILE}
    echo "Step 3: Running sliding window test..." | tee -a ${LOG_FILE}
    
    ./build/tests/test_sliding_window \
        ${DATA_TYPE} \
        ${DATA_FILE} \
        ${L_MEM} \
        ${R_MEM} \
        ${ALPHA_MEM} \
        ${L_DISK} \
        ${R_DISK} \
        ${ALPHA_DISK} \
        ${INITIAL_POINTS} \
        ${BATCH_SIZE} \
        ${NUM_BATCHES} \
        ${NUM_SHARDS} \
        ${DATASET_PQ_CHUNKS} \
        ${NODES_TO_CACHE} \
        ${SYNC_INDEX_PREFIX} \
        ${QUERY_FILE} \
        ${GT_DIR} \
        ${RECALL_AT} \
        ${BEAM_WIDTH} \
        ${LOG_FILE} \
        ${SEARCH_L_LIST}
    
    local EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "Test completed successfully" | tee -a ${LOG_FILE}
        echo "${DATASET_NAME} completed successfully at $(date)" | tee -a ${MASTER_LOG}
    else
        echo "Test FAILED with exit code ${EXIT_CODE}" | tee -a ${LOG_FILE}
        echo "${DATASET_NAME} FAILED with exit code ${EXIT_CODE} at $(date)" | tee -a ${MASTER_LOG}
    fi
    
    echo ""
    echo "" | tee -a ${MASTER_LOG}
}

# Example: Test your dataset
# Modify the parameters below for your dataset
# Parameters explanation:
#   DATASET_NAME: Name for output directory
#   DATA_FILE: Path to base vectors (.bin format)
#   QUERY_FILE: Path to query vectors (.bin format)
#   GT_DIR: Directory containing ground truth files (gt_0.bin, gt_2000.bin, ...)
#   DATA_TYPE: Data type (float/int8/uint8)
#   R_BUILD: Degree for disk index build
#   L_BUILD: Search list size for disk index build
#   B: Compression ratio for PQ (0.03 for 128-dim, 0.06 for 256-dim, 0.25 for 960-dim)
#   DATASET_PQ_CHUNKS: Number of PQ chunks (typically dim/8 or dim/16)

prepare_and_test "my_dataset" \
    "/path/to/your/dataset_base.bin" \
    "/path/to/your/dataset_query.bin" \
    "/path/to/your/gt_directory" \
    "float" \
    48 128 0.03 16

echo "=========================================="  | tee -a ${MASTER_LOG}
echo "All tests completed!"  | tee -a ${MASTER_LOG}
echo "Finished at: $(date)"  | tee -a ${MASTER_LOG}
echo "Results saved in ${OUTPUT_DIR}"  | tee -a ${MASTER_LOG}
echo "Master log: ${MASTER_LOG}"  | tee -a ${MASTER_LOG}
echo "=========================================="  | tee -a ${MASTER_LOG}
