#!/bin/bash

# Generate ground truth for sliding window update experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_local.sh"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --initial N        Initial index size (default: 800000)"
    echo "  --total N          Total vectors in dataset (default: 1002000)"
    echo "  --batch N          Batch size per step (default: 2000)"
    echo "  --topk N           Top-K for ground truth (default: 1000)"
    echo "  --output-dir DIR   Output directory (default: ./data/\${PREFIX}/gt_update)"
    echo ""
    echo "Example:"
    echo "  $0 --initial 500000 --total 700000 --batch 1000"
    exit 1
}

# Default parameters
INITIAL_SIZE=800000
TOTAL_SIZE=1002000
BATCH_SIZE=2000
TOPK=1000
OUTPUT_DIR="${DATA_DIR}/${PREFIX}/gt_update"
FULL_GT_FILE="${DATA_DIR}/${PREFIX}/gt_full_top${TOPK}.bin"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --initial)
            INITIAL_SIZE="$2"
            shift 2
            ;;
        --total)
            TOTAL_SIZE="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --topk)
            TOPK="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "=== Generate Ground Truth ==="
echo "Dataset: ${PREFIX}"
echo "Initial: ${INITIAL_SIZE}"
echo "Total: ${TOTAL_SIZE}"
echo "Batch: ${BATCH_SIZE}"
echo "Top-K: ${TOPK}"
echo "Output: ${OUTPUT_DIR}"
echo "============================="
echo ""

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname ${FULL_GT_FILE})"

# Generate full GT if not exists
if [ ! -f "${FULL_GT_FILE}" ]; then
    echo "Generating top-${TOPK} ground truth..."
    LOG_FILE="${OUTPUT_DIR}/generate_full_gt_$(date +%Y%m%d_%H%M%S).log"
    
    { time ${BUILD_DIR}/tests/utils/compute_groundtruth \
        ${DATA_TYPE} \
        ${DIST_FN} \
        ${BASE_PATH} \
        ${QUERY_FILE} \
        ${TOPK} \
        ${FULL_GT_FILE} \
        null \
        null; } 2>&1 | tee "${LOG_FILE}"
    
    echo "Full GT: ${FULL_GT_FILE}"
else
    echo "Using existing GT: ${FULL_GT_FILE}"
fi

echo ""
echo "Extracting per-step ground truth..."
LOG_FILE="${OUTPUT_DIR}/extract_gt_$(date +%Y%m%d_%H%M%S).log"

{ time ${BUILD_DIR}/tests/utils/gt_update \
    ${FULL_GT_FILE} \
    ${INITIAL_SIZE} \
    ${TOTAL_SIZE} \
    ${BATCH_SIZE} \
    10 \
    ${OUTPUT_DIR} \
    0; } 2>&1 | tee "${LOG_FILE}"

echo ""
echo "=== Complete ==="
echo "Output: ${OUTPUT_DIR}"
echo "Files: $(ls ${OUTPUT_DIR}/gt_*.bin 2>/dev/null | wc -l)"
echo "================"

