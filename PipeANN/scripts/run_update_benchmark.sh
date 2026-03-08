#!/bin/bash

# Overall performance test with sliding window updates
# Works with any dataset configured in config_dataset.sh and config_local.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_local.sh"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --steps N          Number of update steps (default: 100)"
    echo "  --batch-size N     Vectors per step (default: 2000)"
    echo "  --l-values \"...\"   L values to test (default: from config_local.sh)"
    echo "  --index-prefix P   Index path (default: from config)"
    echo "  --gt-prefix P      Ground truth path (default: from config)"
    echo ""
    echo "Example:"
    echo "  $0 --steps 50 --batch-size 1000 --l-values \"20 30 40\""
    exit 1
}

# Default parameters
STEPS=100
BATCH_SIZE=2000
L_DISK=128
RECALL_AT=10
BEAM_WIDTH=4
L_SEARCH_VALUES="$L_VALUES"
INDEX_PATH="$INDEX_PREFIX"
GT_PATH="${DATA_DIR}/${PREFIX}/gt_update"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --l-values)
            L_SEARCH_VALUES="$2"
            shift 2
            ;;
        --index-prefix)
            INDEX_PATH="$2"
            shift 2
            ;;
        --gt-prefix)
            GT_PATH="$2"
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

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRIPT_DIR}/results_update_${PREFIX}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/experiment.log"

# Print configuration
echo "=== Overall Performance Test ===" | tee "${LOG_FILE}"
echo "Dataset: ${PREFIX}" | tee -a "${LOG_FILE}"
echo "Data: ${BASE_PATH}" | tee -a "${LOG_FILE}"
echo "Query: ${QUERY_FILE}" | tee -a "${LOG_FILE}"
echo "Index: ${INDEX_PATH}" | tee -a "${LOG_FILE}"
echo "Ground Truth: ${GT_PATH}" | tee -a "${LOG_FILE}"
echo "Steps: ${STEPS} (${BATCH_SIZE} vectors/step)" | tee -a "${LOG_FILE}"
echo "L values: ${L_SEARCH_VALUES}" | tee -a "${LOG_FILE}"
echo "===============================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Check files
if [ ! -f "${BUILD_DIR}/tests/overall_performance" ]; then
    echo "Error: overall_performance binary not found" | tee -a "${LOG_FILE}"
    exit 1
fi

if [ ! -f "${BASE_PATH}" ]; then
    echo "Error: Data file not found: ${BASE_PATH}" | tee -a "${LOG_FILE}"
    exit 1
fi

if [ ! -f "${QUERY_FILE}" ]; then
    echo "Error: Query file not found: ${QUERY_FILE}" | tee -a "${LOG_FILE}"
    exit 1
fi

if [ ! -f "${INDEX_PATH}_disk.index" ]; then
    echo "Error: Index not found: ${INDEX_PATH}_disk.index" | tee -a "${LOG_FILE}"
    exit 1
fi

if [ ! -d "${GT_PATH}" ]; then
    echo "Warning: Ground truth directory not found: ${GT_PATH}" | tee -a "${LOG_FILE}"
    echo "Recall metrics will be unavailable" | tee -a "${LOG_FILE}"
fi

# Run experiment
echo "Starting at $(date)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

CMD="${BUILD_DIR}/tests/overall_performance ${DATA_TYPE} ${BASE_PATH} ${L_DISK} ${INDEX_PATH} ${QUERY_FILE} ${GT_PATH} ${RECALL_AT} ${BEAM_WIDTH} ${STEPS} ${L_SEARCH_VALUES}"

echo "Command: ${CMD}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

${CMD} 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "${LOG_FILE}"
echo "===============================" | tee -a "${LOG_FILE}"
echo "Finished at $(date)" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Results: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "===============================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
