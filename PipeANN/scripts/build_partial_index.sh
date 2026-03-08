#!/bin/bash

# Build partial index for update experiments
# Creates an index with subset of vectors for sliding window tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_local.sh"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --num-vectors N    Number of vectors to include (default: 800000)"
    echo "  --output-dir DIR   Output directory (default: ./data/\${PREFIX}_partial)"
    echo ""
    echo "Example:"
    echo "  $0 --num-vectors 500000"
    exit 1
}

# Default parameters
NUM_VECTORS=800000
OUTPUT_DIR="${DATA_DIR}/${PREFIX}_partial"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-vectors)
            NUM_VECTORS="$2"
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

echo "=== Building Partial Index ==="
echo "Dataset: ${PREFIX}"
echo "Vectors: ${NUM_VECTORS}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================="
echo ""

mkdir -p "${OUTPUT_DIR}"

TEMP_DATA="${OUTPUT_DIR}/temp_subset.bin"
INDEX_PREFIX="${OUTPUT_DIR}/index"

# Create subset
echo "Creating ${NUM_VECTORS}-vector subset..."
python3 - <<EOF
import numpy as np
import struct

with open('${BASE_PATH}', 'rb') as f:
    npts = struct.unpack('i', f.read(4))[0]
    dim = struct.unpack('i', f.read(4))[0]
    print(f"Original: {npts} vectors, {dim} dims")
    
    n_subset = min(${NUM_VECTORS}, npts)
    
    if '${DATA_TYPE}' == 'float':
        dtype = np.float32
    elif '${DATA_TYPE}' == 'int8':
        dtype = np.int8
    elif '${DATA_TYPE}' == 'uint8':
        dtype = np.uint8
    else:
        raise ValueError(f"Unknown type: ${DATA_TYPE}")
    
    data = np.fromfile(f, dtype=dtype, count=n_subset * dim).reshape(n_subset, dim)
    print(f"Subset: {n_subset} vectors")

with open('${TEMP_DATA}', 'wb') as f:
    f.write(struct.pack('i', n_subset))
    f.write(struct.pack('i', dim))
    data.tofile(f)
    print(f"Saved to ${TEMP_DATA}")
EOF

echo ""
echo "Building index..."
LOG_FILE="${OUTPUT_DIR}/build_$(date +%Y%m%d_%H%M%S).log"

{ time ${BUILD_DIR}/tests/build_disk_index \
    ${DATA_TYPE} \
    ${TEMP_DATA} \
    ${INDEX_PREFIX} \
    ${BUILD_R} \
    ${BUILD_L} \
    ${N_PQ_CODE} \
    2 \
    ${BUILD_T} \
    ${DIST_FN} \
    pq; } 2>&1 | tee "${LOG_FILE}"

# Fix naming if needed
if [ -f "${INDEX_PREFIX}_mem.index.tags" ] && [ ! -f "${INDEX_PREFIX}_disk.index.tags" ]; then
    mv "${INDEX_PREFIX}_mem.index.tags" "${INDEX_PREFIX}_disk.index.tags"
fi

rm -f "${TEMP_DATA}"

echo ""
echo "=== Build Complete ==="
echo "Index: ${INDEX_PREFIX}_disk.index"
echo "Log: ${LOG_FILE}"
echo "======================"
