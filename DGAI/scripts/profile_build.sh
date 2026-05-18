#!/usr/bin/env bash
#
# Profile build_disk_index and summarize:
#   - PQ build time
#   - Vamana build time
#   - peak memory from /usr/bin/time -v
#
# Usage:
#   bash scripts/profile_build.sh
#
# Optional environment overrides:
#   DATASET_FUNC=dataset_sift1m BUILD_T=32 CLEAN_INDEX=1 bash scripts/profile_build.sh

set -euo pipefail

source "$(dirname "$0")/config_dataset.sh"

PROJECT_PATH=${PROJECT_PATH:-~/DGAI/DGAI}
INDEX_OUT_PATH=${INDEX_OUT_PATH:-~/DGAI/DGAI/indices}
LOG_PATH=${LOG_PATH:-~/DGAI/DGAI/log}
DATASET_FUNC=${DATASET_FUNC:-dataset_sift1m}
CLEAN_INDEX=${CLEAN_INDEX:-0}

# Build parameters. Keep these aligned with scripts/preprocess.sh by default.
R=${R:-48}
BUILD_L=${BUILD_L:-100}
B=${B:-64}
M=${M:-64}
BUILD_T=${BUILD_T:-128}
SINGLE_FILE_INDEX=${SINGLE_FILE_INDEX:-0}

"$DATASET_FUNC"

original_index_path="${INDEX_OUT_PATH}/${PREFIX}/original_index"
index_path_prefix="${original_index_path}/${PREFIX}"
mkdir -p "$original_index_path" "$LOG_PATH"

if [ "$CLEAN_INDEX" = "1" ]; then
    rm -rf "$original_index_path"
    mkdir -p "$original_index_path"
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
run_log="${LOG_PATH}/profile_build_${PREFIX}_${timestamp}.log"
time_log="${LOG_PATH}/profile_build_${PREFIX}_${timestamp}.time"
summary_file="${LOG_PATH}/profile_build_${PREFIX}_${timestamp}.summary"

cmd=(
    "$PROJECT_PATH/build/tests/build_disk_index"
    "$DATA_TYPE"
    "$BASE_FILE"
    "$index_path_prefix"
    "$R"
    "$BUILD_L"
    "$B"
    "$M"
    "$BUILD_T"
    "$DIST_FN"
    "$SINGLE_FILE_INDEX"
)

echo "=========================================="
echo " dataset      : $PREFIX"
echo " base file    : $BASE_FILE"
echo " data_type    : $DATA_TYPE"
echo " dim / npts   : $DATA_DIM / $DATA_N"
echo " R / L / B/M  : $R / $BUILD_L / $B / $M"
echo " build threads: $BUILD_T"
echo " log          : $run_log"
echo " time log     : $time_log"
echo "=========================================="
echo "[build_disk_index] ${cmd[*]}"

set +e
/usr/bin/time -v "${cmd[@]}" 2> >(tee "$time_log" >&2) | \
    awk '{ print systime() "\t" $0; fflush(); }' | tee "$run_log"
cmd_status=${PIPESTATUS[0]}
set -e

awk -v run_log="$run_log" -v time_log="$time_log" -v status="$cmd_status" '
function last_field_number(line,    n, a, v) {
    n = split(line, a, " ")
    v = a[n]
    sub(/s[.]?$/, "", v)
    return v + 0
}
BEGIN {
    pq_pivots = -1
    pq_compress = -1
    vamana = -1
    max_rss_kb = -1
}
FILENAME == run_log {
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
        vamana = last_field_number(line)
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
    print "Build profile summary"
    print "=========================================="
    print "pq_total_sec             : " pq_total
    print "vamana_total_sec         : " vamana
    print "max_rss_kb               : " max_rss_kb
    print "max_rss_gb               : " max_rss_gb
}
' "$run_log" "$time_log" | tee "$summary_file"

echo "[done] summary saved to $summary_file"
exit "$cmd_status"
