# Sliding Window Test Guide

This guide explains how to run the sliding window test for FreshDISKANN, which continuously inserts new points and deletes old points while measuring search performance.

## Overview

The sliding window test simulates a real-world scenario where:
- Start with 800K points (configurable)
- Each batch: insert N new points, delete N oldest points
- Run for multiple batches (maintaining constant index size)
- Measure Search (S), Update (U), Maintenance (M), and Total (T) time
- Perform continuous queries during updates and merges

## Quick Start

### 1. Prepare Your Data

You need:
- **Base vectors**: Binary file with all vectors (at least initial_points + num_batches × batch_size)
- **Query vectors**: Binary file with query vectors
- **Ground truth directory**: Contains `gt_0.bin`, `gt_2000.bin`, `gt_4000.bin`, etc.
  - File naming: `gt_{offset}.bin` where offset = batch_number × batch_size

### 2. Run the Test

```bash
./prepare_and_test_disk_sliding_window.sh
```

This script performs three steps automatically:
1. **Build disk index** (if not exists) using `build_disk_index`
2. **Convert to SyncIndex format** using `static_sync_index`
3. **Run sliding window test** using `test_sliding_window`

### 3. Customize for Your Dataset

Edit `prepare_and_test_disk_sliding_window.sh` and modify the `prepare_and_test` call:

```bash
prepare_and_test "your_dataset_name" \
    "/path/to/your/base_vectors.bin" \
    "/path/to/your/query_vectors.bin" \
    "/path/to/your/groundtruth_directory" \
    "float" \
    48 128 0.03 16
```

**Parameter Guide:**
- `DATASET_NAME`: Name for output directory
- `DATA_FILE`: Path to base vectors (.bin format)
- `QUERY_FILE`: Path to query vectors (.bin format)
- `GT_DIR`: Directory with ground truth files (must end with `/`)
- `DATA_TYPE`: `float`, `int8`, or `uint8`
- `R_BUILD`: Degree for disk index build (typically 48)
- `L_BUILD`: Search list size for disk index build (typically 128)
- `B`: PQ compression ratio
  - 128-dim: 0.03
  - 256-dim: 0.06
  - 960-dim: 0.25
- `PQ_CHUNKS`: Number of PQ chunks (typically dim/8 or dim/16)

## Configuration Parameters

### Script-Level Parameters (in prepare_and_test_disk_sliding_window.sh)

Edit the top of the script to adjust test parameters:

```bash
# Memory index parameters (for in-memory navigation graph)
L_MEM=128           # Search list size for memory index
R_MEM=24            # Degree for memory index
ALPHA_MEM=1.2       # Alpha parameter for memory index

# Disk index parameters  
L_DISK=128          # Search list size for disk index
R_DISK=48           # Degree for disk index
ALPHA_DISK=1.2      # Alpha parameter for disk index

# Sliding window parameters
INITIAL_POINTS=800000    # Starting number of points
BATCH_SIZE=2000          # Points to insert/delete per batch
NUM_BATCHES=100          # Number of batches to run

# System parameters
NUM_SHARDS=1             # Number of shards (typically 1)
NODES_TO_CACHE=0         # Nodes to cache in memory (0 = no caching)
BEAM_WIDTH=4             # Beam width for search
RECALL_AT=10             # Recall@K metric

# Search L values
SEARCH_L_LIST="20 30 40 50 60"  # List of L values to test
```

### Hardcoded Parameters (in test_sliding_window.cpp)

These parameters are hardcoded and cannot be changed without recompiling:

```cpp
#define NUM_INSERT_THREADS 9              // Threads for insertion operations
#define NUM_DELETE_THREADS 1              // Threads for deletion operations
#define NUM_SEARCH_THREADS 16             // Threads for search operations
#define MERGE_ROUND 20                    // Merge every N batches
#define UPDATE_QUERY_INTERVAL_MS 200      // Query interval during updates (ms)
#define MERGE_QUERY_INTERVAL_S 10         // Query interval during merge (seconds)

```

**Note:** To change these parameters, you need to:
1. Edit `tests/test_sliding_window.cpp`
2. Recompile: `make -C build test_sliding_window -j8`

## Output

Results are saved in `./disk_sliding_window_results/`:

```
disk_sliding_window_results/
├── master_YYYYMMDD_HHMMSS.log          # Summary of all datasets
└── your_dataset_name/
    └── your_dataset_name_YYYYMMDD_HHMMSS.log  # Detailed results
```

### Log File Contents

The log file contains comprehensive performance metrics:

**Per-Batch Metrics:**
- QPS (Queries Per Second) for each L value
- Recall@K for each L value
- Mean latency and 99.9th percentile latency
- Time breakdown: S (Search), U (Update), M (Maintenance)
- Number of queries processed during updates and merges

**Final Summary:**
```
========== Time Summary (seconds) ==========
S (Search):           XXX.XX s
U (Update):           XXX.XX s
M (Maintenance):      XXX.XX s
T (Total):            XXX.XX s
============================================

========== Time Summary (hours) ==========
S (Search):           X.XXXX h
U (Update):           X.XXXX h
M (Maintenance):      X.XXXX h
T (Total):            X.XXXX h
==========================================
```

### Understanding the Metrics

- **S (Search)**: Total time spent on search queries
- **U (Update)**: Total time spent on insert + delete operations (pure update time)
- **M (Maintenance)**: Total time spent on merge operations
- **T (Total)**: Wall-clock time from start to finish
- **Queries during updates**: Continuous queries executed while inserting/deleting
- **Queries during merges**: Continuous queries executed during merge operations

## Understanding the Three Steps

### Step 1: Build Disk Index
**Program:** `build_disk_index`  
**Purpose:** Create the initial disk-based index from your data file.

**Command:**
```bash
./build/tests/build_disk_index \
    <data_type> <data_file> <index_prefix> \
    <R> <L> <B> <M> <T>
```

**Parameters:**
- `R`: Degree (typically 48)
- `L`: Search list size (typically 128)
- `B`: PQ compression ratio (0.03-0.25 depending on dimension)
- `M`: Number of threads (typically 2)
- `T`: Temp memory limit (typically 128 GB)

**Output:** `{prefix}_disk.index` and related PQ files

### Step 2: Convert to SyncIndex
**Program:** `static_sync_index`  
**Purpose:** Convert the disk index into SyncIndex format, which supports dynamic updates.

**Command:**
```bash
./build/tests/static_sync_index \
    <data_type> <data_file> \
    <L_mem> <R_mem> <alpha_mem> \
    <L_disk> <R_disk> <alpha_disk> \
    <num_start> <num_shards> <pq_chunks> <nodes_to_cache> \
    <save_path>
```

**Key Points:**
- `num_start`: Number of initial points to load (e.g., 800000)
- Creates a SyncIndex that can handle insert/delete operations
- Clears old SyncIndex files before building

**Output:** `{prefix}0_lti_disk.index` and related shard files

### Step 3: Run Sliding Window Test
**Program:** `test_sliding_window`  
**Purpose:** Perform continuous insert/delete operations while measuring performance.

**Command:**
```bash
./build/tests/test_sliding_window \
    <data_type> <data_file> \
    <L_mem> <R_mem> <alpha_mem> \
    <L_disk> <R_disk> <alpha_disk> \
    <initial_points> <batch_size> <num_batches> \
    <num_shards> <pq_chunks> <nodes_to_cache> \
    <index_prefix> <query_file> <gt_prefix> \
    <recall@> <beam_width> <log_file> \
    <L1> <L2> <L3> ...
```

**Key Features:**
- Loads existing SyncIndex
- For each batch:
  1. Insert new points
  2. Delete old points
  3. Perform continuous queries during updates
  4. Merge every N batches (default: 20)
  5. Perform continuous queries during merge
  6. Evaluate search quality with ground truth
- Outputs detailed performance metrics to log file

**Output:** Performance metrics in specified log file

## Troubleshooting

### Index Already Exists
The script checks if indices exist and skips rebuilding. To rebuild from scratch:
```bash
# Remove disk index
rm -f ./disk_indices/your_dataset_disk*

# Remove SyncIndex
rm -f ./disk_indices/your_dataset_index*
```

### Out of Memory
- Reduce `NODES_TO_CACHE` (set to 0 for no caching)
- Reduce `NUM_SHARDS` (but typically should be 1)
- Reduce `NUM_INSERT_THREADS` in source code and recompile

### Slow Performance
- Ensure indices are on SSD (check `INDEX_DIR` path in script)
- Adjust thread counts in `tests/test_sliding_window.cpp`:
  - `NUM_INSERT_THREADS` (default: 9)
  - `NUM_DELETE_THREADS` (default: 1)
  - `NUM_SEARCH_THREADS` (default: 16)
- Reduce `NUM_BATCHES` for faster testing
- Increase `MERGE_ROUND` to merge less frequently

### Ground Truth File Not Found
Ensure your ground truth directory:
- Ends with `/` in the path
- Contains files named exactly: `gt_0.bin`, `gt_2000.bin`, `gt_4000.bin`, etc.
- File naming matches: `gt_{batch_number × batch_size}.bin`

### Merge Failures
If merge operations fail:
- Check disk space (merges create temporary files)
- Verify index files are not corrupted
- Check log file for specific error messages
- Try reducing `MERGE_ROUND` to merge more frequently with smaller changes

## Advanced Usage

### Test Multiple Datasets
Add multiple `prepare_and_test` calls in the script:

```bash
prepare_and_test "dataset1" \
    "/path/to/dataset1_base.bin" \
    "/path/to/dataset1_query.bin" \
    "/path/to/dataset1_gt/" \
    "float" 48 128 0.03 16

prepare_and_test "dataset2" \
    "/path/to/dataset2_base.bin" \
    "/path/to/dataset2_query.bin" \
    "/path/to/dataset2_gt/" \
    "float" 48 128 0.06 32
```

### Modify Search Parameters
Edit `SEARCH_L_LIST` to test different search quality/speed tradeoffs:
```bash
SEARCH_L_LIST="10 20 30 40 50 100"  # More values = more comprehensive test
```

### Adjust Merge Frequency
Edit `MERGE_ROUND` in `tests/test_sliding_window.cpp`:
```cpp
#define MERGE_ROUND 20  // Merge every 20 batches (default)
```
- Smaller value: More frequent merges, slower but more stable
- Larger value: Less frequent merges, faster but may accumulate more changes

### Adjust Query Intervals
Edit query intervals in `tests/test_sliding_window.cpp`:
```cpp
#define UPDATE_QUERY_INTERVAL_MS 200  // Query every 200ms during updates
#define MERGE_QUERY_INTERVAL_S 10     // Query every 10s during merge
```
- Smaller intervals: More queries, better QPS measurement, but slower overall
- Larger intervals: Fewer queries, faster test, but less comprehensive

### Run Without Continuous Queries
To disable continuous queries during updates/merges:
1. Set very large intervals in source code
2. Or comment out the query loops in `test_sliding_window.cpp`
3. Recompile

### Debug Mode
For debugging, use `debug_gist_crash.sh` to run with gdb:
```bash
./debug_gist_crash.sh
```
This will catch crashes and print stack traces.

## File Structure

```
FreshDISKANN/
├── prepare_and_test_disk_sliding_window.sh  # Main test script
├── debug_gist_crash.sh                      # Debug helper
├── build/tests/
│   ├── build_disk_index                     # Step 1 executable
│   ├── static_sync_index                    # Step 2 executable
│   └── test_sliding_window                  # Step 3 executable
├── tests/
│   ├── build_disk_index.cpp                 # Step 1 source
│   ├── static_sync_index.cpp                # Step 2 source
│   └── test_sliding_window.cpp              # Step 3 source
└── disk_sliding_window_results/             # Output directory
```

## Citation

If you use this code in your research, please cite:
```
[Add your citation here]
```
