# Performance Measurement Guide

## IO Utilization Measurement

### Step 1: Generate Baseline Data

Run DiskANN to generate baseline data (1000 queries):

```bash
./scripts/run_benchmark.sh release search
```

This generates 4 files:
- `exact_data_nodes.txt` - Exact vector nodes accessed per query
- `adjacency_nodes.txt` - Adjacency list nodes accessed per query  
- `exact_data_frequency.txt` - Exact vector node access frequency
- `adjacency_frequency.txt` - Adjacency list node access frequency

### Step 2: Enable IO Statistics

Edit `scripts/config_local.sh`:

```bash
ENABLE_IO_UTILIZATION_STATS=1
```

Set baseline directory:

```bash
export BASELINE_DIR="/path/to/baseline/data"
```

### Step 3: Run Test

```bash
export BASELINE_DIR="/path/to/baseline/data"
./scripts/run_benchmark.sh release search
```

### Step 4: View Results

```bash
grep -A 10 "IO Utilization Statistics" gorgeous/*/search/*.log
```

Example output:

```
========================================
Baseline IO Hit Rate Statistics
========================================
Total disk bytes accessed: 524288000 (500.00 MB)
Total disk bytes hit: 419430400 (400.00 MB)
Overall disk hit ratio: 80.00%

IO Utilization Statistics:
Mean IO Utilization: 80.00%
Mean Useful IO Nodes: 8000
Mean Total IO Nodes: 10000
========================================
```

**Supported Systems:**
- Gorgeous (DECO_IMPL=1): ✅
- Starling (USE_PAGE_SEARCH=1): ✅

---

## Cache Hit Rate Measurement

### Step 1: Generate Cache List

**Option 1: Using the automated script (recommended)**

```bash
# Generate cache list for 10000 nodes
bash scripts/generate_cache_10000.sh
```

This script will:
- Check frequency files exist
- Generate cache list using `generate_cache_list.py`
- Display statistics and next steps

**Option 2: Using Python script directly**

```bash
python3 scripts/generate_cache_list.py \
    /path/to/adjacency_frequency.txt \
    /path/to/cache_list_10000.txt \
    100 \
    1220
```

**Parameters:**
- Arg 1: Frequency file path
- Arg 2: Output cache list file path
- Arg 3: Cache size in MB (used for calculation)
- Arg 4: Bytes per node (default: 708)
  - For exact vectors: data_dim * sizeof(T)
  - For adjacency lists: (max_degree + 1) * sizeof(unsigned)
  - Example: 128-dim float + degree 48 = 512 + 196 = 708 bytes

### Step 2: Configure Cache List

Edit `scripts/config_local.sh`:

```bash
CACHE_LIST_FILE="dataset_name/M2_R48_L128/cache_list_100mb.txt"
```

### Step 3: Run Test

```bash
./scripts/run_benchmark.sh release search
```

### Step 4: View Results

Look for cache hit rate statistics in the log output.

---

## Complete Test Workflows

### Workflow 1: Measure IO Utilization

```bash
# Generate baseline
./scripts/run_benchmark.sh release search

# Enable IO stats in config_local.sh
# ENABLE_IO_UTILIZATION_STATS=1

# Set baseline directory and run
export BASELINE_DIR="/path/to/baseline/data"
./scripts/run_benchmark.sh release search

# View results
grep -A 10 "IO Utilization Statistics" gorgeous/*/search/*.log
```

### Workflow 2: Measure Cache Hit Rate

```bash
# Generate cache list (10000 nodes)
bash scripts/generate_cache_10000.sh

# Configure in config_local.sh
# CACHE_LIST_FILE="dataset_name/M2_R48_L128/cache_list_10000.txt"

# Run test
./scripts/run_benchmark.sh release search
```

### Workflow 3: Full Performance Analysis

```bash
# Configure all statistics in config_local.sh:
# ENABLE_IO_UTILIZATION_STATS=1
# CACHE_LIST_FILE="dataset/cache_list_100mb.txt"

# Set baseline directory
export BASELINE_DIR="/path/to/baseline/data"

# Run test
./scripts/run_benchmark.sh release search

# View all results
LOG_FILE="gorgeous/*/search/*.log"
echo "=== IO Utilization ==="
grep -A 10 "IO Utilization Statistics" $LOG_FILE
```

### Workflow 4: Compare Different Cache Configurations

```bash
#!/bin/bash
# compare_configs.sh

NODE_COUNTS=(5000 10000 15000 20000)

for NODES in "${NODE_COUNTS[@]}"; do
    echo "Testing cache with ${NODES} nodes"
    
    # Generate cache list
    python3 scripts/generate_cache_list.py \
        /path/to/adjacency_frequency.txt \
        gorgeous/dataset/cache_list_${NODES}.txt \
        100 \
        1220
    
    # Update config
    sed -i "s|CACHE_LIST_FILE=.*|CACHE_LIST_FILE=\"dataset/cache_list_${NODES}.txt\"|" scripts/config_local.sh
    
    # Run test
    ./scripts/run_benchmark.sh release search > results_cache_${NODES}.log 2>&1
done

# Extract comparison data
echo "Node Count,Hit Rate,QPS,Mean Latency" > comparison.csv
for NODES in "${NODE_COUNTS[@]}"; do
    HIT_RATE=$(grep "Cache hit rate:" results_cache_${NODES}.log | awk '{print $4}')
    QPS=$(grep -A 1 "^  50" results_cache_${NODES}.log | tail -1 | awk '{print $3}')
    LATENCY=$(grep -A 1 "^  50" results_cache_${NODES}.log | tail -1 | awk '{print $4}')
    echo "${NODES},$HIT_RATE,$QPS,$LATENCY" >> comparison.csv
done
```

---

## Configuration Reference

### config_local.sh Example

```bash
#!/bin/sh
source config_dataset.sh

dataset_example

# Disk Build
R=48
BUILD_L=128
M=2
BUILD_T=128

# In-Memory Navigation Graph
MEM_R=24
MEM_BUILD_L=128
MEM_ALPHA=1.2
MEM_RAND_SAMPLING_RATE=0.005

# Graph Partition
GP_TIMES=16
GP_T=128
GP_LOCK_NUMS=0
GP_CUT=4096

# Search
BM_LIST=(4)
T_LIST=(1)
CACHE=0
MEM_L=128

# Disk Separation Implementation
DECO_IMPL=1
MEM_GRAPH_USE_RATIO=0.066
MEM_EMB_USE_RATIO=0
EMB_SEARCH_RATIO=0.4
USE_DISK_GRAPH_CACHE_INDEX=1
PQ_FILTER_RATIO=0.9

# Page Search (Starling)
USE_PAGE_SEARCH=1
PS_USE_RATIO=0.3

# Cache List File
CACHE_LIST_FILE="dataset_name/M2_R48_L128/cache_list_10000.txt"

LS="50 "

# Dynamic Cache Search
DYNAMIC_CACHE_PREFETCH_WINDOW=4
DYNAMIC_CACHE_CAPACITY=1000
DYNAMIC_CACHE_IO_LIMIT=10000

# Statistics Control
ENABLE_IO_UTILIZATION_STATS=1
```

### Environment Variables

```bash
export BASELINE_DIR="/path/to/baseline/data"
```


---
