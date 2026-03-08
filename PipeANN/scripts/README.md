# PipeANN Scripts Usage Guide

## Quick Start

```bash
# 1. Configure your dataset
scripts/config_dataset.sh
# Edit paths to point to your data files

# 2. Build index
./scripts/run_pipeann.sh build

# 3. Run search benchmark
./scripts/run_pipeann.sh search
```

## Main Scripts

### Basic Operations

- **run_pipeann.sh**: Main script for build/search/clean
  ```bash
  ./scripts/run_pipeann.sh build   # Build index
  ./scripts/run_pipeann.sh search  # Run search
  ./scripts/run_pipeann.sh clean   # Clean files
  ```

### Update Experiments (Advanced)

For sliding window update experiments:

- **build_partial_index.sh**: Build index with subset of vectors
  ```bash
  ./scripts/build_partial_index.sh --num-vectors 800000
  ```

- **generate_update_gt.sh**: Generate ground truth for updates
  ```bash
  ./scripts/generate_update_gt.sh --initial 800000 --total 1000000 --batch 2000
  ```

- **run_update_benchmark.sh**: Run update benchmark
  ```bash
  ./scripts/run_update_benchmark.sh --steps 100 --batch-size 2000
  ```

## Configuration Files

### config_dataset.sh
Dataset configuration (paths, dimensions, data types):
```bash
BASE_PATH=/path/to/your/dataset/base.bin
QUERY_FILE=/path/to/your/dataset/query.bin
GT_FILE=/path/to/your/dataset/groundtruth.bin
PREFIX=sift1m
DATA_TYPE=float
DATA_DIM=128
```

### config_local.sh
Algorithm parameters:
```bash
BUILD_R=48          # Graph degree
BUILD_L=128         # Build complexity
SEARCH_MODE=2       # 0=beam, 1=page, 2=pipe, 3=coro
SEARCH_THREADS=1    # Search threads
L_VALUES="20 30 40 50"  # L values to test
```

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| BUILD_R | Graph degree (out-neighbors) | 32-128 |
| BUILD_L | Build complexity | 100-200 |
| SEARCH_MODE | Search algorithm | 2 (PipeANN) |
| L_VALUES | Search complexity values | "20 30 40 50" |

## Output Metrics

Search results show:
- **QPS**: Queries per second
- **AvgLat**: Average latency (microseconds)
- **Mean IOs**: Average disk I/O operations
- **Recall@10**: Recall rate at top-10

## Examples

### Basic Search Test
```bash

# Build and search
./scripts/run_pipeann.sh build
./scripts/run_pipeann.sh search


### Update Experiment
```bash
# Build partial index (800k vectors)
./scripts/build_partial_index.sh --num-vectors 800000

# Generate ground truth
./scripts/generate_update_gt.sh \
  --initial 800000 \
  --total 1000000 \
  --batch 2000

# Run sliding window test
./scripts/run_update_benchmark.sh \
  --steps 100 \
  --batch-size 2000 \
  --l-values "20 30 40 50"
```

### Test Different Search Modes
```bash
# Set SEARCH_MODE=0 for beam search
# Set SEARCH_MODE=2 for pipe search

./scripts/run_pipeann.sh search
```

