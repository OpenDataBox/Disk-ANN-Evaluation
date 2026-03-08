# AISAQ Usage Guide

Quick guide for using the AISAQ experiment script.

## Quick Start

### 1. Build

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..
```

### 2. Configure

Edit `run_aisaq.sh` and set your dataset paths:

```bash
DATA_PATH="/path/to/your/dataset_base.bin"
QUERY_FILE="/path/to/your/dataset_query.bin"
GT_FILE="/path/to/your/dataset_groundtruth.bin"
DIM=128                     # Your vector dimension
```

### 3. Run

```bash
./run_aisaq.sh both         # Build and search
./run_aisaq.sh build        # Build only
./run_aisaq.sh search       # Search only
```

## Key Parameters

Edit these in `run_aisaq.sh`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `BUILD_R` | Max degree | 48 |
| `BUILD_L` | Build complexity | 128 |
| `B` | Search DRAM budget (GB) | 0.015 |
| `QD` | PQ chunks | DIM/8 |
| `INLINE_PQ` | Inline PQ vectors | 16 |
| `L_VALUES` | Search L values | "250 300 350..." |

### Parameter Tips

- **B calculation**: `B = (QD * num_points) / (1024^3)`
  - 1M points, QD=16: B ≈ 0.015 GB
  - 10M points, QD=16: B ≈ 0.149 GB
  
- **QD**: Usually `DIM / 8`
  - 128-dim → QD=16
  - 256-dim → QD=32
  
- **INLINE_PQ**: 
  - 0 = minimum memory
  - 16-32 = balanced
  - -1 = auto

## Changing Page Size

Page size (default: 4096 bytes) is a compile-time constant.

To change it:

1. Edit `include/defaults.h`:
   ```cpp
   const uint64_t SECTOR_LEN = 4096;  // Change to 8192 or 16384
   ```

2. Recompile:
   ```bash
   cd build && rm -rf * && cmake .. && make -j
   ```

**Note**: Different page sizes require different binaries and rebuilding indexes.

## Output

Results are saved to:
- `./indexes/aisaq_index*` - Index files
- `./results/aisaq_experiment_*.log` - Search logs


## References

- [AISAQ Paper](https://arxiv.org/abs/2404.06004)
- [Main README](README.md) - Full DiskANN documentation
