# DGAI Run Guide

This repository has a working evaluation pipeline for the main **HPQ rerank search** path. The workflow has three stages:

1. Configure the dataset
2. Build the dataset index
3. Run search

Repository path:

```bash
cd /storage/vector_data/DGAI
```

Note that `scripts/preprocess.sh` and `scripts/run_search.sh` currently use the following paths at the top of each script:

```bash
PROJECT_PATH=/storage_ssd_old/DGAI
INDEX_OUT_PATH=/storage_ssd_old/DGAI/indices
LOG_PATH=/storage_ssd_old/DGAI/log
```

If you want to run the scripts directly from this repository path, change them to:

```bash
PROJECT_PATH=/storage/vector_data/DGAI
INDEX_OUT_PATH=/storage/vector_data/DGAI/indices
LOG_PATH=/storage/vector_data/DGAI/log
```

Path variables:

| Variable | Description |
|---|---|
| `PROJECT_PATH` | Repository root. The scripts build under `${PROJECT_PATH}/build`. |
| `INDEX_OUT_PATH` | Index output directory. |
| `LOG_PATH` | Log output directory. |

## 1. Configure the Dataset

Dataset configurations are defined in:

```bash
scripts/config_dataset.sh
```

Each dataset is configured as a `dataset_xxx` function. Example:

```bash
dataset_sift1m() {
    BASE_FILE=/storage/vector_data/sift1m/sift1m_base.bin
    QUERY_FILE=/storage/vector_data/sift1m/sift1m_query1k.bin
    GT_FILE=/storage/vector_data/sift1m/sift1m_groundtruth1k.bin
    PREFIX=sift1m
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=128
    DATA_N=1000000
    SECTOR_LEN=4096
    N_PQ_CHUNKS=16
}
```

Fields:

| Variable | Description |
|---|---|
| `BASE_FILE` | Base vector file. |
| `QUERY_FILE` | Query vector file. |
| `GT_FILE` | Ground-truth file. |
| `PREFIX` | Index filename prefix. |
| `DATA_TYPE` | Data type: `float`, `int8`, or `uint8`. |
| `DIST_FN` | Distance function, for example `l2`. |
| `DATA_DIM` | Vector dimensionality. |
| `DATA_N` | Number of base vectors. |
| `SECTOR_LEN` | Disk sector size. |
| `N_PQ_CHUNKS` | Number of HPQ/PQ chunks. |

To select a dataset, update both:

```bash
scripts/preprocess.sh
scripts/run_search.sh
```

Find `DATASET_FUNCS=(...)` and keep the dataset function you want to run:

```bash
DATASET_FUNCS=(
    dataset_sift1m
)
```

To run multiple datasets, add multiple function names to the array. The scripts will process them in order.

## 2. Build the Dataset Index

Build script:

```bash
scripts/preprocess.sh
```

Run:

```bash
cd /storage/vector_data/DGAI
bash scripts/preprocess.sh
```

The script runs the following stages:

| Stage | Switch | Program | Description |
|---|---|---|---|
| Compile | `do_compile=1` | `cmake .. && make -j` | Builds binaries under `build/tests/`. |
| Build disk index | `do_build=1` | `build/tests/build_disk_index` | Builds the disk index, PQ/HPQ files, and map file. |
| Build memory index | `do_build_mem=1` | `gen_random_slice` + `build_memory_index` | Builds `${PREFIX}_mem.index`. |
| Split index | `do_split=1` | `build/tests/split_index` | Splits the disk index into `disk_index_graph` and `disk_index_data`. |
| Reorder topology | `do_topo_reorder=1` | `build/tests/reorder_by_map` | Produces `reordered_disk_index_graph_2`. |
| Reorder vectors | `do_coord_reorder=1` | `build/tests/reorder_by_map` | Produces `reordered_disk_index_data_2`. |

Default build macros:

```bash
ADDITIONAL_DEFS="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_DOUBLE_PQ"
```

Index output directory:

```bash
${INDEX_OUT_PATH}/${PREFIX}/original_index/
```

For example, when `PREFIX=sift1m`, the output path is:

```bash
/storage/vector_data/DGAI/indices/sift1m/original_index/
```

Main output files:

| File | Description |
|---|---|
| `${PREFIX}_disk.index` | Original disk index. |
| `${PREFIX}_pq_pivots.bin` | Original PQ codebook. |
| `${PREFIX}_pq_compressed.bin` | Original PQ codes. |
| `${PREFIX}_pq_pivots_refined.bin` | Refined HPQ codebook. |
| `${PREFIX}_pq_compressed_refined.bin` | Refined HPQ codes. |
| `${PREFIX}_map.bin` | HPQ codebook selection map. |
| `${PREFIX}_mem.index` | Sampled memory navigation index. |
| `disk_index_graph` | Split graph topology file. |
| `disk_index_data` | Split vector data file. |
| `reorder_map_graph_2` | Topology reorder map. |
| `reorder_map_data_2` | Vector reorder map. |
| `reordered_disk_index_graph_2` | Reordered graph topology file. |
| `reordered_disk_index_data_2` | Reordered vector data file. |

## 3. Run Search

Search script:

```bash
scripts/run_search.sh
```

Run:

```bash
cd /storage/vector_data/DGAI
bash scripts/run_search.sh
```

The search script reads the index from:

```bash
${INDEX_OUT_PATH}/${PREFIX}/original_index/${PREFIX}
```

and calls:

```bash
${PROJECT_PATH}/build/tests/search_disk_index
```

`search_disk_index` argument order:

```bash
search_disk_index \
  ${DATA_TYPE} \
  ${work_prefix} \
  ${nt} \
  ${bw} \
  ${QUERY_FILE} \
  ${GT_FILE} \
  ${K} \
  ${DIST_FN} \
  ${SEARCH_MODE} \
  ${MEM_L} \
  ${strategy} \
  ${L_LIST}
```

The main evaluated path is **HPQ rerank search**:

| Parameter | Meaning |
|---|---|
| `SEARCH_MODE=3` | Enables rerank search. |
| `USE_RERANK=1` | Enables reranking. |
| `USE_TOPO_REORDER=1` | Uses reordered graph topology. |
| `USE_DOUBLE_PQ=1` | Uses HPQ / double PQ. |
| `USE_COORD_REORDER=1` | Uses reordered vector data. |
| `MEM_L=0` | Does not load the memory navigation index by default. |
| `K` | Recall@K. |
| `L_LIST` | Search L values. |
| `BEAM_WIDTHS` | Pipeline width / beam width values. |

Compared with the original implementation, this version limits the maximum number of disk requests issued per batch in the second search stage. The current maximum is:

```bash
32
```

This means the second stage does not submit an unbounded number of disk requests. Instead, each batch is capped at 32 requests to evaluate HPQ rerank search under controlled I/O batching.

Logs are written to:

```bash
${LOG_PATH}/${PREFIX}_search.log
${LOG_PATH}/${PREFIX}_search_t_${nt}_bm_${bw}.log
```

If search fails because index files cannot be found, check:

1. `INDEX_OUT_PATH` is the same in `preprocess.sh` and `run_search.sh`.
2. `DATASET_FUNCS` selects the same dataset in both scripts.
3. `${INDEX_OUT_PATH}/${PREFIX}/original_index/` contains the files produced by the build stage.
