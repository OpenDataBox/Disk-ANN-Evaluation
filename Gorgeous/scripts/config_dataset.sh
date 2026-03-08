#!/bin/sh

# Dataset configuration template
# Switch dataset by uncommenting and calling the desired function

# Example dataset configuration template
dataset_example() {
  # Path to base vectors file
  BASE_PATH=/path/to/dataset/base.bin
  
  # Path to query vectors file
  QUERY_FILE=/path/to/dataset/query.bin
  
  # Path to ground truth file
  GT_FILE=/path/to/dataset/groundtruth.bin
  
  # Dataset prefix for output files
  PREFIX=example_dataset
  
  # Data type: float, uint8, int8
  DATA_TYPE=float
  
  # Distance function: l2, mips, cosine
  DIST_FN=l2
  
  # Build parameter (graph degree ratio)
  B=0.03
  
  # Number of nearest neighbors to retrieve
  K=100
  
  # Dimension of vectors
  DATA_DIM=128
  
  # Number of vectors in base dataset
  DATA_N=1000000
  
  # Sector length for disk page (default: 4096)
  SECTOR_LEN=4096
  
  # Sector length for graph replicated layout
  GR_SECTOR_LEN=4096
  
  # PQ code dimension (typically 4 or 8)
  # 4: for single-modal datasets (optimal)
  # 8: for higher dimensional data
  N_PQ_CODE=8
}

# Uncomment to activate a dataset configuration
# dataset_example

