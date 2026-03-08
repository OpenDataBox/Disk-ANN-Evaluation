#!/bin/sh

# PipeANN Dataset Configuration Template
# Copy this file and modify paths to match your dataset location

# Example: SIFT-1M Dataset
BASE_PATH=/path/to/your/dataset/sift1m_base.bin
QUERY_FILE=/path/to/your/dataset/sift1m_query.bin
GT_FILE=/path/to/your/dataset/sift1m_groundtruth.bin

PREFIX=sift1m
DATA_TYPE=float      # Options: float, uint8, int8
DIST_FN=l2          # Options: l2, ip (inner product)
K=100               # Top-K for ground truth
DATA_DIM=128        # Vector dimension
DATA_N=1000000      # Number of vectors
N_PQ_CODE=16        # PQ compression bytes (typically dim/8)