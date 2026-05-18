#!/bin/bash
#
# 数据集卡片：每个数据集一个函数，把这个数据集相关的所有变量写齐。
# 在 preprocess.sh / 后续 search 脚本中通过 `dataset_xxx` 调用即可。
#
# 每个函数必须导出（可不全用，但变量名固定）：
#   BASE_FILE      base 向量文件
#   QUERY_FILE     query 文件
#   GT_FILE        groundtruth 文件
#   PREFIX         索引文件前缀（仅文件名部分）
#   DATA_TYPE      float / int8 / uint8
#   DIST_FN        l2 / cosine
#   DATA_DIM       向量维度
#   DATA_N         点数
#   SECTOR_LEN     磁盘 sector 大小，目前固定 4096
#   N_PQ_CHUNKS    PQ 子空间数（即 num_pq_chunks）
#   TRUNC_LEN      rerank 候选截断长度（搜索用）
#

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

dataset_sift1m_8kb() {
    BASE_FILE=/storage/vector_data/sift1m/sift1m_base.bin
    QUERY_FILE=/storage/vector_data/sift1m/sift1m_query1k.bin
    GT_FILE=/storage/vector_data/sift1m/sift1m_groundtruth1k.bin
    PREFIX=sift1m_8kb
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=128
    DATA_N=1000000
    SECTOR_LEN=4096*2
    N_PQ_CHUNKS=16
}

dataset_sift1m_16kb() {
    BASE_FILE=/storage/vector_data/sift1m/sift1m_base.bin
    QUERY_FILE=/storage/vector_data/sift1m/sift1m_query1k.bin
    GT_FILE=/storage/vector_data/sift1m/sift1m_groundtruth1k.bin
    PREFIX=sift1m_16kb
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=128
    DATA_N=1000000
    SECTOR_LEN=4096*4
    N_PQ_CHUNKS=16
}


dataset_sift1b() {
    BASE_FILE=/storage/vector_data/sift1b/sift1b_base.bin
    QUERY_FILE=/storage/vector_data/sift1b/sift1b_query.bin
    GT_FILE=/storage/vector_data/sift1b/sift1b_groundtruth.bin
    PREFIX=sift1b
    DATA_TYPE=uint8
    DIST_FN=l2
    DATA_DIM=128
    DATA_N=1000000000
    SECTOR_LEN=4096
    N_PQ_CHUNKS=16
}

dataset_gist() {
    BASE_FILE=/storage/vector_data/gist/gist_base.bin
    QUERY_FILE=/storage/vector_data/gist/gist_query.bin
    GT_FILE=/storage/vector_data/gist/gist_groundtruth.bin
    PREFIX=gist
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=960
    DATA_N=1000000
    SECTOR_LEN=4096
    N_PQ_CHUNKS=120
}

dataset_gist_8kb() {
    BASE_FILE=/storage/vector_data/gist/gist_base.bin
    QUERY_FILE=/storage/vector_data/gist/gist_query.bin
    GT_FILE=/storage/vector_data/gist/gist_groundtruth.bin
    PREFIX=gist_8kb
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=960
    DATA_N=1000000
    SECTOR_LEN=4096*2
    N_PQ_CHUNKS=120
}

dataset_gist_16kb() {
    BASE_FILE=/storage/vector_data/gist/gist_base.bin
    QUERY_FILE=/storage/vector_data/gist/gist_query.bin
    GT_FILE=/storage/vector_data/gist/gist_groundtruth.bin
    PREFIX=gist_16kb
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=960
    DATA_N=1000000
    SECTOR_LEN=4096*4
    N_PQ_CHUNKS=120
}

dataset_deep1m() {
    BASE_FILE=/storage/vector_data/deep1m/deep1m_base.bin
    QUERY_FILE=/storage/vector_data/deep1m/deep1m_query.bin
    GT_FILE=/storage/vector_data/deep1m/deep1m_groundtruth.bin
    PREFIX=deep1m
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=256
    DATA_N=1000000
    SECTOR_LEN=4096
    N_PQ_CHUNKS=32
}

dataset_deep1m_8kb() {
    BASE_FILE=/storage/vector_data/deep1m/deep1m_base.bin
    QUERY_FILE=/storage/vector_data/deep1m/deep1m_query.bin
    GT_FILE=/storage/vector_data/deep1m/deep1m_groundtruth.bin
    PREFIX=deep1m_8kb
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=256
    DATA_N=1000000
    SECTOR_LEN=4096*2
    N_PQ_CHUNKS=32
}

dataset_deep1m_16kb() {
    BASE_FILE=/storage/vector_data/deep1m/deep1m_base.bin
    QUERY_FILE=/storage/vector_data/deep1m/deep1m_query.bin
    GT_FILE=/storage/vector_data/deep1m/deep1m_groundtruth.bin
    PREFIX=deep1m_16kb
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=256
    DATA_N=1000000
    SECTOR_LEN=4096*4
    N_PQ_CHUNKS=32
}

dataset_msong() {
    BASE_FILE=/storage/vector_data/msong/msong_base.bin
    QUERY_FILE=/storage/vector_data/msong/msong_query.bin
    GT_FILE=/storage/vector_data/msong/msong_groundtruth.bin
    PREFIX=msong
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=420
    DATA_N=994185
    SECTOR_LEN=4096
    N_PQ_CHUNKS=64
}

dataset_glove() {
    BASE_FILE=/storage/vector_data/glove1m/glove1m_base.bin
    QUERY_FILE=/storage/vector_data/glove1m/glove1m_query1k.bin
    GT_FILE=/storage/vector_data/glove1m/glove1m_groundtruth1k.bin
    PREFIX=glove
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=100
    DATA_N=1183514
    SECTOR_LEN=100
    N_PQ_CHUNKS=25
}

dataset_tiny5m() {
    BASE_FILE=/storage/vector_data/tiny5m/tiny5m_base.bin
    QUERY_FILE=/storage/vector_data/tiny5m/tiny5m_query.bin
    GT_FILE=/storage/vector_data/tiny5m/tiny5m_groundtruth.bin
    PREFIX=tiny5m
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=384
    DATA_N=5000000
    SECTOR_LEN=4096
    N_PQ_CHUNKS=48
}

dataset_OpenAI(){
    BASE_FILE=/storage/vector_data/OpenAI3072/OpenAI3072.bin
    QUERY_FILE=/storage/vector_data/OpenAI3072/OpenAI3072_query.bin
    GT_FILE=/storage/vector_data/OpenAI3072/OpenAI3072_groundtruth.bin
    PREFIX=OpenAI3072
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=3072
    DATA_N=1000000
    SECTOR_LEN=4096*4
    N_PQ_CHUNKS=384
}

dataset_OpenAI_300(){
    BASE_FILE=/storage/vector_data/OpenAI/OpenAI300/OpenAI300_base.bin
    QUERY_FILE=/storage/vector_data/OpenAI/OpenAI300/OpenAI300_query.bin
    GT_FILE=/storage/vector_data/OpenAI/OpenAI300/OpenAI300_groundtruth.bin
    PREFIX=OpenAI3072_300
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=300
    DATA_N=1000000
    SECTOR_LEN=4096
    N_PQ_CHUNKS=75
}

dataset_OpenAI_500(){
    BASE_FILE=/storage/vector_data/OpenAI/OpenAI500/OpenAI500_base.bin
    QUERY_FILE=/storage/vector_data/OpenAI/OpenAI500/OpenAI500_query.bin
    GT_FILE=/storage/vector_data/OpenAI/OpenAI500/OpenAI500_groundtruth.bin
    PREFIX=OpenAI3072_500
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=500
    DATA_N=1000000
    SECTOR_LEN=4096
    N_PQ_CHUNKS=125
}

dataset_OpenAI_700(){
    BASE_FILE=/storage/vector_data/OpenAI/OpenAI700/OpenAI700_base.bin
    QUERY_FILE=/storage/vector_data/OpenAI/OpenAI700/OpenAI700_query.bin
    GT_FILE=/storage/vector_data/OpenAI/OpenAI700/OpenAI700_groundtruth.bin
    PREFIX=OpenAI3072_700
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=700
    DATA_N=1000000
    SECTOR_LEN=4096
    N_PQ_CHUNKS=175
}

dataset_OpenAI_900(){
    BASE_FILE=/storage/vector_data/OpenAI/OpenAI900/OpenAI900_base.bin
    QUERY_FILE=/storage/vector_data/OpenAI/OpenAI900/OpenAI900_query.bin
    GT_FILE=/storage/vector_data/OpenAI/OpenAI900/OpenAI900_groundtruth.bin
    PREFIX=OpenAI3072_900
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=900
    DATA_N=1000000
    SECTOR_LEN=4096
    N_PQ_CHUNKS=225
}

dataset_OpenAI_1500(){
    BASE_FILE=/storage/vector_data/OpenAI/OpenAI1500/OpenAI1500_base.bin
    QUERY_FILE=/storage/vector_data/OpenAI/OpenAI1500/OpenAI1500_query.bin
    GT_FILE=/storage/vector_data/OpenAI/OpenAI1500/OpenAI1500_groundtruth.bin
    PREFIX=OpenAI3072_1500
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=1500
    DATA_N=1000000
    SECTOR_LEN=4096*2
    N_PQ_CHUNKS=375
}

dataset_OpenAI_2000(){
    BASE_FILE=/storage/vector_data/OpenAI/OpenAI2000/OpenAI2000_base.bin
    QUERY_FILE=/storage/vector_data/OpenAI/OpenAI2000/OpenAI2000_query.bin
    GT_FILE=/storage/vector_data/OpenAI/OpenAI2000/OpenAI2000_groundtruth.bin
    PREFIX=OpenAI3072_2000
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=2000
    DATA_N=1000000
    SECTOR_LEN=4096*4
    N_PQ_CHUNKS=500
}

dataset_OpenAI_3000(){
    BASE_FILE=/storage/vector_data/OpenAI/OpenAI3000/OpenAI3000_base.bin
    QUERY_FILE=/storage/vector_data/OpenAI/OpenAI3000/OpenAI3000_query.bin
    GT_FILE=/storage/vector_data/OpenAI/OpenAI3000/OpenAI3000_groundtruth.bin
    PREFIX=OpenAI3072_3000
    DATA_TYPE=float
    DIST_FN=l2
    DATA_DIM=3000
    DATA_N=1000000
    SECTOR_LEN=4096*4
    N_PQ_CHUNKS=750
}