#!/bin/sh
source config_dataset.sh

# Choose the dataset by uncommenting the line below
# If multiple lines are uncommented, only the last dataset is effective
# dataset_example

##################
#   Disk Build   #
##################
R=48         # graph degree
BUILD_L=128  # build complexity
M=2          # memory budget in GB for graph build
BUILD_T=128  # number of threads for build

##################################
#   In-Memory Navigation Graph   #
##################################
MEM_R=24                     # graph degree
MEM_BUILD_L=128              # build complexity
MEM_ALPHA=1.2                # alpha parameter
MEM_RAND_SAMPLING_RATE=0.005 # sampling rate for building in-memory navigation graph

#######################
#   Graph Partition   #
#######################
GP_TIMES=16      # number of times to partition
GP_T=128         # number of threads
GP_LOCK_NUMS=0   # lock nodes at init, lock_node_nums = partition_size * GP_LOCK_NUMS
GP_CUT=4096      # graph degree limit

##############
#   Search   #
##############
BM_LIST=(1)  # beam width
T_LIST=(1)   # number of threads
CACHE=0      # number of node cache (set to 0 when using cache_list_file)
MEM_L=0      # non-zero to enable in-memory navigation index

#####################
#   Disk Sep Impl   #
#####################
DECO_IMPL=1                     # 1 to enable Gorgeous
MEM_GRAPH_USE_RATIO=0           # graph cached ratio
MEM_EMB_USE_RATIO=0             # embedding cached ratio
EMB_SEARCH_RATIO=0.4            # ratio of embedding being searched when using mem graph
USE_DISK_GRAPH_CACHE_INDEX=1    # enable cache neighbor graph in a page
PQ_FILTER_RATIO=0.9             # Gorgeous PQ filter ratio

# Page Search
USE_PAGE_SEARCH=1               # 0 for beam search, 1 for page search
PS_USE_RATIO=0.3

# Baseline Directory (for IO statistics)
export BASELINE_DIR="/path/to/baseline/data"

# Cache List File (relative to project root)
CACHE_LIST_FILE="dataset_name/M2_R48_L128/cache_list_10000.txt"
STATIC_CACHE_FILE="${CACHE_LIST_FILE}"  # HybridCache uses the same cache list
STATIC_CACHE_MAX_NODES=0                # 0 = load all nodes from cache list

LS="250 "

#############################
#   Dynamic Cache Search    #
#############################
DYNAMIC_CACHE_PREFETCH_WINDOW=4     # prefetch window size
DYNAMIC_CACHE_CAPACITY=1000         # cache capacity in pages
DYNAMIC_CACHE_IO_LIMIT=10000        # I/O limit

#############################
#   Statistics Control      #
#############################
ENABLE_IO_UTILIZATION_STATS=1       # 1 to enable IO utilization statistics (requires BASELINE_DIR)
ENABLE_PER_HOP_CACHE_STATS=0        # 1 to enable per-hop cache hit rate statistics

#############################
#   Clustering Parameters   #
#############################
USE_CLUSTERING=1                     # 1 to enable clustering-based partitioning
CLUSTERING_SIMILARITY_THRESHOLD=0.7  # Similarity threshold for clustering (0.0-1.0)
CLUSTERING_DISTANCE_FUNC=L2          # Distance function: L2 or cosine
CLUSTERING_PAGE_SIZE=0               # Page size for clustering (0 = auto)
CLUSTERING_VERBOSE=0                 # 1 to enable verbose logging for clustering

#############################
#   HybridCache Parameters  #
#############################
HYBRIDCACHE_THETA=0.3                # Phase transition threshold (0.0-1.0)
HYBRIDCACHE_PREFETCH_PAGES=3         # Number of pages to prefetch
HYBRIDCACHE_STATIC_CACHE=100         # Static cache size (MB)
HYBRIDCACHE_DYNAMIC_CACHE=500        # Dynamic cache size (pages)
HYBRIDCACHE_ENABLE_STATS=1           # 1 to enable statistics collection

