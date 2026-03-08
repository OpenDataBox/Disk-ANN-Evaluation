-#!/bin/bash
# Clustering Search Configuration
# Configure parameters for the clustering-based search algorithm

#############################
#   Phase Transition
#############################

# Theta: Phase transition threshold (0 < theta < 1)
# - Switch to Phase 2 when the number of visited candidates reaches l_search * theta
# - Smaller value: earlier transition, more prefetching
# - Larger value: later transition, less prefetching
# Recommended value: 0.3 (transition after 30% of candidates)
CLUSTERING_THETA=0.3

#############################
#   Similarity-Aware Prefetching
#############################

# Prefetch Pages: Number of consecutive pages to prefetch in Phase 2
# - 1-2: Conservative prefetching, suitable for random access patterns
# - 3-5: Balanced prefetching (recommended), suitable for most scenarios
# - 6-10: Aggressive prefetching, suitable for sequential access patterns
# Note: Excessively large values may cause cache pollution
CLUSTERING_PREFETCH_PAGES=1

#############################
#   Hybrid Caching
#############################

# Static Cache Capacity: Static cache capacity (in pages)
# - Used to store entry nodes and their multi-hop neighbors
# - Recommended value: 50-200 pages
# - Calculation method: number of entry nodes + 1-2 hop neighbors
CLUSTERING_STATIC_CACHE=100

# Dynamic Cache Capacity: Dynamic cache capacity (in pages)
# - Used to store pages accessed during search (Phase 2 only)
# - Uses LFU replacement policy
# - Recommended value: 500-2000 pages
# - Calculation method: available_memory / page_size
# Example: If 8GB available memory, page size 16KB, capacity is approximately 8*1024*1024/16 = 524288 pages
CLUSTERING_DYNAMIC_CACHE=1000

#############################
#   Statistics & Debugging
#############################

# Enable Statistics: Whether to enable detailed statistics
# - 0: Disabled (faster)
# - 1: Enabled (for analysis and tuning)
CLUSTERING_ENABLE_STATS=1

# Verbose Logging: Detailed logging
# - 0: Output only critical information
# - 1: Output detailed phase transition and cache information
CLUSTERING_VERBOSE=0

#############################
#   Advanced Tuning
#############################

# Phase Detection Method: Phase detection method
# - "threshold": Based on candidate count threshold (default)
# - "stagnation": Based on distance improvement stagnation
# - "hybrid": Hybrid method (transition when either condition is met)
CLUSTERING_PHASE_METHOD="hybrid"

# Stagnation Threshold: Stagnation detection threshold
# - Number of consecutive times distance improvement is below threshold to consider stagnation
# - Only effective when CLUSTERING_PHASE_METHOD includes "stagnation"
CLUSTERING_STAGNATION_COUNT=5

# Improvement Threshold: Distance improvement threshold
# - Consider stagnation when distance improvement is below this percentage
# - Example: 0.01 means 1% improvement
CLUSTERING_IMPROVEMENT_THRESHOLD=0.01

# Prefetch Strategy: Prefetch strategy
# - "centered": Centered on target page, expand forward and backward (default)
# - "forward": Prefetch forward only
# - "backward": Prefetch backward only
CLUSTERING_PREFETCH_STRATEGY="centered"

# Cache Warmup: Cache warmup
# - 0: No warmup
# - 1: Preload hot pages before search
CLUSTERING_CACHE_WARMUP=0

# Warmup Pages: Number of pages to warmup
# - Only effective when CLUSTERING_CACHE_WARMUP=1
CLUSTERING_WARMUP_PAGES=50


#############################
#   Export Variables
#############################

export CLUSTERING_THETA
export CLUSTERING_PREFETCH_PAGES
export CLUSTERING_STATIC_CACHE
export CLUSTERING_DYNAMIC_CACHE
export CLUSTERING_ENABLE_STATS
export CLUSTERING_VERBOSE
export CLUSTERING_PHASE_METHOD
export CLUSTERING_STAGNATION_COUNT
export CLUSTERING_IMPROVEMENT_THRESHOLD
export CLUSTERING_PREFETCH_STRATEGY
export CLUSTERING_CACHE_WARMUP
export CLUSTERING_WARMUP_PAGES

#############################
#   Configuration Summary
#############################

if [ "$CLUSTERING_VERBOSE" = "1" ]; then
    echo "========================================="
    echo "Clustering Search Configuration"
    echo "========================================="
    echo "Phase Transition:"
    echo "  Theta: $CLUSTERING_THETA"
    echo "  Method: $CLUSTERING_PHASE_METHOD"
    echo ""
    echo "Prefetching:"
    echo "  Pages per operation: $CLUSTERING_PREFETCH_PAGES"
    echo "  Strategy: $CLUSTERING_PREFETCH_STRATEGY"
    echo ""
    echo "Caching:"
    echo "  Static capacity: $CLUSTERING_STATIC_CACHE pages"
    echo "  Dynamic capacity: $CLUSTERING_DYNAMIC_CACHE pages"
    echo "  Warmup: $CLUSTERING_CACHE_WARMUP"
    echo ""
    echo "Statistics:"
    echo "  Enabled: $CLUSTERING_ENABLE_STATS"
    echo "  Verbose: $CLUSTERING_VERBOSE"
    echo "========================================="
    echo ""
fi
