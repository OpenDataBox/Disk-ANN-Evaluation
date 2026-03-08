#!/usr/bin/env python3
"""
Generate cache list from adjacency frequency file.
Usage: python generate_cache_list.py <frequency_file> <output_file> <cache_size_mb> [bytes_per_node]
"""

import sys
import os

def main():
    if len(sys.argv) < 4:
        print("Usage: python generate_cache_list.py <frequency_file> <output_file> <cache_size_mb> [bytes_per_node]")
        print("Example: python generate_cache_list.py tests_data/sift1m/recall_99/adjacency_frequency.txt cache_list_10mb.txt 10")
        sys.exit(1)
    
    freq_file = sys.argv[1]
    output_file = sys.argv[2]
    cache_size_mb = float(sys.argv[3])
    
    # Default bytes per node: vector (128 * 4) + adjacency list ((48+1) * 4) = 512 + 196 = 708
    # For SIFT1M: 128-dim float vectors
    bytes_per_node = int(sys.argv[4]) if len(sys.argv) > 4 else 708
    
    cache_size_bytes = int(cache_size_mb * 1024 * 1024)
    max_nodes = cache_size_bytes // bytes_per_node
    
    print(f"Cache size: {cache_size_mb} MB = {cache_size_bytes} bytes")
    print(f"Bytes per node: {bytes_per_node}")
    print(f"Max nodes to cache: {max_nodes}")
    
    # Read frequency file
    nodes = []
    with open(freq_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                node_id = int(parts[0].strip())
                freq = int(parts[1].strip())
                nodes.append((node_id, freq))
    
    print(f"Total nodes in frequency file: {len(nodes)}")
    
    # Take top N nodes
    max_nodes=10000
    selected_nodes = nodes[:max_nodes]
    
    print(f"Selected {len(selected_nodes)} nodes for cache")
    if selected_nodes:
        print(f"Frequency range: {selected_nodes[-1][1]} - {selected_nodes[0][1]}")
    
    # Write output file (just node IDs, one per line)
    with open(output_file, 'w') as f:
        for node_id, freq in selected_nodes:
            f.write(f"{node_id}\n")
    
    print(f"Cache list written to: {output_file}")
    
    # Calculate actual cache size
    actual_size = len(selected_nodes) * bytes_per_node
    print(f"Actual cache size: {actual_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
